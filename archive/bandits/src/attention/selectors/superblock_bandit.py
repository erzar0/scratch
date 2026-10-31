"""Superblock bandit: double indexing whose coarse step is a two-round bandit.

    Changes  CSA's select step, like double indexing (double_indexing.py): the
             indexer scores only the blocks of s kept superblocks and of the
             query's own. Only the coarse step differs: no summary keys, the
             superblocks are bandit arms and a pull scores one of their block keys
    Keeps    everything else; the fine step is double indexing's
    Reads    first_pulls · G + survivors · second_pulls + (s + 1) · S block keys,
             instead of n
    Trained  nothing, and no extra cache: pulls read the indexer keys

An arm's value is its best block score (what the fine top_k wants). The two rounds
come from attention/dense/chunk_bandit.py:

    round 1  first_pulls pulls of every complete superblock; the survivors best by
             UCB = best pull + ucb_width · std of the pulls go on
    round 2  second_pulls more pulls of each survivor; the kept_superblocks best by
             their best pull are kept

Blocks are pulled in a fixed random order per superblock (pull_order), a function of
the superblock index only, so prefill and decode pull the same blocks.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from attention.csa import (
    IndexerInputs,
    Params,
    Picks,
    score_candidates,
    score_keys,
)
from attention.selectors import Selection, indexer_view
from attention.selectors.double_indexing import (
    pick_in_superblocks,
    superblock_candidates,
)


@dataclass(frozen=True)
class SuperblockBandit:
    """Superblock bandit settings and selector.

    Attributes:
        superblock_size: Blocks per superblock (S).
        kept_superblocks: Superblocks kept for the fine level (s).
        first_pulls: Pulls per superblock in round 1.
        survivors: Superblocks that reach round 2.
        second_pulls: Extra pulls per survivor in round 2.
        ucb_width: Weight of the pulls' std in the round-1 UCB.
        seed: Seed of the per-superblock pull order.

    Raises:
        ValueError: If pulls are not distinct blocks of one superblock.
    """

    superblock_size: int
    kept_superblocks: int
    first_pulls: int
    survivors: int
    second_pulls: int
    ucb_width: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.first_pulls <= self.superblock_size - self.second_pulls:
            raise ValueError(
                "need 1 <= first_pulls and first_pulls + second_pulls <= S"
            )

    def select(self, params: Params, x: jax.Array) -> Selection:
        """Blocks of the kept superblocks and of each query's own; reads."""
        indexer, finished = indexer_view(params, x)
        kept = bandit_superblocks(indexer, finished, self)
        candidates = superblock_candidates(
            kept, finished, self.superblock_size, indexer.block_keys.shape[0]
        )
        return Selection(candidates, reads(finished, self))


def pull_order(n_superblocks: int, superblock_size: int, seed: int) -> jax.Array:
    """(G, S): the order in which each superblock's blocks get pulled."""
    if n_superblocks == 0:
        return jnp.zeros((0, superblock_size), jnp.int32)
    base = jax.random.PRNGKey(seed)
    return jax.vmap(
        lambda g: jax.random.permutation(jax.random.fold_in(base, g), superblock_size)
    )(jnp.arange(n_superblocks))


def bandit_superblocks(
    indexer: IndexerInputs, finished_blocks: jax.Array, route: SuperblockBandit
) -> jax.Array:
    """Run both bandit rounds for every query.

    Args:
        indexer: Indexer inputs.
        finished_blocks: (T,) blocks finished before each query's block.
        route: Bandit settings.

    Returns:
        (T, min(s, survivors, G)) kept superblock ids; ids at or past the query's
        complete superblocks are padding.

    Example:
        >>> route = SuperblockBandit(16, 2, 2, 6, 4)
        >>> kept = bandit_superblocks(indexer, jnp.arange(T) // 4, route)
    """
    query, head_weight, block_keys = jax.lax.stop_gradient(tuple(indexer))
    size, seq_len = route.superblock_size, query.shape[0]
    n_superblocks = block_keys.shape[0] // size
    if n_superblocks == 0 or route.kept_superblocks == 0:
        return jnp.zeros((seq_len, 0), jnp.int32)
    complete = finished_blocks // size
    order = pull_order(n_superblocks, size, route.seed)
    grouped = block_keys[: n_superblocks * size].reshape(n_superblocks, size, -1)
    first_keys = jnp.take_along_axis(grouped, order[:, : route.first_pulls, None], 1)
    first = score_keys(query, head_weight, first_keys.reshape(-1, grouped.shape[-1]))
    first = first.reshape(seq_len, n_superblocks, route.first_pulls)
    ucb = first.max(-1) + route.ucb_width * first.std(-1)
    visible = jnp.arange(n_superblocks) < complete[:, None]
    n_survivors = min(route.survivors, n_superblocks)
    survivors = jax.lax.top_k(jnp.where(visible, ucb, -jnp.inf), n_survivors)[1]

    pulled = order[
        survivors, route.first_pulls : route.first_pulls + route.second_pulls
    ]
    second_blocks = survivors[..., None] * size + pulled  # (T, survivors, second)
    second_keys = block_keys[second_blocks.reshape(seq_len, -1)]
    second = score_candidates(query, head_weight, second_keys)
    second = second.reshape(seq_len, n_survivors, route.second_pulls)
    best_first = jnp.take_along_axis(first.max(-1), survivors, 1)
    best = jnp.maximum(best_first, second.max(-1, initial=-jnp.inf))
    best = jnp.where(survivors < complete[:, None], best, -jnp.inf)
    keep = jax.lax.top_k(best, min(route.kept_superblocks, n_survivors))[1]
    return jnp.take_along_axis(survivors, keep, 1)


def pick(
    indexer: IndexerInputs,
    finished_blocks: jax.Array,
    top_k: int,
    route: SuperblockBandit,
) -> Picks:
    """Bandit rounds, then double indexing's fine step scoring only the candidates.

    Args:
        indexer: Indexer inputs.
        finished_blocks: (T,) blocks finished before each query's block.
        top_k: Blocks to pick.
        route: Bandit settings.

    Returns:
        The picks, as from csa.pick_flat.
    """
    kept = bandit_superblocks(indexer, finished_blocks, route)
    return pick_in_superblocks(
        indexer, kept, finished_blocks, top_k, route.superblock_size
    )


def reads(finished_blocks: jax.Array, route: SuperblockBandit) -> jax.Array:
    """Indexer keys a query scores: first_pulls per complete superblock,
    second_pulls per survivor, then every block of the kept superblocks and the
    finished blocks of its own superblock."""
    size = route.superblock_size
    complete = finished_blocks // size
    return (
        route.first_pulls * complete
        + jnp.minimum(route.survivors, complete) * route.second_pulls
        + jnp.minimum(route.kept_superblocks, complete) * size
        + finished_blocks % size
    )
