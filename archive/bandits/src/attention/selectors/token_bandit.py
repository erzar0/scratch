"""Token bandit: replaces CSA's indexer with the main query and sampled raw tokens.

    Changes  CSA's select step: the indexer is not used at all; the bandit picks
             the top_k candidates itself, so the indexer's top_k keeps them all
    Keeps    compression, window and attention
    Reads    first_pulls · n + survivors · second_pulls token keys, at
             main-attention cost each
    Trained  nothing

attention/dense/chunk_bandit.py's two rounds applied to CSA blocks. An arm is a
compressed block; a pull is the main attention's query (every head) against one raw
token key inside the block, as the sliding window reads it (window_kv, KV norm,
RoPE). A block's value is its estimated log attention mass, summed over heads.

    round 1  first_pulls tokens of every finished block; the survivors best by
             UCB = log-mass + ucb_width · std of the pull logits go on
    round 2  second_pulls more tokens of each survivor; the top_k by the pooled
             log-mass estimate are the candidates

Tokens are pulled in a fixed random order per block, a function of the block index
only. Caveat: a decoder would have to keep the raw KV entry of every token, which is
the memory CSA's compression exists to avoid.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from attention.csa import (
    Params,
    attention_keys,
    attention_queries,
    block_size_of,
    pad_to_blocks,
)
from attention.selectors import (
    QUERIES_PER_MAP,
    Selection,
    main_read_cost,
    top_candidates,
)


@dataclass(frozen=True)
class TokenBandit:
    """Token bandit settings and selector.

    Attributes:
        top_k: Candidate blocks per query.
        first_pulls: Tokens per block in round 1.
        second_pulls: Tokens per survivor in round 2.
        survivors: Blocks reaching round 2.
        ucb_width: Weight of the pull logits' std in the round-1 UCB.
        seed: Seed of the per-block pull order.
    """

    top_k: int = 8
    first_pulls: int = 1
    second_pulls: int = 1
    survivors: int = 16
    ucb_width: float = 1.0
    seed: int = 0

    def select(self, params: Params, x: jax.Array) -> Selection:
        """The top_k blocks by estimated attention mass, and the pulls' reads.

        Raises:
            ValueError: If pulls are not distinct tokens of one block, or fewer
                survivors than top_k.
        """
        block_size = block_size_of(params)
        if not 1 <= self.first_pulls <= block_size - self.second_pulls:
            raise ValueError(
                "need 1 <= first_pulls and first_pulls + second_pulls <= block"
            )
        if self.survivors < self.top_k:
            raise ValueError("need survivors >= top_k")
        x = pad_to_blocks(x, block_size)
        positions = jnp.arange(x.shape[0])
        finished = positions // block_size
        n_blocks = x.shape[0] // block_size
        first_pulls, second_pulls = self.first_pulls, self.second_pulls
        token_keys = attention_keys(params, x @ params["window_kv"], positions)
        grouped = token_keys.reshape(n_blocks, block_size, -1)
        order = token_order(n_blocks, block_size, self.seed)
        first_keys = jnp.take_along_axis(grouped, order[:, :first_pulls, None], 1)
        queries = attention_queries(params, x @ params["query_down"], positions)
        scale = grouped.shape[-1] ** -0.5
        log_block = jnp.log(block_size)
        n_survivors = min(self.survivors, n_blocks)

        def block_score(args: tuple[jax.Array, jax.Array]) -> jax.Array:
            query, n_finished = args
            visible = jnp.arange(n_blocks) < n_finished
            first = jnp.einsum("hc,nsc->nhs", query, first_keys) * scale
            log_mass = (
                jax.nn.logsumexp(first, (1, 2)) - jnp.log(first_pulls) + log_block
            )
            ucb = log_mass + self.ucb_width * first.std((1, 2))
            survivors = jax.lax.top_k(jnp.where(visible, ucb, -jnp.inf), n_survivors)[1]
            pulled = order[survivors, first_pulls : first_pulls + second_pulls]
            second_keys = grouped[survivors[:, None], pulled]
            second = jnp.einsum("hc,msc->mhs", query, second_keys) * scale
            pooled = jnp.concatenate([first[survivors], second], -1)
            n_pulls = first_pulls + second_pulls
            estimate = jax.nn.logsumexp(pooled, (1, 2)) - jnp.log(n_pulls) + log_block
            estimate = jnp.where(visible[survivors], estimate, -jnp.inf)
            return jnp.full(n_blocks, -jnp.inf, query.dtype).at[survivors].set(estimate)

        score = jax.lax.map(
            block_score, (queries, finished), batch_size=QUERIES_PER_MAP
        )
        mask = top_candidates(score, finished, self.top_k)
        pulls = (
            first_pulls * finished + jnp.minimum(n_survivors, finished) * second_pulls
        )
        return Selection(mask, pulls * main_read_cost(params))


def token_order(n_blocks: int, block_size: int, seed: int) -> jax.Array:
    """(n_blocks, block_size): the order in which each block's tokens get pulled."""
    base = jax.random.PRNGKey(seed)
    return jax.vmap(
        lambda b: jax.random.permutation(jax.random.fold_in(base, b), block_size)
    )(jnp.arange(n_blocks))
