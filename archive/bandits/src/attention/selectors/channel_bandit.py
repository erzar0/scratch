"""Channel bandit: CSA's indexer reads each block key a few channels at a time.

    Changes  CSA's select step: blocks that provably cannot reach the top_k are
             dropped before their key is read in full; the survivors are the
             candidates. Single level, no superblocks
    Keeps    everything else, including every indexer weight
    Reads    channel reads in full-key units, data-dependent, at most n
    Trained  nothing

Every round reads dims_per_round more indexer channels (in a fixed random order) of
every block still active. Per indexer head the dot product q_h·k_b is estimated from
the partial sum, with a confidence radius from the partial squares: the per-key
bandit's bound (attention/dense/key_bandit.py dot_bounds). That bounds the block's
score Σ_h w_h·relu(q_h·k_b) from above and below. A block is eliminated when its
upper bound falls below the top_k-th best lower bound minus tolerance.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from attention.csa import IndexerInputs, Params
from attention.dense.key_bandit import dot_bounds
from attention.selectors import QUERIES_PER_MAP, Selection, indexer_view


@dataclass(frozen=True)
class ChannelBandit:
    """Channel bandit settings and selector.

    Attributes:
        top_k: Blocks csa_attention will pick (the elimination threshold's rank).
        ucb_width: Confidence radius in standard errors.
        tolerance: Score slack below the top_k-th best lower bound.
        dims_per_round: Indexer channels read per round.
        seed: Seed of the channel order.
    """

    top_k: int = 8
    ucb_width: float = 1.0
    tolerance: float = 0.0
    dims_per_round: int = 4
    seed: int = 0

    def select(self, params: Params, x: jax.Array) -> Selection:
        """The surviving blocks and the channel reads."""
        indexer, finished = indexer_view(params, x)
        return Selection(*channel_bandit_candidates(indexer, finished, self))


def channel_bandit_candidates(
    indexer: IndexerInputs, finished_blocks: jax.Array, config: ChannelBandit
) -> tuple[jax.Array, jax.Array]:
    """Run successive elimination for every query.

    Args:
        indexer: Indexer inputs.
        finished_blocks: (T,) blocks finished before each query's block.
        config: Bandit settings.

    Returns:
        Surviving blocks (T, n_blocks) and reads per query in full-key units.

    Example:
        >>> config = ChannelBandit()
        >>> survivors, reads = channel_bandit_candidates(indexer, finished, config)
    """
    query, head_weight, keys = jax.lax.stop_gradient(tuple(indexer))
    n_heads, dim = query.shape[1:]
    n_blocks, step = keys.shape[0], config.dims_per_round
    order = jax.random.permutation(jax.random.PRNGKey(config.seed), dim)
    # dot_bounds works in q·k / sqrt(dim); this converts to Eq. 16 score units.
    unit = dim**0.5 * float(n_heads * dim) ** -0.5
    k = min(config.top_k, n_blocks)

    def one_query(
        args: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        q, weight, n_finished = args
        active = jnp.arange(n_blocks) < n_finished
        positive = (weight >= 0)[:, None]
        weight = weight[:, None]
        partial_dot = partial_sq = jnp.zeros((n_heads, n_blocks), q.dtype)
        reads = jnp.zeros((), q.dtype)
        # The last round is skipped: survivors get the full score in csa_attention.
        for start in range(0, dim - step, step):
            channels = order[start : start + step]
            reads += active.sum() * step / dim
            partial_dot += q[:, channels] @ keys[:, channels].T / dim**0.5
            partial_sq += (q[:, channels] ** 2) @ (keys[:, channels] ** 2).T / dim
            estimate, radius = dot_bounds(
                partial_dot, partial_sq, start + step, dim, config.ucb_width
            )
            low, high = jax.nn.relu(estimate - radius), jax.nn.relu(estimate + radius)
            upper = jnp.where(positive, weight * high, weight * low).sum(0) * unit
            lower = jnp.where(positive, weight * low, weight * high).sum(0) * unit
            kth_lower = jax.lax.top_k(jnp.where(active, lower, -jnp.inf), k)[0][-1]
            active &= upper >= kth_lower - config.tolerance
        return active, reads + active.sum() * step / dim

    return jax.lax.map(
        one_query, (query, head_weight, finished_blocks), batch_size=QUERIES_PER_MAP
    )
