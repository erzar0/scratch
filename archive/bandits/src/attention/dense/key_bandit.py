"""Per-key bandit attention: successive elimination on partial dot products.

    Changes  softmax's "read every key" step: keys that cannot matter are dropped
             after reading a few of their channels
    Keeps    the q, k, v projections and one softmax over the survivors
    Reads    channel reads in exact-logit units, data-dependent
    Trained  nothing zero-shot; the "bandit" mixer trains a model with it inside

Keys are arms. Each round reads dims_per_round more channels of q·k (in a random
channel order) for every key still active, estimates the full logit from the partial
sum with a confidence radius from the partial squares, and drops keys whose upper
bound falls more than `tolerance` below the best lower bound. Survivors get the exact
logit and one softmax. Dense emulation on (T, T) matrices: it reproduces the
selection and counts its reads, it is not faster.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class KeyBanditConfig:
    """Successive-elimination settings.

    Attributes:
        ucb_width: Width of the confidence radius in standard errors.
        tolerance: Logit slack: keys within it of the best lower bound stay.
        dims_per_round: Channels read per round.
    """

    ucb_width: float = 2.0
    tolerance: float = 3.0
    dims_per_round: int = 8


def causal_mask(seq_len: int) -> jax.Array:
    """(T, T) bool, True where key j <= query i."""
    return jnp.tril(jnp.ones((seq_len, seq_len), bool))


def scaled_logits(q: jax.Array, k: jax.Array) -> jax.Array:
    """(T, d), (S, d) -> (T, S) logits q·k / sqrt(d)."""
    return q @ k.T / q.shape[-1] ** 0.5


def masked_softmax(logits: jax.Array, mask: jax.Array) -> jax.Array:
    """Row softmax over the True entries of mask; every row needs at least one."""
    return jax.nn.softmax(jnp.where(mask, logits, -jnp.inf), axis=-1)


def exact_causal(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """Exact causal softmax attention for one head, (T, d) each -> (T, d)."""
    return masked_softmax(scaled_logits(q, k), causal_mask(len(q))) @ v


def dot_bounds(
    partial_dot: jax.Array,
    partial_sq: jax.Array,
    n_read: int,
    dim: int,
    ucb_width: float,
) -> tuple[jax.Array, jax.Array]:
    """Estimate of a scaled dot product from n_read of its dim channels.

    Args:
        partial_dot: Sum of q_c·k_c / sqrt(dim) over the channels read.
        partial_sq: Sum of q_c²·k_c² / dim over the channels read.
        n_read: Channels read so far.
        dim: Total channels.
        ucb_width: Radius in standard errors of the per-channel mean.

    Returns:
        The estimate of the full logit and its confidence radius.
    """
    estimate = partial_dot * dim / n_read
    variance = jnp.maximum(partial_sq / n_read * dim - estimate**2 / dim, 1e-12)
    radius = ucb_width * jnp.sqrt(variance * dim) / n_read**0.5
    return estimate, radius


def key_bandit_cost(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    channel_order: jax.Array,
    config: KeyBanditConfig,
) -> tuple[jax.Array, jax.Array]:
    """Per-key successive elimination for one causal head.

    Args:
        q: (T, d) queries.
        k: (T, d) keys.
        v: (T, d) values.
        channel_order: (d,) permutation of the channels, the order they are read.
        config: Elimination settings.

    Returns:
        Output (T, d) and the reads per query in exact-logit equivalents.
    """
    seq_len, dim = q.shape
    step = config.dims_per_round
    active = causal_mask(seq_len)
    partial_dot = partial_sq = jnp.zeros((seq_len, seq_len), q.dtype)
    cost = jnp.zeros((), q.dtype)
    # The last round is skipped: survivors get the full dot product instead.
    for start in range(0, dim - step, step):
        channels = channel_order[start : start + step]
        cost += active.sum() * step / dim
        partial_dot += q[:, channels] @ k[:, channels].T / dim**0.5
        partial_sq += (q[:, channels] ** 2) @ (k[:, channels] ** 2).T / dim
        estimate, radius = dot_bounds(
            partial_dot, partial_sq, start + step, dim, config.ucb_width
        )
        lower = jnp.where(active, estimate - radius, -jnp.inf)
        active &= estimate + radius >= lower.max(1, keepdims=True) - config.tolerance
    cost += active.sum() * step / dim
    return masked_softmax(scaled_logits(q, k), active) @ v, cost / seq_len


def key_bandit(
    q: jax.Array, k: jax.Array, v: jax.Array, key: jax.Array, config: KeyBanditConfig
) -> jax.Array:
    """key_bandit_cost's output, with a channel order drawn from key."""
    order = jax.random.permutation(key, q.shape[-1])
    return key_bandit_cost(q, k, v, order, config)[0]
