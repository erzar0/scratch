"""Chunk-level bandit attention for one head: chunks are arms, sampled keys are pulls.

    Changes  softmax's "read every key" step: each query chunk reads a few old
             chunks token by token and every other old chunk as one mean key
    Keeps    the q, k, v projections and one softmax per query
    Reads    chunk_reads: summaries + window + refined chunks + pulls
    Trained  nothing (zero-shot on the softmax model)

Per-key selection needs a top-k over a (T, T) score matrix (2.2 ms at T=4k, 13 ms at
8k, against 0.30 / 0.47 ms for flash). Here every query chunk picks whole key chunks
and reads only those exactly. Query chunk Q sees:

    [summaries (mean key/value) of the unrefined old chunks ;
     window chunks token by token ; refined chunks token by token]

The refined chunks come from a selector:

    bandit  round 1: first_pulls sampled keys of every old chunk; the survivors best
            by UCB on log-mass go on. Round 2: second_pulls more keys of each
            survivor; the refined_chunks best by the pooled estimate are refined.
    oracle  exact mass of every chunk (reads every key: a bound, not a method)
    random  random old chunks (baseline)

Caveat: a chunk's picks depend on all of its queries, so query i sees later tokens of
its own chunk through the selection.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ChunkBanditConfig:
    """Chunk attention settings, static under jit.

    Attributes:
        selector: How refined chunks are chosen.
        chunk: Tokens per chunk.
        refined_chunks: Old chunks read token by token per query chunk.
        window: Recent tokens read exactly (a multiple of chunk); None: 2 chunks.
        first_pulls: Sampled keys per chunk in round 1.
        second_pulls: Sampled keys per survivor in round 2.
        survivors: Chunks reaching round 2; None: 3 · refined_chunks.
        ucb_width: Weight of the pulls' std in the round-1 UCB.
        sink: Always refine chunk 0 (attention sink).
        summaries: Unrefined old chunks enter as mean key/value; False drops them.
        selection_query: "all" scores with every query of the chunk (summed mass),
            "mean" with the chunk's mean query.
        query_chunks_per_map: Query chunks per lax.map step in chunk_attn.
        impl: chunk_sparse's attention kernel: "split" (fastest on GPU), "cudnn",
            or "xla" (float64 / CPU).
    """

    selector: Literal["bandit", "oracle", "random"] = "bandit"
    chunk: int = 32
    refined_chunks: int = 8
    window: int | None = None
    first_pulls: int = 2
    second_pulls: int = 8
    survivors: int | None = None
    ucb_width: float = 1.0
    sink: bool = False
    summaries: bool = True
    selection_query: Literal["all", "mean"] = "all"
    query_chunks_per_map: int = 64
    impl: Literal["split", "cudnn", "xla"] = "cudnn"

    @property
    def window_len(self) -> int:
        """Window length in tokens."""
        return self.window or 2 * self.chunk


type Aggregate = Callable[[jax.Array, jax.Array], jax.Array]


def _chunked(a: jax.Array, chunk: int) -> jax.Array:
    """(T, d) -> (n_chunks, chunk, d), zero-padded at the end."""
    padded = -(-a.shape[0] // chunk) * chunk
    return jnp.pad(a, ((0, padded - a.shape[0]), (0, 0))).reshape(-1, chunk, a.shape[1])


def _sample_keys(
    key_chunks: jax.Array, key: jax.Array, config: ChunkBanditConfig
) -> tuple[jax.Array, jax.Array]:
    """Round-1 (n, first_pulls, d) and round-2 (n, second_pulls, d) sampled keys,
    the same positions for every query chunk."""
    n_chunks, rows = key_chunks.shape[0], jnp.arange(key_chunks.shape[0])[:, None]
    first_key, second_key = jax.random.split(key)
    first = jax.random.randint(
        first_key, (n_chunks, config.first_pulls), 0, config.chunk
    )
    second = jax.random.randint(
        second_key, (n_chunks, config.second_pulls), 0, config.chunk
    )
    return key_chunks[rows, first], key_chunks[rows, second]


def _bandit_score(
    queries: jax.Array,
    samples: tuple[jax.Array, jax.Array],
    visible: jax.Array,
    aggregate: Aggregate,
    config: ChunkBanditConfig,
) -> jax.Array:
    """Two-round bandit estimate of each chunk's log-mass for one query chunk.

    Args:
        queries: (Cs, d) selection queries.
        samples: Sampled keys from _sample_keys.
        visible: (n,) bool, old chunks the query chunk may refine.
        aggregate: (Cs, n') per-query estimates and chunk ids -> (n',) scores.
        config: Bandit settings.

    Returns:
        (n,) scores, -inf for chunks eliminated in round 1.
    """
    first_keys, second_keys = samples
    n_chunks = first_keys.shape[0]
    scale = queries.shape[-1] ** -0.5
    log_chunk = jnp.log(jnp.asarray(config.chunk, queries.dtype))
    n_survivors = min(config.survivors or 3 * config.refined_chunks, n_chunks)
    first = jnp.einsum("id,csd->ics", queries, first_keys) * scale
    log_mass = jax.nn.logsumexp(first, -1) - jnp.log(config.first_pulls) + log_chunk
    ucb = log_mass + config.ucb_width * first.std(-1)
    all_chunks = jnp.arange(n_chunks)
    masked_ucb = jnp.where(visible, aggregate(ucb, all_chunks), -jnp.inf)
    survivors = jax.lax.top_k(masked_ucb, n_survivors)[1]
    second = jnp.einsum("id,csd->ics", queries, second_keys[survivors]) * scale
    pooled = jnp.concatenate([first[:, survivors], second], -1)
    n_pulls = config.first_pulls + config.second_pulls
    estimate = jax.nn.logsumexp(pooled, -1) - jnp.log(n_pulls) + log_chunk
    score = jnp.full(n_chunks, -jnp.inf, queries.dtype)
    return score.at[survivors].set(aggregate(estimate, survivors))


@partial(jax.jit, static_argnames="config")
def chunk_attn(
    q: jax.Array, k: jax.Array, v: jax.Array, key: jax.Array, config: ChunkBanditConfig
) -> jax.Array:
    """Chunk attention with summaries, one causal head.

    Args:
        q: (T, d) queries.
        k: (T, d) keys.
        v: (T, d) values.
        key: PRNG key of the bandit's samples and the random selector.
        config: Chunk settings.

    Returns:
        (T, d) output.
    """
    seq_len, dim = q.shape
    chunk, window = config.chunk, config.window_len
    n_local = window // chunk
    query_chunks, key_chunks, value_chunks = (_chunked(a, chunk) for a in (q, k, v))
    n_chunks = query_chunks.shape[0]
    n_refined = min(config.refined_chunks, n_chunks)
    scale, chunk_ids = dim**-0.5, jnp.arange(n_chunks)
    log_chunk = jnp.log(jnp.asarray(chunk, q.dtype))
    mean_keys, mean_values = key_chunks.mean(1), value_chunks.mean(1)
    window_causal = jnp.tril(jnp.ones((chunk, window), bool), window - chunk)
    samples = _sample_keys(key_chunks, key, config)
    random_scores = jax.random.uniform(key, (n_chunks, n_chunks), q.dtype)

    def summed_mass(estimate: jax.Array, window_logits: jax.Array) -> jax.Array:
        """(C, n) per-query log-mass -> (n,) mass summed over the chunk's queries."""
        logits = jnp.concatenate([estimate, window_logits], -1)
        return jax.nn.softmax(logits, -1)[:, : estimate.shape[1]].sum(0)

    def select(
        queries: jax.Array,
        chunk_id: jax.Array,
        old: jax.Array,
        window_logits: jax.Array,
    ) -> jax.Array:
        """(n,) selection score of every chunk for one query chunk."""
        if config.selector == "random":
            return random_scores[chunk_id]
        if config.selector == "oracle":
            positions = chunk_id * chunk + jnp.arange(chunk)
            visible = positions[:, None] >= jnp.arange(n_chunks * chunk)
            logits = queries @ key_chunks.reshape(-1, dim).T * scale
            weight = jax.nn.softmax(jnp.where(visible, logits, -jnp.inf), -1)
            return weight.reshape(chunk, n_chunks, chunk).sum((0, 2))
        if config.selection_query == "all":
            return _bandit_score(
                queries,
                samples,
                old,
                lambda est, ids: summed_mass(
                    jnp.where(old[ids], est, -jnp.inf), window_logits
                ),
                config,
            )
        mean_query = queries.mean(0, keepdims=True)
        return _bandit_score(mean_query, samples, old, lambda est, ids: est[0], config)

    def one_chunk(queries: jax.Array, chunk_id: jax.Array) -> jax.Array:
        old = chunk_ids <= chunk_id - n_local
        local = chunk_id - n_local + 1 + jnp.arange(n_local)
        local_keys, local_values = (
            a[jnp.maximum(local, 0)].reshape(window, dim)
            for a in (key_chunks, value_chunks)
        )
        local_ok = window_causal & jnp.repeat(local >= 0, chunk)
        window_logits = jnp.where(local_ok, queries @ local_keys.T * scale, -jnp.inf)
        score = jnp.where(old, select(queries, chunk_id, old, window_logits), -jnp.inf)
        if config.sink:
            score = score.at[0].set(jnp.where(old[0], 1e9, -jnp.inf))
        top_score, refined = jax.lax.top_k(jax.lax.stop_gradient(score), n_refined)
        ok = top_score > -jnp.inf  # fewer old chunks than n_refined: padding
        refined_keys = key_chunks[refined].reshape(-1, dim)
        refined_logits = jnp.where(
            jnp.repeat(ok, chunk), queries @ refined_keys.T * scale, -jnp.inf
        )
        refined_values = value_chunks[refined].reshape(-1, dim)
        if not config.summaries:
            w = jax.nn.softmax(jnp.concatenate([window_logits, refined_logits], -1), -1)
            return w[:, :window] @ local_values + w[:, window:] @ refined_values
        is_refined = ((refined[:, None] == chunk_ids) & ok[:, None]).any(0)
        summary_logits = jnp.where(
            old & ~is_refined, queries @ mean_keys.T * scale + log_chunk, -jnp.inf
        )
        logits = jnp.concatenate([summary_logits, window_logits, refined_logits], -1)
        w = jax.nn.softmax(logits, -1)
        return (
            w[:, :n_chunks] @ mean_values
            + w[:, n_chunks : n_chunks + window] @ local_values
            + w[:, n_chunks + window :] @ refined_values
        )

    out = jax.lax.map(
        lambda a: one_chunk(*a),
        (query_chunks, chunk_ids),
        batch_size=config.query_chunks_per_map,
    )
    return out.reshape(-1, dim)[:seq_len]


@partial(jax.jit, static_argnames="config")
def chunk_sparse(
    q: jax.Array, k: jax.Array, v: jax.Array, key: jax.Array, config: ChunkBanditConfig
) -> jax.Array:
    """Fast path of chunk_attn with summaries=False and selection_query="mean".

    Every query chunk's picks come at once from (n, n) chunk scores, then fused
    attention calls. Same output as chunk_attn up to bf16 rounding; oracle is not
    supported.

    Args:
        q: (T, d) queries.
        k: (T, d) keys.
        v: (T, d) values.
        key: PRNG key.
        config: Chunk settings; config.impl picks the kernel.

    Returns:
        (T, d) output.

    Raises:
        ValueError: If config.selector is "oracle".
    """
    if config.selector == "oracle":
        raise ValueError("chunk_sparse has no oracle selector; use chunk_attn")
    seq_len, dim = q.shape
    chunk, n_local = config.chunk, config.window_len // config.chunk
    query_chunks, key_chunks, value_chunks = (_chunked(a, chunk) for a in (q, k, v))
    n_chunks = query_chunks.shape[0]
    n_refined = min(config.refined_chunks, n_chunks)
    chunk_ids = jnp.arange(n_chunks)

    def at_least_f32(a: jax.Array) -> jax.Array:
        # bf16 scores rank chunks badly.
        return a.astype(jnp.promote_types(a.dtype, jnp.float32))

    mean_queries = at_least_f32(query_chunks).mean(1)
    old = chunk_ids[None, :] <= chunk_ids[:, None] - n_local  # (query, key chunk)
    if config.selector == "bandit":
        first, second = _sample_keys(key_chunks, key, config)
        samples = (at_least_f32(first), at_least_f32(second))
        score = jax.vmap(
            lambda query, visible: _bandit_score(
                query[None], samples, visible, lambda est, ids: est[0], config
            )
        )(mean_queries, old)
    else:
        score = jax.random.uniform(key, (n_chunks, n_chunks), mean_queries.dtype)
    score = jnp.where(old, score, -jnp.inf)
    if config.sink:
        score = score.at[:, 0].set(jnp.where(old[:, 0], 1e9, -jnp.inf))
    top_score, refined = jax.lax.top_k(jax.lax.stop_gradient(score), n_refined)
    if config.impl == "split":
        out = _split_attention(query_chunks, key_chunks, value_chunks, refined, n_local)
        return out[:seq_len]
    window = config.window_len
    local = chunk_ids[:, None] - n_local + 1 + jnp.arange(n_local)
    chunk_index = jnp.concatenate([jnp.maximum(local, 0), refined], 1)
    keys, values = (
        a[chunk_index].reshape(n_chunks, -1, dim) for a in (key_chunks, value_chunks)
    )
    window_causal = jnp.tril(jnp.ones((chunk, window), bool), window - chunk)
    local_mask = window_causal[None] & jnp.repeat(local >= 0, chunk, 1)[:, None]
    refined_ok = jnp.repeat(top_score > -jnp.inf, chunk, 1)[:, None]
    refined_mask = jnp.broadcast_to(refined_ok, (n_chunks, chunk, n_refined * chunk))
    mask = jnp.concatenate([local_mask, refined_mask], -1)
    if config.impl == "cudnn":
        out = jax.nn.dot_product_attention(
            query_chunks[:, :, None],
            keys[:, :, None],
            values[:, :, None],
            mask=mask[:, None],
            implementation="cudnn",
        )
    else:  # written out: dot_product_attention's xla path runs float64 in float32
        logits = jnp.einsum("ncd,nkd->nck", query_chunks, keys) * dim**-0.5
        out = jax.nn.softmax(jnp.where(mask, logits, -jnp.inf), -1) @ values
    return out.reshape(-1, dim)[:seq_len]


def _flash(
    q: jax.Array, k: jax.Array, v: jax.Array, causal: bool
) -> tuple[jax.Array, jax.Array]:
    """cuDNN attention over (batch, T, d) with one head; returns output and lse."""
    return jax.nn.dot_product_attention(
        q[:, :, None],
        k[:, :, None],
        v[:, :, None],
        is_causal=causal,
        implementation="cudnn",
        return_residual=True,
    )


def _split_attention(
    query_chunks: jax.Array,
    key_chunks: jax.Array,
    value_chunks: jax.Array,
    refined: jax.Array,
    n_local: int,
) -> jax.Array:
    """chunk_sparse's "split" kernel: no mask tensors.

    Chunks with at most refined_chunks old chunks are plain causal attention (one
    flash call on that prefix). Every later query chunk is two fused calls, [its
    refined chunks] and causal [its window chunks], merged by log-sum-exp. Forward
    only: the lse residual carries no gradient.

    Returns:
        (n_chunks · chunk, d) output.
    """
    n_chunks, chunk, dim = query_chunks.shape
    n_refined, window = refined.shape[1], n_local * chunk
    exact = min(n_refined + n_local, n_chunks)
    prefix_q, prefix_k, prefix_v = (
        a[:exact].reshape(1, exact * chunk, dim)
        for a in (query_chunks, key_chunks, value_chunks)
    )
    out_prefix = _flash(prefix_q, prefix_k, prefix_v, True)[0]
    out_prefix = out_prefix.reshape(exact * chunk, dim)
    if exact == n_chunks:
        return out_prefix
    later = jnp.arange(exact, n_chunks)
    n_later = n_chunks - exact
    refined_k, refined_v = (
        a[refined[later]].reshape(n_later, n_refined * chunk, dim)
        for a in (key_chunks, value_chunks)
    )
    out_refined, lse_refined = _flash(query_chunks[later], refined_k, refined_v, False)
    local = later[:, None] - n_local + 1 + jnp.arange(n_local)
    local_q, local_k, local_v = (
        a[local].reshape(n_later, window, dim)
        for a in (query_chunks, key_chunks, value_chunks)
    )
    out_local, lse_local = _flash(local_q, local_k, local_v, True)
    out_local, lse_local = (
        out_local[:, window - chunk :],
        lse_local[:, window - chunk :],
    )
    lse_refined = lse_refined.astype(jnp.float32)
    lse_local = lse_local.astype(jnp.float32)
    top = jnp.maximum(lse_refined, lse_local)
    w_refined = jnp.exp(lse_refined - top)[..., None]
    w_local = jnp.exp(lse_local - top)[..., None]
    merged = (out_refined * w_refined + out_local * w_local) / (w_refined + w_local)
    return jnp.concatenate(
        [out_prefix, merged.astype(query_chunks.dtype).reshape(-1, dim)]
    )


def chunk_reads(seq_len: int, config: ChunkBanditConfig) -> float:
    """Mean q·k dot products per query of chunk_attn; exact causal reads (T + 1) / 2.

    Counts the visible window keys, the refined tokens, the summaries of the other
    old chunks, and the selector: oracle reads every visible key, bandit its pulls
    (per query with selection_query "all", per query chunk with "mean"), random none.

    Args:
        seq_len: Sequence length T.
        config: Chunk settings.

    Returns:
        Reads per query, averaged over the T queries.
    """
    chunk, window = config.chunk, config.window_len
    n_chunks = -(-seq_len // chunk)
    position = np.arange(seq_len)
    n_old = np.maximum(position // chunk - window // chunk + 1, 0)
    refined = np.minimum(min(config.refined_chunks, n_chunks), n_old)
    window_keys = np.minimum(position + 1, window - chunk + position % chunk + 1)
    reads = (window_keys + refined * chunk).astype(np.float64)
    if config.summaries:
        reads += n_old - refined
    if config.selector == "oracle":
        reads += position + 1
    elif config.selector == "bandit":
        survivors = min(config.survivors or 3 * config.refined_chunks, n_chunks)
        pulls = config.first_pulls * n_old
        pulls += config.second_pulls * np.minimum(survivors, n_old)
        reads += pulls if config.selection_query == "all" else pulls / chunk
    return float(reads.mean())
