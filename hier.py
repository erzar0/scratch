"""Hierarchical chunk attention in pure JAX, a token mixer for transformer.py.

Ported from modded-nanogpt (track_1_short/model/hier_attention.py).

Idea: cut the sequence into chunks. Recent tokens are read exactly; older chunks are first read through a
cheap one-token summary each, and the chunks that look relevant through their summary are then re-read
token by token ("refined"). Nothing is dropped: an unrefined chunk still contributes through its summary.
Cost per query: n_chunks summaries + window local + n_refined·chunk_size refined keys, vs T.

Two variants (mixers in transformer.py), the same code with a different window:
- hierarchical only ("hier", window=None): the exact part is just the previous and the current chunk
- window + hierarchical ("hier_window", e.g. window=512): a sliding window of whole chunks is exact, and
  the hierarchy covers only what lies before it"""
import jax
import jax.numpy as jnp
from jax.experimental.compute_on import compute_on


def hier_attention(params, x, n_heads, chunk_size=32, n_refined=8, window=None, chunks_per_block=64):
    """Hierarchical chunk attention, one document per sequence. x: (T, D) -> (T, D).
    Query i in chunk Q sees the last L = window // chunk_size chunks (Q-L+1 ... Q) exactly (causal): a sliding
    window of window - chunk_size + 1 to window tokens. Every older chunk is seen through one summary token
    (mean key/value, logit + log chunk_size), except the n_refined chunks with the most summary mass (summed
    over Q's queries), which it sees token by token. One exact softmax over that mixed key set.
    window=None means 2 · chunk_size, the original design (chunks Q-1 and Q); window >= T is exact attention.
    Caveat (same as the original): Q's picks depend on all of Q's queries, so query i's output depends on
    later tokens of its own chunk through which chunks get refined.
    Memory: query chunks run chunks_per_block at a time, and the backward recomputes each block instead of
    storing it, so memory is O(T) at a fixed block size. Compute is still O(T² / chunk_size)."""
    seq_len, d_model = x.shape
    window = window or 2 * chunk_size
    assert window % chunk_size == 0 and window > 0, "the window is made of whole chunks"
    n_local = window // chunk_size  # chunks read exactly
    # Round T up to whole chunks; zero padding at the end is only visible to the (discarded) padded rows
    padded_len = -(-seq_len // chunk_size) * chunk_size
    x = jnp.pad(x, ((0, padded_len - seq_len), (0, 0)))
    n_chunks, head_dim = padded_len // chunk_size, d_model // n_heads
    n_refined = min(n_refined, n_chunks)
    # Same q, k, v projections as standard attention (transformer.attention), laid out by chunk
    query, key, value = (a.reshape(n_chunks, chunk_size, n_heads, head_dim)
                         for a in jnp.split(x @ params["qkv"], 3, axis=-1))
    scale, chunk = head_dim**-0.5, jnp.arange(n_chunks)
    # Summaries: each chunk as one token, with the mean key and the mean value of its tokens.
    # The mean key gives the exact mean logit (q·mean(k) = mean(q·k)), but exp of a mean is at most the
    # mean of exps, so a summary underrates a chunk with one sharply matching key. Refinement fixes that.
    key_mean, value_mean = key.mean(1), value.mean(1)  # (chunks, heads, head_dim)
    # Head-major copies for gathering a head's refined chunks: (heads, chunks, chunk_size, head_dim)
    key_by_head, value_by_head = key.transpose(2, 0, 1, 3), value.transpose(2, 0, 1, 3)
    # Local keys = chunks q-L+1 ... q in order: key j of L·chunk_size is visible to query i iff j <= i + (L-1)·chunk_size
    causal = jnp.tril(jnp.ones((chunk_size, window), bool), window - chunk_size)

    def one_chunk(query_q, q):
        """Output of query chunk q (scalar index), query_q: (chunk_size, heads, head_dim).
        Everything a query chunk needs is its own queries plus K/V, so chunks are computed independently.
        Logit shapes below: (heads, query within chunk, key slot)."""
        # --- Summaries of chunks older than the window. + log chunk_size: a chunk of identical keys weighs
        # what its tokens would. Chunks q-L+1 ... q are read exactly below, so only chunks c <= q-L appear as
        # summaries. A summary covers a whole chunk, so it must be entirely in the past: c < q guarantees that.
        summary_logits = jnp.einsum("ihd,chd->hic", query_q, key_mean) * scale + jnp.log(chunk_size)
        summary_logits = jnp.where(chunk <= q - n_local, summary_logits, -jnp.inf)

        # --- Local: the window's chunks q-L+1 ... q, token by token (ordinary causal attention).
        # Keeping recent chunks exact means a query at the start of its chunk still sees its recent past.
        # Near the start, chunks before 0 don't exist: they are clamped to chunk 0 here and masked out.
        local_chunk = q - n_local + 1 + jnp.arange(n_local)
        local_key = key[jnp.maximum(local_chunk, 0)].reshape(window, n_heads, head_dim)
        local_value = value[jnp.maximum(local_chunk, 0)].reshape(window, n_heads, head_dim)
        exists = jnp.repeat(local_chunk >= 0, chunk_size)  # per local key
        local_logits = jnp.einsum("ihd,jhd->hij", query_q, local_key) * scale
        local_logits = jnp.where(causal & exists, local_logits, -jnp.inf)

        # --- Selection: which old chunks deserve a token-by-token read?
        # Run the cheap attention (summaries + local) once and see how much weight each summary gets. Weight
        # is summed over all queries of the chunk, so the whole chunk shares one set of picks (one gather per
        # chunk instead of per query); that sharing is also where the in-chunk leak in the docstring comes from.
        # Per head, keep the n_refined chunks with the most mass. Discrete: no gradient.
        probs = jax.nn.softmax(jnp.concatenate([summary_logits, local_logits], -1), -1)
        summary_mass = jax.lax.stop_gradient(probs[..., :n_chunks].sum(1))  # (heads, chunks)
        top_mass, refined_chunk = jax.lax.top_k(summary_mass, n_refined)  # (heads, n_refined)
        refined_valid = top_mass > 0  # zero mass = invisible chunk (fewer than n_refined old chunks)
        is_refined = ((refined_chunk[..., None] == chunk) & refined_valid[..., None]).any(1)  # (heads, chunks)

        # --- Refined chunks: their raw keys, scored by their own logits
        # Gather the picked chunks' tokens per head: (heads, n_refined, chunk_size, head_dim)
        head = jnp.arange(n_heads)[:, None]
        refined_key, refined_value = key_by_head[head, refined_chunk], value_by_head[head, refined_chunk]
        refined_logits = jnp.einsum("ihd,hsjd->hisj", query_q, refined_key) * scale  # s: pick, j: its token
        refined_logits = jnp.where(refined_valid[:, None, :, None], refined_logits, -jnp.inf)
        # Flatten (pick, token) into one key axis so it can join the other logits
        refined_logits = refined_logits.reshape(n_heads, chunk_size, -1)
        refined_value = refined_value.reshape(n_heads, n_refined * chunk_size, head_dim)

        # --- One softmax over [unrefined summaries ; local tokens ; refined tokens]
        # A refined chunk's summary is removed, so its tokens aren't counted twice.
        summary_logits = jnp.where(is_refined[:, None], -jnp.inf, summary_logits)
        # Every query sees at least itself, so no row is all -inf (no NaN from the softmax)
        attn = jax.nn.softmax(jnp.concatenate([summary_logits, local_logits, refined_logits], -1), -1)
        attn_summary, attn_local = attn[..., :n_chunks], attn[..., n_chunks : n_chunks + window]
        attn_refined = attn[..., n_chunks + window :]
        # Each part of the weights multiplies its own values; the sum is the attention output
        return (jnp.einsum("hic,chd->ihd", attn_summary, value_mean)
                + jnp.einsum("hij,jhd->ihd", attn_local, local_value)
                + jnp.einsum("hij,hjd->ihd", attn_refined, refined_value))

    # chunks_per_block query chunks at a time (vmap within a block, a loop over blocks): only one block's
    # logits, of shape (block tokens, n_chunks) instead of (T, n_chunks), exist at once. checkpoint: the
    # backward recomputes a block's forward instead of keeping its logits and gathered K/V for every block.
    out = jax.lax.map(lambda args: jax.checkpoint(one_chunk)(*args), (query, chunk), batch_size=chunks_per_block)
    # Merge heads back to (T, D), drop the padding rows, then the output projection
    return out.reshape(padded_len, d_model)[:seq_len] @ params["proj"]


def hier_cache(params, context, n_heads, chunk_size=32, window=None, offload=False):
    """Decode cache for the tokens so far, context: (T, D) layer inputs, for the next token at position T.
    With Q = T // chunk_size and L = window // chunk_size (window=None: 2 · chunk_size, as in hier_attention):
    - key_mean / value_mean: mean key/value of every complete chunk
    - chunk_mass: summary mass summed over the current chunk's queries so far (starts at 0)
    - recent_key / recent_value: raw keys/values from the start of chunk Q-L+1 on (the window)
    - old_key / old_value: raw keys/values of chunks older than the window, read only when refined. Head-major
      and flattened to (heads · old chunks, chunk_size, head_dim), so one row = one head's chunk.
    offload=True keeps old_* in pinned host RAM: the GPU then holds only the summaries and the recent tokens
    (about 1/chunk_size of a full KV cache), and each step copies just the picked chunks over.
    ponytail: built from scratch; a real decoder appends one key/value per token and one mean per chunk."""
    seq_len, d_model = context.shape
    head_dim, window = d_model // n_heads, window or 2 * chunk_size
    _, key, value = (a.reshape(seq_len, n_heads, head_dim) for a in jnp.split(context @ params["qkv"], 3, axis=-1))
    n_complete = seq_len // chunk_size
    n_old = max(n_complete - window // chunk_size + 1, 0)  # chunks only reachable as summaries or refined
    start = n_old * chunk_size
    chunk_mean = lambda a: a[: n_complete * chunk_size].reshape(n_complete, chunk_size, n_heads, head_dim).mean(1)
    by_row = lambda a: (a[:start].reshape(n_old, chunk_size, n_heads, head_dim).transpose(2, 0, 1, 3)
                        .reshape(n_heads * n_old, chunk_size, head_dim))
    old_key, old_value = by_row(key), by_row(value)
    if offload:
        old_key, old_value = jax.device_put((old_key, old_value), jax.memory.Space.Host)
    return {"key_mean": chunk_mean(key), "value_mean": chunk_mean(value),
            "chunk_mass": jnp.zeros((n_heads, n_complete), context.dtype),
            "recent_key": key[start:], "recent_value": value[start:], "old_key": old_key, "old_value": old_value}


@compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Device)
def _rows_on_host(table, rows):
    return table[rows]


def fetch_rows(table, rows):
    """table[rows]. A table in host memory is gathered on the host, so only the picked rows reach the GPU
    (copying the whole table over first would need it on the GPU after all)."""
    if jax.typeof(table).memory_space == jax.memory.Space.Host:
        return _rows_on_host(table, jax.device_put(rows, jax.memory.Space.Host))
    return table[rows]


def hier_decode_step(params, cache, x_t, t, chunk_size=32, n_refined=8, window=None):
    """Output for the token at position t (int), x_t: (D,) its layer input, cache: hier_cache of tokens < t.
    Returns (output (D,), updated chunk_mass): pass that mass on to the next token of the same chunk.
    Prefill picks a chunk's refined chunks from all of its queries, including later ones, which a decoder
    doesn't have yet. Here the picks use this chunk's queries so far (causal), so the output equals
    hier_attention's at the last token of each chunk and can differ before it."""
    n_heads, head_dim = cache["key_mean"].shape[1:]
    window = window or 2 * chunk_size
    query_chunk = t // chunk_size
    n_visible = max(query_chunk - window // chunk_size + 1, 0)  # summaries of chunks older than the window
    assert cache["old_key"].shape[0] == n_heads * n_visible, "cache built for another position or window"
    q, key_t, value_t = (a.reshape(n_heads, head_dim) for a in jnp.split(x_t @ params["qkv"], 3))
    scale = head_dim**-0.5

    summary_logits = jnp.einsum("hd,chd->hc", q, cache["key_mean"][:n_visible]) * scale + jnp.log(chunk_size)
    # local: the window's tokens, up to and including t
    local_key = jnp.concatenate([cache["recent_key"], key_t[None]])  # (local tokens, heads, head_dim)
    local_value = jnp.concatenate([cache["recent_value"], value_t[None]])
    local_logits = jnp.einsum("hd,jhd->hj", q, local_key) * scale

    # selection: add this query's summary mass to the chunk's running total, then take the top n_refined
    probs = jax.nn.softmax(jnp.concatenate([summary_logits, local_logits], -1), -1)
    chunk_mass = cache["chunk_mass"].at[:, :n_visible].add(probs[:, :n_visible].astype(cache["chunk_mass"].dtype))
    n_picks = min(n_refined, n_visible)
    if n_picks > 0:
        top_mass, refined_chunk = jax.lax.top_k(chunk_mass[:, :n_visible], n_picks)  # (heads, picks)
        refined_valid = top_mass > 0
        is_refined = ((refined_chunk[..., None] == jnp.arange(n_visible)) & refined_valid[..., None]).any(1)
        rows = (jnp.arange(n_heads)[:, None] * n_visible + refined_chunk).reshape(-1)  # row of (head, chunk)
        refined_key = fetch_rows(cache["old_key"], rows).reshape(n_heads, n_picks, chunk_size, head_dim)
        refined_value = fetch_rows(cache["old_value"], rows).reshape(n_heads, -1, head_dim)
        refined_logits = jnp.einsum("hd,hsjd->hsj", q, refined_key) * scale
        refined_logits = jnp.where(refined_valid[..., None], refined_logits, -jnp.inf).reshape(n_heads, -1)
        summary_logits = jnp.where(is_refined, -jnp.inf, summary_logits)
    else:
        refined_value = jnp.zeros((n_heads, 0, head_dim), q.dtype)
        refined_logits = jnp.zeros((n_heads, 0), q.dtype)

    # one softmax over [unrefined summaries ; local tokens ; refined tokens]
    attn = jax.nn.softmax(jnp.concatenate([summary_logits, local_logits, refined_logits], -1), -1)
    n_local = local_key.shape[0]
    out = (jnp.einsum("hc,chd->hd", attn[:, :n_visible], cache["value_mean"][:n_visible])
           + jnp.einsum("hj,jhd->hd", attn[:, n_visible : n_visible + n_local], local_value)
           + jnp.einsum("hj,hjd->hd", attn[:, n_visible + n_local :], refined_value))
    return out.reshape(-1) @ params["proj"], chunk_mass
