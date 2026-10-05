"""Compressed Sparse Attention (DeepSeek-V4 §2.3.1) in pure JAX, a token mixer for transformer.py.

Every block_size tokens are compressed into one KV entry; each token attends to the top_k entries picked
by a small indexer, plus a sliding window of raw tokens.

Why: full attention reads all T past tokens per query, and a KV cache stores all of them. Here a query
reads only top_k + window entries, and the cache holds T / block_size compressed entries + the window.
The pieces, in the order csa_attention runs them:
  compress     block_size tokens -> one KV entry (a learned, per-channel weighted average)
  indexer      a cheap scorer that ranks the compressed blocks for each query; top_k are kept
  window       the most recent raw tokens, at full resolution (compression blurs nearby detail)
  attention    one softmax over [picked blocks ; window], with an attention sink
  indexer loss teaches the indexer to pick the blocks the attention actually uses"""
import jax
import jax.numpy as jnp


def init_csa(key, d_model, n_heads, block_size=4, n_groups=2):
    """Compressed Sparse Attention weights, sized from d_model like the scaled-down reference
    (d_model=256: kv_dim=64, latent_dim=128, 4 indexer heads of 32, group_dim=64).
    Paper: d_model=4096, kv_dim=192, 128 heads."""
    kv_dim = d_model // n_heads  # c: the one KV head shared by all query heads
    latent_dim = d_model // 2  # d_c: compressed query
    n_indexer_heads = max(1, n_heads // 2)  # n_h^I
    indexer_dim = kv_dim // 2  # c^I
    group_dim = d_model // 4  # d_g
    keys = iter(jax.random.split(key, 9))  # one random key per dense matrix below

    def dense(n_in, *shape):  # (*batch, n_in, n_out), scaled to keep unit variance
        return jax.random.normal(next(keys), (*shape[:-1], n_in, shape[-1])) * n_in**-0.5

    def compressor(out_dim):
        # Each token produces a value (C) and a score (Z) per channel, twice: once for its own block's
        # entry (a) and once for the next block's entry (b), so neighbouring entries overlap.
        return {
            # Eqs. 9-10: the four projections in one matrix [C^a | C^b | Z^a | Z^b]
            "proj": dense(d_model, 4 * out_dim),
            # Learned offsets per position within the block, so e.g. the last token can count more.
            # Zero at init: every token in the window starts with equal weight.
            "bias_own": jnp.zeros((block_size, out_dim)),  # B^a: per position in the block
            "bias_prev": jnp.zeros((block_size, out_dim)),  # B^b
        }

    return {
        "kv_compressor": compressor(kv_dim),  # C^Comp
        # Queries go through a small latent first (d_model -> latent_dim -> heads), as in DeepSeek's MLA.
        # The latent is computed once and reused by both the indexer and the attention.
        "query_down": dense(d_model, latent_dim),  # Eq. 13 W^DQ: shared by indexer and attention
        # Everything the indexer owns sits under one key, so train_lm.py can leave it out of clipping.
        "indexer": {  # trained only by its KL loss (see csa_attention)
            "key_compressor": compressor(indexer_dim),  # K^IComp: same algorithm, own weights
            "query_up": dense(latent_dim, n_indexer_heads * indexer_dim),  # Eq. 14 W^IUQ
            "head_weight": dense(d_model, n_indexer_heads),  # Eq. 15 W^w
        },
        "query_up": dense(latent_dim, n_heads * kv_dim),  # Eq. 18 W^UQ
        "window_kv": dense(d_model, kv_dim),  # raw KV entries for the sliding window
        "query_norm": jnp.ones(kv_dim),  # RMSNorm gains (start at 1 = plain normalization)
        "kv_norm": jnp.ones(kv_dim),
        "sink": jnp.zeros(n_heads),  # Eq. 27: attention sink logit per head
        # Output projection in two steps instead of one big (heads·kv_dim, d_model) matrix: fewer weights.
        "group_proj": dense(n_heads // n_groups * kv_dim, n_groups, group_dim),  # per group of heads
        "out_proj": dense(n_groups * group_dim, d_model),  # all groups -> d_model
    }


def rms_norm(gain, x, eps=1e-6):
    # Scale each vector to unit root-mean-square, then apply a learned per-channel gain. Unlike layer
    # norm there's no mean subtraction. Normalized queries and keys keep the logits from blowing up.
    return gain * x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + eps)


def rope(x, positions):
    """Partial RoPE: rotate the last quarter of x's channels by angle position·freq.
    x: (..., dim), positions: x.shape[:-1]."""
    # RoPE treats channel pairs as 2D points and rotates them by an angle proportional to the position.
    # A dot product of two rotated vectors then depends only on the difference of their positions.
    # Only the last quarter is rotated: the other channels carry content with no position mixed in.
    n_rot = x.shape[-1] // 4
    half = n_rot // 2
    # One frequency per pair, from 1 (fast, local) down to 1/10000 (slow, long range)
    angle = positions[..., None] * 10000.0 ** (-jnp.arange(half) / half)
    x1, x2 = x[..., -n_rot:-half], x[..., -half:]  # pair i = (x1[i], x2[i])
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    return jnp.concatenate([x[..., :-n_rot], x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def compress(params, x):
    """Eqs. 9-12: every block_size tokens -> one entry. Each channel is a softmax-weighted sum over the
    block's own tokens and the previous block's tokens.
    x: (T, D), T % block_size == 0 -> (T / block_size, out_dim)."""
    block_size = params["bias_own"].shape[0]
    n_blocks = x.shape[0] // block_size
    projected = (x @ params["proj"]).reshape(n_blocks, block_size, -1)
    # value_*: what each token contributes; score_*: how much it should count (before the softmax)
    value_own, value_prev, score_own, score_prev = jnp.split(projected, 4, axis=-1)

    def shift_to_next_block(a, fill):  # block i gets block i-1's tokens; block 0 gets fill
        return jnp.concatenate([jnp.full_like(a[:1], fill), a[:-1]])

    # Entry i averages over 2·block_size tokens: its own block (branch a) and the block before (branch b).
    # Block 0 has no block before it: score -inf gives those slots weight 0, and their values are 0.
    scores = jnp.concatenate([score_own + params["bias_own"],
                              shift_to_next_block(score_prev, -jnp.inf) + params["bias_prev"]], 1)
    values = jnp.concatenate([value_own, shift_to_next_block(value_prev, 0.0)], 1)  # (blocks, 2·block_size, out)
    # Softmax over the token axis (1), separately for every channel: each channel picks its own mix
    # of tokens. The entry only uses tokens up to the end of its block, so it never sees the future.
    return (jax.nn.softmax(scores, axis=1) * values).sum(1)


# How the indexer loss reaches the gradient without changing what the model returns: the loss is
# attached to the layer output, and on the way back it is handed a gradient of exactly 1, the same
# gradient it would get if it had been added to the training loss.
@jax.custom_vjp
def with_aux_loss(y, aux_loss):
    """Returns y unchanged; backward adds gradient 1 to aux_loss, as if it were added to the loss."""
    return y


# forward: return y, keep aux_loss (only its shape is needed). backward: y's gradient passes through.
with_aux_loss.defvjp(lambda y, aux_loss: (y, aux_loss),
                     lambda aux_loss, grad_y: (grad_y, jnp.ones_like(aux_loss)))


def csa_attention(params, x, top_k=8, window=32):
    """Compressed Sparse Attention. x: (T, D) -> (T, D).
    Token t attends to (a) the top_k compressed blocks, among those finished before t's block, that a
    lightweight indexer scores highest, and (b) the `window` most recent raw tokens. Every query head shares
    one KV head, and each entry is both key and value. Reference config: block_size=4, top_k=8, window=32."""
    seq_len, _ = x.shape
    block_size = params["kv_compressor"]["bias_own"].shape[0]
    # Compression needs whole blocks. Round T up and pad with zeros at the end: only later (discarded)
    # tokens can see the padding, so the real outputs are the same as without it.
    padded_len = -(-seq_len // block_size) * block_size
    x = jnp.pad(x, ((0, padded_len - seq_len), (0, 0)))
    n_heads, kv_dim = params["sink"].shape[0], params["query_norm"].shape[0]
    pos = jnp.arange(padded_len)
    n_blocks = padded_len // block_size
    k = min(top_k, n_blocks)  # short sequences have fewer blocks than top_k

    # --- Indexer (Eqs. 13-17): score every compressed block for every query, keep the top k.
    # Its inputs are detached, as in DeepSeek-V3.2: top_k has no gradient, so the indexer learns only
    # from the KL loss below, and that loss doesn't move the main model.
    latent_query = x @ params["query_down"]  # c^Q: (T, latent_dim)
    indexer = params["indexer"]
    x_detached, latent_detached = jax.lax.stop_gradient(x), jax.lax.stop_gradient(latent_query)
    n_indexer_heads = indexer["head_weight"].shape[1]
    indexer_query = (latent_detached @ indexer["query_up"]).reshape(padded_len, n_indexer_heads, -1)
    indexer_keys = compress(indexer["key_compressor"], x_detached)  # (blocks, indexer_dim)
    # Each indexer head: how well the query matches the block, clipped at 0 (ReLU, not softmax: cheap).
    match = jax.nn.relu(jnp.einsum("thc,bc->thb", indexer_query, indexer_keys))
    # Then a per-token weighted sum over heads; the weights come from the token itself (head_weight).
    # Eq. 16; the 1/sqrt(heads · dim) scale is from DeepSeek-V3.2's reference indexer, not in the equation
    block_score = jnp.einsum("th,thb->tb", x_detached @ indexer["head_weight"], match)
    block_score = block_score * indexer_query[0].size**-0.5  # (T, blocks)
    # Causality: token t may only use blocks that are complete before its own block starts.
    # block b holds tokens <= block_size·(b+1) - 1, so it's finished before t's block iff b < t // block_size
    n_finished = pos // block_size  # (T,)
    finished = jnp.arange(n_blocks) < n_finished[:, None]
    picked_score, picked_block = jax.lax.top_k(jnp.where(finished, block_score, -jnp.inf), k)  # Eq. 17
    # Early tokens have fewer than k finished blocks, but top_k still returns k indices: the extras point
    # at unfinished (future) blocks and must be masked. (The reference PyTorch code leaked here.)
    picked_valid = picked_block < n_finished[:, None]

    # --- KV entries per token: k picked compressed blocks ++ the `window` raw tokens ending at t
    window_pos = pos[:, None] - jnp.arange(window)[::-1]  # (T, window): t-window+1, ..., t
    compressed_kv = compress(params["kv_compressor"], x)  # (blocks, kv_dim)
    # Positions before 0 are clamped to token 0 here and masked out below (window_pos >= 0)
    window_kv = (x @ params["window_kv"])[jnp.maximum(window_pos, 0)]
    kv = jnp.concatenate([compressed_kv[picked_block], window_kv], 1)  # (T, k + window, kv_dim)
    # Each entry needs a position for RoPE; a compressed block uses the middle of its block.
    kv_pos = jnp.concatenate([picked_block * block_size + block_size // 2, window_pos], 1)  # block: its centre
    kv_valid = jnp.concatenate([picked_valid, window_pos >= 0], 1)

    # --- Core attention (Eqs. 18-19): QK norm, partial RoPE, MQA with shared key = value
    # MQA (multi-query attention): n_heads query heads, but only one KV head that all of them read.
    query = (latent_query @ params["query_up"]).reshape(padded_len, n_heads, kv_dim)
    query = rope(rms_norm(params["query_norm"], query), pos[:, None])
    kv = rope(rms_norm(params["kv_norm"], kv), kv_pos)
    logits = jnp.einsum("thc,tkc->thk", query, kv) / jnp.sqrt(kv_dim)
    logits = jnp.where(kv_valid[:, None], logits, -jnp.inf)  # masked entries get weight 0
    # Eq. 27: the sink is one more logit with no value, so a head can put weight on nothing
    # (Plain softmax forces every head to spend weight 1 somewhere, even when nothing is relevant.)
    sink = jnp.broadcast_to(params["sink"][:, None], (padded_len, n_heads, 1))
    attn = jax.nn.softmax(jnp.concatenate([logits, sink], -1), -1)[..., :-1]  # (T, heads, k + window)
    # The same kv tensor serves as values: "shared KV" halves what has to be stored per entry.
    out = jnp.einsum("thk,tkc->thc", attn, kv)
    # The values were rotated to their own positions. Rotating the output back by -t leaves each value's
    # rotation relative to the query: what's mixed in carries "how far back", not "where in the sequence".
    out = rope(out, -pos[:, None])

    # --- Indexer loss: cross-entropy (= KL up to a constant) from the attention's weight on the picked
    # blocks, summed over heads and renormalized, to the indexer's softmax over the same blocks.
    # In words: blocks the attention found useful should be the ones the indexer scores highest.
    target = jax.lax.stop_gradient(attn[..., :k].sum(1))
    target = target / jnp.maximum(target.sum(-1, keepdims=True), 1e-9)  # rows with no finished block: all 0
    # -1e9 instead of -inf: a row with no valid pick then gives a finite (uniform) log-softmax, times target 0
    indexer_logp = jax.nn.log_softmax(jnp.where(picked_valid, picked_score, -1e9))
    indexer_loss = -(target * indexer_logp).sum(-1).mean()

    # --- Grouped output projection: each group of heads -> group_dim, then all groups -> D
    n_groups = params["group_proj"].shape[0]
    grouped = jnp.einsum("tgi,gio->tgo", out.reshape(padded_len, n_groups, -1), params["group_proj"])
    y = grouped.reshape(padded_len, -1) @ params["out_proj"]
    # Under vmap each sequence's loss gets gradient 1 (a sum over the batch). Only the indexer sees it, so
    # with Adam the scale doesn't matter, but keep the indexer out of global-norm clipping (train_lm.py).
    return with_aux_loss(y[:seq_len], indexer_loss)  # drop the padding rows


def csa_cache(params, context, window=32):
    """Decode cache for the tokens so far, context: (T, D) layer inputs. Holds the compressed KV entry and
    indexer key of every complete block, and the raw KV entries of the last window - 1 tokens: about
    T / block_size · (kv_dim + indexer_dim) values instead of the T · 2 · d_model of a full KV cache.
    ponytail: built from scratch; a real decoder appends one window entry per token and compresses one
    block every block_size tokens (constant work per token)."""
    seq_len = context.shape[0]
    block_size = params["kv_compressor"]["bias_own"].shape[0]
    complete = context[: seq_len // block_size * block_size]
    recent = context[max(0, seq_len - (window - 1)) :] @ params["window_kv"]
    def blocks(compressor):  # compress can't reshape zero blocks
        if complete.shape[0] == 0:
            return jnp.zeros((0, compressor["bias_own"].shape[1]), context.dtype)
        return compress(compressor, complete)

    return {
        "compressed_kv": blocks(params["kv_compressor"]),  # (complete blocks, kv_dim)
        "indexer_keys": blocks(params["indexer"]["key_compressor"]),  # (complete blocks, indexer_dim)
        # oldest first; zero rows stand in for positions before 0 and are masked in the step
        "window_kv": jnp.pad(recent, ((window - 1 - recent.shape[0], 0), (0, 0))),
    }


def csa_decode_step(params, cache, x_t, t, top_k=8):
    """Output for the token at position t (int), x_t: (D,) its layer input, cache: csa_cache of tokens < t.
    Same math as row t of csa_attention, for one query. Every cached block is finished before t's block:
    block b holds tokens up to block_size·(b+1) - 1 < t, and t's own block isn't complete in the cache."""
    block_size = params["kv_compressor"]["bias_own"].shape[0]
    n_heads, kv_dim = params["sink"].shape[0], params["query_norm"].shape[0]
    window = cache["window_kv"].shape[0] + 1
    indexer = params["indexer"]
    latent_query = x_t @ params["query_down"]

    # Indexer: score the cached blocks, keep the top k (fewer if fewer blocks exist; prefill masks the rest)
    k = min(top_k, cache["indexer_keys"].shape[0])
    if k > 0:
        indexer_query = (latent_query @ indexer["query_up"]).reshape(indexer["head_weight"].shape[1], -1)
        match = jax.nn.relu(indexer_query @ cache["indexer_keys"].T)  # (indexer heads, blocks)
        block_score = (x_t @ indexer["head_weight"]) @ match * indexer_query.size**-0.5
        picked_block = jax.lax.top_k(block_score, k)[1]
    else:
        picked_block = jnp.zeros(0, jnp.int32)

    # KV entries: picked blocks ++ the window (cached window - 1 tokens, then this token's own entry)
    window_pos = t - jnp.arange(window)[::-1]
    window_kv = jnp.concatenate([cache["window_kv"], (x_t @ params["window_kv"])[None]])
    kv = jnp.concatenate([cache["compressed_kv"][picked_block], window_kv])  # (k + window, kv_dim)
    kv_pos = jnp.concatenate([picked_block * block_size + block_size // 2, window_pos])
    kv_valid = jnp.concatenate([jnp.ones(k, bool), window_pos >= 0])

    # Core attention, as in csa_attention
    query = (latent_query @ params["query_up"]).reshape(n_heads, kv_dim)
    query = rope(rms_norm(params["query_norm"], query), jnp.full(n_heads, t))
    kv = rope(rms_norm(params["kv_norm"], kv), kv_pos)
    logits = jnp.where(kv_valid, query @ kv.T / jnp.sqrt(kv_dim), -jnp.inf)  # (heads, k + window)
    attn = jax.nn.softmax(jnp.concatenate([logits, params["sink"][:, None]], -1), -1)[:, :-1]
    out = rope(attn @ kv, jnp.full(n_heads, -t))
    n_groups = params["group_proj"].shape[0]
    grouped = jnp.einsum("gi,gio->go", out.reshape(n_groups, -1), params["group_proj"])
    return grouped.reshape(-1) @ params["out_proj"]
