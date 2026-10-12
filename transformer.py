"""Minimal decoder-only transformer (GPT-style) in pure JAX.

Three token mixers ("attention" layers) are available:
- "softmax": standard causal multi-head attention. Cost O(T²) per sequence of T tokens.
- "csa": DeepSeek-V4 Compressed Sparse Attention. Top-k of the m-token compressed blocks
  (picked by a small indexer) plus a sliding window of raw tokens.
- "hier": hierarchical chunk attention (from modded-nanogpt). Chunk summaries, with the
  top-S chunks re-read token by token; only the previous and current chunk are read exactly.
- "hier_window": the same with a 512-token sliding window read exactly; the hierarchy covers
  what lies before the window.

The MLP is trained with nested dropout, so its hidden layer can be cut down at inference.
Run this file to train all mixers on a toy task and run the self-checks.
"""
import jax
import jax.numpy as jnp

from attn_methods_jax import bandit, bitonic_topk, ppr, sparse_softmax
from csa import csa_attention, init_csa
from hier import hier_attention


def init_params(key, vocab, d_model=64, n_heads=4, n_layers=2, max_len=128, mixer="softmax"):
    """Create all weights as a nested dict. JAX treats nested dicts/lists as a "pytree",
    so jax.grad and jax.tree.map work on the whole model without any wrapper classes."""

    def dense(key, n_in, n_out):
        # Scaling by 1/sqrt(n_in) keeps activations at roughly unit variance through the layer.
        return jax.random.normal(key, (n_in, n_out)) * n_in**-0.5

    # One random key per matrix: 4 per layer, plus token and position embeddings.
    keys = iter(jax.random.split(key, 2 + 4 * n_layers))
    layers = [
        {
            "ln1": {"g": jnp.ones(d_model), "b": jnp.zeros(d_model)},  # layer norm before attention
            "qkv": dense(next(keys), d_model, 3 * d_model),  # query, key, value projections in one matrix
            "proj": dense(next(keys), d_model, d_model),  # attention output projection
            "ln2": {"g": jnp.ones(d_model), "b": jnp.zeros(d_model)},  # layer norm before the MLP
            "fc1": dense(next(keys), d_model, 4 * d_model),  # MLP: expand to 4·d_model hidden units
            "fc2": dense(next(keys), 4 * d_model, d_model),  # MLP: project back to d_model
        }
        for _ in range(n_layers)
    ]
    if mixer == "csa":  # own projections replace qkv/proj; separate keys keep the other mixers' init unchanged
        for layer, k in zip(layers, jax.random.split(jax.random.fold_in(key, 1), n_layers)):
            del layer["qkv"], layer["proj"]
            layer["csa"] = init_csa(k, d_model, n_heads)
    if mixer in ("hier", "hier_window") + tuple(APPROX):  # same weights as softmax; the empty dict only marks the layer (no leaves)
        for layer in layers:
            layer[mixer] = {}
    return {
        "tok_emb": jax.random.normal(next(keys), (vocab, d_model)) * 0.02,  # token id -> vector
        "pos_emb": jax.random.normal(next(keys), (max_len, d_model)) * 0.02,  # position -> vector; caps T at max_len
        "layers": layers,
        "ln_f": {"g": jnp.ones(d_model), "b": jnp.zeros(d_model)},  # final layer norm before the output
    }


def layer_norm(p, x, eps=1e-5):
    # Rescale each token's vector to zero mean and unit variance, then apply a learned
    # scale g and shift b. Keeps activations in a stable range as layers stack up.
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return p["g"] * (x - mean) / jnp.sqrt(var + eps) + p["b"]


def attention(p, x, n_heads):
    """Causal softmax attention. x: (T, D) -> (T, D)."""
    T, D = x.shape
    q, k, v = jnp.split(x @ p["qkv"], 3, axis=-1)
    # Split the D channels into n_heads independent heads: (T, D) -> (H, T, D/H)
    q, k, v = (a.reshape(T, n_heads, D // n_heads).transpose(1, 0, 2) for a in (q, k, v))
    # scores[h, i, j]: how well token i's query matches token j's key.
    # Dividing by sqrt(head size) keeps the scores from growing with the head size.
    scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(D // n_heads)
    # Causal mask: token i may only look at tokens j <= i. -inf becomes weight 0 after softmax.
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores = jnp.where(mask, scores, -jnp.inf)
    # Each row of softmax weights sums to 1: every output is a weighted average of values.
    out = jax.nn.softmax(scores, axis=-1) @ v
    # Merge heads back to (T, D) and mix them with the output projection.
    return out.transpose(1, 0, 2).reshape(T, D) @ p["proj"]


HIER_WINDOW = 512  # tokens read exactly by "hier_window"

# Approximate attentions from attn_methods_jax.py, same qkv/proj weights as softmax. The random keys are fixed, so every
# step sees the same random landmarks. lowrank, nystrom and stationary are not here: they read the future (test_attn_methods.py).
APPROX = {
    "sparse": lambda q, k, v, key: sparse_softmax(q, k, v, key, m=16, window=32),
    "ppr": lambda q, k, v, key: ppr(q, k, v, key, m=16, window=32, alpha=0.7),
    "bitonic": lambda q, k, v, key: bitonic_topk(q, k, v, key, topk=16, tau=0.1, pool=64),
    "bandit": lambda q, k, v, key: bandit(q, k, v, key, z=2.0),
}


def approx_attention(p, x, n_heads, method):
    """Like `attention`, with the per-head softmax replaced by APPROX[method]."""
    T, D = x.shape
    q, k, v = (a.reshape(T, n_heads, D // n_heads).transpose(1, 0, 2) for a in jnp.split(x @ p["qkv"], 3, axis=-1))
    out = jax.vmap(APPROX[method], in_axes=(0, 0, 0, None))(q, k, v, jax.random.PRNGKey(0))
    return out.transpose(1, 0, 2).reshape(T, D) @ p["proj"]


def mix(p, x, n_heads, evicted_at=None):
    """Dispatch on which weights the layer has (see init_params). evicted_at: CSA only, see csa_attention."""
    if "csa" in p:  # recomputed in the backward, like softmax below: at T = 2048 its stored activations hit 3.9 GiB
        return jax.checkpoint(lambda p, x, e: csa_attention(p, x, evicted_at=e))(p["csa"], x, evicted_at)
    for name in APPROX:
        if name in p:
            return jax.checkpoint(approx_attention, static_argnums=(2, 3))(p, x, n_heads, name)
    if "hier" in p:
        return hier_attention(p, x, n_heads)
    if "hier_window" in p:
        return hier_attention(p, x, n_heads, window=HIER_WINDOW)
    # Recompute the T x T scores in the backward instead of storing them for every layer: same gradients;
    # at T = 2048, batch 2, the stored scores (6.6 GiB) exceed WSL's 4 GiB buffer limit.
    return jax.checkpoint(attention, static_argnums=2)(p, x, n_heads)


def forward(params, tokens, k=None, n_heads=4, evicted_at=None):
    """tokens: (T,) int -> logits (T, vocab). vmap for batches.
    k: keep only the first k MLP hidden units (nested dropout during training).
    evicted_at: CSA only, (n_layers, blocks) from evict.py's decode; row l goes to layer l."""
    x = params["tok_emb"][tokens] + params["pos_emb"][: len(tokens)]
    for l, p in enumerate(params["layers"]):
        # Pre-norm residual block: each sublayer reads a normalized copy of x and adds
        # its result back to x, so information and gradients flow straight through.
        a = layer_norm(p["ln1"], x)
        x = x + mix(p, a, n_heads, None if evicted_at is None else evicted_at[l])
        h = jax.nn.gelu(layer_norm(p["ln2"], x) @ p["fc1"])
        if k is not None:
            # Nested dropout: zero every hidden unit from index k on. Units with low indices
            # are always used, so training makes them the most important ones.
            h = h * (jnp.arange(h.shape[-1]) < k)
        x = x + h @ p["fc2"]
    x = layer_norm(params["ln_f"], x)
    # Tied embeddings: the same matrix maps token -> vector at the input
    # and vector -> score per token at the output.
    return x @ params["tok_emb"].T  # tied embeddings


def loss_fn(params, batch, k=None):
    """batch: (B, T+1) tokens; predict next token."""
    # Inputs are tokens 0..T-1 and targets are tokens 1..T: every position predicts the next one.
    # vmap runs forward on each sequence in the batch; params and k are shared (None).
    logits = jax.vmap(forward, in_axes=(None, 0, None))(params, batch[:, :-1], k)
    logp = jax.nn.log_softmax(logits)
    targets = batch[:, 1:]
    # Cross-entropy: average negative log-probability the model gave the correct next token.
    return -jnp.take_along_axis(logp, targets[..., None], axis=-1).mean()


# ponytail: k drawn from 4 cut points, so ordering is only trained at these sizes;
# sample k from 1..hidden for finer-grained truncation
NESTED_SIZES = jnp.array([32, 64, 128, 256])


@jax.jit  # compile the whole step (forward, backward, update) into one XLA program
def train_step(params, batch, key, lr=0.1):
    # ponytail: plain SGD, swap for optax.adamw on real data
    # Each step trains a randomly sized MLP (first k hidden units), so all cut points work.
    k = jax.random.choice(key, NESTED_SIZES)
    loss, grads = jax.value_and_grad(loss_fn)(params, batch, k)
    # grads has the same nested-dict shape as params: update every weight in one tree.map.
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params, loss


def truncate(params, k):
    """Keep the first k MLP hidden units: real smaller matrices for inference."""
    # Slicing (not masking) gives genuinely smaller matrices, so inference does less work.
    layers = [{**p, "fc1": p["fc1"][:, :k], "fc2": p["fc2"][:k]} for p in params["layers"]]
    return {**params, "layers": layers}


if __name__ == "__main__":
    # Toy task: count up mod vocab by a per-sequence step s in {1, 2, 3}, e.g. 3 5 7 9 ...
    # s is only visible from past tokens: without mixing, loss stays >= ln 3 ≈ 1.1
    # (With mixing, only the first prediction is unknowable: floor ≈ ln 3 / 32 ≈ 0.034.)
    vocab, seq_len, batch_size = 16, 32, 32
    STEPS = 500
    # Evaluation set: every start token (16) with every step size (3) = 48 sequences.
    starts, steps = jnp.meshgrid(jnp.arange(vocab), jnp.arange(1, 4))
    eval_batch = (starts.reshape(-1, 1) + steps.reshape(-1, 1) * jnp.arange(seq_len + 1)) % vocab

    # Train both mixers from the same seed on the same data, so their losses are comparable.
    for mixer in ["softmax", "csa", "hier"]:
        key = jax.random.PRNGKey(0)
        params = init_params(key, vocab, mixer=mixer)
        for step in range(STEPS):
            # Fresh random keys each step: batch starts, step sizes, nested-dropout size k.
            key, sub, s_key, k_key = jax.random.split(key, 4)
            starts = jax.random.randint(sub, (batch_size, 1), 0, vocab)
            steps = jax.random.randint(s_key, (batch_size, 1), 1, 4)
            batch = (starts + steps * jnp.arange(seq_len + 1)) % vocab
            params, loss = train_step(params, batch, k_key)
            if step % 100 == 0:
                print(f"[{mixer}] step {step}: loss {loss:.4f}")

        # Nested dropout check: the model must still solve the task with its MLP cut down.
        for k in NESTED_SIZES.tolist():
            k_loss = loss_fn(truncate(params, k), eval_batch)
            print(f"[{mixer}] hidden units {k:>3}/256: loss {k_loss:.4f}")
            assert k_loss < 0.1, f"[{mixer}] truncated model (k={k}) failed random-step task"

        # causal: running a prefix must reproduce the full run's outputs
        # If any future token leaked into an earlier position, the first 10 outputs would
        # differ between these two runs. Tolerance 1e-2 sits above GPU rounding noise
        # (up to ~7e-4 measured) and below the smallest real leak measured (~0.29).
        tokens = eval_batch[0, :-1]
        diff = jnp.abs(forward(params, tokens[:10]) - forward(params, tokens)[:10]).max()
        print(f"[{mixer}] prefix vs full max diff: {diff:.2e}")
        assert diff < 1e-2, f"[{mixer}] outputs depend on future tokens"

    # full fp32 matmuls: GPU default (TF32) leaves ~1e-3 error vs the dense reference
    with jax.default_matmul_precision("highest"):
        # The toy is too short for CSA/hier to compress or select anything (32 tokens: 8 blocks <= top_k,
        # one 32-token chunk), so check the sparse paths at the layer level with small settings.
        x = jax.random.normal(key, (40, 32))
        p_csa = init_csa(key, 32, 4)
        y = csa_attention(p_csa, x, top_k=2, window=4)
        # 6: fewer than top_k visible blocks, so masked picks (block t // block_size, holding later tokens) must stay masked
        # 13: cut mid-block (padded); 24: deep enough that top_k=2 drops blocks
        for n in [6, 13, 24]:
            diff = jnp.abs(csa_attention(p_csa, x[:n], top_k=2, window=4) - y[:n]).max()
            assert diff < 1e-4, f"[csa] layer depends on future tokens (prefix {n}: {diff:.2e})"
        g = jax.grad(lambda p: csa_attention(p, x, top_k=2, window=4).sum())(p_csa)
        assert jnp.abs(g["indexer"]["query_up"]).max() > 0, "[csa] indexer gets no gradient from its KL loss"

        p_att = init_params(key, vocab, d_model=32, mixer="softmax")["layers"][0]
        # window=8: two 4-token chunks (the default 512 would cover all 40 tokens: plain exact attention)
        # n_refined >= every older chunk: all chunks refined, so hier must equal exact causal attention
        full = hier_attention(p_att, x, 4, chunk_size=4, n_refined=10, window=8)
        assert jnp.allclose(full, attention(p_att, x, 4), atol=1e-4), "[hier] all-refined != softmax attention"
        # sparse (n_refined=1): picks depend on the whole query chunk, so prefixes must end on a chunk boundary
        y = hier_attention(p_att, x, 4, chunk_size=4, n_refined=1, window=8)
        for n in [16, 28]:
            diff = jnp.abs(hier_attention(p_att, x[:n], 4, chunk_size=4, n_refined=1, window=8) - y[:n]).max()
            assert diff < 1e-4, f"[hier] layer depends on future chunks (prefix {n}: {diff:.2e})"
        # 10 chunks in blocks of 3: three full blocks plus a remainder must give the same output as one block
        blocked = hier_attention(p_att, x, 4, chunk_size=4, n_refined=1, window=8, chunks_per_block=3)
        assert jnp.allclose(blocked, y, atol=1e-5), "[hier] output depends on chunks_per_block"
        print("csa / hier layer checks passed")
