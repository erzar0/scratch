"""Minimal decoder-only transformer (GPT-style) in pure JAX."""
import jax
import jax.numpy as jnp


def init_params(key, vocab, d_model=64, n_heads=4, n_layers=2, max_len=128, mixer="softmax"):
    def dense(key, n_in, n_out):
        return jax.random.normal(key, (n_in, n_out)) * n_in**-0.5

    keys = iter(jax.random.split(key, 2 + 4 * n_layers))
    layers = [
        {
            "ln1": {"g": jnp.ones(d_model), "b": jnp.zeros(d_model)},
            "qkv": dense(next(keys), d_model, 3 * d_model),
            "proj": dense(next(keys), d_model, d_model),
            "ln2": {"g": jnp.ones(d_model), "b": jnp.zeros(d_model)},
            "fc1": dense(next(keys), d_model, 4 * d_model),
            "fc2": dense(next(keys), 4 * d_model, d_model),
        }
        for _ in range(n_layers)
    ]
    if mixer == "frac":  # per head: α = 1.5, diffusion time t from 0.5 (local) to 32 (global)
        for layer in layers:
            layer.update(alpha=jnp.zeros(n_heads), t=jnp.log(jnp.expm1(jnp.geomspace(0.5, 32.0, n_heads))))
    return {
        "tok_emb": jax.random.normal(next(keys), (vocab, d_model)) * 0.02,
        "pos_emb": jax.random.normal(next(keys), (max_len, d_model)) * 0.02,
        "layers": layers,
        "ln_f": {"g": jnp.ones(d_model), "b": jnp.zeros(d_model)},
    }


def layer_norm(p, x, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return p["g"] * (x - mean) / jnp.sqrt(var + eps) + p["b"]


def attention(p, x, n_heads):
    T, D = x.shape
    q, k, v = jnp.split(x @ p["qkv"], 3, axis=-1)
    # (T, D) -> (H, T, D/H)
    q, k, v = (a.reshape(T, n_heads, D // n_heads).transpose(1, 0, 2) for a in (q, k, v))
    scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(D // n_heads)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores = jnp.where(mask, scores, -jnp.inf)
    out = jax.nn.softmax(scores, axis=-1) @ v
    return out.transpose(1, 0, 2).reshape(T, D) @ p["proj"]


def causal_mix(w, v):
    """y_i = Σ_{j≤i} w[i-j]·v_j / Σ_{j≤i} w[j] per head. w: (H, T) lag weights, v: (T, H, Dh).
    FFT padded to 2T so the circular convolution never wraps future tokens into the past."""
    T = v.shape[0]
    w = w.T[..., None]  # (T, H, 1)
    y = jnp.fft.irfft(jnp.fft.rfft(v, 2 * T, axis=0) * jnp.fft.rfft(w, 2 * T, axis=0), 2 * T, axis=0)[:T]
    return y / jnp.cumsum(w, axis=0)


def frac_attention(p, x, n_grid):
    """Softmax weights replaced by the causal fractional heat kernel exp(-t·L^(α/2)),
    L = graph Laplacian of a cycle of n_grid positions. O(T log T) instead of O(T²)."""
    T, D = x.shape
    H = p["alpha"].shape[0]
    alpha = 1 + jax.nn.sigmoid(p["alpha"])  # (H,) in (1, 2)
    t = jax.nn.softplus(p["t"])  # (H,) diffusion time: window width
    # fixed grid, not 2T: kernel must not depend on sequence length
    lam = (2 * jnp.sin(jnp.pi * jnp.arange(n_grid // 2 + 1) / n_grid)) ** 2  # eigenvalues of L
    lam = jnp.maximum(lam, 1e-12)  # λ=0 makes d/dα λ^(α/2) NaN
    w = jnp.fft.irfft(jnp.exp(-t[:, None] * lam ** (alpha[:, None] / 2)), n_grid)[:, :T]
    q, k, v = jnp.split(x @ p["qkv"], 3, axis=-1)
    mixed = causal_mix(w, (k * v).reshape(T, H, D // H)).reshape(T, D)
    return (q * mixed) @ p["proj"]


def forward(params, tokens, k=None, n_heads=4):
    """tokens: (T,) int -> logits (T, vocab). vmap for batches.
    k: keep only the first k MLP hidden units (nested dropout during training)."""
    x = params["tok_emb"][tokens] + params["pos_emb"][: len(tokens)]
    for p in params["layers"]:
        a = layer_norm(p["ln1"], x)
        x = x + (frac_attention(p, a, 2 * len(params["pos_emb"])) if "alpha" in p else attention(p, a, n_heads))
        h = jax.nn.gelu(layer_norm(p["ln2"], x) @ p["fc1"])
        if k is not None:
            h = h * (jnp.arange(h.shape[-1]) < k)
        x = x + h @ p["fc2"]
    x = layer_norm(params["ln_f"], x)
    return x @ params["tok_emb"].T  # tied embeddings


def loss_fn(params, batch, k=None):
    """batch: (B, T+1) tokens; predict next token."""
    logits = jax.vmap(forward, in_axes=(None, 0, None))(params, batch[:, :-1], k)
    logp = jax.nn.log_softmax(logits)
    targets = batch[:, 1:]
    return -jnp.take_along_axis(logp, targets[..., None], axis=-1).mean()


# ponytail: k drawn from 4 cut points, so ordering is only trained at these sizes;
# sample k from 1..hidden for finer-grained truncation
NESTED_SIZES = jnp.array([32, 64, 128, 256])


@jax.jit
def train_step(params, batch, key, lr=0.1):
    # ponytail: plain SGD, swap for optax.adamw on real data
    k = jax.random.choice(key, NESTED_SIZES)
    loss, grads = jax.value_and_grad(loss_fn)(params, batch, k)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params, loss


def truncate(params, k):
    """Keep the first k MLP hidden units: real smaller matrices for inference."""
    layers = [{**p, "fc1": p["fc1"][:, :k], "fc2": p["fc2"][:k]} for p in params["layers"]]
    return {**params, "layers": layers}


if __name__ == "__main__":
    # Toy task: count up mod vocab by a per-sequence step s in {1, 2, 3}, e.g. 3 5 7 9 ...
    # s is only visible from past tokens: without mixing, loss stays >= ln 3 ≈ 1.1
    vocab, seq_len, batch_size = 16, 32, 32
    STEPS = 500
    starts, steps = jnp.meshgrid(jnp.arange(vocab), jnp.arange(1, 4))
    eval_batch = (starts.reshape(-1, 1) + steps.reshape(-1, 1) * jnp.arange(seq_len + 1)) % vocab

    for mixer in ["softmax", "frac"]:
        key = jax.random.PRNGKey(0)
        params = init_params(key, vocab, mixer=mixer)
        for step in range(STEPS):
            key, sub, s_key, k_key = jax.random.split(key, 4)
            starts = jax.random.randint(sub, (batch_size, 1), 0, vocab)
            steps = jax.random.randint(s_key, (batch_size, 1), 1, 4)
            batch = (starts + steps * jnp.arange(seq_len + 1)) % vocab
            params, loss = train_step(params, batch, k_key)
            if step % 100 == 0:
                print(f"[{mixer}] step {step}: loss {loss:.4f}")

        for k in NESTED_SIZES.tolist():
            k_loss = loss_fn(truncate(params, k), eval_batch)
            print(f"[{mixer}] hidden units {k:>3}/256: loss {k_loss:.4f}")
            assert k_loss < 0.1, f"[{mixer}] truncated model (k={k}) failed random-step task"

        # causal: running a prefix must reproduce the full run's outputs
        tokens = eval_batch[0, :-1]
        diff = jnp.abs(forward(params, tokens[:10]) - forward(params, tokens)[:10]).max()
        print(f"[{mixer}] prefix vs full max diff: {diff:.2e}")
        assert diff < 1e-2, f"[{mixer}] outputs depend on future tokens"

    # flat weights must give the running mean: catches FFT wrap-around and normalization bugs
    v = jax.random.normal(key, (seq_len, 2, 3))
    running_mean = jnp.cumsum(v, axis=0) / jnp.arange(1, seq_len + 1)[:, None, None]
    assert jnp.allclose(causal_mix(jnp.ones((2, seq_len)), v), running_mean, atol=1e-5)
