"""Minimal decoder-only transformer (GPT-style) in pure JAX."""
import jax
import jax.numpy as jnp


def init_params(key, vocab, d_model=64, n_heads=4, n_layers=2, max_len=128):
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


def forward(params, tokens, k=None, n_heads=4):
    """tokens: (T,) int -> logits (T, vocab). vmap for batches.
    k: keep only the first k MLP hidden units (nested dropout during training)."""
    x = params["tok_emb"][tokens] + params["pos_emb"][: len(tokens)]
    for p in params["layers"]:
        x = x + attention(p, layer_norm(p["ln1"], x), n_heads)
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
    # Toy task: sequences count up mod vocab, e.g. 3 4 5 6 ...
    vocab, seq_len, batch_size = 16, 32, 32
    STEPS = 500
    key = jax.random.PRNGKey(0)
    params = init_params(key, vocab)
    for step in range(STEPS):
        key, sub, k_key = jax.random.split(key, 3)
        starts = jax.random.randint(sub, (batch_size, 1), 0, vocab)
        batch = (starts + jnp.arange(seq_len + 1)) % vocab
        params, loss = train_step(params, batch, k_key)
        if step % 100 == 0:
            print(f"step {step}: loss {loss:.4f}")

    eval_batch = (jnp.arange(vocab)[:, None] + jnp.arange(seq_len + 1)) % vocab
    for k in NESTED_SIZES.tolist():
        k_loss = loss_fn(truncate(params, k), eval_batch)
        print(f"hidden units {k:>3}/256: loss {k_loss:.4f}")
        assert k_loss < 0.1, f"truncated model (k={k}) failed counting task"
