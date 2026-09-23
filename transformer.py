"""Minimal decoder-only transformer (GPT-style) in pure JAX.

Two token mixers ("attention" layers) are available:
- "softmax": standard causal multi-head attention. Cost O(T²) per sequence of T tokens.
- "frac": fractional Laplacian attention. Each head averages past tokens with weights
  from a fractional heat kernel, computed with an FFT convolution. Cost O(T log T).

The MLP is trained with nested dropout, so its hidden layer can be cut down at inference.
Run this file to train both mixers on a toy task and run the self-checks.
"""
import jax
import jax.numpy as jnp


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
    if mixer == "frac":  # per head: α = 1.5, diffusion time t from 0.5 (local) to 32 (global)
        # Both are stored unconstrained; frac_attention maps them into their valid ranges.
        # alpha = 0 gives α = 1 + sigmoid(0) = 1.5. t is stored as inverse softplus, so
        # softplus(t) recovers 0.5 ... 32: heads start with different window widths.
        # The presence of "alpha" in a layer is what makes forward() use frac_attention.
        for layer in layers:
            layer.update(alpha=jnp.zeros(n_heads), t=jnp.log(jnp.expm1(jnp.geomspace(0.5, 32.0, n_heads))))
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


def causal_mix(w, v):
    """y_i = Σ_{j≤i} w[i-j]·v_j / Σ_{j≤i} w[j] per head. w: (H, T) lag weights, v: (T, H, Dh).
    FFT padded to 2T so the circular convolution never wraps future tokens into the past."""
    # The weight between tokens i and j depends only on their distance i - j (the lag),
    # so the whole weighted sum is a convolution. An FFT turns a convolution into an
    # elementwise product: O(T log T) instead of building the O(T²) weight matrix.
    # An FFT convolution is circular: the end of the sequence wraps around to the start.
    # Zero-padding to length 2T leaves room for that wrap to land on zeros, and since w
    # only has lags 0..T-1 (present and past), no future token reaches the output.
    T = v.shape[0]
    w = w.T[..., None]  # (T, H, 1)
    y = jnp.fft.irfft(jnp.fft.rfft(v, 2 * T, axis=0) * jnp.fft.rfft(w, 2 * T, axis=0), 2 * T, axis=0)[:T]
    # Token i sees only lags 0..i, so its weights sum to cumsum(w)[i]. Dividing by that makes
    # every output a true weighted average (weights sum to 1), like a softmax row.
    return y / jnp.cumsum(w, axis=0)


def frac_attention(p, x, n_grid):
    """Softmax weights replaced by the causal fractional heat kernel exp(-t·L^(α/2)),
    L = graph Laplacian of a cycle of n_grid positions. O(T log T) instead of O(T²)."""
    # Picture each token dropping ink on its position and letting it diffuse along the
    # sequence for time t. The ink that reaches each earlier position is that position's weight.
    # L^(α/2) is the fractional Laplacian: α = 2 is ordinary diffusion (local, fast decay);
    # α -> 1 allows long jumps, so weights decay as a power law and reach far back.
    T, D = x.shape
    H = p["alpha"].shape[0]
    alpha = 1 + jax.nn.sigmoid(p["alpha"])  # (H,) in (1, 2)
    t = jax.nn.softplus(p["t"])  # (H,) diffusion time: window width
    # fixed grid, not 2T: kernel must not depend on sequence length
    # (otherwise appending a token would change the weights, and so every earlier output).
    # On a cycle of n_grid positions, L's eigenvectors are Fourier modes with these eigenvalues:
    lam = (2 * jnp.sin(jnp.pi * jnp.arange(n_grid // 2 + 1) / n_grid)) ** 2  # eigenvalues of L
    lam = jnp.maximum(lam, 1e-12)  # λ=0 makes d/dα λ^(α/2) NaN
    # Apply exp(-t·λ^(α/2)) to each Fourier mode, then inverse FFT to get the kernel over
    # positions. Keep lags 0..T-1 only: that makes it causal. The kernel is positive and
    # sums to 1 over the whole cycle (it's a random walk's transition probabilities).
    w = jnp.fft.irfft(jnp.exp(-t[:, None] * lam ** (alpha[:, None] / 2)), n_grid)[:, :T]
    # Gating adds content dependence: each token sends the message k ⊙ v, messages are
    # averaged with the kernel weights, and the reader's q scales which channels it keeps.
    q, k, v = jnp.split(x @ p["qkv"], 3, axis=-1)
    mixed = causal_mix(w, (k * v).reshape(T, H, D // H)).reshape(T, D)
    return (q * mixed) @ p["proj"]


def forward(params, tokens, k=None, n_heads=4):
    """tokens: (T,) int -> logits (T, vocab). vmap for batches.
    k: keep only the first k MLP hidden units (nested dropout during training)."""
    x = params["tok_emb"][tokens] + params["pos_emb"][: len(tokens)]
    for p in params["layers"]:
        # Pre-norm residual block: each sublayer reads a normalized copy of x and adds
        # its result back to x, so information and gradients flow straight through.
        a = layer_norm(p["ln1"], x)
        x = x + (frac_attention(p, a, 2 * len(params["pos_emb"])) if "alpha" in p else attention(p, a, n_heads))
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
    for mixer in ["softmax", "frac"]:
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

    # flat weights must give the running mean: catches FFT wrap-around and normalization bugs
    # With every weight equal, output i must be the plain average of v_0..v_i.
    v = jax.random.normal(key, (seq_len, 2, 3))
    running_mean = jnp.cumsum(v, axis=0) / jnp.arange(1, seq_len + 1)[:, None, None]
    assert jnp.allclose(causal_mix(jnp.ones((2, seq_len)), v), running_mean, atol=1e-5)
