"""Train the transformer from transformer.py on WikiText-103 (GPT-2 BPE tokens): softmax, csa, hier or
hier_window mixer.

Prep once (downloads ~320 MB of parquet, writes data/wt103_*.bin):
    uv run --with tiktoken --with pyarrow --with numpy python train_lm.py prep
Train (JAX with CUDA from .env):
    .env/bin/python train_lm.py softmax
    .env/bin/python train_lm.py csa
    .env/bin/python train_lm.py hier
    .env/bin/python train_lm.py hier_window   # hier + a 512-token sliding window: exact attention unless SEQ > 512
SEQ=2048 BATCH=2 trains on 2048-token windows (same tokens per step) and logs to runs/<mixer>_seq2048.csv.
Logs go to runs/<mixer>.csv, final weights to runs/<mixer>.pkl. Loss is nats per BPE token, so perplexities are per BPE token,
not the word-level perplexities that papers report on WikiText-103.
"""
import math
import os
import pickle
import sys
import time
from functools import partial

import numpy as np

DATA = "data"
SPLITS = {"train": 2, "validation": 1, "test": 1}  # split -> number of parquet shards
VOCAB = 50304  # GPT-2's 50257 padded to a multiple of 64
D_MODEL, N_HEADS, N_LAYERS, SEQ = 512, 8, 8, int(os.environ.get("SEQ", 512))
BATCH, STEPS, WARMUP = int(os.environ.get("BATCH", 8)), int(os.environ.get("STEPS", 8000)), 300
PEAK_LR = float(os.environ.get("PEAK_LR", 1e-3))
EVAL_EVERY, EVAL_BATCHES = 500, 40


def prep():
    import urllib.request

    import pyarrow as pa
    import pyarrow.parquet as pq
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    base = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/"
    os.makedirs(DATA, exist_ok=True)
    for split, shards in SPLITS.items():
        lines = []
        for i in range(shards):
            raw = urllib.request.urlopen(f"{base}{split}-{i:05d}-of-{shards:05d}.parquet").read()
            lines += pq.read_table(pa.BufferReader(raw)).column("text").to_pylist()
        ids = np.concatenate([np.array(t, np.uint16) for t in enc.encode_ordinary_batch(lines) if t])
        ids.tofile(f"{DATA}/wt103_{split}.bin")
        print(f"{split}: {len(ids):,} tokens")


def load(split):
    return np.memmap(f"{DATA}/wt103_{split}.bin", dtype=np.uint16, mode="r")


if __name__ == "__main__" and sys.argv[1:] == ["prep"]:
    prep()
    sys.exit()

# Allocate on demand: under WSL only ~7 of the card's 16 GB could be allocated, and JAX's default
# 75% preallocation fails outright.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402  (prep runs in an env without jax)
import jax.numpy as jnp  # noqa: E402

from transformer import forward, init_params  # noqa: E402


def loss_fn(params, batch):
    """batch: (B, T+1) tokens -> mean next-token cross-entropy. Full MLP (no nested dropout)."""
    logits = jax.vmap(partial(forward, n_heads=N_HEADS), in_axes=(None, 0))(params, batch[:, :-1])
    target = jnp.take_along_axis(logits, batch[:, 1:, None], axis=-1)[..., 0]
    return (jax.nn.logsumexp(logits, axis=-1) - target).mean()


def lr_at(step):
    """Linear warmup, then cosine decay to 10% of peak."""
    warm = step / WARMUP
    cos = 0.1 + 0.45 * (1 + jnp.cos(jnp.pi * jnp.minimum(step / STEPS, 1.0)))
    return PEAK_LR * jnp.where(step < WARMUP, warm, cos)


@partial(jax.jit, donate_argnums=(0,), static_argnames="loss")
def train_step(state, batch, loss=loss_fn):
    """AdamW (b2 = 0.95, weight decay 0.1 on matrices only) with global-norm clipping at 1.
    loss(params, batch) -> scalar; evict.py passes its own to fine-tune under eviction."""
    params, m, v, step = state
    loss, grads = jax.value_and_grad(loss)(params, batch)
    # The CSA indexer learns only from its own loss, at a much larger gradient scale: left out of the
    # clip, or it would shrink every other gradient. Adam normalizes its scale anyway.
    lm = lambda path: all(getattr(k, "key", None) != "indexer" for k in path)
    norm = jnp.sqrt(sum(jnp.sum(g**2) for path, g in jax.tree_util.tree_leaves_with_path(grads) if lm(path)))
    clip = jnp.minimum(1.0, 1.0 / (norm + 1e-6))
    grads = jax.tree_util.tree_map_with_path(lambda path, g: g * clip if lm(path) else g, grads)
    step = step + 1
    m = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, m, grads)
    v = jax.tree.map(lambda v, g: 0.95 * v + 0.05 * g**2, v, grads)
    lr, c1, c2 = lr_at(step), 1 - 0.9**step, 1 - 0.95**step

    def update(p, m, v):
        decay = 0.1 * p if p.ndim == 2 else 0.0  # skip norms, biases, sink logits
        return p - lr * (m / c1 / (jnp.sqrt(v / c2) + 1e-8) + decay)

    params = jax.tree.map(update, params, m, v)
    return (params, m, v, step), loss, norm


eval_loss = jax.jit(loss_fn)


def evaluate(params, data, n_batches=None):
    """Mean loss over consecutive non-overlapping windows of SEQ tokens (all of them if n_batches is None)."""
    n_win = (len(data) - 1) // SEQ
    windows = np.arange(n_win)[: None if n_batches is None else n_batches * BATCH]
    losses = []
    for i in range(0, len(windows) - BATCH + 1, BATCH):  # ponytail: drops the last partial batch
        idx = windows[i : i + BATCH, None] * SEQ + np.arange(SEQ + 1)
        losses.append(float(eval_loss(params, jnp.asarray(data[idx], jnp.int32))))
    return float(np.mean(losses))


def main(mixer):
    train, val, test = load("train"), load("validation"), load("test")
    params = init_params(jax.random.PRNGKey(0), VOCAB, D_MODEL, N_HEADS, N_LAYERS, SEQ, mixer)
    zeros = lambda: jax.tree.map(jnp.zeros_like, params)  # m and v need separate buffers (donated)
    state = (params, zeros(), zeros(), jnp.zeros((), jnp.int32))
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"[{mixer}] {n_params / 1e6:.1f}M params, {STEPS} steps x {BATCH * SEQ} tokens")

    rng = np.random.default_rng(0)  # same data order for both mixers
    os.makedirs("runs", exist_ok=True)
    run = f"runs/{mixer}" if SEQ == 512 else f"runs/{mixer}_seq{SEQ}"
    log = open(f"{run}.csv", "w")
    log.write("step,tokens,train_loss,val_loss,seconds\n")
    t0, seen = time.time(), 0
    for step in range(1, STEPS + 1):
        starts = rng.integers(0, len(train) - SEQ - 1, BATCH)
        batch = jnp.asarray(train[starts[:, None] + np.arange(SEQ + 1)], jnp.int32)
        state, loss, norm = train_step(state, batch)
        seen += BATCH * SEQ
        if step % 100 == 0:
            loss = float(loss)
            assert math.isfinite(loss), f"[{mixer}] loss diverged at step {step}"
            el = time.time() - t0
            print(f"[{mixer}] step {step}: loss {loss:.3f}  |g| {float(norm):.2f}  {seen / el:,.0f} tok/s")
            val_loss = evaluate(state[0], val, EVAL_BATCHES) if step % EVAL_EVERY == 0 else ""
            if val_loss:
                print(f"[{mixer}] step {step}: val loss {val_loss:.3f} (ppl {math.exp(val_loss):.1f})")
            log.write(f"{step},{seen},{loss:.4f},{val_loss},{el:.1f}\n")
            log.flush()

    params = state[0]
    with open(f"{run}.pkl", "wb") as f:  # weights only, for evict.py; the optimizer state is dropped
        pickle.dump(jax.device_get(params), f)
    val_loss, test_loss = evaluate(params, val), evaluate(params, test)
    print(f"[{mixer}] final val {val_loss:.4f} (ppl {math.exp(val_loss):.2f}), "
          f"test {test_loss:.4f} (ppl {math.exp(test_loss):.2f}), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
