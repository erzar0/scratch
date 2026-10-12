"""Val loss of the trained softmax model (runs/train/softmax_seq1024.pkl) with its attention swapped for an approximation.
    .env/bin/python eval_fast.py         -> runs/approx/fast_loss.csv
    T=4096 .env/bin/python eval_fast.py  -> runs/approx/fast_loss_T4096.csv (model runs/train/softmax_seq4096.pkl)"""
import csv, os, pickle, sys
from functools import partial
import jax, jax.numpy as jnp, numpy as np
import transformer
from attn_fast import chunk_attn
from attn_methods_jax import bandit, exact_causal

T, H, B = int(os.environ.get("T", 1024)), 8, 2
params = pickle.load(open(f"runs/train/softmax_seq{T}.pkl", "rb"))
data = np.fromfile("data/wt103_validation.bin", np.uint16).astype(np.int32)
N_SEQ = min(64, (len(data) - 1) // T // B * B)  # validation is 247k tokens: 60 windows at T=4096
batches = [jnp.asarray(data[(np.arange(i, i + B)[:, None]) * T + np.arange(T + 1)]) for i in range(0, N_SEQ, B)]


def val_loss(fn, name):
    transformer.APPROX[name] = fn  # a fresh name per config: jax.checkpoint in transformer.mix caches traces by name
    p = dict(params, layers=[dict(l, **{name: {}}) for l in params["layers"]])
    @jax.jit
    def loss(p, batch):
        logits = jax.vmap(partial(transformer.forward, n_heads=H), in_axes=(None, 0))(p, batch[:, :-1])
        return (jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, batch[:, 1:, None], -1)[..., 0]).mean()
    return float(np.mean([loss(p, b) for b in batches]))


CONFIGS = {"exact": lambda q, k, v, key: exact_causal(q, k, v)}
if T == 1024: CONFIGS["bandit per-key z=2 (dense)"] = partial(bandit, z=2.0)  # (T, T) buffers per head: OOM at 4096
for c, nr in (((16, 2), (16, 4), (16, 8), (32, 2), (32, 4), (32, 8)) if T == 1024 else
              ((32, 4), (32, 8), (32, 16), (64, 4), (64, 8))):
    for m in ("oracle", "hier", "bandit", "active", "stationary"):
        CONFIGS[f"{m} c{c} r{nr}"] = partial(chunk_attn, method=m, chunk=c, n_ref=nr)
if __name__ == "__main__":
    rows = []
    for name, fn in CONFIGS.items():
        rows.append((name, val_loss(fn, name))); print(f"{name:30s} {rows[-1][1]:.4f}", flush=True)
    with open("runs/approx/fast_loss.csv" if T == 1024 else f"runs/approx/fast_loss_T{T}.csv", "w", newline="") as f:
        csv.writer(f).writerows([("config", "val_loss")] + rows)
