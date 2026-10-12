"""Fidelity and loss of approximations swapped into the T=4096 softmax model: val loss, and KL(exact model || approx model)
over next-token distributions (0 = identical predictions). Baselines: sliding window only, random chunk picks.
    .env/bin/python eval_kl.py   -> runs/approx/fast_kl.csv"""
import csv, os
os.environ["T"] = "4096"
from functools import partial
import jax, jax.numpy as jnp, numpy as np
import transformer
from attn_fast import chunk_sparse
from attn_methods_jax import exact_causal
from eval_fast import H, batches, params


def run(fn, name):
    transformer.APPROX[name] = fn
    pa = dict(params, layers=[dict(l, **{name: {}}) for l in params["layers"]])
    @jax.jit
    def f(pa, pe, batch):
        fwd = lambda p: jax.vmap(partial(transformer.forward, n_heads=H), in_axes=(None, 0))(p, batch[:, :-1])
        la, le = jax.nn.log_softmax(fwd(pa)), jax.nn.log_softmax(fwd(pe))
        loss = -jnp.take_along_axis(la, batch[:, 1:, None], -1).mean()
        return loss, (jnp.exp(le) * (le - la)).sum(-1).mean()
    r = np.array([f(pa, params, b) for b in batches])
    return r[:, 0].mean(), r[:, 1].mean()


def window(W):  # sliding window of W tokens (token-level), exact softmax inside
    return lambda q, k, v, key: jax.nn.dot_product_attention(q[None, :, None], k[None, :, None], v[None, :, None], is_causal=True,
                                                             local_window_size=(W - 1, 0), implementation="xla")[0, :, 0]


CONFIGS = {"exact": lambda q, k, v, key: exact_causal(q, k, v)}
for W in (256, 512, 1024): CONFIGS[f"window {W}"] = window(W)
for c, r in ((128, 2), (64, 4), (64, 8), (64, 16)):
    for m in ("random", "hier", "bandit", "active", "stationary"):
        CONFIGS[f"{m} c{c} r{r}"] = partial(chunk_sparse, method=m, chunk=c, n_ref=r, impl="xla")
if __name__ == "__main__":
    with open("runs/approx/fast_kl.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["config", "val_loss", "kl"])
        for name, fn in CONFIGS.items():
            loss, kl = run(fn, name)
            w.writerow([name, loss, kl]); fh.flush()
            print(f"{name:24s} loss {loss:.4f}  KL {kl:.4f}", flush=True)
