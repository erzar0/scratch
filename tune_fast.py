"""Accuracy (val loss of the T=4096 softmax model) and time (8 heads, T=4k/16k/32k) of chunk_attn configs.
    .env/bin/python tune_fast.py   -> runs/approx/fast_tune.csv (appends)"""
import csv, os, sys
os.environ["T"] = "4096"
from functools import partial
import jax, jax.numpy as jnp
from attn_fast import chunk_attn, chunk_sparse
from bench_fast import bench, FNS, H
from eval_fast import val_loss

key = jax.random.PRNGKey(0)
GRID = [dict(method=m, chunk=c, n_ref=r, summaries=su, sel=se) for m in sys.argv[1].split(",")
        for c, r in [tuple(map(int, x.split("x"))) for x in sys.argv[2].split(",")]
        for su, se in [(su == "T", se) for su, se in (x.split("/") for x in sys.argv[3].split(","))]]
if __name__ == "__main__":
    out = "runs/approx/fast_tune.csv"
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["method", "chunk", "n_ref", "summaries", "sel", "val_loss", "ms_4096", "ms_16384", "ms_32768"])
        for cfg in GRID:
            name = "{method} c{chunk} r{n_ref} {summaries} {sel}".format(**cfg)
            loss = val_loss(partial(chunk_attn, **cfg), name)
            fast = not cfg["summaries"] and cfg["sel"] == "mean"  # chunk_sparse computes exactly this config, fused
            f8 = jax.vmap(partial(chunk_sparse, impl="split", **{k: v for k, v in cfg.items() if k not in ("summaries", "sel")})
                          if fast else partial(chunk_attn, **cfg), (0, 0, 0, None))
            ms = [bench(f8, *jax.random.normal(key, (3, H, T, 64), jnp.bfloat16), key)[0] for T in (4096, 16384, 32768)]
            w.writerow([*cfg.values(), loss, *ms]); f.flush()
            print(f"{name:32s} loss {loss:.4f}  ms " + " ".join(f"{x:.2f}" for x in ms), flush=True)
