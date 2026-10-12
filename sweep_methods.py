"""Accuracy and speed sweeps of attn_methods_jax.py on real heads of the trained softmax LM, vs flash and exact.
    .env/bin/python sweep_methods.py   -> runs/approx/methods_sweep.csv, methods_timing.csv, methods_sweep.png
Accuracy: 12 heads x 3 seeds, T=1024, d=64, f32, output rel. error vs exact causal softmax; cost = exact-logit equivalents
per row (exact causal attention averages 512.5). Timing: random q, k, v, queued calls, temp memory from XLA."""
import csv
import os
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from attn_lowrank_jax import bench, exact, flash  # noqa: E402
from attn_methods_jax import (active_cost, bandit_cost, bitonic_topk, exact_causal, lowrank_attn, nystrom, ppr,  # noqa: E402
                              sampled, sparse_softmax, stationary)
from heads import T, get_heads  # noqa: E402

SEEDS = 3
EXACT_COST = (T + 1) / 2
ms = (16, 32, 64, 128, 256)
# name -> list of (label, fn(q,k,v,key) -> (Y, cost))
cost_of = lambda c: (lambda q, k, v, key: (None, c))
CONFIGS = {
    "sampled": [(f"m={m}", lambda q, k, v, key, m=m: (sampled(q, k, v, key, m=m), m)) for m in ms],
    "nystrom": [(f"m={m}", lambda q, k, v, key, m=m: (nystrom(q, k, v, key, m=m), m)) for m in ms],
    "lowrank": [(f"m={m}", lambda q, k, v, key, m=m: (lowrank_attn(q, k, v, key, m=m), m)) for m in ms],
    "sparse softmax": [(f"w={w}+{m}", lambda q, k, v, key, w=w, m=m: (sparse_softmax(q, k, v, key, m=m, window=w), w / 1 + m - w * w / (2 * T)))
                       for w, m in ((16, 8), (32, 16), (64, 64), (128, 128))],
    "ppr a=0.7": [(f"w={w}+{m}", lambda q, k, v, key, w=w, m=m: (ppr(q, k, v, key, m=m, window=w), w + m - w * w / (2 * T)))
                  for w, m in ((16, 8), (32, 16), (64, 64), (128, 128))],
    "stationary b=0.5": [("all", lambda q, k, v, key: (stationary(q, k, v, key, beta=0.5), EXACT_COST))],
    "bandit": [(f"z={z}", lambda q, k, v, key, z=z: bandit_cost(q, k, v, key, z=z)) for z in (0.5, 1.0, 2.0, 3.0)],
    "bitonic top-k": [(f"k={k_} all keys", lambda q, k, v, key, k_=k_: (bitonic_topk(q, k, v, key, topk=k_, tau=0.05), EXACT_COST)) for k_ in (16, 64)]
                     + [("k=32 pool 128", lambda q, k, v, key: (bitonic_topk(q, k, v, key, topk=32, tau=0.05, pool=128), 125.0))],
    "active": [(f"p={p}", lambda q, k, v, key, p=p: active_cost(q, k, v, key, p=p)) for p in (8, 16, 32)],
}
CONFIGS = {n: [(lbl, jax.jit(f)) for lbl, f in cs] for n, cs in CONFIGS.items()}


def accuracy():
    heads = get_heads(with_qk=True)
    rows = []
    for l, h, S, V, Q, K in heads:
        q, k, v = (jnp.asarray(a, jnp.float32) for a in (Q, K, V))
        R = exact_causal(q, k, v)
        for name, cs in CONFIGS.items():
            for lbl, f in cs:
                for seed in range(SEEDS):
                    Y, c = f(q, k, v, jax.random.PRNGKey(seed))
                    rows.append((name, lbl, float(c), seed, l, h, float(jnp.linalg.norm(Y - R) / jnp.linalg.norm(R))))
        print(f"head L{l} H{h} done", flush=True)
    with open("runs/approx/methods_sweep.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "config", "cost", "seed", "layer", "head", "rel_err"]); w.writerows(rows)
    return rows


def timing():
    key = jax.random.PRNGKey(0)
    fns = {"flash": flash, "exact (xla f32)": exact,
           "sampled m=64": partial(sampled, m=64), "nystrom m=64": partial(nystrom, m=64), "lowrank m=64": partial(lowrank_attn, m=64),
           "sparse softmax w=32+16": partial(sparse_softmax, m=16, window=32), "ppr w=32+16": partial(ppr, m=16, window=32),
           "stationary": stationary, "bandit z=2": lambda q, k, v, key: bandit_cost(q, k, v, key, z=2.0)[0],
           "bitonic k=32 pool 128": partial(bitonic_topk, topk=32, tau=0.05, pool=128), "bitonic k=16 all keys": partial(bitonic_topk, topk=16, tau=0.05),
           "active p=16": lambda q, k, v, key: active_cost(q, k, v, key)[0]}
    rows = []
    for Tn in (1024, 2048, 4096):
        q, k, v = jax.random.normal(key, (3, Tn, 64))
        for name, f in fns.items():
            if name == "bitonic k=16 all keys" and Tn > 1024: continue  # gathers a (T, T, d) tensor
            try:
                args = tuple(a.astype(jnp.bfloat16) for a in (q, k, v)) if name == "flash" else (q, k, v) + (() if name in ("flash", "exact (xla f32)") else (key,))
                ms_, gib = bench(f, *args, reps=3)
            except Exception as e:  # out of memory etc.
                ms_, gib = float("nan"), float("nan"); print(f"  {name} T={Tn}: {type(e).__name__}")
            rows.append((name, Tn, ms_, gib)); print(f"T={Tn:5d} {name:24s} {ms_:9.2f} ms {gib:6.2f} GiB", flush=True)
    with open("runs/approx/methods_timing.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "T", "ms", "temp_gib"]); w.writerows(rows)
    return rows


def plot(acc, tim):
    col = plt.cm.tab20(np.linspace(0, 1, 20))
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.5))
    names = list(dict.fromkeys(r[0] for r in acc))
    for i, name in enumerate(names):
        pts = {}
        for n, lbl, c, seed, l, h, e in acc:
            if n == name: pts.setdefault(lbl, []).append((c, e))
        cs = np.array([np.mean([c for c, _ in v]) for v in pts.values()]); es = np.array([np.mean([e for _, e in v]) for v in pts.values()])
        o = np.argsort(cs)
        ax[0].plot(cs[o], es[o], "o-", label=name, ms=4, color=col[2 * i % 20])
    ax[0].plot([EXACT_COST], [1e-3], "k*", ms=12, label="exact (error 0)")
    ax[0].set(xscale="log", yscale="log", xlabel="exact-logit equivalents per row", ylabel="output rel. error (mean of 12 heads x 3 seeds)",
              title="accuracy vs cost, real heads T=1024"); ax[0].legend(fontsize=7)
    tn = list(dict.fromkeys(r[0] for r in tim))
    for i, n in enumerate(tn):
        pts = [(r[1], r[2], r[3]) for r in tim if r[0] == n]
        for j, a in enumerate(ax[1:]):
            a.plot([p[0] for p in pts], [p[1 + j] for p in pts], "o-" if i < 10 else "s--", label=n, ms=4, color=col[i % 20])
    ax[1].set(xscale="log", yscale="log", xlabel="T", ylabel="forward ms", title="time (d=64, f32; flash bf16)"); ax[1].legend(fontsize=7)
    ax[2].set(xscale="log", yscale="log", xlabel="T", ylabel="temp GiB", title="XLA temp memory")
    fig.tight_layout(); fig.savefig("runs/approx/methods_sweep.png", dpi=110)


def read_csvs():
    acc = [(r["method"], r["config"], float(r["cost"]), int(r["seed"]), int(r["layer"]), int(r["head"]), float(r["rel_err"]))
           for r in csv.DictReader(open("runs/approx/methods_sweep.csv"))]
    tim = [(r["method"], int(r["T"]), float(r["ms"]), float(r["temp_gib"])) for r in csv.DictReader(open("runs/approx/methods_timing.csv"))]
    return acc, tim


if __name__ == "__main__":
    import sys
    plot(*(read_csvs() if sys.argv[1:] == ["plot"] else (accuracy(), timing())))
