"""Low-rank-completion attention in JAX vs flash attention (cuDNN) and exact softmax.

lowrank_attn: observe m exact logits per query row (random causal keys + the diagonal), fit S ~ U V^T by ALS
on those entries only (O(T*m*r^2) per sweep), then causal softmax(U V^T) @ v. The final softmax is still dense O(T^2).
    .env/bin/python attn_lowrank_jax.py   -> table on stdout, runs/approx/lowrank_vs_flash.png
"""
import os
import time
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@partial(jax.jit, static_argnames=("m", "iters"))
def lowrank_attn(q, k, v, key, m, iters=10, lam=0.1, idx=None, U0=None, V0=None):
    """q, k, v: (T, d) -> (T, d). Rank r = min(64, m // 2), as in attn_approx.lowrank.
    idx (T, m) observed keys per row and U0, V0 (T, r) initial factors can be passed in (tests); else drawn from key."""
    T, d = q.shape
    r = min(64, max(m // 2, 1))
    i = jnp.arange(T, dtype=jnp.int32)
    k0, k1, k2 = jax.random.split(key, 3)
    if idx is None:
        idx = jnp.floor(jax.random.uniform(k0, (T, m), q.dtype) * (i[:, None] + 1)).astype(jnp.int32)  # with replacement
        idx = idx.at[:, 0].set(i)
    s = jnp.einsum("td,tmd->tm", q, k[idx]) / d ** 0.5  # the only logits computed exactly
    U = 0.1 * jax.random.normal(k1, (T, r), q.dtype) if U0 is None else U0
    V = 0.1 * jax.random.normal(k2, (T, r), q.dtype) if V0 is None else V0
    eye = lam * jnp.eye(r, dtype=q.dtype)

    def sweep(UV, _):
        U, V = UV
        Vo = V[idx]  # row update: each row's observed columns
        U = jnp.linalg.solve(jnp.einsum("tmr,tms->trs", Vo, Vo) + eye, jnp.einsum("tm,tmr->tr", s, Vo)[..., None])[..., 0]

        def col(gb, kk):  # column update: scatter row i's outer product into its observed column idx[i, kk]
            G, b = gb
            j = idx[:, kk]
            return (G.at[j].add(U[:, :, None] * U[:, None, :]), b.at[j].add(s[:, kk, None] * U)), None

        (G, b), _ = jax.lax.scan(col, (jnp.zeros((T, r, r), q.dtype) + eye, jnp.zeros((T, r), q.dtype)), jnp.arange(m))
        return (U, jnp.linalg.solve(G, b[..., None])[..., 0]), None

    (U, V), _ = jax.lax.scan(sweep, (U, V), None, length=iters)
    P = jax.nn.softmax(jnp.where(i[:, None] >= i[None, :], U @ V.T, -jnp.inf), axis=-1)
    return P @ v


def flash(q, k, v):  # (T, d) bf16 -> (T, d), cuDNN fused kernel
    return jax.nn.dot_product_attention(q[None, :, None], k[None, :, None], v[None, :, None],
                                        is_causal=True, implementation="cudnn")[0, :, 0]


def exact(q, k, v):  # f32 reference
    return jax.nn.dot_product_attention(q[None, :, None], k[None, :, None], v[None, :, None],
                                        is_causal=True, implementation="xla")[0, :, 0]


def relerr(O, R): return float(jnp.linalg.norm(O.astype(jnp.float32) - R) / jnp.linalg.norm(R))


def bench(f, *args, reps=5):
    """-> (median ms, temp GiB)."""
    fn = jax.jit(f).lower(*args).compile()
    temp = fn.memory_analysis().temp_size_in_bytes / 2**30
    fn(*args).block_until_ready()
    n = 20  # calls queued back to back, one sync at the end: a sync costs ~1 ms under WSL (see bench_attention.py)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n): out = fn(*args)
        out.block_until_ready(); ts.append((time.perf_counter() - t0) / n)
    return sorted(ts)[reps // 2] * 1e3, temp


if __name__ == "__main__":
    from heads import get_heads
    key = jax.random.PRNGKey(0)
    ms = [16, 32, 64, 128, 256]
    heads = get_heads(with_qk=True)
    print(f"{jax.devices()[0].device_kind}; accuracy on {len(heads)} real heads (T=1024, d=64), rel. output error vs f32 exact")
    err = {"flash (bf16)": np.zeros(len(heads))} | {m: np.zeros(len(heads)) for m in ms}
    for j, (*_, q, k, v) in enumerate(heads):
        q, k, v = (jnp.asarray(a, jnp.float32) for a in (q, k, v))
        R = exact(q, k, v)
        err["flash (bf16)"][j] = relerr(flash(*(a.astype(jnp.bfloat16) for a in (q, k, v))), R)
        for m in ms: err[m][j] = relerr(lowrank_attn(q, k, v, key, m), R)
    for name, e in err.items(): print(f"  {name!s:>13}: mean {e.mean():.4f}  min {e.min():.4f}  max {e.max():.4f}")

    print("\ntime / temp memory, synthetic q,k,v (d=64)")
    Ts = [1024, 2048, 4096, 8192]
    rows = {n: [] for n in ("flash", "lowrank m=64", "lowrank m=128", "exact xla f32")}
    for T in Ts:
        q, k, v = jax.random.normal(key, (3, T, 64))
        qb, kb, vb = (a.astype(jnp.bfloat16) for a in (q, k, v))
        rows["flash"].append(bench(flash, qb, kb, vb))
        rows["lowrank m=64"].append(bench(partial(lowrank_attn, m=64), q, k, v, key))
        rows["lowrank m=128"].append(bench(partial(lowrank_attn, m=128), q, k, v, key))
        rows["exact xla f32"].append(bench(exact, q, k, v))
        print(f"  T={T:5d} " + "  ".join(f"{n}: {t:8.2f} ms {g:5.2f} GiB" for n, (t, g) in ((n, rows[n][-1]) for n in rows)), flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    mean = [err[m].mean() for m in ms]
    ax[0].plot(ms, mean, "o-", label="lowrank (ALS, JAX)")
    ax[0].fill_between(ms, [err[m].min() for m in ms], [err[m].max() for m in ms], alpha=0.15)
    ax[0].axhline(err["flash (bf16)"].mean(), color="C3", ls="--", label="flash bf16 rounding")
    ax[0].set(xscale="log", yscale="log", xlabel="exact logits per row (m)", ylabel="output rel. error", title="accuracy, 12 real heads"); ax[0].legend()
    for n, r in rows.items():
        ax[1].plot(Ts, [t for t, _ in r], "o-", label=n); ax[2].plot(Ts, [g for _, g in r], "o-", label=n)
    ax[1].set(xscale="log", yscale="log", xlabel="T", ylabel="ms", title="forward time"); ax[1].legend()
    ax[2].set(xscale="log", yscale="log", xlabel="T", ylabel="temp GiB", title="temp memory")
    fig.tight_layout(); fig.savefig("runs/approx/lowrank_vs_flash.png", dpi=120)
