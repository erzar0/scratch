"""Tests for attn_methods_jax.py against independent loop-style float64 NumPy references (same style as test_attention.py).
    .env/bin/python test_attn_methods.py        (no pytest; exit code 1 if anything fails)
Random draws (sample indices, dim permutation, projection, initial observations) are passed to both sides, so
parity is exact up to float64 rounding. Also: exactness limits, causality, gradients, jit/vmap, edge cases, real heads.
Causality is asserted both ways: methods that must be causal, and methods that leak the future (documented findings)."""
import os
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from attn_methods_jax import (active_cost, bandit_cost, bitonic_topk, exact_causal, idx_mask, lowrank_attn,  # noqa: E402
                              nystrom, ppr, sample_idx, sampled, sparse_softmax, stationary, METHODS)

TESTS = []
key = jax.random.PRNGKey(0)
T, D = 48, 32


def test(f):
    TESTS.append(f)
    return f


def qkv(seed=0, t=T, d=D, scale=1.0):
    q, k, v = jax.random.normal(jax.random.fold_in(key, seed), (3, t, d), jnp.float64)
    return q * scale, k * scale, v


def np_(*a): return [np.asarray(x, np.float64) for x in a]


def sm(z):
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    return float(np.abs(a - b).max())


def close(a, b, tol, what=""):
    d = diff(a, b)
    assert d < tol, f"{what} max diff {d:.3e} >= {tol:.0e}"
    return f"max diff {d:.1e}"


# ----- numpy references, written from the definitions, one query at a time ---------------------------------------------

def ref_exact(q, k, v):
    return np.stack([sm(q[i] @ k[:i + 1].T / np.sqrt(q.shape[1])) @ v[:i + 1] for i in range(len(q))])


def ref_sampled(q, k, v, idx):
    out = []
    for i in range(len(q)):
        js = np.unique(idx[i])
        out.append(sm(q[i] @ k[js].T / np.sqrt(q.shape[1])) @ v[js])
    return np.stack(out)


def ref_nystrom(q, k, v, m):
    T = len(q)
    L = np.linspace(0, T - 1, max(m // 2, 1)).astype(int)
    S = q @ k.T / np.sqrt(q.shape[1])
    c = max(S[:, L].max(), S[L, :].max())
    E = np.exp(S - c)
    P = E[:, L] @ np.linalg.pinv(E[np.ix_(L, L)], rcond=1e-3) @ E[L, :]
    P = np.where(np.tril(np.ones((T, T), bool)), np.maximum(P, 0), 0) + 1e-12 * np.eye(T)
    return (P / P.sum(1, keepdims=True)) @ v


def ref_ppr(q, k, v, idx, window, alpha):
    T = len(q)
    S = q @ k.T / np.sqrt(q.shape[1])
    P = np.zeros((T, T))
    for i in range(T):
        js = sorted(set(idx[i]) | {j for j in range(i + 1) if i - j < window})
        P[i, js] = sm(S[i, js])
    return (1 - alpha) * np.linalg.inv(np.eye(T) - alpha * P) @ v


def ref_stationary(q, k, v, beta, damp, iters):
    T = len(q)
    P = np.stack([np.concatenate([sm(q[i] @ k[:i + 1].T / np.sqrt(q.shape[1])), np.zeros(T - i - 1)]) for i in range(T)])
    pi = np.full(T, 1 / T)
    for _ in range(iters): pi = damp * pi @ P + (1 - damp) / T
    if beta == 0: return np.tile(pi @ v, (T, 1))
    W = P * pi[None] ** beta
    return (W / W.sum(1, keepdims=True)) @ v


def ref_bandit(q, k, v, perm, z, tau=3.0, chunk=8):
    T, d = q.shape
    Y, cost = np.zeros_like(v), 0.0
    for i in range(T):
        act = np.ones(i + 1, bool); s1 = np.zeros(i + 1); s2 = np.zeros(i + 1)
        for t in range(0, d - chunk, chunk):
            c = perm[t:t + chunk]
            cost += act.sum() * chunk / d
            x = q[i, c][None] * k[:i + 1][:, c] / np.sqrt(d)  # per-dimension products
            s1 += x.sum(1); s2 += (x ** 2).sum(1)
            n = t + chunk
            est = s1 * d / n
            rad = z * np.sqrt(np.maximum(s2 / n * d - est ** 2 / d, 1e-12) * d) / np.sqrt(n)
            best = (est - rad)[act].max()
            act &= est + rad >= best - tau
        cost += act.sum() * chunk / d
        Y[i] = sm(q[i] @ k[:i + 1][act].T / np.sqrt(d)) @ v[:i + 1][act]
    return Y, cost / T


def ref_bitonic(q, k, v, topk, tau, pool, payload):
    T, d = q.shape
    n = int(2 ** np.ceil(np.log2(T))) if pool is None else pool
    sig = lambda x: 1 / (1 + np.exp(-x))
    Y = np.zeros_like(v)
    for i in range(T):
        pidx = np.arange(n) if pool is None else i - (n - 1) + np.arange(n)
        ok = (pidx >= 0) & (pidx <= i) & (pidx < T)
        pc = np.clip(pidx, 0, T - 1)
        s = np.where(ok, k[pc] @ q[i] / np.sqrt(d), -50.0)
        s0 = s.copy()
        pay = v[pc].copy()
        size = 2
        while size <= n:
            step = size // 2
            while step >= 1:
                for x in range(n):
                    y = x ^ step
                    if y > x:
                        lo, hi = (x, y) if (x & size) == 0 else (y, x)  # smaller must end at lo
                        g = sig((s[lo] - s[hi]) / tau)
                        s[lo], s[hi] = g * s[hi] + (1 - g) * s[lo], g * s[lo] + (1 - g) * s[hi]
                        pay[lo], pay[hi] = g * pay[hi] + (1 - g) * pay[lo], g * pay[lo] + (1 - g) * pay[hi]
                step //= 2
            size *= 2
        w = sm(s[-topk:])
        if payload: Y[i] = w @ pay[-topk:]
        else:
            top = np.argsort(-s0, kind="stable")[:topk]
            Y[i] = sm(s[-topk:][::-1]) @ v[pc][top]
    return Y


def ref_active(q, k, v, R, init_obs, m0, rounds, b, p, ucb, lam):
    T, d = q.shape
    F, S = k @ R, q @ k.T / np.sqrt(d)
    Y = np.zeros_like(v)
    for i in range(T):
        obs = init_obs[i].copy()
        for r in range(rounds + 1):
            Fo, so = F[obs], S[i][obs]
            Ai = np.linalg.inv(Fo.T @ Fo + lam * np.eye(p))
            mu = F[:i + 1] @ (Ai @ Fo.T @ so)
            if r == rounds: break
            sd = np.sqrt(np.einsum("jp,pq,jq->j", F[:i + 1], Ai, F[:i + 1]))
            acq = np.where(obs[:i + 1], -np.inf, mu + ucb * sd)
            for j in np.argsort(-acq, kind="stable")[:min(b, i + 1)]:
                if acq[j] > -np.inf: obs[j] = True
        logit = np.where(obs[:i + 1], S[i, :i + 1], mu)
        Y[i] = sm(logit) @ v[:i + 1]
    return Y, m0 + rounds * b + p / d * T


def ref_lowrank(q, k, v, idx, U, V, lam, iters):
    T, d = q.shape
    r = U.shape[1]
    s = np.stack([q[i] @ k[idx[i]].T / np.sqrt(d) for i in range(T)])
    U, V = U.copy(), V.copy()
    for _ in range(iters):
        for i in range(T):
            Vo = V[idx[i]]
            U[i] = np.linalg.solve(Vo.T @ Vo + lam * np.eye(r), Vo.T @ s[i])
        G = np.stack([lam * np.eye(r)] * T); bb = np.zeros((T, r))
        for i in range(T):
            for kk in range(idx.shape[1]):
                j = idx[i, kk]
                G[j] += np.outer(U[i], U[i]); bb[j] += s[i, kk] * U[i]
        V = np.stack([np.linalg.solve(G[j], bb[j]) for j in range(T)])
    Sh = U @ V.T
    return np.stack([sm(Sh[i, :i + 1]) @ v[:i + 1] for i in range(T)])


# ----- 1. parity --------------------------------------------------------------------------------------------------------

@test
def parity_sampled():
    q, k, v = qkv(1)
    idx = sample_idx(key, T, 8, jnp.float64)
    return close(sampled(q, k, v, key, idx=idx), ref_sampled(*np_(q, k, v), np.asarray(idx)), 1e-10)


@test
def parity_nystrom():
    q, k, v = qkv(2, scale=0.3)
    return close(nystrom(q, k, v, key, m=16), ref_nystrom(*np_(q, k, v), 16), 1e-8)


@test
def parity_ppr():
    q, k, v = qkv(3)
    idx = sample_idx(key, T, 4, jnp.float64)
    return close(ppr(q, k, v, key, window=6, alpha=0.7, idx=idx), ref_ppr(*np_(q, k, v), np.asarray(idx), 6, 0.7), 1e-9)


@test
def parity_stationary():
    q, k, v = qkv(4)
    out = []
    for beta in (0, 0.5, 2):
        out.append(close(stationary(q, k, v, key, beta=beta, iters=60), ref_stationary(*np_(q, k, v), beta, 0.85, 60), 1e-10, f"beta={beta}"))
    return out[-1]


@test
def parity_bandit():
    q, k, v = qkv(5)
    perm = jax.random.permutation(key, D)
    for z in (0.5, 2.0):
        Y, c = bandit_cost(q, k, v, key, z=z, perm=perm)
        Yr, cr = ref_bandit(*np_(q, k, v), np.asarray(perm), z)
        close(Y, Yr, 1e-10, f"z={z}"); assert abs(float(c) - cr) < 1e-9, f"cost {float(c)} vs {cr}"
    return f"cost {float(c):.2f} (z=2)"


@test
def parity_bitonic_payload():
    q, k, v = qkv(6, t=32)
    return close(bitonic_topk(q, k, v, key, topk=8, tau=0.5, payload=True), ref_bitonic(*np_(q, k, v), 8, 0.5, None, True), 1e-10)


@test
def parity_bitonic_scores_only():
    q, k, v = qkv(7, t=32)
    return close(bitonic_topk(q, k, v, key, topk=8, tau=0.5, payload=False), ref_bitonic(*np_(q, k, v), 8, 0.5, None, False), 1e-10)


@test
def parity_bitonic_window_pool():
    q, k, v = qkv(8, t=40)  # pool 16 < T and T not a power of two
    a = close(bitonic_topk(q, k, v, key, topk=4, tau=0.3, pool=16, payload=True), ref_bitonic(*np_(q, k, v), 4, 0.3, 16, True), 1e-10)
    close(bitonic_topk(q, k, v, key, topk=4, tau=0.3, pool=16), ref_bitonic(*np_(q, k, v), 4, 0.3, 16, False), 1e-10)
    return a


@test
def parity_active():
    q, k, v = qkv(9, t=40)
    R = jax.random.normal(key, (D, 8), jnp.float64) / np.sqrt(8)
    init = idx_mask(sample_idx(key, 40, 6, jnp.float64)) & jnp.tril(jnp.ones((40, 40), bool))
    kw = dict(m0=6, rounds=2, b=5, p=8, ucb=1.0, lam=1.0)
    Y, c = active_cost(q, k, v, key, R=R, init_obs=init, **kw)
    Yr, cr = ref_active(*np_(q, k, v, R), np.asarray(init), **kw)
    close(Y, Yr, 1e-8); assert abs(float(c) - cr) < 1e-9
    return f"cost {float(c):.1f}"


@test
def parity_lowrank():
    q, k, v = qkv(10, t=40, scale=0.5)
    m, r = 8, 4
    idx = sample_idx(key, 40, m, jnp.float64)
    U0, V0 = 0.1 * jax.random.normal(key, (2, 40, r), jnp.float64)
    Y = lowrank_attn(q, k, v, key, m=m, iters=3, lam=0.1, idx=idx, U0=U0, V0=V0)
    return close(Y, ref_lowrank(*np_(q, k, v), np.asarray(idx), np.asarray(U0), np.asarray(V0), 0.1, 3), 1e-8)


# ----- 2. exactness limits ----------------------------------------------------------------------------------------------

@test
def exact_sampled_all_keys():
    q, k, v = qkv(11)
    idx = jnp.minimum(jnp.arange(T)[None, :], jnp.arange(T)[:, None])  # every causal key at least once
    return close(sampled(q, k, v, key, idx=idx), exact_causal(q, k, v), 1e-12)


@test
def exact_sparse_softmax_full_window():
    q, k, v = qkv(12)
    return close(sparse_softmax(q, k, v, key, m=1, window=T), exact_causal(q, k, v), 1e-12)


@test
def exact_bandit_huge_z():
    q, k, v = qkv(13)
    Y, c = bandit_cost(q, k, v, key, z=1e6)
    close(Y, exact_causal(q, k, v), 1e-12)
    return f"cost {float(c):.1f} of {(T + 1) / 2:.1f} causal keys/row"


@test
def exact_active_all_observed():
    q, k, v = qkv(14)
    return close(active_cost(q, k, v, key, init_obs=jnp.tril(jnp.ones((T, T), bool)))[0], exact_causal(q, k, v), 1e-12)


@test
def exact_bitonic_all_keys_small_tau():
    q, k, v = qkv(15, t=32)
    e = float(np.linalg.norm(bitonic_topk(q, k, v, key, topk=32, tau=1e-3, payload=True) - exact_causal(q, k, v)) /
              np.linalg.norm(exact_causal(q, k, v)))
    assert e < 1e-2, f"rel err {e:.2e}"
    return f"rel err {e:.1e}"


@test
def exact_lowrank_full_observation():
    q, k, v = qkv(16, t=24, d=8, scale=0.5)  # logit rank <= 8, rank 12 factors, every entry observed
    idx = jnp.broadcast_to(jnp.arange(24), (24, 24)); idx = jnp.minimum(idx, jnp.arange(24)[:, None])
    Y = lowrank_attn(q, k, v, key, m=24, iters=40, lam=1e-9, idx=idx)
    e = float(np.linalg.norm(Y - exact_causal(q, k, v)) / np.linalg.norm(exact_causal(q, k, v)))
    assert e < 1e-3, f"rel err {e:.2e}"
    return f"rel err {e:.1e}"


@test
def limits_ppr_stationary():
    q, k, v = qkv(17)
    close(ppr(q, k, v, key, alpha=0.0), v, 1e-12, "ppr alpha=0 is the identity (Y = V)")
    Y = stationary(q, k, v, key, beta=0)
    close(Y, np.tile(np.asarray(Y)[:1], (T, 1)), 1e-14, "beta=0: same output for every query")
    close(stationary(q, k, v, key, beta=0, damp=0.0), np.tile(np.asarray(v).mean(0), (T, 1)), 1e-12, "damp=0: pi uniform")
    return "ok"


# ----- 3. causality -----------------------------------------------------------------------------------------------------

def leak(name, **cfg):
    """Max change in rows <= i0 when q, k, v after i0 are replaced."""
    i0 = T // 2
    q, k, v = qkv(20)
    q2, k2, v2 = (a.at[i0 + 1:].set(5.0 * jax.random.normal(jax.random.fold_in(key, 99 + j), (T - i0 - 1, D))) for j, a in enumerate((q, k, v)))
    f = METHODS[name]
    return diff(f(q, k, v, key, **cfg)[:i0 + 1], f(q2, k2, v2, key, **cfg)[:i0 + 1])


CAUSAL = dict(sampled=dict(m=8), ppr=dict(m=4, window=8), sparse_softmax=dict(m=4, window=8), bandit=dict(z=1.0),
              bitonic=dict(topk=8, tau=0.5), active=dict(m0=6, rounds=2, b=5))
LEAKY = dict(nystrom=dict(m=16), lowrank=dict(m=8, iters=3), stationary=dict(beta=0.5))


@test
def causal_methods_do_not_leak():
    worst = {n: leak(n, **c) for n, c in CAUSAL.items()}
    bad = {n: d for n, d in worst.items() if d > 1e-9}
    assert not bad, f"future tokens change past outputs: {bad}"
    return "max change " + ", ".join(f"{n} {d:.0e}" for n, d in worst.items())


@test
def known_leaks_nystrom_lowrank_stationary():
    """Findings, not bugs to hide: these three read the future, so they cannot be used as autoregressive mixers as written.
    nystrom: landmark queries/keys beyond i enter row i. lowrank: V_j is fit from rows > j. stationary: pi_j sums rows > j."""
    got = {n: leak(n, **c) for n, c in LEAKY.items()}
    assert all(d > 1e-6 for d in got.values()), f"expected a leak in each: {got}"
    return "leak " + ", ".join(f"{n} {d:.1e}" for n, d in got.items())


# ----- 4. gradients -----------------------------------------------------------------------------------------------------

def fd_check(f, q, k, v, eps=1e-6):
    loss = lambda q_: jnp.sum(f(q_, k, v) ** 2)
    g = jax.grad(loss)(q)
    u = jax.random.normal(jax.random.fold_in(key, 77), q.shape, jnp.float64)
    fd = (loss(q + eps * u) - loss(q - eps * u)) / (2 * eps)
    an = jnp.vdot(g, u)
    return float(abs(fd - an) / (abs(fd) + 1e-12))


@test
def gradients_match_finite_differences():
    q, k, v = qkv(30, t=32)
    idx = sample_idx(key, 32, 4, jnp.float64)
    cases = {
        "sampled": lambda q, k, v: sampled(q, k, v, key, idx=idx),
        "ppr": lambda q, k, v: ppr(q, k, v, key, window=6, idx=idx),
        "stationary": lambda q, k, v: stationary(q, k, v, key, beta=0.5, iters=40),
        "bitonic payload": lambda q, k, v: bitonic_topk(q, k, v, key, topk=8, tau=0.5, payload=True),
        "bitonic scores": lambda q, k, v: bitonic_topk(q, k, v, key, topk=8, tau=0.5),
    }
    errs = {n: fd_check(f, q, k, v) for n, f in cases.items()}
    bad = {n: e for n, e in errs.items() if e > 1e-4}
    # the scores-only variant treats the hard top-k selection and its values as constants, so only finiteness holds
    bad.pop("bitonic scores", None)
    assert not bad, f"gradient vs finite difference: {bad}"
    return ", ".join(f"{n} {e:.0e}" for n, e in errs.items())


@test
def gradients_finite_for_selection_methods():
    q, k, v = qkv(31, t=32)
    for name, f in {"bandit": lambda q, k, v: bandit_cost(q, k, v, key, z=1.0)[0],
                    "active": lambda q, k, v: active_cost(q, k, v, key, m0=6, rounds=2, b=4)[0],
                    "lowrank": lambda q, k, v: lowrank_attn(q, k, v, key, m=8, iters=2),
                    "nystrom": lambda q, k, v: nystrom(q, k, v, key, m=8)}.items():
        g = jax.grad(lambda q, k, v: jnp.sum(f(q, k, v) ** 2), argnums=(0, 1, 2))(q, k, v)
        assert all(bool(jnp.isfinite(x).all()) for x in g), f"{name}: non-finite gradient"
        assert float(jnp.abs(g[2]).max()) > 0, f"{name}: zero gradient to v"
    return "finite, nonzero"


# ----- 5. plumbing and edge cases ---------------------------------------------------------------------------------------

ALL = dict(sampled=dict(m=8), nystrom=dict(m=8), lowrank=dict(m=8, iters=2), ppr=dict(m=4, window=8),
           sparse_softmax=dict(m=4, window=8), stationary=dict(beta=0.5), bandit=dict(z=1.0),
           bitonic=dict(topk=8, tau=0.5), active=dict(m0=6, rounds=2, b=5))


@test
def jit_equals_eager_and_vmap_equals_loop():
    qs, ks, vs = (jax.random.normal(jax.random.fold_in(key, 40 + j), (3, 32, D), jnp.float64) for j in range(3))
    keys = jax.random.split(key, 3)
    worst = 0.0
    for n, cfg in ALL.items():
        f = lambda q, k, v, kk: METHODS[n](q, k, v, kk, **cfg)
        eager = jnp.stack([f(qs[h], ks[h], vs[h], keys[h]) for h in range(3)])
        worst = max(worst, diff(jax.jit(jax.vmap(f))(qs, ks, vs, keys), eager))
    assert worst < 1e-8, worst
    return f"max diff {worst:.0e}"


@test
def float32_finite_and_shapes():
    q, k, v = (a.astype(jnp.float32) for a in qkv(41))
    for n, cfg in ALL.items():
        Y = jax.jit(lambda q, k, v: METHODS[n](q, k, v, key, **cfg))(q, k, v)
        assert Y.shape == v.shape and Y.dtype == jnp.float32 and bool(jnp.isfinite(Y).all()), (n, Y.dtype)
    return f"{len(ALL)} methods"


@test
def edge_cases_t1_odd_lengths_big_budget():
    for t in (1, 2, 7, 33):
        q, k, v = qkv(50 + t, t=t)
        for n, cfg in ALL.items():
            if n == "nystrom" and t < 2: continue
            cfg = {**cfg, **({"m": 64} if "m" in cfg else {}), **({"topk": min(8, 1 << int(np.ceil(np.log2(t))))} if n == "bitonic" else {})}
            if n == "active": cfg = dict(m0=64, rounds=1, b=64)
            Y = METHODS[n](q, k, v, key, **cfg)
            assert Y.shape == v.shape and bool(jnp.isfinite(Y).all()), (n, t)
    q, k, v = qkv(60, t=1)
    close(exact_causal(q, k, v), v, 1e-12, "T=1 attends to itself")
    return "T in 1, 2, 7, 33; m, m0 > T"


@test
def bandit_cost_is_below_exact_and_monotone_in_z():
    q, k, v = qkv(61, t=64, scale=1.5)
    cs = [float(bandit_cost(q, k, v, key, z=z)[1]) for z in (0.5, 1, 2, 4)]
    assert all(a <= b + 1e-9 for a, b in zip(cs, cs[1:])), cs
    assert cs[0] < (64 + 1) / 2, cs
    return "cost " + " <= ".join(f"{c:.1f}" for c in cs)


# ----- 6. real heads ----------------------------------------------------------------------------------------------------

@test
def real_heads_match_numpy_sweep_ranges():
    from heads import get_heads
    heads = [h for h in get_heads(with_qk=True) if (h[0], h[1]) in ((0, 2), (3, 2))]
    errs = {}
    for l, h, S, V, Q, K in heads:
        q, k, v = (jnp.asarray(a[:256], jnp.float64) for a in (Q, K, V))
        R = exact_causal(q, k, v)
        e = lambda Y: float(jnp.linalg.norm(Y - R) / jnp.linalg.norm(R))
        errs[(l, h)] = dict(bandit=e(bandit_cost(q, k, v, key, z=2.0)[0]), lowrank=e(lowrank_attn(q, k, v, key, m=128)),
                            sampled=e(sampled(q, k, v, key, m=64)), active=e(active_cost(q, k, v, key)[0]))
    for lh, r in errs.items():
        assert r["bandit"] < 0.05, (lh, r)       # numpy sweep: 0.009 mean at z=2
        assert r["lowrank"] < 0.6, (lh, r)       # 0.15 mean at m=128, T=1024 (T=256 here)
        assert r["sampled"] < 1.5 and r["active"] < 0.3, (lh, r)
    return "; ".join(f"L{l}H{h}: " + " ".join(f"{k} {v:.3f}" for k, v in r.items()) for (l, h), r in errs.items())


# ----- 7. chunk-level versions (attn_fast.py) ---------------------------------------------------------------------------

from attn_fast import chunk_attn, chunk_sparse  # noqa: E402

CHUNK_METHODS = ("hier", "bandit", "active", "stationary")


@test
def chunk_exact_when_every_chunk_refined():
    q, k, v = qkv(70, t=45)  # 6 chunks of 8, last one padded
    R = exact_causal(q, k, v)
    worst = max(diff(chunk_attn(q, k, v, key, method=m, chunk=8, n_ref=6, summaries=su, sel=se), R)
                for m in CHUNK_METHODS for su in (True, False) for se in ("all", "mean"))
    worst = max(worst, *(diff(chunk_sparse(q, k, v, key, method=m, chunk=8, n_ref=6, impl="xla"), R) for m in CHUNK_METHODS))
    assert worst < 1e-12, worst
    return f"max diff {worst:.0e}"


@test
def chunk_sparse_equals_chunk_attn_sparse_mean():
    """Two implementations of the same config: per-chunk lax.map (chunk_attn) vs all chunks at once + one fused call."""
    q, k, v = qkv(71, t=45)
    worst = max(diff(chunk_sparse(q, k, v, key, method=m, chunk=8, n_ref=2, impl="xla"),
                     chunk_attn(q, k, v, key, method=m, chunk=8, n_ref=2, summaries=False, sel="mean")) for m in CHUNK_METHODS)
    assert worst < 1e-12, worst
    return f"max diff {worst:.0e}"


@test
def chunk_split_path_bf16_matches_f32():
    if jax.devices()[0].platform != "gpu": return "skipped: cuDNN needs a GPU"
    q, k, v = (a.astype(jnp.float32) for a in qkv(72, t=1000, d=64, scale=1.5))
    b = lambda a: a.astype(jnp.bfloat16)
    errs = {}
    for m in CHUNK_METHODS:
        R = chunk_sparse(q, k, v, key, method=m, chunk=64, n_ref=4, impl="xla")
        Y = chunk_sparse(b(q), b(k), b(v), key, method=m, chunk=64, n_ref=4, impl="split").astype(jnp.float32)
        errs[m] = float(jnp.linalg.norm(Y - R) / jnp.linalg.norm(R))
    assert max(errs.values()) < 0.02, errs  # bf16 inputs, outputs and lse residuals
    return "rel err " + ", ".join(f"{m} {e:.3f}" for m, e in errs.items())


@test
def chunk_causal_across_chunks():
    """Rows of chunks < c never see chunks >= c (within a chunk the picks use the chunk's mean query: a documented leak).
    stationary's pi sums every chunk, so it can leak across chunks (only through a flipped pick: not seen on this data)."""
    q, k, v = qkv(73, t=64)
    c0 = 6 * 8  # chunks 6, 7 replaced; chunk 5 picks 1 of 4 old chunks, so a leak can show
    q2, k2, v2 = (a.at[c0:].set(5.0 * jax.random.normal(jax.random.fold_in(key, 80 + j), (64 - c0, D))) for j, a in enumerate((q, k, v)))
    ch = {m: diff(chunk_sparse(q, k, v, key, method=m, chunk=8, n_ref=1, impl="xla")[:c0],
                  chunk_sparse(q2, k2, v2, key, method=m, chunk=8, n_ref=1, impl="xla")[:c0]) for m in CHUNK_METHODS}
    assert all(ch[m] < 1e-12 for m in ("hier", "bandit", "active")), ch  # stationary: reads later chunks, not asserted
    return ", ".join(f"{m} {d:.0e}" for m, d in ch.items())


if __name__ == "__main__":
    failed = []
    for f in TESTS:
        t0 = time.time()
        try:
            note = f() or ""
            print(f"PASS {f.__name__:45s} {time.time() - t0:6.1f}s  {note}", flush=True)
        except Exception:
            failed.append(f.__name__)
            print(f"FAIL {f.__name__:45s} {time.time() - t0:6.1f}s", flush=True)
            traceback.print_exc()
    print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    raise SystemExit(1 if failed else 0)
