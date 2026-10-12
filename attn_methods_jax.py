"""JAX versions of the attention approximations in attn_approx.py / attn_tournament.py.
Every method: f(q, k, v, key, **cfg) -> (T, d) for one head, causal, jit- and vmap-able. Randomness comes from `key`,
or can be passed in (idx, perm, R, init_obs) so tests can feed the numpy references the same draws.
Several methods emulate the sparsity densely (a (T, T) mask): they match the numpy results, they are not faster.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from attn_lowrank_jax import lowrank_attn  # noqa: F401  (re-exported: ALS completion of the logit matrix)

NEG = -50.0  # score of a masked key inside the bitonic network (finite, as in attn_tournament.py)


def causal(T): return jnp.tril(jnp.ones((T, T), bool))


def logits(q, k): return q @ k.T / q.shape[-1] ** 0.5


def masked_softmax(S, M):
    """Row softmax over the True entries of M. Every row of M must have at least one."""
    return jax.nn.softmax(jnp.where(M, S, -jnp.inf), axis=-1)


def exact_causal(q, k, v): return masked_softmax(logits(q, k), causal(len(q))) @ v


def sample_idx(key, T, m, dtype=jnp.float32):
    """(T, m) keys per row, uniform over the causal prefix with replacement; column 0 is the diagonal."""
    i = jnp.arange(T, dtype=jnp.int32)
    idx = jnp.floor(jax.random.uniform(key, (T, m), dtype) * (i[:, None] + 1)).astype(jnp.int32)
    return idx.at[:, 0].set(i)


def idx_mask(idx):
    T = len(idx)
    return jnp.zeros((T, T), bool).at[jnp.arange(T)[:, None], idx].set(True)


# 1. sampled softmax: exact logits only at m random causal keys per row (self-normalised importance sampling)
def sampled(q, k, v, key, m=64, idx=None):
    idx = sample_idx(key, len(q), m, q.dtype) if idx is None else idx
    s = jnp.einsum("td,tmd->tm", q, k[idx]) / q.shape[-1] ** 0.5
    S = jnp.full((len(q),) * 2, -jnp.inf, q.dtype).at[jnp.arange(len(q))[:, None], idx].set(s)
    return jax.nn.softmax(S, axis=-1) @ v


# 2. Nystrom: exp-kernel completed from m/2 landmark rows and columns, then causal mask and row-normalise
def nystrom(q, k, v, key, m=64):
    T = len(q)
    L = jnp.asarray(np.linspace(0, T - 1, max(m // 2, 1)).astype(int))
    Ec, Er = logits(q, k[L]), logits(q[L], k)  # (T, l), (l, T): the only logits computed
    c = jnp.maximum(Ec.max(), Er.max())        # one global shift; cancels in the row normalisation
    Ec, Er = jnp.exp(Ec - c), jnp.exp(Er - c)
    P = Ec @ jnp.linalg.pinv(Ec[L], rtol=1e-3) @ Er
    P = jnp.where(causal(T), jnp.maximum(P, 0), 0) + 1e-12 * jnp.eye(T, dtype=q.dtype)
    return (P / P.sum(1, keepdims=True)) @ v


# 3. personalised PageRank on the sparse logits: Pi V = (1-a)(I - aP)^-1 V; I - aP is lower triangular
def ppr(q, k, v, key, m=16, window=32, alpha=0.7, idx=None):
    T = len(q)
    M = idx_mask(sample_idx(key, T, m, q.dtype) if idx is None else idx)
    i = jnp.arange(T)
    M |= (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)
    S = jnp.where(M, logits(q, k), -jnp.inf)
    P = jax.nn.softmax(S, axis=-1)
    return (1 - alpha) * solve_triangular(jnp.eye(T, dtype=q.dtype) - alpha * P, v, lower=True)


def sparse_softmax(q, k, v, key, m=16, window=32, idx=None):
    """Baseline for ppr: softmax over the same window + random set."""
    T = len(q)
    M = idx_mask(sample_idx(key, T, m, q.dtype) if idx is None else idx)
    i = jnp.arange(T)
    M |= (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)
    return masked_softmax(logits(q, k), M) @ v


# 4. stationary distribution of the (teleporting) causal attention chain
def stationary(q, k, v, key, beta=0.5, damp=0.85, iters=None):  # iters: unused, kept for the numpy-parity tests
    T = len(q)
    P = masked_softmax(logits(q, k), causal(T))
    # fixed point of pi = damp * pi P + (1 - damp) / T: P^T is upper triangular, so one triangular solve instead of `iters` matvecs
    pi = (1 - damp) / T * solve_triangular(jnp.eye(T, dtype=q.dtype) - damp * P.T, jnp.ones(T, q.dtype), lower=False)
    if beta == 0:  # the attention IS pi: one output for every query
        return jnp.broadcast_to(pi @ v, v.shape)
    W = P * pi[None, :] ** beta
    return (W / W.sum(1, keepdims=True)) @ v


# 5. bandit: successive elimination on partial dot products (chunks of dims), survivors get the full dot product
def bandit_cost(q, k, v, key, z=2.0, tau=3.0, chunk=8, perm=None):
    """-> (output (T, d), cost in exact-logit equivalents per row)."""
    T, d = q.shape
    perm = jax.random.permutation(key, d) if perm is None else perm
    active = causal(T)
    sm = ss = jnp.zeros((T, T), q.dtype)
    cost = 0.0
    for t in range(0, d - chunk, chunk):  # last chunk is skipped: survivors get the full dot product instead
        c = perm[t:t + chunk]
        cost += active.sum() * chunk / d
        sm += q[:, c] @ k[:, c].T / d ** 0.5
        ss += (q[:, c] ** 2) @ (k[:, c] ** 2).T / d
        n = t + chunk
        est = sm * d / n
        rad = z * jnp.sqrt(jnp.maximum(ss / n * d - est ** 2 / d, 1e-12) * d) / n ** 0.5
        best = jnp.where(active, est - rad, -jnp.inf).max(1, keepdims=True)
        active &= est + rad >= best - tau
    cost += active.sum() * chunk / d
    return masked_softmax(logits(q, k), active) @ v, cost / T


def bandit(q, k, v, key, **cfg): return bandit_cost(q, k, v, key, **cfg)[0]


# 6. bitonic sorting network with soft compare-swap -> differentiable top-k attention
def _stages(n):
    """Per stage of the bitonic network: partner index of every position, and +1/-1 for 'ends smaller'/'ends larger'."""
    out, sz = [], 2
    while sz <= n:
        step = sz // 2
        while step >= 1:
            idx = np.arange(n); partner = idx ^ step
            first = np.where((idx & sz) == 0, idx < partner, idx > partner)  # this position receives the smaller of the pair
            out.append((partner, np.where(first, 1.0, -1.0)))
            step //= 2
        sz *= 2
    return out


def bitonic_topk(q, k, v, key, topk=16, tau=0.05, pool=None, payload=False):
    """Per query: score the pool keys, soft-sort the scores with a bitonic network, softmax over the topk largest.
    pool=None: every key (padded to a power of two). pool=n: the n most recent keys (n a power of two).
    payload=True drags the values through the soft swaps (faithful, O(T * n * d) memory); payload=False sorts
    only the scores and takes the values of the hard top-k (gradient flows through the scores)."""
    T, d = q.shape
    n = int(2 ** np.ceil(np.log2(T))) if pool is None else pool
    i = jnp.arange(T)[:, None]
    pidx = jnp.arange(n)[None, :] if pool is None else i - (n - 1) + jnp.arange(n)[None, :]
    valid = (pidx >= 0) & (pidx <= i) & (pidx < T)
    pidx = jnp.clip(jnp.broadcast_to(pidx, (T, n)), 0, T - 1)
    if pool is None:  # scores of every key (padded with zero rows)
        raw = q @ jnp.pad(k, ((0, n - T), (0, 0))).T
    else:  # band: key t-(n-1)+p for p = 0..n-1, one shifted elementwise product per p (never builds a (T, n, d) tensor)
        kp = jnp.pad(k, ((n - 1, 0), (0, 0)))
        raw = jnp.stack([jnp.sum(q * kp[p:p + T], axis=-1) for p in range(n)], axis=1)
    s = jnp.where(valid, raw / d ** 0.5, NEG)
    s0 = s
    pay = v[pidx] if payload else None
    for partner, sign in _stages(n):
        sp = s[:, partner]
        g = jax.nn.sigmoid(jnp.asarray(sign, s.dtype) * (s - sp) / tau)  # soft "swap" gate: ~1 if this pair is in the wrong order
        if payload: pay = g[..., None] * pay[:, partner] + (1 - g[..., None]) * pay
        s = g * sp + (1 - g) * s
    w = jax.nn.softmax(s[:, -topk:], axis=-1)  # ascending order: the last topk are the largest
    if payload:
        return jnp.einsum("tk,tkd->td", w, pay[:, -topk:])
    top = jax.lax.stop_gradient(jax.lax.top_k(s0, topk)[1])  # hard indices of the largest, descending
    vals = v[jnp.take_along_axis(pidx, top, axis=1)]  # (T, topk, d)
    return jnp.einsum("tk,tkd->td", w[:, ::-1], vals)


def _spd_inv(A):
    """Batched inverse of SPD matrices by unrolled Gauss-Jordan (cuSOLVER's batched LU/Cholesky is slower for 16x16)."""
    p = A.shape[-1]
    M = jnp.concatenate([A, jnp.broadcast_to(jnp.eye(p, dtype=A.dtype), A.shape)], -1)
    for i in range(p):
        row = M[:, i, :] / M[:, i, i, None]
        M = (M - M[:, :, i, None] * row[:, None, :]).at[:, i, :].set(row)
    return M[..., p:]


# 7. active sampling: random sketch, then UCB acquisition under a Bayesian ridge model on a p-dim key projection
def active_cost(q, k, v, key, m0=16, rounds=3, b=16, p=16, ucb=1.0, lam=1.0, R=None, init_obs=None):
    """-> (output (T, d), cost in exact-logit equivalents per row: observed logits + predicting the rest)."""
    T, d = q.shape
    kR, kI = jax.random.split(key)
    R = jax.random.normal(kR, (d, p), q.dtype) / p ** 0.5 if R is None else R
    F = k @ R
    S = logits(q, k)
    if init_obs is None:
        rnd = jax.random.uniform(kI, (T, T), q.dtype)
        valid_all = causal(T)
        top = jax.lax.top_k(jnp.where(valid_all, rnd, -1.0), min(m0, T))[1]
        init_obs = idx_mask(top) & valid_all

    FF = (F[:, :, None] * F[:, None, :]).reshape(T, p * p)  # F_j F_j^T per key, so every row's Gram matrix is one matmul
    valid = causal(T)
    obs = init_obs
    for r in range(rounds + 1):  # all rows at once: batched ridge posterior from matmuls + an unrolled Gauss-Jordan inverse
        G = (obs.astype(q.dtype) @ FF).reshape(T, p, p) + lam * jnp.eye(p, dtype=q.dtype)
        Ai = _spd_inv(G)
        mu = jnp.einsum("tp,jp->tj", jnp.einsum("tpq,tq->tp", Ai, jnp.where(obs, S, 0.0) @ F), F)
        if r == rounds: break
        sd = jnp.sqrt(jnp.maximum(Ai.reshape(T, p * p) @ FF.T, 0))
        acq = jnp.where(obs | ~valid, -jnp.inf, mu + ucb * sd)
        top = jax.lax.top_k(acq, min(b, T))
        picked = jnp.zeros((T, T), bool).at[jnp.arange(T)[:, None], top[1]].set(top[0] > -jnp.inf)  # only real candidates
        obs = obs | picked
    P = jax.nn.softmax(jnp.where(valid, jnp.where(obs, S, mu), -jnp.inf), axis=-1)
    return P @ v, m0 + rounds * b + p / d * T


def active(q, k, v, key, **cfg): return active_cost(q, k, v, key, **cfg)[0]


METHODS = dict(sampled=sampled, nystrom=nystrom, lowrank=lowrank_attn, ppr=ppr, sparse_softmax=sparse_softmax,
               stationary=stationary, bandit=bandit, bitonic=bitonic_topk, active=active)
