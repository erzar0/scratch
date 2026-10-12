"""Chunk-level bandit / active / stationary: versions that skip work, so they can beat exact attention at long T.

Per-key selection (attn_methods_jax.py) needs a top-k over a (T, T) score matrix: 2.2 ms at T=4k, 13 ms at 8k,
vs 0.30 / 0.47 ms for flash. So these choose whole key chunks per query chunk and read only those exactly.
The executor is hier.py's (single head, f(q, k, v, key) -> (T, d)). Query chunk Q sees:
    [summary (mean key/value) of every old chunk not refined ; window chunks token by token ; n_ref chunks token by token]
Only the choice of the n_ref refined chunks (and for active, the summaries) differs:
  hier        summary mass: softmax over summaries + window, summed over the chunk's queries (hier.py)
  bandit      chunks = arms, sampled keys = pulls. s1 keys from every chunk -> keep the m best by UCB on log-mass
              -> s2 more keys from each survivor -> refine the n_ref best by the pooled estimate
  active      Gaussian model of each chunk's keys (mean + diagonal variance): predicted log-mass q.mu + q^2.var/2
              (lognormal) and UCB = + kappa * sd. Refining = observing the exact logits of the top-UCB chunks
  stationary  stationary distribution pi of the chunk graph (summary attention between chunks) as a prior:
              score = summary logit + beta * log pi
Caveats: as in hier.py, a chunk's picks depend on all of its queries, so query i sees later tokens of its own
chunk through the selection. stationary's pi (and the chunk-mean queries) also read later chunks."""
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


@partial(jax.jit, static_argnames=("method", "chunk", "n_ref", "window", "s1", "s2", "m", "z", "kappa", "beta", "block", "sink", "summaries", "sel"))
def chunk_attn(q, k, v, key, method="hier", chunk=32, n_ref=8, window=None, s1=2, s2=8, m=None, z=1.0, kappa=1.0,
               beta=0.5, block=64, sink=False, summaries=True, sel="all"):
    T, d = q.shape
    window = window or 2 * chunk
    n_local = window // chunk
    Tp = -(-T // chunk) * chunk
    pad = lambda a: jnp.pad(a, ((0, Tp - T), (0, 0)))
    n = Tp // chunk
    n_ref = min(n_ref, n)
    m = min(m or 3 * n_ref, n)
    qc, kc, vc = (pad(a).reshape(n, chunk, d) for a in (q, k, v))
    scale, logC, ch = d ** -0.5, jnp.log(jnp.asarray(chunk, q.dtype)), jnp.arange(n)
    kmean, vmean = kc.mean(1), vc.mean(1)
    kvar = kc.var(1) * scale ** 2
    causal = jnp.tril(jnp.ones((chunk, window), bool), window - chunk)
    k1, k2 = jax.random.split(key)
    if method == "bandit":  # the same sampled positions for every query chunk: one gather up front
        ks1 = kc[ch[:, None], jax.random.randint(k1, (n, s1), 0, chunk)]  # (n, s1, d)
        ks2 = kc[ch[:, None], jax.random.randint(k2, (n, s2), 0, chunk)]  # (n, s2, d)
    if method == "stationary":
        A = jnp.where(ch[:, None] >= ch[None, :], qc.mean(1) @ kmean.T * scale, -jnp.inf)
        P = jax.nn.softmax(A, -1)
        pi = solve_triangular(jnp.eye(n, dtype=q.dtype) - 0.85 * P.T, jnp.full(n, 0.15 / n, q.dtype), lower=False)
        log_pi = jnp.log(pi / pi.sum())

    def mass(est, ll):  # (C, n) per-query log-mass of each chunk -> (n,) mass summed over the chunk's queries
        return jax.nn.softmax(jnp.concatenate([est, ll], -1), -1)[:, :est.shape[1]].sum(0)

    def one_chunk(qq, Q):
        old = ch <= Q - n_local
        loc = Q - n_local + 1 + jnp.arange(n_local)
        lk, lv = (a[jnp.maximum(loc, 0)].reshape(window, d) for a in (kc, vc))
        ll = jnp.where(causal & jnp.repeat(loc >= 0, chunk), qq @ lk.T * scale, -jnp.inf)
        # Selection queries: every query of the chunk (score = estimated mass summed over them), or only the chunk's
        # mean query (score = its estimated log-mass): (1, n) instead of (C, n) selection logits
        qs = qq if sel == "all" else qq.mean(0, keepdims=True)
        agg = (lambda est, idx=slice(None): mass(jnp.where(old[idx], est, -jnp.inf), ll)) if sel == "all" else \
              (lambda est, idx=slice(None): est[0])

        if method == "oracle":  # exact mass of each chunk (reads every key: a bound for the scorers, not a method)
            pos = Q * chunk + jnp.arange(chunk)
            fl = jnp.where(pos[:, None] >= jnp.arange(Tp)[None], qq @ kc.reshape(-1, d).T * scale, -jnp.inf)
            score = jax.nn.softmax(fl, -1).reshape(chunk, n, chunk).sum((0, 2))
        elif method == "bandit":
            L1 = jnp.einsum("id,csd->ics", qs, ks1) * scale  # (Cs, n, s1) round 1: s1 pulls of every arm
            ucb = jax.nn.logsumexp(L1, -1) - jnp.log(s1) + logC + z * L1.std(-1)
            surv = jax.lax.top_k(jnp.where(old, agg(ucb), -jnp.inf), m)[1]  # (m,)
            L2 = jnp.einsum("id,csd->ics", qs, ks2[surv]) * scale  # round 2: s2 more pulls of the m survivors
            est2 = jax.nn.logsumexp(jnp.concatenate([L1[:, surv], L2], -1), -1) - jnp.log(s1 + s2) + logC
            score = jnp.full(n, -jnp.inf, q.dtype).at[surv].set(agg(est2, surv))
        else:
            est = qs @ kmean.T * scale + logC  # mean key: log-mass of a chunk of identical keys
            if method == "active":
                var = (qs ** 2) @ kvar.T  # lognormal: log E exp(q.k) = q.mu + q^T Sigma q / 2 (diagonal Sigma), + UCB
                est = est + var / 2 + kappa * jnp.sqrt(var)
            if method == "stationary":
                est = est + beta * log_pi
            score = agg(est)
        score = jnp.where(old, score, -jnp.inf)
        if sink: score = score.at[0].set(jnp.where(old[0], 1e9, -jnp.inf))  # chunk 0 (attention sink) always refined
        top_s, ref = jax.lax.top_k(jax.lax.stop_gradient(score), n_ref)
        ok = top_s > -jnp.inf  # fewer than n_ref old chunks: the rest are padding
        rl = jnp.where(jnp.repeat(ok, chunk), qq @ kc[ref].reshape(-1, d).T * scale, -jnp.inf)
        rv = vc[ref].reshape(-1, d)
        if not summaries:  # sparse: window + refined chunks only, unrefined old chunks are dropped
            w = jax.nn.softmax(jnp.concatenate([ll, rl], -1), -1)
            return w[:, :window] @ lv + w[:, window:] @ rv
        is_ref = ((ref[:, None] == ch) & ok[:, None]).any(0)
        sl = jnp.where(old & ~is_ref, qq @ kmean.T * scale + logC, -jnp.inf)  # summaries of the unrefined old chunks
        w = jax.nn.softmax(jnp.concatenate([sl, ll, rl], -1), -1)
        return w[:, :n] @ vmean + w[:, n:n + window] @ lv + w[:, n + window:] @ rv

    out = jax.lax.map(lambda a: one_chunk(*a), (qc, ch), batch_size=block)
    return out.reshape(Tp, d)[:T]


METHODS = ("hier", "bandit", "active", "stationary")


@partial(jax.jit, static_argnames=("method", "chunk", "n_ref", "window", "s1", "s2", "m", "z", "kappa", "beta", "sink", "impl"))
def chunk_sparse(*args, impl="cudnn", **kw):
    """Fast path of chunk_attn(summaries=False, sel="mean"): same output (up to bf16 rounding). impl:
    "split" (default for bf16 on GPU via chunk_sparse(impl="cudnn")): rows whose old chunks all fit in n_ref are plain causal
      attention (one flash call on that prefix); every later query chunk is two fused calls without a mask tensor, merged
      by their log-sum-exp: [its n_ref refined chunks] and causal [its window chunks]. Forward only: the lse residual
      carries no gradient.
    "cudnn" / "xla": one fused call per query chunk over [window ; refined] with a mask (xla for float64 / CPU)."""
    return _chunk_sparse(*args, impl=impl, **kw)


def _chunk_sparse(q, k, v, key, method="hier", chunk=64, n_ref=8, window=None, s1=2, s2=8, m=None, z=1.0, kappa=1.0,
                 beta=0.5, sink=False, impl="cudnn"):
    """Every query chunk's picks come at once from (n, n) chunk scores (chunk-mean queries), then fused attention calls."""
    T, d = q.shape
    window = window or 2 * chunk
    n_local = window // chunk
    Tp = -(-T // chunk) * chunk
    n = Tp // chunk
    n_ref, m = min(n_ref, n), min(m or 3 * min(n_ref, n), n)
    qc, kc, vc = (jnp.pad(a, ((0, Tp - T), (0, 0))).reshape(n, chunk, d) for a in (q, k, v))
    scale, ch = d ** -0.5, jnp.arange(n)
    f32 = lambda a: a.astype(jnp.promote_types(a.dtype, jnp.float32))  # scores in >= f32: bf16 ranks chunks badly
    qm, kmean = f32(qc).mean(1), f32(kc).mean(1)
    old = ch[None, :] <= ch[:, None] - n_local  # (query chunk, key chunk)
    if method == "bandit":
        k1, k2 = jax.random.split(key)
        ks1 = f32(kc[ch[:, None], jax.random.randint(k1, (n, s1), 0, chunk)])
        ks2 = f32(kc[ch[:, None], jax.random.randint(k2, (n, s2), 0, chunk)])
        L1 = jnp.einsum("qd,csd->qcs", qm, ks1) * scale  # (n, n, s1)
        ucb = jax.nn.logsumexp(L1, -1) - jnp.log(s1) + z * L1.std(-1)
        surv = jax.lax.top_k(jnp.where(old, ucb, -jnp.inf), m)[1]  # (n, m)
        L2 = jnp.einsum("qd,qmsd->qms", qm, ks2[surv]) * scale
        L1s = jnp.take_along_axis(L1, surv[..., None], 1)
        est2 = jax.nn.logsumexp(jnp.concatenate([L1s, L2], -1), -1)
        score = jnp.full((n, n), -jnp.inf, qm.dtype).at[ch[:, None], surv].set(est2)
    elif method == "random":  # baseline: random old chunks
        score = jax.random.uniform(key, (n, n), qm.dtype)
    else:
        score = qm @ kmean.T * scale
        if method == "active":
            var = (qm ** 2) @ (f32(kc).var(1) * scale ** 2).T
            score = score + var / 2 + kappa * jnp.sqrt(var)
        if method == "stationary":
            P = jax.nn.softmax(jnp.where(ch[:, None] >= ch[None, :], qm @ kmean.T * scale, -jnp.inf), -1)
            pi = solve_triangular(jnp.eye(n, dtype=qm.dtype) - 0.85 * P.T, jnp.full(n, 0.15 / n, qm.dtype), lower=False)
            score = score + beta * jnp.log(pi / pi.sum())
    score = jnp.where(old, score, -jnp.inf)
    if sink: score = score.at[:, 0].set(jnp.where(old[:, 0], 1e9, -jnp.inf))
    top_s, ref = jax.lax.top_k(jax.lax.stop_gradient(score), n_ref)  # (n, n_ref)
    if impl == "split":
        return _split(qc, kc, vc, ref, n_local, T)
    loc = ch[:, None] - n_local + 1 + jnp.arange(n_local)  # (n, n_local)
    kid = jnp.concatenate([jnp.maximum(loc, 0), ref], 1)
    K, V = (a[kid].reshape(n, -1, d) for a in (kc, vc))
    causal = jnp.tril(jnp.ones((chunk, window), bool), window - chunk)
    mask = jnp.concatenate([causal[None] & jnp.repeat(loc >= 0, chunk, 1)[:, None],
                            jnp.broadcast_to(jnp.repeat(top_s > -jnp.inf, chunk, 1)[:, None], (n, chunk, n_ref * chunk))], -1)
    if impl == "cudnn":
        out = jax.nn.dot_product_attention(qc[:, :, None], K[:, :, None], V[:, :, None], mask=mask[:, None], implementation=impl)
    else:  # written out: jax.nn.dot_product_attention's xla path computes float64 inputs in float32
        out = jax.nn.softmax(jnp.where(mask, jnp.einsum("ncd,nkd->nck", qc, K) * scale, -jnp.inf), -1) @ V
    return out.reshape(Tp, d)[:T]


def _split(qc, kc, vc, ref, n_local, T):
    n, C, d = qc.shape
    n_ref, W = ref.shape[1], n_local * C
    dpa = lambda q, k, v, **kw: jax.nn.dot_product_attention(q[:, :, None], k[:, :, None], v[:, :, None], implementation="cudnn", **kw)
    P = min(n_ref + n_local, n)  # chunks 0 .. P-1 have at most n_ref old chunks: all refined = exact causal attention
    out_p = dpa(*(a[:P].reshape(1, P * C, d) for a in (qc, kc, vc)), is_causal=True).reshape(P * C, d)
    if P == n:
        return out_p[:T]
    Qs = jnp.arange(P, n)
    o1, l1 = dpa(qc[Qs], *(a[ref[Qs]].reshape(n - P, n_ref * C, d) for a in (kc, vc)), return_residual=True)
    loc = Qs[:, None] - n_local + 1 + jnp.arange(n_local)
    o2, l2 = dpa(*(a[loc].reshape(n - P, W, d) for a in (qc, kc, vc)), is_causal=True, return_residual=True)
    o2, l2 = o2[:, W - C:], l2[:, W - C:]
    l1, l2 = l1.astype(jnp.float32), l2.astype(jnp.float32)
    mx = jnp.maximum(l1, l2)
    a1, a2 = jnp.exp(l1 - mx)[..., None], jnp.exp(l2 - mx)[..., None]
    out = ((o1 * a1 + o2 * a2) / (a1 + a2)).astype(qc.dtype)
    return jnp.concatenate([out_p, out.reshape(-1, d)])[:T]
