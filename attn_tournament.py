"""Four 'tournament' attentions on real heads of the trained softmax LM (heads.py), vs exact softmax. numpy.
Error = ||Yhat - Y|| / ||Y|| on the head output; cost = exact-logit equivalents per query row (one logit = d MACs).
  bandit    successive elimination on partial dot products (chunks of 8 dims, CLT confidence radius z)
  bitonic   soft-swap bitonic sorting network -> differentiable top-k attention (uses every logit; T=256 prefix)
  stationary  pi = pi G of the causal attention chain (global vector), plain and as a key re-weighting
  active    Bayesian ridge on a p=16 random projection of k: random sketch, UCB acquisition, exact on acquired keys
    .env/bin/python attn_tournament.py   -> table on stdout, runs/approx/tournament.png
"""
import numpy as np
import matplotlib.pyplot as plt
from attn_approx import softmax_masked
from heads import get_heads

rng = np.random.default_rng(0)
nrm = lambda Y, R: np.linalg.norm(Y - R) / np.linalg.norm(R)


def bandit(Q, K, V, z, tau=3.0, chunk=8):
    n, d = Q.shape
    S_full = Q @ K.T / np.sqrt(d)
    causal = np.tril(np.ones((n, n), bool))
    active, sm, ss, cost = causal.copy(), np.zeros((n, n)), np.zeros((n, n)), 0.0
    perm = rng.permutation(d)
    for t in range(0, d - chunk, chunk):  # last chunk is skipped: survivors get the full dot product instead
        c = perm[t:t + chunk]
        cost += active.sum() * chunk / d
        sm += Q[:, c] @ K[:, c].T / np.sqrt(d); ss += (Q[:, c] ** 2) @ (K[:, c] ** 2).T / d
        m = t + chunk
        est = sm * d / m
        rad = z * np.sqrt(np.maximum(ss / m * d - est ** 2 / d, 1e-12) * d) / np.sqrt(m)
        best = np.where(active, est - rad, -np.inf).max(1, keepdims=True)
        active &= est + rad >= best - tau
    cost += active.sum() * chunk / d  # survivors: finish the dot product
    P = softmax_masked(S_full, active)
    return P @ V, cost / n


def bitonic_topk(S, V, k, tau, qs):
    """S (n,n) logits (n a power of 2), V (n,d). Soft bitonic sort per query, then softmax over the top-k outputs."""
    n = S.shape[0]
    s = np.where(np.tril(np.ones_like(S, bool))[qs], S[qs], -50.0)  # (nq, n), masked keys sink to the bottom
    pay = np.broadcast_to(V, (len(qs), n, V.shape[1])).copy()      # payload follows the scores
    ns = 0
    sz = 2
    while sz <= n:
        step = sz // 2
        while step >= 1:
            idx = np.arange(n); partner = idx ^ step
            lo = idx < partner
            a, b = idx[lo], partner[lo]
            asc = (a & sz) == 0 if sz < n else np.ones(len(a), bool)  # direction of this sub-block
            a2, b2 = np.where(asc, a, b), np.where(asc, b, a)           # a2 should end smaller
            g = 1 / (1 + np.exp(-(s[:, a2] - s[:, b2]) / tau))          # soft "swap" gate
            sa, sb = s[:, a2].copy(), s[:, b2].copy()
            pa, pb = pay[:, a2].copy(), pay[:, b2].copy()
            s[:, a2], s[:, b2] = g * sb + (1 - g) * sa, g * sa + (1 - g) * sb
            pay[:, a2], pay[:, b2] = g[..., None] * pb + (1 - g[..., None]) * pa, g[..., None] * pa + (1 - g[..., None]) * pb
            ns += 1; step //= 2
        sz *= 2
    top = s[:, -k:]
    w = np.exp(top - top.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return np.einsum("qk,qkd->qd", w, pay[:, -k:]), ns


def stationary(S, V, beta, damp=0.85):
    n = len(S)
    P = softmax_masked(S)
    pi = np.full(n, 1 / n)
    for _ in range(100): pi = damp * (pi @ P) + (1 - damp) / n
    if beta == 0: return np.broadcast_to(pi @ V, V.shape)  # the spec: the attention IS pi, same for every query
    W = P * pi[None, :] ** beta
    return (W / W.sum(1, keepdims=True)) @ V


def active(Q, K, V, m0=16, rounds=3, b=16, p=16, ucb=1.0, lam=1.0):
    n, d = Q.shape
    R = rng.normal(size=(d, p)) / np.sqrt(p)
    F = K @ R                                  # (n, p) cheap key features
    S = Q @ K.T / np.sqrt(d)                   # only read at acquired entries below
    Y = np.zeros_like(V)
    for i in range(n):
        keys = np.arange(i + 1)
        obs = np.zeros(i + 1, bool)
        obs[rng.choice(i + 1, min(m0, i + 1), replace=False)] = True
        for r in range(rounds + 1):
            Fo, so = F[:i + 1][obs], S[i, :i + 1][obs]
            A = Fo.T @ Fo + lam * np.eye(p); Ai = np.linalg.inv(A)
            mu = F[:i + 1] @ (Ai @ Fo.T @ so)
            if r == rounds or obs.all(): break
            sd = np.sqrt(np.einsum("jp,pq,jq->j", F[:i + 1], Ai, F[:i + 1]))
            acq = np.where(obs, -np.inf, mu + ucb * sd)
            obs[np.argsort(-acq)[:min(b, (~obs).sum())]] = True
        logit = np.where(obs, S[i, :i + 1], mu)  # exact where acquired, GP mean elsewhere
        w = np.exp(logit - logit.max()); Y[i] = (w / w.sum()) @ V[:i + 1]
    return Y, m0 + rounds * b + p / d * n


if __name__ == "__main__":
    heads = get_heads(with_qk=True)
    pts = {}  # method label -> (cost, mean err)

    def add(lbl, cost, errs):
        pts[lbl] = (cost, float(np.mean(errs))); print(f"  {lbl:28s} cost {cost:7.1f} logits/row   err {np.mean(errs):.3f}  (min {np.min(errs):.3f} max {np.max(errs):.3f})", flush=True)

    print("1. bandit / successive elimination (T=1024)")
    for z in (0.5, 1, 2, 3):
        out = [(bandit(q, k, v, z), softmax_masked(S) @ v) for l, h, S, v, q, k in heads]
        add(f"bandit z={z}", np.mean([c for (_, c), _ in out]), [nrm(Y, R) for (Y, _), R in out])

    print("2. soft bitonic top-k attention (T=256 prefix, queries 128..255, all logits used)")
    qs = np.arange(128, 256)
    for k_, tau in ((16, 0.05), (16, 0.5), (64, 0.05)):
        e, hard = [], []
        for l, h, S, v, q, k in heads:
            S2, v2 = S[:256, :256], v[:256]
            R = (softmax_masked(S2) @ v2)[qs]
            Yb, ns = bitonic_topk(S2, v2, k_, tau, qs)
            top = np.argsort(-np.where(np.tril(np.ones_like(S2, bool)), S2, -np.inf)[qs], 1)[:, :k_]
            Mh = np.zeros_like(S2, bool); Mh[qs[:, None], top] = True
            e.append(nrm(Yb, R)); hard.append(nrm((softmax_masked(S2, Mh) @ v2)[qs], R))
        add(f"bitonic k={k_} tau={tau} [{ns} stages]", 256, e)
        print(f"{'':30s} (hard top-{k_} reference err {np.mean(hard):.3f})")

    print("3. stationary distribution (T=1024)")
    for beta in (0, 0.5, 1, 2):
        add(f"stationary beta={beta}", 1024, [nrm(stationary(S, v, beta), softmax_masked(S) @ v) for l, h, S, v, q, k in heads])

    print("4. active sampling, Bayesian ridge on p=16 projection (T=1024)")
    for ucb in (0.0, 1.0):
        e, c = zip(*[(nrm((y := active(q, k, v, ucb=ucb))[0], softmax_masked(S) @ v), y[1]) for l, h, S, v, q, k in heads[:6]])
        add(f"active ucb={ucb}", np.mean(c), e)

    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl, (c, e) in pts.items(): ax.scatter(c, e); ax.annotate(lbl, (c, e), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set(xscale="log", yscale="log", xlabel="exact-logit equivalents per row", ylabel="output rel. error", title="tournament attentions, real heads")
    fig.tight_layout(); fig.savefig("runs/approx/tournament.png", dpi=120)
