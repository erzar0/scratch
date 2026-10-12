"""Approximate causal attention from a budget of exact logits. numpy only, synthetic clustered Q/K/V.

Budget m = exact q.k dot products per query row (n*m total). Methods:
  sampled   self-normalised importance sampling over m random keys (Reformer/HyperAttention-style)
  nystrom   m/2 landmark rows + m/2 landmark cols of exp(S), Nystrom completion, then causal mask
  lowrank   ALS completion of the logit matrix S from m observed entries/row, then causal softmax
  pagerank  observed exp-weights -> row-stochastic G -> (1-a)(I-aG)^-1, i.e. personalised PageRank
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n, d, n_clusters = 512, 32, 16
mask = np.tril(np.ones((n, n), bool))


def make_qkv(noise=0.5, scale=2.0):
    c = rng.normal(size=(n_clusters, d))
    x = c[rng.integers(0, n_clusters, n)] + noise * rng.normal(size=(n, d))
    Wq, Wk = rng.normal(size=(2, d, d)) / np.sqrt(d)
    return x @ Wq, x @ Wk, rng.normal(size=(n, d)), scale / np.sqrt(d)


def softmax_masked(S, M=None):
    M = np.tril(np.ones(S.shape, bool)) if M is None else M
    S = np.where(M, S, -np.inf)
    E = np.exp(S - S.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


def sample_mask(m, n):
    """m random causal keys per row, plus the diagonal."""
    M = np.eye(n, dtype=bool)
    for i in range(n):
        M[i, rng.choice(i + 1, min(m, i + 1), replace=False)] = True
    return M


def sampled(S, m):
    M = sample_mask(m, len(S))
    return softmax_masked(S, M)


def nystrom(S, m):
    n, mask = len(S), np.tril(np.ones(S.shape, bool))
    L = np.linspace(0, n - 1, max(m // 2, 1)).astype(int)
    E = np.exp(S - S.max())                      # global shift only; cancels in the row normalisation
    P = E[:, L] @ np.linalg.pinv(E[np.ix_(L, L)], rcond=1e-3) @ E[L, :]
    P = np.where(mask, np.maximum(P, 0), 0) + 1e-12 * np.eye(n)
    return P / P.sum(1, keepdims=True)


def lowrank(S, m, iters=15, lam=1e-1):
    n = len(S)
    rank = min(64, max(m // 2, 1))
    M = sample_mask(m, len(S))
    U, V = rng.normal(size=(2, n, rank)) * 0.1
    for _ in range(iters):
        for A, B, Mk in ((U, V, M), (V, U, M.T)):    # solve rows of A given B, observed pattern Mk
            for i in range(n):
                o = Mk[i]
                Bo = B[o]
                A[i] = np.linalg.solve(Bo.T @ Bo + lam * np.eye(rank), Bo.T @ (S[i, o] if Mk is M else S[o, i]))
    return softmax_masked(U @ V.T)


def pagerank(S, m, a=0.9):
    n, mask = len(S), np.tril(np.ones(S.shape, bool))
    M = sample_mask(m, len(S))
    W = np.where(M, np.exp(S - np.where(M, S, -np.inf).max(1, keepdims=True)), 0)
    G = W / W.sum(1, keepdims=True)
    P = (1 - a) * np.linalg.inv(np.eye(n) - a * G)
    P = np.maximum(P, 0) * mask
    return P / P.sum(1, keepdims=True)


def tv(P, Ph): return 0.5 * np.abs(P - Ph).sum(1).mean()


if __name__ == "__main__":
    methods = dict(sampled=sampled, nystrom=nystrom, lowrank=lowrank, pagerank=pagerank)
    ms = [8, 16, 32, 64, 128]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for scale, name in ((2.0, "peaked (scale 2)"), (0.5, "flat (scale 0.5)")):
        Q, K, V, s = make_qkv(scale=scale)
        S = Q @ K.T * s
        P = softmax_masked(S)
        O = P @ V
        print(f"\n{name}   entropy={-(P * np.log(P + 1e-12)).sum(1).mean():.2f} nats  (uniform≈{np.log(n/2):.2f})")
        print(f"{'m':>4} " + " ".join(f"{k:>16}" for k in methods) + "    (out relerr / TV)")
        res = {k: [] for k in methods}
        for m in ms:
            row = []
            for k, f in methods.items():
                Ph = f(S, m)
                e = np.linalg.norm(Ph @ V - O) / np.linalg.norm(O)
                res[k].append(e)
                row.append(f"{e:6.3f} /{tv(P, Ph):6.3f}")
            print(f"{m:>4} " + " ".join(f"{r:>16}" for r in row))
        ax = axs[0 if scale == 2.0 else 1]
        for k, v in res.items(): ax.plot(ms, v, "o-", label=k)
        ax.set(xscale="log", yscale="log", xlabel="exact logits per row (m)", ylabel="output rel. error", title=name)
        ax.legend()
    fig.tight_layout(); fig.savefig("runs/approx/attn_approx.png", dpi=130)
