"""PageRank in matrix form: A -> M (column-stochastic) -> G (Google) -> G^k r0 -> r."""
import numpy as np
import matplotlib.pyplot as plt

names = list("ABCDEF")
n, d = len(names), 0.85
# A[i, j] = weight of link j -> i; random weights 1..5 on the diagonal and below, rest 0
A = np.tril(np.random.default_rng(0).integers(1, 6, (n, n))).astype(float)
for i, j in [(0, 0), (-1, 0), (-1, -1)]: A[i, j] = 0  # no entries in the triangle's three corners
cs = A.sum(0)
M = np.where(cs > 0, A / np.where(cs > 0, cs, 1), 1 / n)  # zero column (dangling) -> uniform
G = d * M + (1 - d) / n                # Google matrix
r = np.full(n, 1 / n)
hist = [r]
for _ in range(200):
    r = G @ r
    hist.append(r)
w, V = np.linalg.eig(G); v = V[:, w.real.argmax()].real
assert np.allclose(r, v / v.sum())

def heat(ax, X, title, fmt="{:.2f}", xl=names, yl=names):
    ax.imshow(X, cmap="Blues", vmin=0, vmax=max(X.max(), 1e-9))
    ax.set_xticks(range(X.shape[1]), xl); ax.set_yticks(range(X.shape[0]), yl)
    ax.set_title(title, fontsize=10)
    for (i, j), v in np.ndenumerate(X):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                color="white" if v > 0.6 * X.max() else "#222")
    for s in ax.spines.values(): s.set_visible(False)

fig, ax = plt.subplots(1, 5, figsize=(18, 4), gridspec_kw={"width_ratios": [6, 6, 6, 1.6, 1.6]})
heat(ax[0], A, "A  (col j = page j's outlinks)", "{:.0f}")
heat(ax[1], M, "M = A / colsum")
heat(ax[2], G, f"G = {d}·M + {1-d:.2f}/n")
heat(ax[3], hist[0][:, None], "r₀", xl=[""])
heat(ax[4], r[:, None], "r = G^200 r₀", "{:.3f}", xl=[""])
fig.tight_layout()
fig.savefig("runs/approx/pagerank.png", dpi=150)
print(dict(zip(names, r.round(3))))
