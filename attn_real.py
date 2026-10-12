"""attn_approx methods on real attention heads of the trained softmax LM (runs/train/softmax_seq1024.pkl)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from attn_approx import lowrank, nystrom, pagerank, sampled, softmax_masked, tv

from heads import T, get_heads

heads = get_heads()
methods = dict(sampled=sampled, nystrom=nystrom, lowrank=lowrank, pagerank=pagerank)
ms = [8, 16, 32, 64, 128, 256]
err = {k: np.zeros((len(ms), len(heads))) for k in methods}
for j, (l, h, S, V) in enumerate(heads):
    P = softmax_masked(S); O = P @ V
    for i, m in enumerate(ms):
        for k, f in methods.items():
            err[k][i, j] = np.linalg.norm(f(S, m) @ V - O) / np.linalg.norm(O)
    ent = -(P * np.log(P + 1e-12)).sum(1).mean()
    print(f"L{l} H{h} entropy {ent:.2f} nats; err@m=64: " + " ".join(f"{k}={err[k][3, j]:.3f}" for k in methods), flush=True)
print(f"\n{'m':>4} " + " ".join(f"{k:>10}" for k in methods) + "   (mean out relerr over heads)")
for i, m in enumerate(ms): print(f"{m:>4} " + " ".join(f"{err[k][i].mean():10.3f}" for k in methods))

fig = plt.figure(figsize=(16, 8)); gs = fig.add_gridspec(2, 4)
ax = fig.add_subplot(gs[:, :2])
for k in methods:
    mu = err[k].mean(1); ax.plot(ms, mu, "o-", label=k); ax.fill_between(ms, err[k].min(1), err[k].max(1), alpha=0.15)
ax.set(xscale="log", yscale="log", xlabel="exact logits per row (m)", ylabel="output rel. error (mean, min-max band over 12 heads)",
       title=f"real attention, T={T}"); ax.legend()
j = next(i for i, (l, h, *_) in enumerate(heads) if (l, h) == (3, 2))  # example head shown below
S, V = heads[j][2], heads[j][3]
for c, (name, Pm) in enumerate([("exact", softmax_masked(S)), ("sampled", sampled(S, 64)), ("nystrom", nystrom(S, 64)),
                                ("lowrank", lowrank(S, 64)), ("pagerank", pagerank(S, 64))][:4]):
    a = fig.add_subplot(gs[c // 2, 2 + c % 2]); a.imshow(np.where(Pm > 0, Pm, np.nan), norm=LogNorm(1e-5, 1), cmap="Blues")
    a.set_title(f"{name}  (L3 H2, m=64)", fontsize=9); a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig("runs/approx/attn_real.png", dpi=120)
