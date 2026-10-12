"""Val-loss cost of the chunk-level methods on the T=1024 softmax model (runs/approx/fast_loss.csv) -> runs/approx/fast_loss.png"""
import csv, re
import matplotlib.pyplot as plt

rows = {r["config"]: float(r["val_loss"]) for r in csv.DictReader(open("runs/approx/fast_loss.csv"))}
ex = rows["exact"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for m in ("oracle", "hier", "bandit", "active", "stationary"):
    pts = sorted((2 * int(c) + int(r) * int(c), rows[k] - ex)  # exact keys per query: window (2 chunks) + refined chunks
                 for k in rows for c, r in re.findall(rf"^{m} c(\d+) r(\d+)$", k))
    ax.plot(*zip(*pts), "o-" if m != "oracle" else "k--", label=m + (" (perfect chunk choice)" if m == "oracle" else ""), ms=4)
ax.axhline(rows["bandit per-key z=2 (dense)"] - ex, color="gray", ls=":", label="per-key bandit z=2 (dense, slower than exact)")
ax.set(yscale="log", xlabel="exact keys per query (of 512 on average)", ylabel="val loss - exact (nats)",
       title=f"chunk-level methods, softmax model T=1024 (exact {ex:.4f})")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("runs/approx/fast_loss.png", dpi=130)

# ----- trade-off on the T=4096 model: fidelity (KL to exact) and val loss vs time, and time vs T -> runs/approx/fast_tradeoff.png
kl = {r["config"]: (float(r["val_loss"]), float(r["kl"])) for r in csv.DictReader(open("runs/approx/fast_kl.csv"))}
ms = {}  # (config, T) -> ms, 8 heads, bf16
for r in csv.DictReader(open("runs/approx/fast_baselines_timing.csv")): ms[r["config"], int(r["T"])] = float(r["ms"])
for r in csv.DictReader(open("runs/approx/fast_tune.csv")):
    for T in (4096, 16384, 32768): ms[f"{r['method']} c{r['chunk']} r{r['n_ref']}", T] = float(r[f"ms_{T}"])
ex4 = kl["exact"][0]
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
groups = {"window (cuDNN)": [k for k in kl if k.startswith("window")]} | {m: [k for k in kl if k.startswith(m + " ")] for m in ("hier", "bandit", "active", "stationary")}
for g, names in groups.items():
    pts = sorted((ms[n, 4096], kl[n][1], kl[n][0] - ex4, n) for n in names if (n, 4096) in ms)
    if not pts: continue
    x, y1, y2, lab = zip(*pts)
    l, = ax[0].plot(x, y1, "o-", label=g, ms=5)
    ax[1].plot(x, y2, "o-", color=l.get_color(), ms=5)
    for xi, yi, n in zip(x, y1, lab): ax[0].annotate(n.split(" ", 1)[1], (xi, yi), fontsize=6, xytext=(3, 2), textcoords="offset points")
    T_cfg = names[min(range(len(names)), key=lambda i: kl[names[i]][1])] if g.startswith("window") else names[-1]
for a in ax[:2]: a.axvline(ms["flash", 4096], color="k", ls="--", lw=1, label="flash (exact)")
ax[0].set(xscale="log", yscale="log", xlabel="ms per layer (8 heads, T=4096, bf16)", ylabel="KL(exact model || approx model), nats/token",
          title="fidelity vs time, T=4096 softmax model"); ax[0].legend(fontsize=8)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set(xscale="log", xlabel="ms per layer (8 heads, T=4096, bf16)", ylabel="val loss - exact (nats)",
          title=f"val loss vs time (exact {ex4:.4f}): under-trained model prefers local attention")
for n in ("flash", "window 1024", "window 2048", "hier c64 r8", "bandit c64 r8", "bandit c64 r16", "active c64 r8", "stationary c64 r8"):
    Ts = sorted(T for (c, T) in ms if c == n)
    ax[2].plot(Ts, [ms[n, T] for T in Ts], "k*-" if n == "flash" else "o-", label=n, ms=6 if n == "flash" else 4)
ax[2].set(xscale="log", yscale="log", xlabel="T", ylabel="ms per layer (8 heads, bf16)", title="time vs sequence length")
ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig("runs/approx/fast_tradeoff.png", dpi=120)
