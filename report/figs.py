"""Figures for attention-survey.typ.

Run: uv run --with matplotlib --with numpy --with scipy python report/figs.py
Writes SVGs to report/figs/. Every literature number below is copied from the cited table;
charts never mix numbers from different papers in one panel.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import nnls

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)
W = 6.5  # text width in inches (A4, 2.1 cm margins)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "svg.fonttype": "path",
    "figure.dpi": 150,
})

# one colour per family, shared with the Typst document
FAM = {
    "softmax": "#4c72b0", "linear": "#8c8c8c", "gated": "#dd8452", "ssm": "#937860",
    "delta": "#c44e52", "conv": "#55a868", "ours": "#2a9d8f", "hybrid": "#8172b3",
}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fractional kernel (numpy)
def frac_kernel(alpha, tau, n):
    """Mirror of frac_attention in transformer.py: heat kernel of L^(alpha/2) on an n-cycle."""
    lam = np.maximum((2 * np.sin(np.pi * np.arange(n // 2 + 1) / n)) ** 2, 1e-12)
    return np.fft.irfft(np.exp(-tau * lam ** (alpha / 2)), n)  # full cycle, length n


for a in (1.1, 1.5, 1.9):
    for t in (0.5, 4.0, 32.0):
        w = frac_kernel(a, t, 512)
        assert w.min() > -1e-12, (a, t, w.min())  # positivity (Markov semigroup)
        assert abs(w.sum() - 1) < 1e-4  # mass 1 on the full cycle (up to the 1e-12 eigenvalue clamp)


# ---------------------------------------------------------------- F1 timeline
def fig_timeline():
    # (year + month/12, lane, label); months from arXiv ids
    papers = [
        (2017.45, "softmax", "Transformer"), (2025.1, "softmax", "TransMLA"),
        (2020.46, "linear", "Linear Transformer"), (2023.54, "linear", "RetNet"),
        (2023.96, "gated", "GLA"), (2024.13, "gated", "Griffin"), (2024.88, "gated", "RWKV survey"),
        (2023.96, "ssm", "Mamba"), (2024.38, "ssm", "Mamba-2 / SSD"),
        (2024.96, "delta", "Gated DeltaNet"), (2025.04, "delta", "Titans"),
        (2025.21, "delta", "RWKV-7"), (2025.79, "delta", "Kimi Linear"),
        (2021.87, "conv", "S4"), (2022.79, "conv", "SGConv"), (2022.96, "conv", "H3"),
        (2023.13, "conv", "Hyena"), (2023.38, "conv", "TNN"), (2023.79, "conv", "M2"),
        (2026.72, "ours", "Frac. Laplacian"),
    ]
    off = {"SGConv": (-14, -10), "H3": (-2, 6), "Hyena": (4, -10), "TNN": (6, 6), "M2": (6, -10),
           "Gated DeltaNet": (-24, 6), "Titans": (-4, -10), "RWKV-7": (14, 6), "Kimi Linear": (4, -10),
           "GLA": (-8, 6), "Griffin": (6, -10), "Mamba": (-17, -2.5), "Mamba-2 / SSD": (33, -2.5)}
    lanes = ["softmax", "linear", "gated", "ssm", "delta", "conv", "ours"]
    names = {"softmax": "Softmax / KV", "linear": "Linear / decay", "gated": "Data gates",
             "ssm": "Selective SSM", "delta": "Delta rule / memory", "conv": "Long convolution",
             "ours": "This work"}
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for i, lane in enumerate(lanes):
        pts = sorted(p for p in papers if p[1] == lane)
        ax.axhline(i, color=FAM[lane], lw=0.6, alpha=0.4, zorder=0)
        for x, _, lab in pts:
            ax.scatter(x, i, s=28, color=FAM[lane], zorder=3, marker="*" if lane == "ours" else "o")
            ax.annotate(lab, (x, i), xytext=off.get(lab, (0, 6)), textcoords="offset points",
                        ha="center", fontsize=6.5)
    ax.set_yticks(range(len(lanes)), [names[l] for l in lanes])
    ax.set_ylim(len(lanes) - 0.4, -0.6)
    ax.set_xlim(2016.5, 2027.5)
    ax.set_xticks(range(2017, 2028))
    ax.grid(axis="y", visible=False)
    save(fig, "timeline")


# ---------------------------------------------------------------- F2 mask gallery
def causal(a):
    return np.tril(a)


def fig_masks():
    T = 24
    rng = np.random.default_rng(0)
    i, j = np.indices((T, T))
    lag = np.clip(i - j, 0, None)
    # (a) softmax attention A for random q, k
    q, k = rng.normal(size=(2, T, 8))
    s = np.where(j <= i, q @ k.T / np.sqrt(8) * 1.5, -np.inf)
    A_soft = np.exp(s - s.max(1, keepdims=True))
    A_soft /= A_soft.sum(1, keepdims=True)
    # (b) linear attention: L = causal ones
    L_lin = causal(np.ones((T, T)))
    # (c) RetNet: L = gamma^(t-s)
    L_ret = causal(0.8 ** lag)
    # (d) Mamba-2: L_ts = prod_{r=s+1}^t a_r, random gates with two near-resets
    a = rng.uniform(0.85, 1.0, T)
    a[[8, 17]] = 0.03
    cum = np.cumsum(np.log(a))
    L_m2 = causal(np.exp(cum[:, None] - cum[None, :]))
    # (e) H3/Hyena-style gated LTI conv (one channel): A_ts = q_t h_(t-s) k_s
    h = np.exp(-0.25 * lag) * (1 + 0.5 * np.cos(0.9 * lag))
    g_q, g_k = 1 / (1 + np.exp(-rng.normal(size=(2, T)) * 2))
    A_conv = causal(g_q[:, None] * h * g_k[None, :])
    # (f) fractional heat kernel, row-normalised (causal_mix)
    w = frac_kernel(1.3, 2.0, 2 * T)[:T]
    A_frac = causal(w[lag])
    A_frac /= A_frac.sum(1, keepdims=True)
    assert np.allclose(A_frac.sum(1), 1)
    panels = [
        ("(a) Softmax\n$A$", A_soft), ("(b) Linear attention\n$L = 1$", L_lin),
        ("(c) RetNet\n$L_{ts}=\\gamma^{t-s}$", L_ret), ("(d) Mamba-2 (random)\n$L_{ts}=a_{s+1}\\cdots a_t$", L_m2),
        ("(e) H3 / Hyena\n$A_{ts}=q_t h_{t-s} k_s$", A_conv), ("(f) Fractional heat\n$A_{ts}=w_{t-s}/\\sum_{m\\leq t} w_m$", A_frac),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(5.0, 3.9))
    for ax, (title, M) in zip(axs.flat, panels):
        M = np.where(j <= i, np.maximum(M / M.max(), 1e-3), np.nan)
        im = ax.imshow(M, cmap="viridis", norm=LogNorm(1e-3, 1), interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4)
    cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.03, label="entry / row max (log)")
    cb.ax.tick_params(labelsize=6)
    save(fig, "masks")


# ---------------------------------------------------------------- F3 decoding cost model
def fig_cost():
    d, H, dk = 4096, 32, 128
    T = np.logspace(2, 6.3, 100)
    R = 16
    mem = {  # floats kept per layer while decoding
        "Softmax MHA  $2Td$": (2 * T * d, FAM["softmax"], "-"),
        "GQA, 8 KV heads": (2 * T * 8 * dk, FAM["softmax"], "--"),
        "MLA  $576\\,T$": (576 * T, FAM["softmax"], ":"),
        "Frac. heat (exact)  $Td$": (T * d, FAM["ours"], "-"),
        "Linear / GLA / GDN  $H d_k^2$": (np.full_like(T, H * dk * dk), FAM["delta"], "-"),
        "Mamba  $2dN$": (np.full_like(T, 2 * d * 16), FAM["ssm"], "-"),
        f"Frac. heat, SoE $R={R}$": (np.full_like(T, R * d), FAM["ours"], "--"),
    }
    flops = {  # multiply-adds per decoded token, mixing only
        "Softmax": (4 * T * d, FAM["softmax"], "-"),
        "Frac. heat (exact)": (2 * T * d, FAM["ours"], "-"),
        "Gated DeltaNet": (np.full_like(T, 6 * H * dk * dk), FAM["delta"], "-"),
        "Mamba": (np.full_like(T, 6 * 2 * d * 16), FAM["ssm"], "-"),
        f"Frac. heat, SoE $R={R}$": (np.full_like(T, 3 * R * d), FAM["ours"], "--"),
    }
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.4))
    for ax, data, yl in [(axs[0], mem, "state / cache (floats per layer)"),
                         (axs[1], flops, "mixing FLOPs per token")]:
        for lab, (y, c, ls) in data.items():
            ax.loglog(T, y, color=c, ls=ls, lw=1.3, label=lab)
        ax.set_xlabel("context length $T$")
        ax.set_ylabel(yl)
        ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    axs[0].set_title("(a) Memory while decoding")
    axs[1].set_title("(b) Compute per generated token")
    save(fig, "cost")


# ---------------------------------------------------------------- F4 synthetic recall
def fig_synthetic():
    # source: Mamba (2312.00752) Table 1, selective copying accuracy (%)
    sel = {"no gate": {"S4": 18.3, "S6": 97.0}, "H3": {"S4": 57.0, "Hyena": 30.1, "S6": 99.7},
           "Mamba": {"S4": 56.4, "Hyena": 28.4, "S6": 99.8}}
    # source: Hyena (2302.10866) Table 4.2, associative recall, vocab 30; None = out of memory
    ar = {"Hyena": [100.0, 100.0, 97.2], "FlashTransformer": [32.4, 26.7, None],
          "RWKV": [12.4, 6.5, 2.3], "H3": [8.4, 4.3, 0.6], "GSS": [5.3, 2.1, 0.1], "AFT": [2.3, 1.2, 0.8]}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    inner = ["S4", "Hyena", "S6"]
    cols = {"S4": FAM["conv"], "Hyena": "#8fd19e", "S6": FAM["ssm"]}
    for gi, (arch, vals) in enumerate(sel.items()):
        for k, lay in enumerate(inner):
            if lay in vals:
                x = gi + (k - 1) * 0.27
                ax.bar(x, vals[lay], 0.25, color=cols[lay], label=lay if gi == 1 else None)
                ax.text(x, vals[lay] + 1.5, f"{vals[lay]:.0f}", ha="center", fontsize=6)
    ax.set_xticks(range(3), list(sel))
    ax.set_ylim(0, 150); ax.set_ylabel("accuracy (%)"); ax.set_yticks(range(0, 101, 20))
    ax.set_title("(a) Selective copying — Mamba T1")
    ax.legend(title="inner layer", frameon=False, title_fontsize=6.5, loc="upper center", ncol=3)
    ax = axs[1]
    lens = [30, 64, 131]
    style = {"Hyena": FAM["conv"], "FlashTransformer": FAM["softmax"], "RWKV": FAM["gated"],
             "H3": "#8fd19e", "GSS": "#bbbbbb", "AFT": "#999999"}
    for m, ys in ar.items():
        xs = [l for l, y in zip(lens, ys) if y is not None]
        ax.plot(xs, [y for y in ys if y is not None], "o-", ms=3, lw=1.2, color=style[m], label=m)
    ax.scatter([131], [2], marker="x", color=FAM["softmax"], s=20)
    ax.annotate("OOM", (131, 2), xytext=(3, 6), textcoords="offset points", fontsize=6, color=FAM["softmax"])
    ax.set_xscale("log"); ax.set_xticks(lens, ["30K", "64K", "131K"]); ax.minorticks_off()
    ax.set_ylim(-3, 105); ax.set_xlabel("sequence length"); ax.set_ylabel("accuracy (%)")
    ax.set_title("(b) Associative recall, vocab 30 — Hyena T4.2")
    ax.legend(frameon=False, ncol=2, fontsize=6, loc="center right")
    save(fig, "synthetic")


# ---------------------------------------------------------------- F5 needle in a haystack
def fig_niah():
    ctx = [1, 2, 4, 8]
    # source: Gated DeltaNet (2412.06464) Table 2, 1.3B models
    s1 = {"Mamba2": [99.2, 98.8, 65.4, 30.4], "DeltaNet": [97.4, 96.8, 99.0, 98.8],
          "Gated DeltaNet": [98.4, 88.4, 91.4, 91.8]}
    s2 = {"Mamba2": [99.4, 98.8, 56.2, 17.0], "DeltaNet": [98.4, 45.6, 18.6, 14.4],
          "Gated DeltaNet": [100.0, 99.8, 92.2, 29.6]}
    # source: Titans (2501.00663) Table 2, S-NIAH-N, 2K-16K
    ctx_t = [2, 4, 8, 16]
    tn = {"Mamba2": [98.4, 55.8, 14.2, 0.0], "DeltaNet": [47.2, 15.4, 12.8, 5.4],
          "TTT": [60.2, 36.6, 10.2, 4.4], "Titans LMM": [100.0, 99.8, 93.4, 80.2],
          "Titans MAC": [99.6, 98.2, 97.6, 97.4]}
    col = {"Mamba2": FAM["gated"], "DeltaNet": "#e39a9c", "Gated DeltaNet": FAM["delta"],
           "TTT": "#999999", "Titans LMM": "#8b1e3f", "Titans MAC": FAM["hybrid"]}
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.2), sharey=True)
    for ax, data, xs, title in [(axs[0], s1, ctx, "(a) S-NIAH-1 — GDN T2"),
                                (axs[1], s2, ctx, "(b) S-NIAH-2 — GDN T2"),
                                (axs[2], tn, ctx_t, "(c) S-NIAH-N — Titans T2")]:
        for m, ys in data.items():
            ax.plot(xs, ys, "o-", ms=3, lw=1.2, color=col[m], label=m)
        ax.set_xscale("log", base=2); ax.set_xticks(xs, [f"{x}K" for x in xs]); ax.minorticks_off()
        ax.set_title(title); ax.set_xlabel("context length")
        ax.legend(frameon=False, fontsize=5.8, loc="center right" if data is tn else "lower left")
    axs[0].set_ylabel("accuracy (%)"); axs[0].set_ylim(-3, 104)
    save(fig, "niah")


# ---------------------------------------------------------------- F6 real-world recall
def fig_recall():
    # source: GLA (2312.06635) Table 3, 1.3B / 100B tokens
    gla = {"Transformer++": [27.4, 66.6, 31.5], "RetNet": [14.3, 42.8, 34.7],
           "Mamba": [6.2, 41.4, 35.2], "GLA": [19.9, 50.6, 42.6]}
    # source: Gated DeltaNet (2412.06464) Tables 4-5, 1.3B / 100B tokens
    gdn = {"Transformer++": (37.0, 11.0), "RetNet": (22.9, 13.2), "Mamba": (21.0, 14.6),
           "Mamba2": (29.8, 13.5), "DeltaNet": (26.2, 13.6), "Gated DeltaNet": (30.6, 16.6),
           "Samba*": (37.3, 15.9), "GDN-H1*": (39.0, 17.8), "GDN-H2*": (40.1, 18.4)}
    fam = {"Transformer++": FAM["softmax"], "RetNet": FAM["linear"], "Mamba": FAM["ssm"],
           "GLA": FAM["gated"], "Mamba2": FAM["gated"], "DeltaNet": "#e39a9c",
           "Gated DeltaNet": FAM["delta"], "Samba*": FAM["hybrid"], "GDN-H1*": FAM["hybrid"],
           "GDN-H2*": FAM["hybrid"]}
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.15, 1, 0.8], "wspace": 0.7})
    ax = axs[0]
    tasks = ["FDA", "SWDE", "SQuAD"]
    for k, (m, ys) in enumerate(gla.items()):
        xs = np.arange(3) + (k - 1.5) * 0.2
        ax.bar(xs, ys, 0.19, color=fam[m], label=m)
    ax.set_xticks(range(3), tasks); ax.set_ylabel("accuracy (%)")
    ax.set_title("(a) Extraction — GLA T3"); ax.set_ylim(0, 92)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    names = list(gdn)
    for ax, idx, title in [(axs[1], 0, "(b) Recall avg — GDN T4"), (axs[2], 1, "(c) LongBench — GDN T5")]:
        vals = [gdn[m][idx] for m in names]
        ax.barh(range(len(names)), vals, color=[fam[m] for m in names])
        ax.axvline(gdn["Transformer++"][idx], color=FAM["softmax"], lw=0.8, ls="--")
        ax.set_yticks(range(len(names)), names if idx == 0 else [""] * len(names))
        ax.invert_yaxis(); ax.set_title(title); ax.grid(axis="y", visible=False)
        for y, v in enumerate(vals):
            ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=5.8)
    save(fig, "recall")


# ---------------------------------------------------------------- F7 hybrid ratio
def fig_hybrid():
    # source: Mamba-2 (2405.21060) Table 2, 350M, 48 layers, 7B tokens
    m2 = [(0, 8.60), (1 / 48, 8.38), (6 / 48, 8.26), (24 / 48, 8.50)]
    m2_trm = 8.68
    # source: Kimi Linear (2510.26692) Table 1, KDA:MLA ratio -> attention share
    kimi = [(1 / 16, 5.82), (1 / 8, 5.70), (1 / 4, 5.65), (1 / 2, 5.66), (1.0, 5.77)]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.1))
    for ax, pts, extra, title, lab in [
        (axs[0], m2, (1.0, m2_trm), "(a) Mamba-2 + attention — Mamba-2 T2", "perplexity"),
        (axs[1], kimi, None, "(b) KDA + MLA — Kimi Linear T1", "validation perplexity"),
    ]:
        xs = [max(x, 1 / 96) for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, "o-", color=FAM["hybrid"], ms=4, lw=1.3, label="hybrid")
        b = int(np.argmin(ys))
        ax.scatter([xs[b]], [ys[b]], s=70, facecolors="none", edgecolors="k", lw=0.8, zorder=4)
        if extra:
            ax.scatter(*extra, marker="s", color=FAM["softmax"], s=22, label="Transformer++", zorder=4)
            ax.annotate("0 attn", (xs[0], ys[0]), xytext=(4, -2), textcoords="offset points", fontsize=6)
        ax.axvspan(0.1, 0.25, color=FAM["hybrid"], alpha=0.08)
        ax.set_xscale("log"); ax.set_xlabel("share of attention layers")
        if extra:
            ax.set_xticks([1 / 96, 1 / 48, 1 / 8, 1 / 4, 1 / 2, 1], ["0", "2%", "12.5%", "25%", "50%", "100%"])
        else:
            ax.set_xticks([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1], ["6.25%", "12.5%", "25%", "50%", "100%"])
        ax.minorticks_off(); ax.set_ylabel(lab); ax.set_title(title)
        ax.legend(frameon=False, loc="upper center")
    save(fig, "hybrid")


# ---------------------------------------------------------------- F8 ablations
def fig_ablations():
    # (paper, ingredient removed, ppl with, ppl without); sources in the Typst provenance table
    rows = [
        ("GLA", "gate → none", 14.77, 23.21), ("GLA", "gate → fixed decay", 14.77, 16.55),
        ("GLA", "gate → scalar", 14.77, 15.56),
        ("Mamba", "selective Δ,B,C", 8.71, 10.93), ("Mamba", "state N=16 → 1", 8.71, 9.73),
        ("Titans", "momentum", 27.01, 28.98), ("Titans", "weight decay (forget)", 27.01, 29.04),
        ("Titans", "short conv", 27.01, 28.73), ("Titans", "deep → linear memory", 27.01, 28.49),
        ("RetNet", "decay γ", 26.05, 27.86), ("RetNet", "swish gate", 26.05, 27.84),
        ("RetNet", "GroupNorm", 26.05, 27.54), ("RetNet", "multi-scale γ", 26.05, 27.02),
        ("Mamba-2", "φ = Performer", 11.58, 12.21), ("Mamba-2", "φ = cosFormer", 11.58, 11.97),
        ("Kimi", "short conv", 5.65, 5.70), ("Kimi", "output gate", 5.65, 5.67),
    ]
    col = {"GLA": FAM["gated"], "Mamba": FAM["ssm"], "Titans": "#8b1e3f", "RetNet": FAM["linear"],
           "Mamba-2": "#e0b060", "Kimi": FAM["delta"]}
    fig, ax = plt.subplots(figsize=(W, 3.1))
    pct = [100 * (b - a) / a for _, _, a, b in rows]
    ys = np.arange(len(rows))
    ax.barh(ys, pct, color=[col[p] for p, *_ in rows], height=0.7)
    ax.set_yticks(ys, [f"{p}: {r}" for p, r, *_ in rows]); ax.invert_yaxis()
    for y, v in zip(ys, pct):
        ax.text(v + 0.6, y, f"+{v:.1f}%", va="center", fontsize=6)
    ax.set_xlabel("perplexity increase when the ingredient is removed (%, within each paper)")
    ax.set_xlim(0, 64); ax.grid(axis="y", visible=False)
    ax.text(62, len(rows) - 1.5, "TNN: removing decay, extrapolation\nto 14K: 23.70 → 672.72 (+2744%)",
            ha="right", fontsize=6.5, style="italic")
    save(fig, "ablations")


# ---------------------------------------------------------------- F9/F10 fractional kernel
def fig_spectrum_kernel():
    n = 4096
    f = np.arange(n // 2 + 1) / n
    lam = (2 * np.sin(np.pi * f)) ** 2
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.4), gridspec_kw={"wspace": 0.45})
    ax = axs[0]
    for a, c in [(1.1, "#1b7a6e"), (1.5, FAM["ours"]), (1.9, "#9ad8cf")]:
        ax.plot(f, np.exp(-4 * lam ** (a / 2)), color=c, lw=1.3, label=f"α={a}")
    ax.plot(f, np.exp(-4 * lam), color="k", lw=0.8, ls=":", label="α=2 (heat)")
    ax.set_xscale("log"); ax.set_xlim(1e-4, 0.5); ax.set_xlabel("frequency $j/n$")
    ax.set_ylabel(r"$\hat w_j$"); ax.set_title("(a) Spectrum, τ = 4")
    ax.legend(frameon=False)
    m = np.arange(1, n // 2)
    ax = axs[1]
    slopes = {}
    for a, c in [(1.1, "#1b7a6e"), (1.5, FAM["ours"]), (1.9, "#9ad8cf")]:
        w = frac_kernel(a, 4.0, n)[: n // 2]
        ax.loglog(m, w[1:], color=c, lw=1.3, label=f"α={a}")
        sel = (m > 30) & (m < 300)
        slopes[a] = np.polyfit(np.log(m[sel]), np.log(w[1:][sel]), 1)[0]
    w2 = frac_kernel(2.0, 4.0, n)[: n // 2]
    ax.loglog(m, np.maximum(w2[1:], 1e-300), color="k", lw=0.8, ls=":", label="α=2")
    ax.set_ylim(1e-10, 1); ax.set_xlabel("lag $m$"); ax.set_ylabel("$w_m$")
    ax.set_title("(b) Kernel, τ = 4"); ax.legend(frameon=False, loc="upper right")
    ax = axs[2]
    wf = frac_kernel(1.5, 4.0, n)[: n // 2]
    ref = {r"frac α=1.5": (wf / wf.sum(), FAM["ours"], "-"),
           r"RetNet $\gamma^m$, γ=1−2$^{-5}$": (0.96875 ** np.arange(n // 2), FAM["linear"], "-"),
           r"TNN $0.99^m$": (0.99 ** np.arange(n // 2), FAM["conv"], "--"),
           r"Hyena window $e^{-0.3m}$": (np.exp(-0.3 * np.arange(n // 2)), "#8fd19e", ":")}
    for lab, (y, c, ls) in ref.items():
        y = y / y.sum()
        ax.loglog(m, y[1:], color=c, ls=ls, lw=1.3, label=lab)
    ax.set_ylim(1e-10, 1); ax.set_xlabel("lag $m$"); ax.set_title("(c) vs exponential decays")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2)
    save(fig, "kernel")
    return slopes


# ---------------------------------------------------------------- F11 sum-of-exponentials fit
def soe_fit(w, R, signed=True):
    """Fit w_m ≈ Σ_r c_r γ_r^m on a geometric grid of R decay rates (relative least squares).

    signed=False forces c_r ≥ 0 (NNLS). That fails here: the kernel's core is log-concave,
    so it is not completely monotone and no positive mixture of decays can match it."""
    m = np.arange(len(w))
    rates = np.geomspace(0.3 / len(w), 3.0, R)
    B = np.exp(-np.outer(m, rates))  # columns γ_r^m with γ_r = e^{-rate}
    A = B / w[:, None]
    c = np.linalg.lstsq(A, np.ones_like(w), rcond=None)[0] if signed else nnls(A, np.ones_like(w))[0]
    return B @ c


def fig_soe():
    T = 1024
    w = frac_kernel(1.5, 4.0, 2 * T)[:T]  # n_grid = 2·max_len, as in the model
    lw = np.log(w)
    concave = np.where(lw[2:] - 2 * lw[1:-1] + lw[:-2] < 0)[0] + 1  # lags where log w is concave
    Rs = [2, 4, 8, 16, 32]
    rng = np.random.default_rng(1)
    x = rng.normal(size=T)
    ymix = np.convolve(x, w)[:T] / np.cumsum(w)  # causal_mix on one channel
    res = {}  # (signed, R) -> (max rel err of w, max err of mixed output, fit)
    for signed in (True, False):
        for R in Rs:
            wh = soe_fit(w, R, signed)
            yh = np.convolve(x, wh)[:T] / np.cumsum(wh)
            res[signed, R] = (np.max(np.abs(wh - w) / w), np.abs(yh - ymix).max() / np.abs(ymix).max(), wh)
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.2))
    ax = axs[0]
    m = np.arange(T)
    ax.loglog(m[1:], w[1:], color="k", lw=1.8, label="exact (α=1.5, τ=4)")
    ax.loglog(m[1:], res[False, 16][2][1:], lw=1.0, ls=":", color=FAM["delta"], label="R=16, $c_r\\geq 0$")
    ax.loglog(m[1:], res[True, 8][2][1:], lw=1.0, ls="--", color="#f4a261", label="R=8, signed")
    ax.loglog(m[1:], res[True, 16][2][1:], lw=1.0, ls="--", color=FAM["ours"], label="R=16, signed")
    ax.axvspan(0.8, concave.max() + 0.5, color="k", alpha=0.07)
    ax.text(1.0, 2e-6, "log-concave\ncore", fontsize=6)
    ax.set_xlabel("lag $m$"); ax.set_ylabel("$w_m$"); ax.set_title("(a) Kernel and sum-of-exponentials fits")
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    ax = axs[1]
    for signed, c, mk in [(True, FAM["ours"], "o"), (False, FAM["delta"], "s")]:
        tag = "signed" if signed else "$c_r\\geq 0$"
        ax.semilogy(Rs, [res[signed, R][0] for R in Rs], mk + "-", color=c, ms=3, label=f"kernel, {tag}")
        ax.semilogy(Rs, [res[signed, R][1] for R in Rs], mk + "--", color=c, ms=3, mfc="none", label=f"output, {tag}")
    ax.set_xscale("log", base=2); ax.set_xticks(Rs, [str(r) for r in Rs]); ax.minorticks_off()
    ax.set_xlabel("number of exponentials $R$"); ax.set_ylabel("max relative error")
    ax.set_title("(b) Error vs R (T = 1024)"); ax.legend(frameon=False, fontsize=6, loc="lower left")
    save(fig, "soe")
    assert res[True, 32][1] < 1e-3  # signed fit is accurate enough to replace the FFT at R=32
    return {k: v[:2] for k, v in res.items()}, concave


if __name__ == "__main__":
    fig_timeline()
    fig_masks()
    fig_cost()
    fig_synthetic()
    fig_niah()
    fig_recall()
    fig_hybrid()
    fig_ablations()
    slopes = fig_spectrum_kernel()
    print("tail slopes (fit on 30<m<300, tau=4):", {a: round(s, 3) for a, s in slopes.items()},
          "theory -(1+alpha):", {a: -(1 + a) for a in slopes})
    res, concave = fig_soe()
    print("log-concave lags:", concave.tolist())
    for (signed, R), (e_w, e_y) in res.items():
        print(f"SoE {'signed' if signed else 'c>=0  '} R={R:>2}: kernel max rel err {e_w:.2e}  output err {e_y:.2e}")
    print("wrote", sorted(os.listdir(OUT)))
