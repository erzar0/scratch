"""Plots of runs/bench/bench_attention.csv (prefill), runs/bench/bench_decode.csv (decode) and the train_lm.py runs
(runs/train/<mixer>.csv and .log): hier vs CSA vs exact/softmax attention; and evict.py's results.
    uv run --with matplotlib python plot_bench.py   -> runs/bench/bench_prefill.png, runs/bench/bench_decode.png, runs/train/train_loss.png,
                                                       runs/evict/evict.png
"""
import csv
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "legend.fontsize": 8, "figure.dpi": 150})
STYLE = {  # same colour per method in every panel; hier in the report's "ours" colour
    "exact (flash)": dict(color="#4c72b0", marker="o"), "exact": dict(color="#4c72b0", marker="o"),
    "csa block 4": dict(color="#c44e52", marker="s"), "csa block 32": dict(color="#dd8452", marker="D"),
    "hier": dict(color="#2a9d8f", marker="^"), "hier + window 512": dict(color="#8e6bbf", marker="v"),
    "hier + window 1024": dict(color="#4b2a7b", marker="P"),
}
# hier_attention before query chunks were processed in blocks (prefill, same config). Memory: compile-time temp
# measured 2026-10-05 with the previous hier.py; time: the 2026-10-04 bench_attention.py run.
HIER_BEFORE_GIB = {4096: 12.5 / 1024, 16384: 80.1 / 1024, 65536: 880.1 / 1024, 131072: 3329.0 / 1024}
HIER_BEFORE_MS = {8192: 0.2, 16384: 0.5, 32768: 2.4, 65536: 9.5, 131072: 35.4}


def load(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r.get("mixer") or r["method"]].append(r)
    return rows


def kilo(ax):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1024:g}k"))


def plot_prefill():
    rows = load("runs/bench/bench_attention.csv")
    fig, (time_ax, mem_ax) = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, rs in rows.items():
        ran = [r for r in rs if r["forward_ms"]]
        T = [int(r["seq_len"]) for r in ran]
        time_ax.plot(T, [float(r["forward_ms"]) for r in ran], label=name, **STYLE[name])
        mem_ax.plot(T, [float(r["temp_gib"]) for r in ran], label=name, **STYLE[name])
        for r in rs:
            if not r["forward_ms"] and r["temp_gib"]:  # compiled, but didn't fit
                mem_ax.plot(int(r["seq_len"]), float(r["temp_gib"]), "x", color=STYLE[name]["color"], ms=9, mew=2)
                mem_ax.annotate(f"{name}: OOM", (int(r["seq_len"]), float(r["temp_gib"])), xytext=(-75, -4),
                                textcoords="offset points", fontsize=7, color=STYLE[name]["color"])
    before = dict(color="#2a9d8f", ls=":", marker=".", label="hier, before blocking")
    time_ax.plot(list(HIER_BEFORE_MS), list(HIER_BEFORE_MS.values()), **before)
    mem_ax.plot(list(HIER_BEFORE_GIB), list(HIER_BEFORE_GIB.values()), **before)
    # slope guides through exact attention's 16k point: 2x per doubling = linear, 4x = quadratic
    t0, y0 = 16384, next(float(r["forward_ms"]) for r in rows["exact (flash)"] if r["seq_len"] == "16384")
    xs = [8192, 131072]
    time_ax.plot(xs, [y0 * x / t0 for x in xs], color="gray", lw=0.8, ls="--")
    time_ax.plot(xs, [y0 * (x / t0) ** 2 for x in xs], color="gray", lw=0.8, ls="-.")
    time_ax.text(131072, y0 * 8, "∝ T", fontsize=7, color="gray", ha="left")
    time_ax.text(131072, y0 * 64, "∝ T²", fontsize=7, color="gray", ha="left")
    mem_ax.axhline(4, color="black", lw=0.8, ls="--")
    mem_ax.text(1100, 4.4, "4 GiB: WSL single-buffer limit", fontsize=7)
    time_ax.set(title="Prefill: forward time of one layer", xlabel="sequence length T", ylabel="ms")
    mem_ax.set(title="Prefill: working memory (XLA temp)", xlabel="sequence length T", ylabel="GiB")
    for ax in (time_ax, mem_ax):
        kilo(ax)
    time_ax.legend()
    fig.suptitle("RTX 4070 Ti SUPER, d_model 64, 1 head, bf16, batch 1", fontsize=8, y=0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("runs/bench/bench_prefill.png", bbox_inches="tight")


def plot_decode():
    rows = load("runs/bench/bench_decode.csv")
    fig, (b1, b64, mem) = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, rs in rows.items():
        T = [int(r["context"]) for r in rs]
        b1.plot(T, [float(r["us_batch1"]) for r in rs], label=name, **STYLE[name])
        b64.plot(T, [float(r["us_batch64"]) for r in rs], label=name, **STYLE[name])
    # Every cache grows linearly in T, and the hier caches coincide (the window only moves K/V between the
    # recent and old part): as lines they overlap, so one bar per method at the longest context.
    longest = max(int(r["context"]) for rs in rows.values() for r in rs)
    names = list(rows)[::-1]
    sizes = [sum(float(r["gpu_cache_mib_per_seq"]) + float(r["host_cache_mib_per_seq"]) for r in rows[n]
                 if int(r["context"]) == longest) for n in names]
    mem.barh(names, sizes, color=[STYLE[n]["color"] for n in names])
    for i, size in enumerate(sizes):
        mem.text(size * 1.1, i, f"{size:.1f} MiB", va="center", fontsize=7)
    mem.set_xscale("log")
    mem.set_xlim(right=max(sizes) * 4)
    mem.grid(axis="y", visible=False)
    # all methods sit at ~100-150 µs of fixed per-step cost: a wider range keeps that noise from looking like a trend
    b1.set(title="Decode, batch 1: µs per new token", xlabel="context length", ylabel="µs", ylim=(30, 3000))
    b64.set(title="Decode, batch 64: µs per step (64 tokens)", xlabel="context length", ylabel="µs")
    mem.set(title=f"Decode cache per sequence at {longest // 1024}k (all ∝ T)", xlabel="MiB")
    for ax in (b1, b64):
        kilo(ax)
    b1.set_ylim(30, 3000)  # after kilo(): the log scale resets limits
    b64.legend()
    fig.suptitle("RTX 4070 Ti SUPER, d_model 64, 1 head, bf16; one attention layer, cache update not timed",
                 fontsize=8, y=0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("runs/bench/bench_decode.png", bbox_inches="tight")


def plot_training():
    """Loss curves of the WikiText-103 runs (train_lm.py): one row per sequence length, one line per mixer run."""
    colors = {"softmax": "#4c72b0", "csa": "#c44e52", "hier": "#2a9d8f", "hier_window": "#4b2a7b"}
    # at SEQ 512 a 512-token window covers the whole sequence: hier_window would be softmax, so it only runs at 2048
    lengths = {"": "512 tokens x batch 8", "_seq2048": "2048 tokens x batch 2"}
    fig, axes = plt.subplots(len(lengths), 3, figsize=(13, 3.6 * len(lengths)), squeeze=False)
    for (suffix, shape), (train_ax, val_ax, clock_ax) in zip(lengths.items(), axes):
        for mixer, color in colors.items():
            name = f"runs/train/{mixer}{suffix}"
            if not os.path.exists(f"{name}.csv"):
                continue
            with open(f"{name}.csv") as f:
                rows = list(csv.DictReader(f))
            if not rows:  # run just started
                continue
            label = mixer.replace("hier_window", "hier + window 512")
            if os.path.exists(f"{name}.log"):  # final full-set numbers, printed at the end of the run
                final = re.search(r"final val [\d.]+ \(ppl ([\d.]+)\), test [\d.]+ \(ppl ([\d.]+)\)",
                                  open(f"{name}.log").read())
                label += f" (val ppl {final[1]}, test {final[2]})" if final else f" (running, step {rows[-1]['step']})"
            tokens = [int(r["tokens"]) / 1e6 for r in rows]
            loss = [float(r["train_loss"]) for r in rows]
            # train loss is one batch every 100 steps: plot it faint, plus a mean over the last 10 points
            smooth = [sum(loss[max(0, i - 9) : i + 1]) / len(loss[max(0, i - 9) : i + 1]) for i in range(len(loss))]
            train_ax.plot(tokens, loss, color=color, alpha=0.2, lw=0.7)
            train_ax.plot(tokens, smooth, color=color, label=label)
            evals = [r for r in rows if r["val_loss"]]
            val_ax.plot([int(r["tokens"]) / 1e6 for r in evals], [float(r["val_loss"]) for r in evals], color=color,
                        marker=".", label=label)
            clock_ax.plot([float(r["seconds"]) / 60 for r in evals], [float(r["val_loss"]) for r in evals],
                          color=color, marker=".", label=label)
        train_ax.set(title=f"Train loss (mean of 10 logs), {shape}", xlabel="tokens (M)", ylabel="nats / BPE token",
                     ylim=(3.3, 7.5))
        val_ax.set(title="Validation loss (40 batches)", xlabel="tokens (M)", ylabel="nats / BPE token", ylim=(3.7, 5.5))
        clock_ax.set(title="Validation loss vs wall clock", xlabel="minutes", ylabel="nats / BPE token", ylim=(3.7, 5.5))
        train_ax.legend()
    fig.suptitle("WikiText-103, GPT-2 BPE; d_model 512, 8 layers, 8 heads; 4096 tokens per step, 24k steps; "
                 "same data order for every mixer of a row", fontsize=8, y=0.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("runs/train/train_loss.png", bbox_inches="tight")


def plot_evict():
    """runs/evict/evict.csv and runs/evict/evict_pool.csv (evict.py eval): CSA with an evicting block pool."""
    with open("runs/evict/evict.csv") as f:
        rows = list(csv.DictReader(f))
    rule_color = {"stat": "#c44e52", "greedy": "#8172b3", "random": "#937860", "oldest": "#4c72b0"}
    kind_style = {"sqrt": dict(ls="-", marker="o"), "log": dict(ls="--", marker="s")}
    model_color = {"base": "#c44e52", "tuned full": "#7f7f7f", "tuned sqrt 4": "#2a9d8f", "tuned log 10": "#e9a03b"}
    # x: indexer work, i.e. blocks scored per sequence and layer, summed over tokens (the full pool scores 524k at
    # T = 2048). Pool size at T alone would favour log: it fills its pool early, so it works more for the same end size.
    work = lambda r: float(r["scanned_blocks_per_seq"]) / 1000
    fig, (rule_ax, tune_ax, pool_ax) = plt.subplots(1, 3, figsize=(13, 3.8))
    for r in rows:  # full-pool reference lines
        if r["budget"] == "full":
            tune_ax.axhline(float(r["val_ppl"]), color=model_color[r["model"]], lw=0.8, ls=":",
                            label=f"{r['model']}, full pool")
            if r["model"] == "base":
                rule_ax.axhline(float(r["val_ppl"]), color="black", lw=0.8, ls=":", label="full pool")
    for kind in kind_style:
        for rule, color in rule_color.items():
            pts = [r for r in rows if r["model"] == "base" and r["budget"] == kind and r["rule"] == rule]
            rule_ax.plot([work(r) for r in pts], [float(r["val_ppl"]) for r in pts], color=color,
                         label=f"{rule}, {kind}", **kind_style[kind])
        for model, color in model_color.items():
            pts = [r for r in rows if r["model"] == model and r["budget"] == kind and r["rule"] == "stat"]
            if pts:
                tune_ax.plot([work(r) for r in pts], [float(r["val_ppl"]) for r in pts], color=color,
                             label=f"{model}, {kind}", **kind_style[kind])
    with open("runs/evict/evict_pool.csv") as f:
        pool = defaultdict(list)
        for r in csv.DictReader(f):
            pool[(r["budget"], r["c"])].append((int(r["t"]), float(r["pool"])))
    for (kind, c), pts in pool.items():
        pool_ax.plot(*zip(*pts), label=f"{kind}, c={c}" if kind != "full" else "full",
                     color="black" if kind == "full" else None, ls=kind_style.get(kind, {}).get("ls", "-"))
    xlabel = "indexer work: blocks scored per sequence (k); full pool 524k"
    rule_ax.set(title="Base model: eviction rule", xlabel=xlabel, ylabel="val ppl")
    tune_ax.set(title="Rule stat: fine-tuned under eviction\n(the 3 tuned models coincide: gain = extra tokens)",
                xlabel=xlabel, ylabel="val ppl")
    pool_ax.set(title="Pool size (base model, rule stat)", xlabel="tokens decoded", ylabel="blocks in pool")
    for ax in (rule_ax, tune_ax, pool_ax):
        ax.legend(fontsize=6)
    fig.suptitle("CSA (block 4, top-8, window 32), WikiText-103 val, 2048-token windows decoded token by token; "
                 "pool per layer, mean over layers and sequences", fontsize=8, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("runs/evict/evict.png", bbox_inches="tight")


if __name__ == "__main__":
    plot_prefill()
    plot_decode()
    plot_training()
    if os.path.exists("runs/evict/evict.csv"):
        plot_evict()
