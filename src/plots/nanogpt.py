"""Figures for src/nanogpt/: the CSA2 layer probe on record 91, next to Qwen's, and
record 91 trained from scratch with dense or CSA2 attention.

uv run python -m plots.nanogpt
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from experiments.results import read
from plots.qwen import probe_layers
from plots.style import REFERENCE, SERIES, apply, save

RUNS = Path("runs/nanogpt")
SPAN = 512  # nanogpt/long.py: scored tokens in the history study
R91_LOGS = Path.home() / "repos" / "modded-nanogpt-r91" / "logs"
# Run ids per model, one per seed (first: random seed, val every 2000 steps; then
# TRAIN_SEED=1, 2, ..., val every 500): same data, schedule and token count
TRAINING = {
    "dense attention (record 91)": (
        "ec348eea-fe95-43b8-b0d4-98b06c489b76",
        "bc39288e-1ed4-4d80-a72c-c97f896c3783",
    ),
    "CSA2 attention": (
        "c89581d5-75d5-4509-bb0b-2e885dbd93e5",
        "296a91a4-55af-4b0b-be1a-3da00a57f95d",
    ),
    # record 91 without what CSA2 had to drop: value embeddings into V, XSA,
    # attention gates, DC, paired heads
    "dense attention, stripped as CSA2": (
        "d14113e7-0492-46f1-bd2e-21a0327d5da6",
        "337449e0-1b7d-41e4-9ea9-192f870a7ebb",
        "9497acb4-9da5-4e9e-82f8-5a4111777428",
    ),
    # SWEEP_LONG_DOCS=1: documents cut at the row length (2048/4096/6144 tokens per
    # stage) instead of 896/2048; seed 1
    "CSA2 attention, documents up to 6144 tokens": (
        "8f1e55cd-1745-4f49-a7cc-0e251587424f",
    ),
    "dense attention, stripped as CSA2, documents up to 6144 tokens": (
        "f5ddc4a1-dbb4-4682-a28e-8d667271293b",
    ),
}
VARIANTS = {
    "base": "Qwen run's sizes, scaled to width 768",
    "all": "all four enlargements",
}


def fig_probe() -> None:
    """Held-out error of a CSA2 layer trained alone, per record 91 layer, at the
    last step; Qwen's probed layers (same sizes, same 600 steps) as a band."""
    path = RUNS / "probe_layer.csv"
    if not path.exists():
        return
    frame = read(path)
    last = frame.filter(pl.col("step") == frame["step"].max())
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    qwen = probe_layers().filter(
        (pl.col("variant") == "base") & (pl.col("step") == 600)
    )["val_mix"]
    low, high = float(qwen.min()), float(qwen.max())  # type: ignore[arg-type]
    ax.axhspan(low, high, color=REFERENCE, alpha=0.15, lw=0)
    ax.text(0, high, " Qwen3-0.6B layers 0, 3, 4, 7", va="bottom", fontsize=7)
    for (variant, label), color in zip(VARIANTS.items(), SERIES, strict=False):
        part = last.filter(pl.col("variant") == variant).sort("layer")
        if not part.is_empty():
            label += f" ({part['params'][0] / 1e6:.1f}M per layer)"
            ax.plot(part["layer"], part["val_error"], "o-", color=color, label=label)
    ax.set(
        xlabel="record 91 layer (layer 6 has no attention)",
        ylabel="squared error / output mean square",
        title="One CSA2 layer imitating record 91's attention, held-out FineWeb",
        ylim=(0, None),
        xticks=range(11),
    )
    ax.legend(fontsize=7)
    save(fig, RUNS / "probe.png")


def val_losses(log: Path) -> pl.DataFrame:
    """Every validation of a record 91 training log: step, val_loss."""
    rows = re.findall(r"^step:(\d+)/\d+ val_loss:([\d.]+)", log.read_text(), re.M)
    return pl.DataFrame(
        {"step": [int(s) for s, _ in rows], "val_loss": [float(v) for _, v in rows]}
    )


# CSA2 attending as transformers' DeepSeek-V4 does (nanogpt/csa2_library.py) against
# csa2_varlen's sparse attention, same seed (TRAIN_SEED=1). The first two warmed up
# stage-1 shapes only (SWEEP_WARMUP_BELOW=3331) and stopped by step 2000: stage 2's
# 4096-token rows do not fit in 16 GB with dense masked attention
LIBRARY = {
    "CSA2, transformers' masked attention": "7d259521-a05a-4002-a484-34f5bd6a6c94",
    "same, second run (val every 250)": "8d24c20d-7eb9-4a8c-b84c-68ced9c0c894",
    "CSA2, sparse attention, same warmup": "0407dcae-a39f-4331-b7c2-439a1da89875",
    "CSA2, sparse attention, full run": "296a91a4-55af-4b0b-be1a-3da00a57f95d",
}


def fig_library() -> None:
    """Val loss through step 2000 of LIBRARY's runs, with their ms per step."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    colors = (SERIES[1], SERIES[3], SERIES[2], REFERENCE)
    for (label, run), color in zip(LIBRARY.items(), colors, strict=True):
        log = R91_LOGS / f"{run}.txt"
        frame = val_losses(log).filter(pl.col("step").is_between(1, 2000))
        if frame.is_empty():
            continue
        last = re.findall(
            rf"^step:{frame['step'][-1]}/\d+ val_loss.* step_avg:([\d.]+)ms",
            log.read_text(),
            re.M,
        )
        ax.plot(
            frame["step"],
            frame["val_loss"],
            "o-",
            color=color,
            label=f"{label}, {float(last[0]):.0f} ms per step",
        )
    ax.set(
        xlabel="training step (stage 1: 2048-token rows)",
        ylabel="FineWeb val loss",
        title="Same CSA2, two attention implementations",
    )
    ax.legend(fontsize=7)
    save(fig, RUNS / "library.png")


def fig_train() -> None:
    """Val loss during training of every TRAINING run."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (label, runs), color in zip(TRAINING.items(), SERIES, strict=False):
        logs = [R91_LOGS / f"{run}.txt" for run in runs]
        frames = [val_losses(log).filter(pl.col("step") > 0) for log in logs]
        frames = [f for f in frames if not f.is_empty()]  # runs not yet validated
        finals = ", ".join(f"{f['val_loss'][-1]:.3f}" for f in frames)
        for seed, frame in enumerate(frames):
            ax.plot(
                frame["step"],
                frame["val_loss"],
                "o-" if seed == 0 else ".--",
                color=color,
                lw=1.5 if seed == 0 else 1.0,
                label=f"{label} (final {finals})" if seed == 0 else None,
            )
    ax.set(
        xlabel="training step (every run: about 336M tokens in all)",
        ylabel="FineWeb val loss",
        title="Record 91 trained from scratch, single-GPU proxy scale",
    )
    ax.legend(fontsize=7, title="solid: first seed, dashed: others", title_fontsize=7)
    save(fig, RUNS / "train.png")


def fig_long() -> None:
    """Loss by position on long val documents (nanogpt/long.py), every TRAINING run."""
    path = RUNS / "long.csv"
    if not path.exists():
        return
    frame = read(path).filter(pl.col("topk").is_null())  # trained pick count
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (label, runs), color in zip(TRAINING.items(), SERIES, strict=False):
        parts = [frame.filter(pl.col("run") == run).sort("start") for run in runs]
        for seed, part in enumerate(p for p in parts if not p.is_empty()):
            ax.plot(
                part["end"],
                part["loss"],
                "o-" if seed == 0 else ".--",
                color=color,
                lw=1.5 if seed == 0 else 1.0,
                label=label if seed == 0 else None,
            )
    ax.axvline(2048, color=REFERENCE, lw=1, ls=":")
    ax.text(2048, ax.get_ylim()[1], " longest training document", va="top", fontsize=7)
    ax.set(
        xscale="log",
        xlabel="token position in the document (mean loss up to this position, "
        "from the previous point)",
        ylabel="FineWeb val loss",
        title="Final checkpoints on the first 8192 tokens of 64 long val documents",
    )
    ax.legend(
        fontsize=7,
        title="solid: first seed, dashed: others",
        title_fontsize=7,
        loc="lower left",
    )
    save(fig, RUNS / "long.png")


def fig_history() -> None:
    """Loss on the same last 512 tokens of each long val document, after more and more
    of the document before them (nanogpt/long.py); CSA2 also at other pick counts."""
    path = RUNS / "history.csv"
    if not path.exists():
        return
    frame = read(path)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (label, runs), color in zip(TRAINING.items(), SERIES, strict=False):
        default = frame.filter(pl.col("topk").is_null())
        parts = [default.filter(pl.col("run") == run).sort("history") for run in runs]
        for seed, part in enumerate(p for p in parts if not p.is_empty()):
            ax.plot(
                part["history"] + SPAN,
                part["loss"],
                "o-" if seed == 0 else ".--",
                color=color,
                lw=1.5 if seed == 0 else 1.0,
                label=label if seed == 0 else None,
            )
    first_csa2 = TRAINING["CSA2 attention"][0]
    picks = frame.filter((pl.col("run") == first_csa2) & pl.col("topk").is_not_null())
    for (topk,), part in picks.sort("history").group_by("topk", maintain_order=True):
        ax.plot(
            part["history"] + SPAN,
            part["loss"],
            ":",
            color=SERIES[1],
            lw=1.0,
            label=f"CSA2 first seed, {topk} picks per query (trained with 64)",
        )
    ax.axvline(2048, color=REFERENCE, lw=1, ls=":")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.25)
    ax.text(
        2048, ax.get_ylim()[0], " longest training document", va="bottom", fontsize=7
    )
    ax.set(
        xscale="log",
        xlabel=f"tokens seen: the scored last {SPAN} plus the document before them",
        ylabel=f"val loss on the last {SPAN} tokens",
        title="Same tokens, more context: 64 long FineWeb val documents",
    )
    ax.legend(
        fontsize=6, title="solid: first seed, dashed: others", title_fontsize=6, ncol=2
    )
    save(fig, RUNS / "history.png")


# nanogpt/bench.py stacks: color, line style
STACKS = {
    "CSA2": (SERIES[1], "o-"),
    "CSA2, not compiled": (SERIES[1], ".:"),
    "CSA2, transformers attention": (SERIES[2], "s-"),
    "flash, record 91 windows": (SERIES[0], "o-"),
    "flash, full causal": (SERIES[4], "o--"),
}


def stacked(ax: plt.Axes, frame: pl.DataFrame, title: str) -> None:
    """Horizontal forward and backward bars of frame's ms, one segment per part;
    parts under 3% of the total go into "rest"."""
    total = float(frame["ms"].sum())
    frame = frame.with_columns(
        part=pl.when(pl.col("ms") >= 0.03 * total)
        .then("part")
        .otherwise(pl.lit("rest"))
    )
    parts = frame.group_by("part").agg(pl.col("ms").sum()).sort("ms", descending=True)
    passes = ("forward", "backward")
    left = [0.0, 0.0]
    colors = iter(plt.get_cmap("tab10").colors)  # type: ignore[attr-defined]
    for part in parts["part"]:
        ms = [
            float(
                frame.filter((pl.col("pass") == p) & (pl.col("part") == part))[
                    "ms"
                ].sum()
            )
            for p in passes
        ]
        ax.barh(passes, ms, left=left, label=part, color=next(colors))
        left = [a + b for a, b in zip(left, ms, strict=True)]
    ax.set_ylim(2.3, -0.6)  # inverted, room for the legend under the bars
    ax.set(xlabel="ms per 6144-token row", title=f"{title}: {total:.0f} ms")
    ax.legend(fontsize=7, loc="lower center", ncol=2, frameon=False)


def fig_bench() -> None:
    """nanogpt/bench.py: the 10 attention layers' forward+backward time per row, and
    where CSA2's goes."""
    path, profile = RUNS / "bench.csv", RUNS / "bench_profile.csv"
    if not path.exists() or not profile.exists():
        return
    frame, parts = read(path), read(profile)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), width_ratios=(1.2, 1, 1))
    ax = axes[0]
    for stack, (color, style) in STACKS.items():
        part = frame.filter(pl.col("stack") == stack).sort("T")
        ax.plot(part["T"], part["train_step_ms"], style, color=color, label=stack)
    csa2 = frame.filter(
        (pl.col("stack") == "CSA2") & pl.col("train_step_ms").is_not_nan()
    )
    last = csa2.sort("T").row(-1, named=True)
    ax.annotate(
        "CSA2: out of memory\nfrom 16K",
        (last["T"], last["train_step_ms"]),
        xytext=(10500, 450),
        arrowprops={"arrowstyle": "-", "color": SERIES[1], "lw": 0.6},
        fontsize=7,
        color=SERIES[1],
    )
    ax.axvspan(2048, 6144, color=REFERENCE, alpha=0.12, lw=0)
    ax.text(2048, ax.get_ylim()[1], " training rows", va="top", fontsize=7)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="tokens per row (training rows: packed documents; longer: one document)",
        ylabel="ms, forward + backward",
        title="10 attention layers, record 91 sizes, bf16",
    )
    lengths = sorted(set(frame["T"]))
    ax.set_xticks(lengths, [f"{t // 1024}K" for t in lengths])
    ax.set_xticks([], minor=True)
    ax.legend(fontsize=7)
    compiled = parts.filter(pl.col("mode") == "compiled")
    stacked(axes[1], compiled, "CSA2 compiled, by kernel kind")
    stacked(axes[2], parts.filter(pl.col("mode") == "eager"), "CSA2 eager, by part")
    save(fig, RUNS / "bench.png")


if __name__ == "__main__":
    apply()
    fig_probe()
    fig_train()
    fig_long()
    fig_history()
    fig_bench()
    fig_library()
