"""Record 91 training runs (worktree ~/repos/modded-nanogpt-r91): validation loss
per training hour for the seed-1 runs, and every run's final validation loss and
training time (the log's train_time: validation excluded), all on one RTX 4070 Ti SUPER;
and the final checkpoints' validation loss by token group
(runs/nanogpt/checkpoints/val_groups.csv).

    uv run python -m nanogpt.plot

Same data, schedule and token count in every run: 10000 steps in 3 stages of
2048, 4096 and 6144-token rows, then 160 extension steps.
Stripped: record 91 without value embeddings into V, XSA, attention gates, DC and
paired heads, all left out of the CSA2 and tournament runs. Long documents: cut at the
row length instead of 896 / 2048 tokens. Tournament: nanogpt/tournament.py; fast
tournament: nanogpt/tournament_fast.py, same output, faster kernels; keys at 32K =
keys each query reads at 32K context. Writes runs/nanogpt/training/training_curves,
training_final, val_groups, attention_cost and, when their CSVs exist, coarse_rounds,
real_qk, skip_tiles, skip_tiles_mass, skip_blocks_mass_2048, skip_blocks_mass_8192,
skip_blocks_work, tile_scores_2048, tile_scores_8192, cutoff_timing,
tile_select_speed, tile_select_long_time, tile_select_long_memory, window_speed_time,
window_speed_memory, kv_cache, union_speed, decode_speed, select_methods,
select_in_kernel, topk_two_kernel, topk_passes, topk_warps, topk_versions,
topk_budget_time, topk_budget_memory, step_profile and step_profile_tiles (.png),
each in its topic folder: runs/nanogpt/training, checkpoints, selection_speed,
long_context, settings.
"""

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

from common import style

LOGS = Path.home() / "repos" / "modded-nanogpt-r91" / "logs"
OUT = Path("runs/nanogpt")
TRAINING, CHECKPOINTS, SPEED, LONG, SETTINGS = (
    OUT / x
    for x in ("training", "checkpoints", "selection_speed", "long_context", "settings")
)
LAST = 10160
MODELS = {  # name -> seed -> run id
    "Dense (record 91), FlashAttention 2": {
        "random": "ec348eea-fe95-43b8-b0d4-98b06c489b76",
        "1": "bc39288e-1ed4-4d80-a72c-c97f896c3783",
    },
    "Dense stripped, FlashAttention 2": {
        "random": "d14113e7-0492-46f1-bd2e-21a0327d5da6",
        "1": "337449e0-1b7d-41e4-9ea9-192f870a7ebb",
        "2": "9497acb4-9da5-4e9e-82f8-5a4111777428",
    },
    "CSA2 stripped, PyTorch": {
        "random": "c89581d5-75d5-4509-bb0b-2e885dbd93e5",
        "1": "296a91a4-55af-4b0b-be1a-3da00a57f95d",
    },
    "Tournament stripped, 512 keys at 32K, blocks of 1024, 8 channels per round": {
        "1": "5ce3c9c3-093a-43c7-b330-e7b3b90e4171",
    },
    "Fast tournament stripped, 512 keys at 32K, blocks of 2048, "
    "16 channels per round": {
        "1": "398ef768-7918-4de0-a888-5118b3487fe6",
    },
    "Fast tournament stripped, 1024 keys at 32K, blocks of 2048, "
    "16 channels per round": {
        "1": "150ce19e-9b78-4df5-9b85-dd9dcdf77e2c",
    },
    "Fast tournament stripped, 1024 keys at 32K, blocks of 2048, "
    "16 adjacent channels per round": {
        "1": "408a1744-fce0-4de3-8eef-67acd7fb6223",
    },
    "Exact top 64 per block stripped, 1024 keys at 32K, blocks of 2048, "
    "two Triton kernels": {
        "1": "43f64f34-15b2-451c-a458-dcd2446f2cf0",
    },
    "Tile selection stripped, 512 keys per 128-query tile, 32-key blocks": {
        "1": "d8a58427-414a-4a38-b0a0-1797027cf6c3",
    },
    "Dense stripped, long documents, FlashAttention 2": {
        "1": "f5ddc4a1-dbb4-4682-a28e-8d667271293b",
    },
    "CSA2 stripped, long documents, PyTorch": {
        "1": "8f1e55cd-1745-4f49-a7cc-0e251587424f",
    },
}
OLD = "Tournament stripped, 512 keys at 32K, blocks of 1024, 8 channels per round"
CSA2 = "CSA2 stripped, PyTorch"
# seed 1, default documents; 7 colors: old tournament and CSA2 left out
CURVES = [m for m in list(MODELS)[:9] if m not in (OLD, CSA2)]
COLORS = ("#eda100", "#eb6834", "#1baf7a", "#2a78d6", "#e87ba4", "#6250d6", "#008300")
# slots 4, 2, 3, 1, 5, 7, 6
MARKERS = {"1": "o", "2": "s", "random": "^"}


def val_losses(run: str) -> list[tuple[int, float, float]]:
    """(step, validation loss, training hours so far) lines of a run's log; empty
    if not started."""
    path = LOGS / f"{run}.txt"
    text = path.read_text() if path.exists() else ""
    found = re.findall(
        r"^step:(\d+)/\d+ val_loss:([\d.]+) train_time:(\d+)ms", text, re.M
    )
    return [(int(step), float(loss), int(ms) / 3.6e6) for step, loss, ms in found]


def curves() -> None:
    """Validation loss per training hour (validation excluded), seed 1; step 0
    (10.8 in every run) left out."""
    fig, ax = plt.subplots(figsize=(5.5, 2.2), layout="constrained")
    for name, color in zip(CURVES, COLORS, strict=True):
        points = val_losses(MODELS[name]["1"])[1:]
        if points:
            _, losses, hours = zip(*points, strict=True)
            ax.plot(hours, losses, color=color, marker="o", ms=1.5, label=name)
    ax.set(xlabel="training time (hours)", ylabel="validation loss", ylim=(3.4, 6.0))
    ax.set_xlim(0, None)
    ax.legend(loc="upper right", handlelength=1.5)
    style.save(fig, TRAINING / "training_curves")


def dot_panel(
    ax: Axes, names: list[str], column: int, xlim: tuple[float, float], xlabel: str
) -> None:
    """One row per model, one dot per seed: column 1 (loss) or 2 (hours) of the
    run's last validation line."""
    for row, name in enumerate(names):
        if row % 2:
            ax.axhspan(row - 0.5, row + 0.5, color=style.BAND, lw=0, zorder=0)
        for seed, run in MODELS[name].items():
            last = val_losses(run)[-1:]
            if last and last[0][0] == LAST:
                fill = "none" if seed == "random" else style.INK
                ax.plot(
                    last[0][column],
                    row,
                    MARKERS[seed],
                    color=style.INK,
                    mfc=fill,
                    ms=3,
                    mew=0.5,
                )
            else:  # not finished
                step = f" at step {last[0][0]}" if last else ""
                at = xlim[0] + 0.02 * (xlim[1] - xlim[0])
                ax.text(at, row, f"training{step}", va="center", style="italic")
    ax.set_ylim(len(names) - 0.5, -0.5)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=style.GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set(xlim=xlim, xlabel=xlabel)


def label_pad(fig: Figure, labels: list[str]) -> float:
    """Inches the widest label takes, plus a gap: the y tick pad that left-aligns
    row labels."""
    probe = [fig.text(0, 0, label) for label in labels]
    pad = max(t.get_window_extent().width for t in probe) / fig.dpi + 0.08
    for t in probe:
        t.remove()
    return pad


def panels(
    names: list[str], count: int, gap: float = 0.3, below: float = 0.42
) -> tuple[Figure, list[Axes]]:
    """count side-by-side axes sharing one row per name, names left-aligned on the
    first; `below` inches at the bottom for x labels and a legend."""
    width = 5.5  # inches
    height = below + 0.155 * len(names)  # x axis and label, then the rows
    fig = plt.figure(figsize=(width, height), dpi=300)  # measure text as saved
    pad = label_pad(fig, names)
    panel = (0.98 * width - 0.03 - pad - gap * (count - 1)) / count
    bottom, rows = below / height, 0.155 * len(names) / height
    left = (0.03 + pad) / width
    axes = [
        fig.add_axes((left + i * (panel + gap) / width, bottom, panel / width, rows))
        for i in range(count)
    ]
    for i, ax in enumerate(axes):
        ax.set_yticks(range(len(names)), [] if i else names, ha="left")
        ax.tick_params(axis="y", length=0, pad=pad * 72)
    return fig, axes


def finals() -> None:
    """Final validation loss and training time, one row per model, one dot per
    seed."""
    names = list(MODELS)
    fig, (loss, hours) = panels(names, 2, below=0.56)
    dot_panel(loss, names, 1, (3.4, 4.0), f"final validation loss\n(step {LAST})")
    dot_panel(hours, names, 2, (0, 3.2), "training time\n(hours)")
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=m,
            color=style.INK,
            ms=3,
            mew=0.5,
            mfc="none" if s == "random" else style.INK,
        )
        for s, m in MARKERS.items()
    ]
    fig.legend(
        handles,
        [f"seed {s}" if s != "random" else "random seed" for s in MARKERS],
        ncols=3,
        loc="lower left",
        bbox_to_anchor=(0, 0),
        handletextpad=0.2,
        columnspacing=1,
    )
    style.save(fig, TRAINING / "training_final")


def row_figure(labels: list[str]) -> tuple[Figure, Axes]:
    """A dot panel: one row per label, labels left-aligned, alternate rows banded."""
    fig, ax = plt.subplots(
        figsize=(5.5, 0.42 + 0.155 * max(len(labels), 4)), dpi=300, layout="constrained"
    )
    rows = range(len(labels))
    for row in rows[1::2]:
        ax.axhspan(row - 0.5, row + 0.5, color=style.BAND, lw=0, zorder=0)
    ax.set_yticks(rows, labels, ha="left")
    ax.tick_params(axis="y", length=0, pad=label_pad(fig, labels) * 72)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=style.GRID, lw=0.4)
    ax.set_axisbelow(True)
    return fig, ax


def groups() -> None:
    """Final validation loss by token group, one row per checkpoint and attention it
    was evaluated with (runs/nanogpt/checkpoints/val_groups.csv): the first 128
    validation rows of 8192 tokens (1M tokens), batched as record 91 validates,
    without the final canonical-token mask. A row's first segment starts
    mid-document, without BOS; training documents start with BOS and end by 2048."""
    table = pl.read_csv(CHECKPOINTS / "val_groups.csv")
    names = table["model"].unique(maintain_order=True).to_list()
    parts = table.select("group", "share").unique("group", maintain_order=True)
    fig, axes = panels(names, parts.height, gap=0.15, below=0.95)
    for ax, (group, share) in zip(axes, parts.iter_rows(), strict=True):
        loss = table.filter(pl.col("group") == group)
        rows = [names.index(m) for m in loss["model"]]
        for row in range(1, len(names), 2):
            ax.axhspan(row - 0.5, row + 0.5, color=style.BAND, lw=0, zorder=0)
        ax.plot(loss["loss"], rows, "o", color=style.INK, ms=3, mew=0)
        low, high = min(loss["loss"]), max(loss["loss"])
        margin = max(0.05, 0.1 * (high - low))
        ax.set_xlim(low - margin, high + margin)
        ax.set_ylim(len(names) - 0.5, -0.5)
        text = f"loss: {group} ({share:.0%} of tokens)"
        ax.set_xlabel(textwrap.fill(text, 15, break_on_hyphens=False), fontsize=6)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color=style.GRID, lw=0.4)
        ax.set_axisbelow(True)
    style.save(fig, CHECKPOINTS / "val_groups")


def dot_rows(
    table: pl.DataFrame,
    rows: dict[str, pl.Expr],
    value: str,
    series: str,
    xlabel: str,
    name: Path,
    log: bool = False,
) -> None:
    """A dot panel: one row per label (rows: label -> filter), one dot per value of
    the series column, the mean of value over the rest (log x axis if log); legend
    below; saved as name.png."""
    fig, ax = row_figure(list(rows))
    keys = table[series].unique(maintain_order=True).to_list()
    # lengths are ordered: one-hue ramp; anything else: categorical slots 1, 2, 3, ...
    colors = (
        style.ramp(len(keys))
        if series in ("n", "shape")
        else [COLORS[i] for i in (3, 1, 2, 0)][: len(keys)]
    )
    for key, color in zip(keys, colors, strict=True):
        means = [
            table.filter(match & (pl.col(series) == key))
            .select(pl.col(value).mean())
            .item()
            for match in rows.values()
        ]
        label = f"n = {key}" if series == "n" else str(key)
        ax.plot(
            means,
            range(len(rows)),
            "o",
            color=color,
            ms=3.5,
            mec=style.SURFACE,
            mew=0.4,
            label=label,
        )
    ax.set(xlabel=xlabel, xscale="log" if log else "linear")
    if log:  # plain numbers at 1, 2, 5 x 10^k: readable over one or two decades
        ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter("")
    widest = max(len(str(k)) for k in keys)  # ~110 characters fit on one line
    ncols = len(keys) if widest * len(keys) <= 110 else max(1, 110 // widest)
    fig.legend(loc="outside lower center", ncols=ncols, handletextpad=0.2)
    style.save(fig, name)


def coarse_rounds() -> None:
    """Synthetic output error of coarser rounds at run 1's setting (blocks of 2048,
    32 winners each) relative to run 1's, at the same data (distribution, length
    2048, 8192 or 32768, seed): mean over lengths and 3 seeds, one dot per
    distribution (runs/nanogpt/settings/coarse_rounds_fidelity.csv; scratch
    coarse_screen.py).
    Passes: rounds over the keys, each one cutoff."""
    table = pl.read_csv(SETTINGS / "coarse_rounds_fidelity.csv")
    same = ["distribution", "n", "seed"]
    run1 = table.filter(pl.col("cut") == 2).select(*same, run1="error")
    table = table.join(run1, on=same).with_columns(
        relative=pl.col("error") / pl.col("run1")
    )
    rows = {}
    for cut, dims in (
        table.select("cut", "dims").unique(maintain_order=True).iter_rows()
    ):
        passes = ((2048 // 32).bit_length() - 1) // (cut.bit_length() - 1)
        label = f"keep 1/{cut} per round, {dims} channels each: {passes} passes"
        rows[label + (" (run 1)" if cut == 2 else "")] = (pl.col("cut") == cut) & (
            pl.col("dims") == dims
        )
    xlabel = "output error vs dense attention, relative to run 1's"
    dot_rows(
        table, rows, "relative", "distribution", xlabel, SETTINGS / "coarse_rounds"
    )


def real_qk() -> None:
    """Output error on run 1's own q, k, v (scratch real_qk.py): last 128 queries of
    2 validation documents at n = 2048 and 8192, 10 layers, 6 heads; runs/nanogpt/
    real_qk.csv."""
    table = pl.read_csv(CHECKPOINTS / "real_qk.csv")
    rows = {
        m: pl.col("method") == m for m in table["method"].unique(maintain_order=True)
    }
    xlabel = "output error vs dense attention"
    dot_rows(table, rows, "error", "n", xlabel, CHECKPOINTS / "real_qk")


def skip_tiles() -> None:
    """Flex tiles (128 x 128) computed and dense softmax mass kept per selection rule
    on run 2's own q, k: 2 validation documents, 10 layers, 6 heads, first n tokens
    (scratch skip_tiles.py; runs/nanogpt/checkpoints/skip_tiles.csv)."""
    table = pl.read_csv(CHECKPOINTS / "skip_tiles.csv")
    rows = {
        m: pl.col("method") == m for m in table["method"].unique(maintain_order=True)
    }
    xlabel = "share of causal tiles computed"
    dot_rows(table, rows, "tiles computed", "n", xlabel, CHECKPOINTS / "skip_tiles")
    xlabel = "dense softmax mass on the keys attended"
    name = CHECKPOINTS / "skip_tiles_mass"
    dot_rows(table, rows, "mass kept", "n", xlabel, name)


def skip_blocks() -> None:
    """Dense softmax mass kept and query x key area flex computes when each
    128-query tile picks key blocks by pooled score, against per-query rules, on
    run 2's own q, k (scratch skip_blocks.py; checkpoints/skip_blocks.csv)."""
    table = pl.read_csv(CHECKPOINTS / "skip_blocks.csv")
    rows = {
        m: pl.col("method") == m for m in table["method"].unique(maintain_order=True)
    }
    for n in (2048, 8192):
        xlabel = f"dense softmax mass on the keys attended, first {n} tokens"
        name = CHECKPOINTS / f"skip_blocks_mass_{n}"
        dot_rows(
            table.filter(pl.col("n") == n), rows, "mass kept", "block", xlabel, name
        )
    blocks = pl.col("block").is_in(["per query", "blocks of 16 keys"])
    same = table.filter(blocks & pl.col("work").is_not_null())  # area: any block size
    rows = {
        m: pl.col("method") == m for m in same["method"].unique(maintain_order=True)
    }
    xlabel = "query x key area computed, share of dense flex's"
    dot_rows(same, rows, "work", "n", xlabel, CHECKPOINTS / "skip_blocks_work")


def tile_scores() -> None:
    """Dense softmax mass kept when each query tile picks key units (16-key blocks
    or single keys) by a tile score, by query tile size and key budget, against each
    query's own best keys, on run 2's own q, k (scratch skip_scores.py;
    checkpoints/tile_scores.csv). Best set: per query in its rows, else shared."""
    order = [
        "Mean query . mean key (current)",
        "Max over the tile's queries",
        "Estimated attention mass",
        "Best set (oracle)",
    ]
    table = pl.read_csv(CHECKPOINTS / "tile_scores.csv").with_columns(
        pl.col("score").str.replace(r"^Best .*\(oracle\)$", "Best set (oracle)")
    )
    rows = {}
    labels = table.select("budget", "keys", "query tile").unique(maintain_order=True)
    for budget, keys, tile in labels.sort("budget", maintain_order=True).iter_rows():
        label = f"{budget} keys, {keys}, {tile} per tile"
        if keys == "per query":
            label = f"{budget} keys per query"
        rows[label] = (
            (pl.col("budget") == budget)
            & (pl.col("keys") == keys)
            & (pl.col("query tile") == tile)
        )
    table = table.with_columns(pl.col("score").cast(pl.Enum(order))).sort("score")
    for n in (2048, 8192):
        xlabel = f"dense softmax mass on the keys attended, first {n} tokens"
        name = CHECKPOINTS / f"tile_scores_{n}"
        at_n = table.filter(pl.col("n") == n)
        dot_rows(at_n, rows, "mass kept", "score", xlabel, name)


def tile_select_speed() -> None:
    """Forward + backward ms per call at the training and validation shapes, and no
    grad ms and peak memory on one long document: dense flex, exact top 64 per
    2048-key block, tile selection (scratch speedups/tile_select_speed.py)."""
    table = pl.read_csv(SPEED / "tile_select_speed.csv")
    rows = {
        v: pl.col("version") == v for v in table["version"].unique(maintain_order=True)
    }
    xlabel = "forward + backward time per call (ms)"
    dot_rows(table, rows, "ms_per_call", "shape", xlabel, SPEED / "tile_select_speed")
    path = LONG / "tile_select_long.csv"
    if not path.exists():
        return
    table = pl.read_csv(path)
    rows = {
        v: pl.col("version") == v for v in table["version"].unique(maintain_order=True)
    }
    xlabel = "time per call, no grad, one document (ms)"
    name = LONG / "tile_select_long_time"
    dot_rows(table, rows, "ms_per_call", "n", xlabel, name, log=True)
    xlabel = "peak memory above q, k, v (MiB)"
    name = LONG / "tile_select_long_memory"
    dot_rows(table, rows, "peak_mib", "n", xlabel, name, log=True)


def window_speed() -> None:
    """No grad ms per call and peak memory on one long document: FlashAttention 2's
    sliding window (the dense stripped model's attention) vs tile selection inside
    the window (tile_window), at the dense model's windows of 768 and 2560 keys
    (scratch speedups/window_speed.py)."""
    table = pl.read_csv(LONG / "window_speed.csv")
    table = table.filter(pl.col("input") == "one document")
    rows = {
        f"{v}, window {w}": (pl.col("version") == v) & (pl.col("window") == w)
        for w in table["window"].unique(maintain_order=True)
        for v in table["version"].unique(maintain_order=True)
    }
    xlabel = "time per call, no grad, one document (ms)"
    name = LONG / "window_speed_time"
    dot_rows(table, rows, "ms_per_call", "n", xlabel, name, log=True)
    xlabel = "peak memory above q, k, v (MiB)"
    name = LONG / "window_speed_memory"
    dot_rows(table, rows, "peak_mib", "n", xlabel, name, log=True)


def kv_cache() -> None:
    """KV cache each method keeps for decode, batch 1, record 91 stripped model (10
    attention layers, 6 heads x 128, bf16), allocated and read from torch (scratch
    kv_cache.py)."""
    table = pl.read_csv(LONG / "kv_cache.csv")
    rows = {
        v: pl.col("version") == v for v in table["version"].unique(maintain_order=True)
    }
    xlabel = "KV cache, batch 1, all 10 attention layers (MiB)"
    dot_rows(table, rows, "cache_mib", "n", xlabel, LONG / "kv_cache", log=True)


def union_speed() -> None:
    """No grad ms per call on one long document's last 2048 queries: dense
    FlashAttention 2 vs picking 1024 keys over the shuffled windows on exact or union
    scores (block_topk_fast's kernels; scratch speedups/union_speed.py)."""
    table = pl.read_csv(SPEED / "union_speed.csv")
    rows = {
        v: pl.col("version") == v for v in table["version"].unique(maintain_order=True)
    }
    xlabel = "time per call, no grad, last 2048 queries of one document (ms)"
    dot_rows(table, rows, "ms_per_call", "n", xlabel, SPEED / "union_speed", log=True)


def decode_speed() -> None:
    """One decode step of one layer (batch 1, 6 heads x 128, CUDA graph, L2 flushed):
    dense FlashAttention 2 decode vs pick_decode on exact, SparQ-rule or union-rule
    scores, 1024 keys (scratch decode_speed.py)."""
    table = pl.read_csv(SPEED / "decode_speed.csv")
    rows = {
        v: pl.col("version") == v for v in table["version"].unique(maintain_order=True)
    }
    xlabel = "time per decode step, one layer, batch 1 (ms)"
    dot_rows(table, rows, "ms_per_call", "n", xlabel, SPEED / "decode_speed", log=True)


def cutoff_timing() -> None:
    """Selection ms per call (q, k -> bit words) at run 1's setting, exact vs each
    approximate cutoff, at the 4 training and validation shapes (scratch
    cutoff_timing.py; runs/nanogpt/selection_speed/cutoff_timing.csv)."""
    table = pl.read_csv(SPEED / "cutoff_timing.csv")
    rows = {
        v: pl.col("version") == v for v in table["version"].unique(maintain_order=True)
    }
    xlabel = "selection time per call (ms)"
    dot_rows(table, rows, "ms_per_call", "shape", xlabel, SPEED / "cutoff_timing")


def select_rebuild() -> None:
    """Halving-rounds ms per call at run 1's setting, mean of 8 average stage-3
    rows: radix select before and after the exact early exit, and ceilings with one
    part of the kernel removed (scratch speedups/round1_ablate.py;
    runs/nanogpt/selection_speed/select_rebuild.csv)."""
    table = pl.read_csv(SPEED / "select_rebuild.csv")
    rows = {
        v: pl.col("version") == v for v in table["version"].unique(maintain_order=True)
    }
    xlabel = "halving rounds, time per call (ms)"
    dot_rows(table, rows, "ms_per_call", "launch", xlabel, SPEED / "select_rebuild")


def channel_blocks_timing() -> None:
    """Selection ms per call at run 2's settings, mean of 8 average rows per shape:
    tournament_fast vs contiguous channel blocks on the same rounds kernels, and the
    dropped first version reading K in place (stages 1-3), whole selection, its
    halving rounds and its channel order kernel (scratch
    speedups/channel_blocks_timing.py; runs/nanogpt/selection_speed/
    channel_blocks_timing.csv, channel_blocks_timing_in_place.csv)."""
    table = pl.concat(
        pl.read_csv(SPEED / f"channel_blocks_timing{x}.csv") for x in ("", "_in_place")
    )
    parts = {
        "whole selection": ["total"],
        "halving rounds": ["_rounds"],
        "channel order": ["_order", "_windows"],
    }
    kernel = pl.col("kernel")
    rows = {
        f"{shape}, {label}": (pl.col("shape") == shape) & kernel.is_in(name)
        for shape in table["shape"].unique(maintain_order=True).to_list()
        for label, name in parts.items()
    }
    xlabel = "time per call (ms)"
    dot_rows(table, rows, "ms", "version", xlabel, SPEED / "channel_blocks_timing")


SELECTORS = {
    "radix": "radix select, early exit (current)",
    "sort": "full bitonic sort",
    "topk": "bitonic top-k (tl.topk)",
    "topk tile": "bitonic top-k, several rows per program",
    "bisect": "bisection on the cutoff",
    "bisect tile": "bisection, several rows per program",
    "delegate": "bisection from 64 group maxima",
    "delegate tile": "group maxima, several rows per program",
    "torch": "torch.topk",
    "RAFT select_k": "RAFT select_k (pylibraft 24.02)",
}


def select_methods() -> None:
    """Standalone top-k kernels on the tournament's selection problems at run 2's
    settings, random scores, ms per call summed over launches and rounds, mean of 8
    rows per shape: each round's top half, and the top 64 of each block (scratch
    speedups/select_methods.py, select_raft.py; runs/nanogpt/selection_speed/
    select_methods.csv, select_methods_raft.csv)."""
    table = (
        pl.concat(pl.read_csv(SPEED / f"select_methods{x}.csv") for x in ("", "_raft"))
        .group_by("shape", "problem", "method", maintain_order=True)
        .agg(pl.col("ms").sum())
    )
    problems = {"top half": "top half per round", "top-64": "top 64 per block"}
    rows = {
        f"{label}: {SELECTORS[m]}": (pl.col("problem") == p) & (pl.col("method") == m)
        for p, label in problems.items()
        for m in SELECTORS
        if len(table.filter((pl.col("problem") == p) & (pl.col("method") == m)))
    }
    xlabel = "time per call (ms)"
    dot_rows(table, rows, "ms", "shape", xlabel, SPEED / "select_methods", log=True)


def select_in_kernel() -> None:
    """Selection ms per call (every kernel of select()) at run 2's settings, mean of
    8 rows per shape: exact cutoffs inside the halving rounds (same winners), the
    rule-changing halving in groups and exact top 64 per block (scratch
    speedups/threshold_in_kernel.py; runs/nanogpt/selection_speed/
    select_in_kernel.csv)."""
    table = pl.read_csv(SPEED / "select_in_kernel.csv")
    rows = {
        m: pl.col("method") == m for m in table["method"].unique(maintain_order=True)
    }
    xlabel = "time per call (ms)"
    dot_rows(table, rows, "total ms", "shape", xlabel, SPEED / "select_in_kernel")


def topk_two_kernel() -> None:
    """Selection ms per call at run 2's settings: the current tournament, exact top
    64 per block in torch and in two Triton kernels (tensor-core scores, bisection
    select); mean of 8 rows per shape, and one long document's last 2048 queries
    (scratch speedups/topk_two_kernel.py; runs/nanogpt/selection_speed/
    topk_two_kernel.csv)."""
    table = pl.read_csv(SPEED / "topk_two_kernel.csv")
    rows = {s: pl.col("shape") == s for s in table["shape"].unique(maintain_order=True)}
    xlabel = "selection time per call (ms)"
    name = SPEED / "topk_two_kernel"
    dot_rows(table, rows, "total ms", "method", xlabel, name, log=True)


def topk_select_tune() -> None:
    """Variants of block_topk_fast's select kernel at run 2's settings, winners equal
    in all: bisection passes per (query, block) at 1, 3, 7 pivots per pass; select
    kernel ms by warps per width; ms per select() call by version; ms and peak MiB
    by score buffer size; mean of 8 rows per shape, one 32K document's last 2048
    queries, run 2's real q, k (scratch speedups/topk_select_tune.py,
    topk_pivots1d.py; runs/nanogpt/selection_speed/topk_select_tune.csv)."""
    table = pl.read_csv(SPEED / "topk_select_tune.csv", infer_schema_length=None)
    test, setting, shape = pl.col("test"), pl.col("setting"), pl.col("shape")
    shapes = {s: shape == s for s in table["shape"].unique(maintain_order=True)}
    passes = table.filter(test == "passes").with_columns(
        setting.str.replace("1 pivots per pass", "1 pivot per pass (bisection)")
    )
    xlabel = "mean passes per (query, block)"
    dot_rows(passes, shapes, "mean passes", "setting", xlabel, SPEED / "topk_passes")
    warps = table.filter(setting.str.starts_with("width ")).with_columns(
        width=setting.str.extract(r"width (\d+)"),
        setting=setting.str.extract(r": (.*)").str.replace("^1 warps", "1 warp"),
    )
    rows = {
        f"{s}, width {w}": (shape == s) & (pl.col("width") == w)
        for s in warps["shape"].unique(maintain_order=True)
        for w in warps["width"].unique(maintain_order=True)
    }
    xlabel = "select kernel time per call (ms)"
    dot_rows(warps, rows, "select ms", "setting", xlabel, SPEED / "topk_warps")
    versions = {
        "Bisection, warps = width / 256 (before)": (test == "pivots 1-D")
        & (setting == "src kernel"),
        "Bisection, tuned warps (adopted)": setting.str.starts_with("tuned {"),
        "3 pivots per pass": setting == "3 pivots per pass, 1-D",
        "2 pivots per pass, counts packed in one sum": setting
        == "2 pivots per pass, counts packed in one reduction",
    }
    timing = pl.concat(
        table.filter(m).with_columns(version=pl.lit(v)) for v, m in versions.items()
    )
    xlabel = "selection time per call (ms)"
    name = SPEED / "topk_versions"
    dot_rows(timing, shapes, "total ms", "version", xlabel, name, log=True)
    budget = table.filter(test == "budget").with_columns(
        pl.col("peak MiB").cast(pl.Float64)
    )
    sizes = {
        f"score buffer {s}": setting == s
        for s in budget["setting"].unique(maintain_order=True)
    }
    xlabel = "selection time per call (ms)"
    dot_rows(budget, sizes, "total ms", "shape", xlabel, SPEED / "topk_budget_time")
    xlabel = "peak GPU memory of one call (MiB)"
    dot_rows(budget, sizes, "peak MiB", "shape", xlabel, SPEED / "topk_budget_memory")


def chunk_memory() -> None:
    """One attention call (no grad) on one document of n tokens: peak GPU memory
    above q, k, v and its time, tournament_fast at run 1's settings on the whole
    row vs query chunks, dense flex for reference (scratch speedups/
    chunk_memory.py; runs/nanogpt/long_context/chunk_memory.csv), and chunks of 2048
    before key_copies() (chunk_memory_reads_in_place.csv). A missing n does not fit: the
    whole row at 131072 needs 19.3 GB on a 16 GB GPU and overflows int32 offsets."""
    chunks = "Fast tournament, query chunks of 2048"
    before = (
        pl.read_csv(LONG / "chunk_memory_reads_in_place.csv")
        .filter(pl.col("version") == chunks)
        .with_columns(version=pl.lit(f"{chunks}, keys read in place (before)"))
    )
    table = pl.concat([pl.read_csv(LONG / "chunk_memory.csv"), before]).with_columns(
        peak_gib=pl.col("peak_mib") / 1024
    )
    most = max(table["n"].to_list())
    rows = {}
    for version in table["version"].unique(maintain_order=True).to_list():
        match = pl.col("version") == version
        short = most not in table.filter(match)["n"].to_list()
        rows[f"{version} (n = {most}: over 16 GB)" if short else version] = match
    xlabel = "peak memory above q, k, v (GiB)"
    dot_rows(table, rows, "peak_gib", "n", xlabel, LONG / "chunk_memory", log=True)
    xlabel = "time per call (ms)"
    dot_rows(table, rows, "ms_per_call", "n", xlabel, LONG / "chunk_time", log=True)


def long_scaling() -> None:
    """GPU time per (query, head, key) on one long document: selection for the last
    2048 queries with blocks shuffled as trained, read in place (before) or from
    key_copies() (now), vs contiguous blocks (timing only), and a whole dense flex
    call (scratch speedups/long_scaling.py, chunk_memory.py;
    runs/nanogpt/long_context/long_scaling.csv, chunk_memory.csv)."""
    picked = (
        pl.read_csv(LONG / "long_scaling.csv")
        .group_by("n", "layout", maintain_order=True)
        .agg(pl.col("ns_per_key").sum())
    )
    dense = (
        pl.read_csv(LONG / "chunk_memory.csv")
        .filter(pl.col("version").str.starts_with("Dense"))
        .select(
            "n",
            layout=pl.lit("dense"),
            ns_per_key=pl.col("ms_per_call") * 1e6 / (3 * pl.col("n") ** 2),
        )
    )
    table = pl.concat([picked, dense.cast(picked.schema)])
    rows = {
        "Selection, shuffled blocks read in place (before)": (
            pl.col("layout") == "shuffled"
        ),
        "Selection, shuffled blocks from key copies (now)": (
            pl.col("layout") == "copies"
        ),
        "Selection, blocks contiguous (timing only)": pl.col("layout") == "contiguous",
        "Selection, contiguous channel blocks": (pl.col("layout") == "channel blocks"),
        "Dense flex attention, whole call": pl.col("layout") == "dense",
    }
    xlabel = "GPU time per query, head and key (ns)"
    dot_rows(table, rows, "ns_per_key", "n", xlabel, LONG / "long_scaling", log=True)


def step_profile() -> None:
    """GPU ms per stage-3 training step by part: run 1, run 2 (current tournament
    and exact top 64 per block in two Triton kernels) and dense stripped
    checkpoints, 10 timed steps each (scratch speedups/step_profile.py; runs/nanogpt/
    step_profile.csv; the full-radix run 1 rows are not plotted)."""
    table = pl.read_csv(TRAINING / "step_profile.csv").filter(
        pl.col("script") != "train_tournament.py full radix"
    )
    kernel = pl.col("kernel")
    parts = {
        "Whole step (wall clock)": kernel == "wall",
        "FlashAttention 2, forward + backward": kernel.str.contains("flash::"),
        "Flex attention, forward + backward": kernel.str.contains("flex_attention"),
        "Selection: halving rounds": kernel == "_rounds",
        "Selection: score + select kernels": kernel.is_in(["_scores", "_select"]),
        "Selection: channel order, bit words, tie redo": kernel.is_in(
            ["_order", "_bits", "_redo"]
        ),
    }
    names = {
        "train_tournament.py": "Run 1: fast tournament, 32 winners",
        "run 2 current tournament": "Run 2: fast tournament, 64 winners",
        "run 2 exact top 64 two kernels": "Run 2: exact top 64 per block, Triton",
        "train_dense_stripped.py": "Dense stripped",
    }
    table = table.filter(pl.col("script").is_in(list(names)))  # tiles: next plot
    summed = pl.concat(
        table.filter(match)
        .group_by("script", maintain_order=True)
        .agg(pl.col("ms_per_step").sum())
        .with_columns(part=pl.lit(label))
        for label, match in parts.items()
    ).with_columns(pl.col("script").replace_strict(names))
    rows = {label: pl.col("part") == label for label in parts}
    xlabel = "GPU time per training step (ms)"
    dot_rows(summed, rows, "ms_per_step", "script", xlabel, TRAINING / "step_profile")


def step_profile_tiles() -> None:
    """Stage-3 training step on the run 2 checkpoint: dense stripped, tile
    selection in 32-key blocks and tile keys (single keys, 128 or 32 queries per
    tile), 512 keys per tile, wall clock and GPU time by part (scratch
    speedups/step_profile.py; runs/nanogpt/step_profile.csv)."""
    names = {
        "train_dense_stripped.py": "Dense stripped",
        "run 2 tile selection, 32-key blocks, 512 keys per tile": (
            "Tiles, 32-key blocks"
        ),
        **{
            f"run 2 tile keys, {q} queries per tile, 512 keys per tile": (
                f"Single keys, {q} queries per tile"
            )
            for q in (128, 32)
        },
    }
    table = pl.read_csv(TRAINING / "step_profile.csv").filter(
        pl.col("script").is_in(list(names))
    )
    kernel = pl.col("kernel")
    attention = kernel.str.contains("flash::") | kernel.str.contains("flex_attention")
    parts = {
        "Whole step (wall clock)": kernel == "wall",
        "Attention kernels (FlashAttention 2 or flex)": attention,
        "Every other GPU kernel": (kernel != "wall") & ~attention,
    }
    summed = pl.concat(
        table.filter(match)
        .group_by("script", maintain_order=True)
        .agg(pl.col("ms_per_step").sum())
        .with_columns(part=pl.lit(label))
        for label, match in parts.items()
    ).with_columns(pl.col("script").replace_strict(names))
    rows = {label: pl.col("part") == label for label in parts}
    xlabel = "time per stage-3 training step (ms)"
    name = TRAINING / "step_profile_tiles"
    dot_rows(summed, rows, "ms_per_step", "script", xlabel, name)


def attention_cost() -> None:
    """Attention's GPU time per training step (80 calls: 10 layers, 8 micro-steps),
    forward + backward, by stage, from a scratch benchmark of one layer on rows
    packed from training documents (runs/nanogpt/training/attention_cost.csv); last row:
    the logs' ms/step, tournament minus dense stripped seed 1."""
    table = pl.read_csv(TRAINING / "attention_cost.csv")
    rows = {  # label -> parts summed
        "Dense: FlashAttention 2, sliding windows": (
            "dense short window",
            "dense long window",
        ),
        "Tournament: whole attention": ("tournament whole",),
        "    selection: halving rounds (Triton)": ("halving rounds (Triton)",),
        "    mask building and other": ("mask building and other",),
        "    flex attention, forward + backward": ("flex forward", "flex backward"),
        "    selection: |q| channel sort": ("|q| channel sort",),
        "Training step, tournament minus dense (logs)": (
            "logged step time difference",
        ),
    }
    fig, ax = row_figure(list(rows))
    stages = table.select("stage", "row_tokens").unique(maintain_order=True)
    for (stage, tokens), color in zip(stages.iter_rows(), style.ramp(3), strict=True):
        part = table.filter(pl.col("stage") == stage)
        ms = [
            float(part.filter(pl.col("part").is_in(parts))["ms_per_step"].sum())
            for parts in rows.values()
        ]
        label = f"stage {stage}: {tokens}-token rows"
        ax.plot(ms, range(len(rows)), "o", color=color, ms=3, mew=0, label=label)
    ax.set(xlim=(0, 620), xlabel="GPU time per training step (ms)")
    ax.legend(loc="center right", handletextpad=0.2)
    style.save(fig, TRAINING / "attention_cost")


if __name__ == "__main__":
    style.apply()
    curves()
    finals()
    groups()
    attention_cost()
    for extra, csv in (
        (coarse_rounds, SETTINGS / "coarse_rounds_fidelity"),
        (real_qk, CHECKPOINTS / "real_qk"),
        (skip_tiles, CHECKPOINTS / "skip_tiles"),
        (skip_blocks, CHECKPOINTS / "skip_blocks"),
        (tile_scores, CHECKPOINTS / "tile_scores"),
        (cutoff_timing, SPEED / "cutoff_timing"),
        (select_rebuild, SPEED / "select_rebuild"),
        (channel_blocks_timing, SPEED / "channel_blocks_timing"),
        (select_methods, SPEED / "select_methods"),
        (select_in_kernel, SPEED / "select_in_kernel"),
        (topk_two_kernel, SPEED / "topk_two_kernel"),
        (tile_select_speed, SPEED / "tile_select_speed"),
        (topk_select_tune, SPEED / "topk_select_tune"),
        (chunk_memory, LONG / "chunk_memory"),
        (window_speed, LONG / "window_speed"),
        (kv_cache, LONG / "kv_cache"),
        (union_speed, SPEED / "union_speed"),
        (decode_speed, SPEED / "decode_speed"),
        (long_scaling, LONG / "long_scaling"),
        (step_profile, TRAINING / "step_profile"),
        (step_profile_tiles, TRAINING / "step_profile"),
    ):
        if csv.with_suffix(".csv").exists():
            extra()
