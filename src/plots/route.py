"""E1 double indexing and the indexer cost bench.

Top: val loss above pure CSA2 vs indexer keys read, per coarse mode (faint: every
config; line: front), model 1, base weights frozen; and indexer KiB read per decoded
token. Bottom: bench/indexer.py decode step (64 sequences), prefill indexer stage,
and its peak memory.

    uv run python -m plots.route  -> runs/route/route.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from experiments.eval_csa import flat_reference
from experiments.results import read
from model.train import TrainConfig
from plots.style import (
    REFERENCE,
    SERIES,
    apply,
    context_axis,
    front_series,
    save,
    target_line,
)

OUT = Path("runs/route")
BENCH = Path("runs/bench/indexer.csv")
MODES = {
    "mean": "mean key (zero-shot)",
    "bound": "bound (zero-shot)",
    "learned_r1": "learned router",
    "mixed_r1": "learned + bound in needle layers",
}


def quality(ax: plt.Axes, frame: pl.DataFrame, seq_len: int) -> None:
    """Loss above flat vs reads at one T, per coarse mode (model 1)."""
    rows = frame.filter((pl.col("T") == seq_len) & (pl.col("seed") == 0))
    for color, (mode, label) in zip(SERIES, MODES.items(), strict=False):
        points = rows.filter(pl.col("coarse") == mode)
        x, y = points["reads_per_seq"], points["delta_vs_flat"]
        front_series(ax, x.to_numpy(), y.to_numpy(), color, label)
    flat_loss, flat_reads = flat_reference(TrainConfig(seq_len=seq_len))
    ax.plot(
        flat_reads, 0, "D", color=REFERENCE, ms=7, label=f"pure CSA2 ({flat_loss:.4f})"
    )
    target_line(ax)
    ax.set(
        xscale="log",
        xlabel="indexer keys read per sequence and layer",
        ylabel="val loss - pure CSA2 (nats)",
        title=f"double indexing quality, T={seq_len}",
    )
    ax.legend()


def bench(ax: plt.Axes, frame: pl.DataFrame, column: str, title: str) -> None:
    """One bench metric vs T, flat in gray."""
    others = iter(SERIES)
    for method in frame["method"].unique(maintain_order=True):
        rows = frame.filter(pl.col("method") == method).drop_nulls(column).sort("T")
        color = REFERENCE if method == "flat" else next(others)
        ax.plot(rows["T"], rows[column], "o-", color=color, label=method)
    context_axis(ax, frame["T"].unique().sort().to_list())
    ax.set(yscale="log", title=title)
    ax.legend()


def main() -> None:
    """2 x 3 grid: quality at both T, memory; decode, prefill, prefill memory."""
    apply()
    frame = read(OUT / "route_loss.csv")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, seq_len in zip(axes[0, :2], (2048, 8192), strict=True):
        if not frame.filter(pl.col("T") == seq_len).is_empty():
            quality(ax, frame, seq_len)
    if BENCH.exists():
        timing = read(BENCH)
        bench(axes[0, 2], timing, "indexer_kib_per_token", "indexer KiB read / token")
        bench(axes[1, 0], timing, "decode_us_batch64", "decode µs, 64 sequences")
        bench(axes[1, 1], timing, "prefill_index_ms", "prefill indexer ms")
        bench(axes[1, 2], timing, "prefill_peak_gib", "prefill peak GiB")
    save(fig, OUT / "route.png")


if __name__ == "__main__":
    main()
