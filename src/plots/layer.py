"""Bench (bench/layer.py): one CSA2 Full layer vs one exact softmax layer (SDPA
flash), bf16, d_model 512.

uv run python -m plots.layer  -> runs/bench/layer.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from experiments.results import read
from plots.style import REFERENCE, SERIES, apply, context_axis, save

BENCH = Path("runs/bench/layer.csv")
PANELS = {
    "prefill_ms": "prefill ms",
    "decode_us_batch1": "decode µs, 1 sequence",
    "decode_us_batch8": "decode µs, 8 sequences",
    "cache_mib": "KV cache MiB per sequence",
}


def panel(ax: plt.Axes, frame: pl.DataFrame, column: str, title: str) -> None:
    """One metric vs T per layer kind, exact in gray."""
    others = iter(SERIES)
    for layer in frame["layer"].unique(maintain_order=True):
        rows = frame.filter(pl.col("layer") == layer).sort("T")
        failed = rows.filter(pl.col(column).is_null())["T"]
        label = f"{layer} (OOM from {failed[0] // 1024}k)" if len(failed) else layer
        rows = rows.drop_nulls(column)
        color = REFERENCE if layer.startswith("exact") else next(others)
        ax.plot(rows["T"], rows[column], "o-", color=color, label=label)
    context_axis(ax, frame["T"].unique().sort().to_list()[::2])
    ax.set(yscale="log", title=title)
    ax.legend()


def main() -> None:
    """One panel per metric, T on x."""
    apply()
    frame = read(BENCH)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 3.8))
    for ax, (column, title) in zip(axes, PANELS.items(), strict=True):
        panel(ax, frame, column, title)
    save(fig, BENCH.with_suffix(".png"))


if __name__ == "__main__":
    main()
