"""E4: chunk bandit as every softmax head (zero-shot), and its kernel bench.

Per T: KL to the exact model vs full-resolution tokens per query (refined_chunks
x chunk), per selector; marker = chunk size; gray: sliding windows. Last panel:
bench/chunk_bandit.py time vs T against cuDNN flash.

    uv run python -m plots.chunk_bandit  -> runs/chunk_bandit/chunk_bandit.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from experiments.results import read
from plots.style import REFERENCE, SERIES, apply, context_axis, save

OUT = Path("runs/chunk_bandit")
BENCH = Path("runs/bench/chunk_bandit.csv")
SELECTORS = {"bandit": SERIES[0], "oracle": SERIES[2], "random": SERIES[1]}
MARKERS = ["o", "s", "^"]


def quality(ax: plt.Axes, rows: pl.DataFrame, seq_len: int) -> None:
    """KL vs refined tokens at one T; pull variants left out (same shape)."""
    shapes = rows.filter(~pl.col("name").str.contains("pulls"))
    shapes = shapes.with_columns(
        (pl.col("chunk") * pl.col("refined_chunks")).alias("tokens")
    )
    chunks = shapes["chunk"].drop_nulls().unique().sort().to_list()
    for selector, color in SELECTORS.items():
        for marker, chunk in zip(MARKERS, chunks, strict=False):
            line = shapes.filter(
                (pl.col("selector") == selector) & (pl.col("chunk") == chunk)
            )
            ax.plot(
                line["tokens"],
                line["kl"],
                marker + "-",
                color=color,
                label=f"{selector} c{chunk}",
            )
    if "window" in rows.columns:  # T=4096 rows written
        windows = rows.filter(pl.col("selector") == "window").sort("window")
        ax.plot(
            windows["window"],
            windows["kl"],
            "x--",
            color=REFERENCE,
            label="sliding window",
        )
    exact = rows.filter(pl.col("selector") == "exact")["val_loss"][0]
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="full-resolution tokens per query",
        ylabel="KL to exact (nats / token)",
        title=f"T={seq_len} (exact val {exact:.3f})",
    )
    ax.legend(ncols=2)


def main() -> None:
    """One quality panel per T, then the bench."""
    apply()
    frame = read(OUT / "chunk_loss.csv")
    lengths = frame["T"].unique().sort().to_list()
    fig, axes = plt.subplots(1, len(lengths) + 1, figsize=(6 * (len(lengths) + 1), 4.5))
    for ax, seq_len in zip(axes, lengths, strict=False):
        quality(ax, frame.filter(pl.col("T") == seq_len), seq_len)
    if BENCH.exists():
        bench = read(BENCH)
        ax, configs = axes[-1], bench["config"].unique(maintain_order=True)
        shapes = list(dict.fromkeys(c.split(" ", 1)[1] for c in configs if " " in c))
        for config in configs:
            rows = bench.filter(pl.col("config") == config).drop_nulls("ms").sort("T")
            selector, _, shape = config.partition(" ")
            color = SERIES[shapes.index(shape)] if shape else REFERENCE
            style = "o--" if selector == "random" else "o-"
            ax.plot(rows["T"], rows["ms"], style, color=color, label=config)
        context_axis(ax, bench["T"].unique().sort().to_list())
        ax.set(yscale="log", ylabel="ms, 8 heads, bf16", title="chunk bandit vs flash")
        ax.legend(ncols=2)
    save(fig, OUT / "chunk_bandit.png")


if __name__ == "__main__":
    main()
