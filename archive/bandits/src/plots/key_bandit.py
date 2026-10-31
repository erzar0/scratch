"""E5: per-key bandit on real softmax heads.

Left: output relative error vs reads (fraction of exact causal attention), mean
over 12 heads x 3 seeds; one line per ucb_width through tolerance, one panel
per dims_per_round. Right: val loss with every head swapped for the bandit
(64 windows), reads from the error sweep.

    uv run python -m plots.key_bandit  -> runs/key_bandit/key_bandit.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from experiments.results import read
from model.heads import SEQ_LEN
from plots.style import REFERENCE, SERIES, apply, front_series, save

OUT = Path("runs/key_bandit")
EXACT_READS = (SEQ_LEN + 1) / 2
KNOBS = ["ucb_width", "tolerance", "dims_per_round"]


def main() -> None:
    """Plot the error sweep and the loss table."""
    apply()
    errors = (
        read(OUT / "key_error.csv")
        .group_by(KNOBS)
        .agg(pl.col("reads").mean() / EXACT_READS, pl.col("rel_err").mean())
        .sort(KNOBS)
    )
    steps = errors["dims_per_round"].unique().sort().to_list()
    widths = errors["ucb_width"].unique().sort().to_list()
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(14, 3.8))
    for ax in axes[1:-1]:
        ax.sharey(axes[0])
    for ax, step in zip(axes, steps, strict=False):
        for color, width in zip(SERIES, widths, strict=False):
            line = errors.filter(
                (pl.col("dims_per_round") == step) & (pl.col("ucb_width") == width)
            )
            ax.plot(
                line["reads"], line["rel_err"], "o-", color=color, label=f"z={width}"
            )
        ax.set(
            xlim=(0, 1.02),
            yscale="log",
            xlabel="reads / exact",
            ylabel="relative error",
            title=f"{step} dims per round",
        )
        ax.legend(title="ucb width (tolerance 1→10)")
    losses = read(OUT / "key_loss.csv")
    bandit = losses.filter(pl.col("method") == "key bandit")
    bandit = bandit.with_columns(pl.col(KNOBS).cast(pl.Float64)).join(
        errors.with_columns(pl.col(KNOBS).cast(pl.Float64)), on=KNOBS
    )
    exact = losses.filter(pl.col("method") == "exact")["val_loss"][0]
    ax = axes[-1]
    ax.axhline(exact, color=REFERENCE, ls="--", label=f"exact {exact:.3f}")
    front_series(
        ax,
        bandit["reads"].to_numpy(),
        bandit["val_loss"].to_numpy(),
        SERIES[0],
        "bandit",
    )
    for row in bandit.iter_rows(named=True):
        ax.annotate(
            "{ucb_width:g}/{tolerance:g}/{dims_per_round:g}".format(**row),
            (row["reads"], row["val_loss"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
        )
    ax.set(
        xlabel="reads / exact",
        ylabel="val loss (nats)",
        title="all heads swapped (labels z/tol/dims)",
    )
    ax.legend()
    save(fig, OUT / "key_bandit.png")


if __name__ == "__main__":
    main()
