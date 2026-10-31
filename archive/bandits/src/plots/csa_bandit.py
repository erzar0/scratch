"""E2 / E6 / E7 headline: every CSA block selector vs flat, base weights frozen.

Column 1: val loss above pure (flat) CSA vs selection work in indexer-key reads
(faint: every seed-0 config; line: front; bars: min-max over seeds 0-2 where stage 3
ran), with double indexing's front over its coarse modes (E1) and E0's random /
attention-oracle selectors at top_k 8. Columns 2-3: superblock-bandit stage-2
ablations of ucb_width and second_pulls around the stage-1 front.

    uv run python -m plots.csa_bandit  -> runs/csa_bandit/csa_bandit.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from experiments.eval_csa_bandit import flat_reference
from experiments.results import read
from plots.style import REFERENCE, SERIES, apply, front_series, save, target_line

OUT = Path("runs/csa_bandit")
STUDIES = {
    "superblock": ("superblock bandit", SERIES[1]),
    "channel": ("channel bandit", SERIES[2]),
    "token": ("token bandit", SERIES[3]),
}
ROUTE_MODES = ["mean", "bound", "learned_r1", "mixed_r1"]
SUPERBLOCK = ["superblock_size", "kept_superblocks", "first_pulls", "survivors"]


def seed_spread(frame: pl.DataFrame) -> pl.DataFrame:
    """Min / max delta over seeds of configs that have more than one seed."""
    knobs = [
        c
        for c in frame.columns
        if c not in ("T", "stage", "seed", "reads_per_seq", "reads_ratio")
        and c not in ("val_loss", "delta_vs_flat")
    ]
    return (
        frame.group_by(knobs)
        .agg(
            pl.col("reads_per_seq").mean(),
            pl.col("delta_vs_flat").min().alias("low"),
            pl.col("delta_vs_flat").max().alias("high"),
            pl.col("seed").n_unique().alias("seeds"),
        )
        .filter(pl.col("seeds") > 1)
    )


def frontier(ax: plt.Axes, seq_len: int) -> None:
    """Every selector's loss-vs-reads front at one T."""
    flat_loss, flat_reads = flat_reference(seq_len)
    route = Path("runs/route/route_loss.csv")
    if route.exists():
        rows = read(route).filter(
            (pl.col("T") == seq_len) & pl.col("coarse").is_in(ROUTE_MODES)
        )
        x, y = rows["reads_per_seq"].to_numpy(), rows["delta_vs_flat"].to_numpy()
        front_series(ax, x, y, SERIES[0], "double indexing (best coarse mode)")
    for name, (label, color) in STUDIES.items():
        path = OUT / f"{name}.csv"
        if not path.exists():
            continue
        rows = read(path).filter(pl.col("T") == seq_len)
        seed0 = rows.filter(pl.col("seed") == 0)
        x, y = seed0["reads_per_seq"].to_numpy(), seed0["delta_vs_flat"].to_numpy()
        front_series(ax, x, y, color, label)
        spread = seed_spread(rows)
        ax.vlines(spread["reads_per_seq"], spread["low"], spread["high"], color=color)
    baselines = read("runs/csa/csa_loss.csv").filter(
        (pl.col("T") == seq_len) & (pl.col("top_k") == 8) & (pl.col("window") == 32)
    )
    for selector, marker in (("oracle", "*"), ("random", "x")):
        row = baselines.filter(pl.col("selector") == selector)
        if row.is_empty():
            continue
        delta = row["val_loss"][0] - flat_loss
        if selector == "random":  # reads nothing: off the log axis, off scale
            bottom, top = ax.get_ylim()
            ax.set_ylim(min(bottom, -0.003), min(top, delta))  # keep flat at 0
            ax.axhline(
                delta,
                color=REFERENCE,
                ls="-.",
                lw=1,
                label=f"random blocks {delta:+.3f} (off scale)",
            )
            continue
        ax.plot(
            row["reads_per_seq"][0],
            delta,
            marker,
            color=REFERENCE,
            ms=10,
            label="oracle (attention's own top 8)",
        )
    ax.plot(
        flat_reads,
        0,
        "D",
        color=REFERENCE,
        ms=7,
        label=f"pure CSA (loss {flat_loss:.3f})",
    )
    target_line(ax)
    ax.set(
        xscale="log",
        xlabel="selection work: index keys read per sequence (left = cheaper)",
        ylabel="extra loss vs pure CSA (nats, lower = better)",
        title=f"block pickers on {seq_len}-token text",
    )
    ax.legend()


def ablation(ax: plt.Axes, seq_len: int, knob: str, fixed: str) -> None:
    """Superblock bandit: delta vs `knob`, one line per front config."""
    path = OUT / "superblock.csv"
    if not path.exists():
        return
    rows = read(path).filter((pl.col("T") == seq_len) & (pl.col("seed") == 0))
    bases = (
        rows.filter(pl.col("stage") == 2).select(SUPERBLOCK).unique(maintain_order=True)
    )
    for color, base in zip(SERIES, bases.iter_rows(named=True), strict=False):
        line = rows.join(pl.DataFrame([base]), on=SUPERBLOCK)
        # the front config's value: shared by most of its stage-2 variants
        center = line.filter(pl.col("stage") == 2)[fixed].mode().sort()[0]
        line = line.filter(pl.col(fixed) == center).unique(knob).sort(knob)
        label = "S={superblock_size} s={kept_superblocks} {first_pulls}+ m={survivors}"
        ax.plot(
            line[knob],
            line["delta_vs_flat"],
            "o-",
            color=color,
            label=label.format(**base),
        )
    target_line(ax)
    ax.set(
        xlabel=knob,
        ylabel="extra loss vs pure CSA (nats, lower = better)",
        title=f"superblock bandit: effect of {knob}, T={seq_len}",
    )
    ax.legend()


def main() -> None:
    """Rows T; columns frontier, ucb_width ablation, second_pulls ablation."""
    apply()
    lengths = read("runs/csa/csa_loss.csv")["T"].unique().sort().to_list()
    fig, axes = plt.subplots(
        len(lengths), 3, figsize=(17, 5 * len(lengths)), squeeze=False
    )
    for row, seq_len in zip(axes, lengths, strict=True):
        frontier(row[0], seq_len)
        ablation(row[1], seq_len, "ucb_width", "second_pulls")
        ablation(row[2], seq_len, "second_pulls", "ucb_width")
    save(fig, OUT / "csa_bandit.png")


if __name__ == "__main__":
    main()
