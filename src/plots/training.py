"""Training curves of the paper's LMs in runs/train: val loss vs tokens.

uv run python -m plots.training  -> runs/train/training.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

from experiments.results import read
from plots.style import SERIES, apply, save

RUNS = Path("runs/train")
PANELS = {
    "CSA2 language models": {
        "csa2_seq2048": ("model 1, 2048 tokens", SERIES[0], "-"),
        "csa2_seq2048_s1": ("model 2, 2048 tokens", SERIES[2], ":"),
        "csa2_seq8192": ("8192-token model", SERIES[0], "--"),
    },
}


def draw(axes: list[plt.Axes]) -> None:
    """One panel per PANELS entry, from the runs that exist."""
    for ax, (title, runs) in zip(axes, PANELS.items(), strict=True):
        for name, (label, color, style) in runs.items():
            path = RUNS / f"{name}.csv"
            if not path.exists() or path.stat().st_size == 0:  # not started
                continue
            frame = read(path).drop_nulls("val_loss")
            if frame.is_empty():  # no validation estimate yet
                continue
            final = frame["val_loss"][-1]
            ax.plot(
                frame["tokens"] / 1e6,
                frame["val_loss"],
                style,
                color=color,
                label=f"{label} (final estimate {final:.3f})",
            )
        ax.set(
            xlabel="training tokens (M)", ylabel="validation loss (nats)", title=title
        )
        ax.set_ylim(top=6.5)
        ax.legend()


def main() -> None:
    """Plot every panel from the runs that exist."""
    apply()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.5, 4.2), squeeze=False)
    draw(list(axes[0]))
    save(fig, RUNS / "training.png")


if __name__ == "__main__":
    main()
