"""Shared plot style: categorical hues in a fixed order, recessive axes and grid.

Color follows the method across every figure: SERIES[i] is the i-th method of a
plot; REFERENCE (gray) is always the baseline (flat CSA, exact softmax).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#4a3aa7"]
REFERENCE, INK, GRID = "#52514e", "#0b0b0b", "#e4e3df"
TARGET = 0.01  # nats above the baseline counted as "matches it"


def apply() -> None:
    """Set the shared rcParams."""
    mpl.use("Agg")
    plt.rcParams.update(
        {
            "axes.edgecolor": REFERENCE,
            "axes.labelcolor": INK,
            "xtick.color": REFERENCE,
            "ytick.color": REFERENCE,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "font.size": 9,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
        }
    )


def front(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points no other point beats on both x and y (lower is better), by x."""
    order = np.lexsort((y, x))
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    keep = y < np.minimum.accumulate(np.concatenate([[np.inf], y[:-1]]))
    return x[keep], y[keep]


def front_series(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, label: str
) -> None:
    """Every point faint, the lower-left front as a line."""
    if len(x) == 0:
        return
    ax.plot(x, y, "o", color=color, ms=3.5, alpha=0.3)
    ax.plot(*front(np.asarray(x), np.asarray(y)), "o-", color=color, label=label)


def context_axis(ax: plt.Axes, lengths: list[int]) -> None:
    """Log-2 context-length x axis labelled 1k, 2k, ..."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths, [f"{n // 1024}k" for n in lengths])
    ax.minorticks_off()
    ax.set_xlabel("context length (tokens)")


def target_line(ax: plt.Axes) -> None:
    """Dotted +TARGET line on a loss-above-baseline axis."""
    ax.axhline(TARGET, color=REFERENCE, ls=":", lw=1)
    ax.annotate(
        f"+{TARGET} nats",
        (0.99, TARGET),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=7,
        color=REFERENCE,
        ha="right",
    )


def save(fig: Figure, path: Path) -> None:
    """Write the figure and print its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(path)


def final_val(run: str) -> float | None:
    """Full-split val loss from a training log's 'final val' line, if present."""
    path = Path("runs/train") / f"{run}.log"
    if not path.exists():
        return None
    lines = [s for s in path.read_text().splitlines() if "final val" in s]
    return float(lines[-1].split("final val ")[1].split()[0]) if lines else None
