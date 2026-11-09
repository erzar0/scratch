"""Shared plot style, sized for a NeurIPS page: 7 pt Times-like serif (STIX, bundled
with matplotlib), thin recessive lines, fonts embedded as TrueType in the PDF.
One blue ramp encodes ordered values.

RAMP is the dataviz reference blue, steps 250 to 700 (light to dark); ramp(k)
picks k evenly spaced steps, validated as an ordinal palette on the light surface.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

INK, MUTED, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#f4f4f2"
SERIES = "#2a78d6"
RAMP = (
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)


def ramp(k: int) -> list[str]:
    """k colors from light to dark; one value gets the mid blue."""
    if k == 1:
        return [SERIES]
    return [RAMP[i] for i in np.linspace(0, len(RAMP) - 1, k).round().astype(int)]


def apply() -> None:
    """Set the shared rcParams."""
    mpl.use("Agg")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.title_fontsize": 6.5,
            "ps.fonttype": 42,
            "figure.facecolor": SURFACE,
            "axes.facecolor": "none",
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "axes.grid": False,
            "lines.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save(fig: Figure, path: Path) -> None:
    """Write the figure as PNG; print its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(path)
