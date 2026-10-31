"""How each tested method chooses what to read, on synthetic scores (paper figures).

methods_csa.svg: pure CSA2 and double indexing choosing main KV entries for
one query over 32 entries (top 8 kept, superblocks of 4). Each row runs the
method's rule (deepseek/double_indexing.py) in a few lines of numpy, shrunk so each
block fits on the page; the reads printed are in index-key units, as in the paper.

    uv run python -m plots.methods  -> report/figs/methods_csa.svg
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from plots.style import GRID, INK, REFERENCE

N_BLOCKS, SIZE, TOP_K, KEPT = 32, 4, 8, 3
SUPERBLOCKS = N_BLOCKS // SIZE
CMAP, NORM = plt.get_cmap("Blues"), Normalize(-0.3, 1.0)
RNG = np.random.default_rng(3)
# Indexer score of each block for the query: low background, high recent blocks,
# and "needles": block 3 sits alone in a dull superblock, so averaging hides it.
SCORE = 0.08 + 0.12 * RNG.random(N_BLOCKS)
SCORE[[3, 13, 14, 21]] = [0.95, 0.85, 0.7, 0.9]
SCORE[28:] = [0.5, 0.55, 0.65, 0.75]
TARGET = set(np.argsort(-SCORE)[:TOP_K].tolist())  # pure CSA's picks


def top(values: np.ndarray, k: int, allowed: np.ndarray | None = None) -> set[int]:
    """Indices of the k largest values among allowed."""
    masked = values if allowed is None else np.where(allowed, values, -np.inf)
    return {int(i) for i in np.argsort(-masked)[:k] if np.isfinite(masked[i])}


def in_superblocks(kept: set[int]) -> np.ndarray:
    """Block mask of the kept superblocks."""
    return np.isin(np.arange(N_BLOCKS) // SIZE, list(kept))


def row(
    ax: Axes,
    y: float,
    values: np.ndarray,
    shown: np.ndarray,
    picked: set[int] = frozenset(),  # type: ignore[assignment]
) -> None:
    """One row of block cells: shaded by value where read, blank where skipped.

    Args:
        ax: Target axes (x = block index).
        y: Bottom of the row.
        values: Value per block.
        shown: Blocks this step read.
        picked: Blocks outlined as the final picks.
    """
    for block in range(N_BLOCKS):
        face = CMAP(NORM(values[block])) if shown[block] else "white"
        ax.add_patch(
            Rectangle((block, y), 0.9, 0.8, facecolor=face, edgecolor=GRID, lw=0.5)
        )
        if block in picked:
            ax.add_patch(
                Rectangle((block, y), 0.9, 0.8, fill=False, edgecolor=INK, lw=1.4)
            )


def frame(ax: Axes, title: str, n_rows: float, reads: str, picked: set[int]) -> None:
    """Axes limits, title, pure-CSA markers on top and the reads/recall note."""
    for block in TARGET:
        ax.plot(block + 0.45, n_rows + 0.15, "v", ms=3, color=REFERENCE)
    for edge in range(0, N_BLOCKS + 1, SIZE):
        ax.axvline(edge - 0.05, color=GRID, lw=0.8, zorder=0)
    found = len(picked & TARGET)
    ax.set_title(title, loc="left", fontsize=8)
    ax.text(
        N_BLOCKS + 0.4,
        n_rows / 2,
        f"reads {reads}\nfinds {found}/{TOP_K}",
        va="center",
        fontsize=6.5,
    )
    ax.set(xlim=(-0.2, N_BLOCKS + 5.5), ylim=(-0.1, n_rows + 0.45))
    ax.axis("off")


def label(ax: Axes, y: float, text: str) -> None:
    ax.text(-0.5, y + 0.4, text, ha="right", va="center", fontsize=6)


def pure_csa(ax: Axes) -> None:
    row(ax, 0, SCORE, np.ones(N_BLOCKS, bool), TARGET)
    label(ax, 0, "score all")
    frame(
        ax,
        "Pure CSA2 (baseline): indexer scores every entry, keeps the top 8",
        1,
        "32",
        TARGET,
    )


def double_indexing(ax: Axes) -> None:
    coarse = SCORE.reshape(SUPERBLOCKS, SIZE).mean(1)  # "mean" coarse mode
    kept = top(coarse, KEPT)
    picked = top(SCORE, TOP_K, in_superblocks(kept))
    row(
        ax, 1, np.repeat(coarse, SIZE), np.isin(np.arange(N_BLOCKS) // SIZE, list(kept))
    )
    label(ax, 1, "1. score groups")
    row(ax, 0, SCORE, in_superblocks(kept), picked)
    label(ax, 0, "2. score inside")
    frame(
        ax,
        "Double indexing: score groups of 4 entries, then entries only in the best "
        f"{KEPT} groups",
        2,
        f"{SUPERBLOCKS} + {KEPT * SIZE}",
        picked,
    )


def fig_csa() -> plt.Figure:
    heights = [1.0, 2.0]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(5.5, 1.9),
        gridspec_kw={"height_ratios": [h + 1 for h in heights]},
    )
    for ax, draw in zip(axes, (pure_csa, double_indexing), strict=True):
        draw(ax)
    legend = [
        Line2D([], [], ls="", marker="v", color=REFERENCE, label="pure CSA2's top 8"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, lw=1.4, label="final picks"),
        Rectangle((0, 0), 1, 1, color=CMAP(0.8), label="entry read (darker = higher)"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=GRID, label="entry not read"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=6)
    return fig


def main() -> None:
    """Write the figure with the paper style."""
    from plots import paper

    paper.style()
    fig = fig_csa()
    paper.save_svg(fig, "methods_csa", bottom=0.1)


if __name__ == "__main__":
    main()
