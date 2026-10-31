"""Tournaments on keys in position order against keys in random slots, keep 512.

    shuffled.png  left: PG-19 loss above dense at 32768 tokens per selection, keys
                  in position order beside keys in random slots
                  (sparse/shuffled.py), with global halving and SparQ 32 as
                  lines; right: recall of the exact top-128, mean over layers
    shuffled.md   the same, with differences paired over books, and RULER (50
                  samples) of shuffled blocks of 1024 against global halving and
                  SparQ 32 top-512, paired by sample

uv run python -m plots.sparse_shuffled
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from experiments.results import paired_delta, read
from plots.sparse import FAMILY, RUNS, below, label, plain, summary
from plots.sparse_kernel import ruler50, ruler_pair
from plots.style import REFERENCE, apply, save

PAIRS = {  # selection -> (position order row, random slots row)
    "halving in blocks of 1024": ("blocks-1024-512", "shuffled-blocks-1024-512"),
    "halving in blocks of 4096": ("blocks-4096-512", "shuffled-blocks-4096-512"),
    "bracket": (
        "tournament-query-bracket-drop-512",
        "shuffled-tournament-query-bracket-drop-512",
    ),
}
GLOBAL = "tournament-query-halving-drop-512"
LINES = {GLOBAL: (FAMILY["tournament"], ":"), "sparq-32-512": (FAMILY["sparq"], "--")}
LINES["sparq-32-256"] = (FAMILY["sparq"], "-.")
RULER_PAIRS = [
    ("shuffled-blocks-1024-512", GLOBAL),
    ("shuffled-blocks-1024-512", "sparq-32-512"),
]
SIDES = {"keys in position order": "blocks", "keys in random slots": "shuffled"}


def ce(row: str) -> np.ndarray:
    """Per-book loss of the row, in book order."""
    books = read(RUNS / "pg19.csv").filter(pl.col("row") == row).sort("book")
    return books["ce"].to_numpy()


def versus(row: str, other: str) -> str:
    """row minus other, paired over books."""
    return "{:+.3f} ± {:.3f}".format(*paired_delta(ce(row), ce(other)))


def main() -> None:
    """Write shuffled.png and shuffled.md."""
    apply()
    frame = summary()
    stats = {r["row"]: r for r in frame.iter_rows(named=True)}
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))
    for side, (words, family) in enumerate(SIDES.items()):
        shown = [(i, pair[side]) for i, pair in enumerate(PAIRS.values())]
        shown = [(i, r) for i, r in shown if r in stats]
        xs = [i + (side - 0.5) * 0.2 for i, _ in shown]
        fill = "none" if side == 0 else FAMILY[family]
        style = {"color": FAMILY[family], "mfc": fill, "label": words}
        extra = [stats[r]["extra"] for _, r in shown]
        left.errorbar(xs, extra, [stats[r]["se"] for _, r in shown], fmt="o", **style)
        right.plot(xs, [stats[r]["recall"] for _, r in shown], "o", **style)
    for row, (color, ls) in LINES.items():
        if row in stats:
            left.axhline(
                stats[row]["extra"], color=color, ls=ls, lw=1, label=label(row)
            )
            right.axhline(stats[row]["recall"], color=color, ls=ls, lw=1)
    for ax in (left, right):
        ax.set_xticks(range(len(PAIRS)), list(PAIRS))
        ax.set_xlim(-0.5, len(PAIRS) - 0.5)
    left.set_yscale("log")
    plain(left.yaxis)
    left.set_ylabel("PG-19 loss above dense at 32768 tokens (nats)")
    left.set_title("Tournament, |q| order, losers dropped, keep 512", fontsize=9)
    right.set_ylabel("recall of the exact top-128 keys, mean over layers")
    right.set_title("Recall", fontsize=9)
    right.axhline(1, color=REFERENCE, lw=0.5)
    below(left)
    save(fig, RUNS / "shuffled.png")

    lines = [
        "| Selection, keep 512 | Keys in position order: loss above dense (nats) "
        "| Keys in random slots: loss above dense (nats) "
        "| Random slots minus position order (nats) "
        f"| Random slots minus {label(GLOBAL)} (nats) "
        "| Recall of the exact top-128, position order / random slots |",
        "|---|---|---|---|---|---|",
    ]
    for words, (stored, mixed) in PAIRS.items():
        if stored not in stats or mixed not in stats:
            continue
        a, b = stats[stored], stats[mixed]
        lines.append(
            f"| {words} | {a['extra']:+.3f} ± {a['se']:.3f} "
            f"| {b['extra']:+.3f} ± {b['se']:.3f} | {versus(mixed, stored)} "
            f"| {versus(mixed, GLOBAL)} | {a['recall']:.3f} / {b['recall']:.3f} |"
        )
    lines += ["", "| Reference | Loss above dense (nats) | Recall |", "|---|---|---|"]
    lines += [
        f"| {label(r)} | {stats[r]['extra']:+.3f} ± {stats[r]['se']:.3f} "
        f"| {stats[r]['recall']:.3f} |"
        for r in LINES
        if r in stats
    ]
    present = set(ruler50()["row"].to_list())
    for pair in RULER_PAIRS:
        if set(pair) <= present:
            lines += ["", *ruler_pair(ruler50(), pair)]
    (RUNS / "shuffled.md").write_text("\n".join(lines) + "\n")
    print(RUNS / "shuffled.md")


if __name__ == "__main__":
    main()
