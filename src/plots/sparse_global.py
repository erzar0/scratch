"""Figure and tables of parts 3 and 4 of the sparse decode study: the user's
sort-based variant (global halving, first round on 8 or 16 dims, losers
dropped, kept with approximate scores or given their mean value, keeps
128/256/512), RULER for its keep-512 rows, cutoffs from a sample against exact
ones, dims read in fixed groups of 8, and the global and sampled halving
kernels' timing.

    global.png  left: PG-19 loss above dense against keep, one line per first
                round and losers rule, SparQ 32 and 64 as horizontal lines;
                middle: RULER by context length; right: microseconds per layer
                from experiments/sparse_kernel.py
    global.md   the rows against SparQ 32 and against the 8-dim first round with
                losers dropped, paired over books; RULER pairs; timing; the
                hybrid kernel with bisection vs radix select; sampled vs exact
                cutoffs

uv run python -m plots.sparse_global
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import ticker

from experiments.results import paired_delta, read
from plots.sparse import FAMILY, RUNS, below, label, plain, summary
from plots.sparse_kernel import (
    TIMING,
    ruler,
    ruler50,
    ruler_pair,
    timing,
    timing_table,
    timings,
)
from plots.style import SERIES, apply, save

KEEPS = (128, 256, 512)
VARIANTS = {  # (first-round dims, losers) -> row, keep left open
    (8, "drop"): "tournament-query-halving-drop-{}",
    (8, "approx"): "tournament-query-halving-approx-{}",
    (16, "drop"): "first-16-drop-{}",
    (16, "approx"): "first-16-approx-{}",
    (8, "meanv"): "first-8-meanv-{}",
    (16, "meanv"): "first-16-meanv-{}",
}
SAMPLED = {  # sampled-median cutoff -> the same row with exact cutoffs
    "sampled-8-drop-512": VARIANTS[8, "drop"].format(512),
    "sampled-16-drop-512": VARIANTS[16, "drop"].format(512),
    "sampled-16-approx-512": VARIANTS[16, "approx"].format(512),
}
GROUPS = {  # fixed groups of 8 dims, keep 512 only -> marker
    "groups-pairs-8-drop-512": "s",
    "groups-contiguous-8-drop-512": "^",
}
LOSERS = {
    "drop": ("losers dropped", SERIES[0]),
    "approx": ("losers keep approximate scores", SERIES[3]),
    "meanv": ("losers' weight on their mean value", SERIES[2]),
}
WIDTH = {8: "-", 16: "--"}
SPARQ = "sparq-32-128"
REFERENCES = {SPARQ: "--", "sparq-64-128": ":"}
BASE = VARIANTS[8, "drop"].format(512)
APPROX = VARIANTS[8, "approx"].format(512)
WIDE = VARIANTS[16, "drop"].format(512)
WIDE_APPROX = VARIANTS[16, "approx"].format(512)
WIDE_MEAN = VARIANTS[16, "meanv"].format(512)
SPARQ_WIDE = "sparq-32-512"  # SparQ with as many exact keys as keep 512
RULER_ROWS = ["dense", SPARQ, SPARQ_WIDE, BASE, APPROX, WIDE, WIDE_APPROX, WIDE_MEAN]
PAIRS = [
    (APPROX, BASE),
    (WIDE, BASE),
    (APPROX, SPARQ),
    (WIDE, SPARQ),
    (WIDE_APPROX, WIDE),
    (WIDE_MEAN, WIDE),
    (BASE, SPARQ_WIDE),
    (WIDE, SPARQ_WIDE),
    (WIDE_APPROX, SPARQ_WIDE),
]
TIMED = {  # kernel.csv method -> legend
    "dense": "dense attention (best of SDPA and bmm-softmax-bmm)",
    "sparq-64-128": "SparQ, 64 dims, top-128",
    "sparq-32-128": "SparQ, 32 dims, top-128",
    "hybrid-512": "tournament, 3 rounds in blocks of 1024 then global, keep 512",
    "global-512": "tournament, global halving, exact cutoffs, keep 512",
    "sampled-8-drop-512": "global halving, sampled cutoffs, 8 first dims, keep 512",
    "sampled-16-drop-512": "global halving, sampled cutoffs, 16 first dims, keep 512",
    "sampled-16-approx-512": "the same, losers keep approximate scores (reads all V)",
    "parallel-32-512": "tournament, every round in 32 shuffled blocks per KV head, "
    "keep 512",
}
BISECTION = RUNS / "kernel_bisection.csv"


def pg19(ax: plt.Axes, frame: pl.DataFrame) -> None:
    """PG-19 loss against keep, per first-round width and losers rule."""
    points = {r: (e, s) for r, e, s in frame.select("row", "extra", "se").iter_rows()}
    for (width, losers), pattern in VARIANTS.items():
        rows = [(k, pattern.format(k)) for k in KEEPS]
        shown = [(k, *points[r]) for k, r in rows if r in points]
        if shown:
            xs, ys, es = zip(*shown, strict=True)
            words, color = LOSERS[losers]
            ax.errorbar(
                xs,
                ys,
                es,
                fmt="o",
                ls=WIDTH[width],
                color=color,
                label=f"tournament, first round on {width} dims, {words}",
            )
    for row, marker in GROUPS.items():
        if row in points:
            extra, se = points[row]
            ax.errorbar(
                512, extra, se, fmt=marker, color=FAMILY["groups"], label=label(row)
            )
    for row, style in REFERENCES.items():  # SparQ at top-k = keep
        dims = row.split("-")[1]
        tops = {k: f"sparq-{dims}-{k}" for k in KEEPS}
        shown = [(k, *points[r]) for k, r in tops.items() if r in points]
        if shown:
            xs, ys, es = zip(*shown, strict=True)
            words = f"SparQ, {dims} dims, top-k = keep"
            ax.errorbar(
                xs, ys, es, fmt="o", ls=style, color=FAMILY["sparq"], label=words
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(KEEPS, [str(k) for k in KEEPS])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_yscale("log")
    plain(ax.yaxis)
    ax.set_xlabel("keys left for exact attention (keep)")
    ax.set_ylabel("PG-19 loss above dense (nats)")
    ax.set_title("Global halving, |q| order, 32768 tokens", fontsize=9)
    below(ax)


def styles(row: str) -> tuple[str, str]:
    """A RULER row's color and line style, as in the PG-19 panel."""
    if row in REFERENCES:
        return FAMILY["sparq"], REFERENCES[row]
    if row == SPARQ_WIDE:
        return FAMILY["sparq"], "-"
    found = [v for v, p in VARIANTS.items() if p.format(512) == row]
    if not found:  # dense
        return FAMILY[row], "-"
    width, losers = found[0]
    return LOSERS[losers][1], WIDTH[width]


def table(frame: pl.DataFrame) -> list[str]:
    """Table lines: every row vs SparQ 32 and vs 8 first dims, losers dropped."""
    books = read(RUNS / "pg19.csv")

    def ce(row: str) -> np.ndarray:
        return books.filter(pl.col("row") == row).sort("book")["ce"].to_numpy()

    def versus(row: str, other: str) -> str:
        if row == other or len(ce(row)) != len(ce(other)):
            return "—"
        return "{:+.3f} ± {:.3f}".format(*paired_delta(ce(row), ce(other)))

    lines = [
        "| Method | Loss above dense (nats) "
        "| Minus SparQ, 32 dims, top-k = keep (nats) "
        "| Minus first round on 8 dims, losers dropped, same keep (nats) "
        "| K and V read (share of dense) | K sectors, best layout (share of dense K) |",
        "|---|---|---|---|---|---|",
    ]
    known = set(frame["row"].to_list())
    rows = [p.format(k) for k in KEEPS for p in VARIANTS.values()]
    rows += [*GROUPS]
    rows += [f"sparq-{d}-{k}" for d in (32, 64) for k in KEEPS]
    for row in [r for r in rows if r in known]:
        r = frame.filter(pl.col("row") == row).row(0, named=True)
        keep = row.rsplit("-", 1)[1]
        sparq = row.startswith("sparq")
        base = row if sparq else VARIANTS[8, "drop"].format(keep)
        cells = [
            label(row),
            f"{r['extra']:+.3f} ± {r['se']:.3f}",
            versus(row, f"sparq-32-{keep}"),
            versus(row, base),
            f"{r['reads']:.3f}",
            f"{r['sectors']:.3f}",
        ]
        lines.append(f"| {' | '.join(cells)} |")
    return lines


def sampled_table() -> list[str]:
    """Table lines: each sampled-cutoff row minus its exact-cutoff row, per book."""
    books = read(RUNS / "pg19.csv")

    def ce(row: str) -> np.ndarray:
        return books.filter(pl.col("row") == row).sort("book")["ce"].to_numpy()

    lines = [
        "| Method | Books | Sampled minus exact cutoffs, paired over books (nats) "
        "| Within 2 standard errors or under 0.01 nats |",
        "|---|---|---|---|",
    ]
    for row, exact in SAMPLED.items():
        if len(ce(row)) and len(ce(row)) == len(ce(exact)):
            mean, se = paired_delta(ce(row), ce(exact))
            ok = "yes" if abs(mean) <= 2 * se or abs(mean) < 0.01 else "no"
            cells = [label(row), str(len(ce(row))), f"{mean:+.4f} ± {se:.4f}", ok]
            lines.append(f"| {' | '.join(cells)} |")
    return lines


def radix_table(after: pl.DataFrame, before: pl.DataFrame) -> list[str]:
    """Table lines: hybrid keep 512 with bisection vs radix select, each run's SparQ."""
    lines = [
        "| Context length | hybrid, bisection (µs) | SparQ 32, same run (µs) "
        "| hybrid, radix select (µs) | SparQ 32, same run (µs) "
        "| hybrid ÷ SparQ 32, bisection | hybrid ÷ SparQ 32, radix select |",
        "|---|---|---|---|---|---|---|",
    ]
    wide = [f.pivot("method", index="n", values="us") for f in (before, after)]
    joined = wide[0].join(wide[1], on="n", suffix="_radix").sort("n")
    for r in joined.iter_rows(named=True):
        old, new = r["hybrid-512"], r["hybrid-512_radix"]
        ref_old, ref_new = r[SPARQ], r[f"{SPARQ}_radix"]
        cells = [f"{x:.1f}" for x in (old, ref_old, new, ref_new)]
        cells += [f"{old / ref_old:.2f}", f"{new / ref_new:.2f}"]
        lines.append(f"| {r['n']} | {' | '.join(cells)} |")
    return lines


def main() -> None:
    """Write global.png and global.md."""
    apply()
    frame = summary()
    scores = ruler50().filter(pl.col("row").is_in(RULER_ROWS))
    lines = table(frame) + ["", *sampled_table()]
    present = set(scores["row"].to_list())
    for pair in PAIRS:
        if set(pair) <= present:
            lines += ["", *ruler_pair(scores, pair)]
    times = timings() if TIMING.exists() else None
    if times is not None:
        lines += ["", *timing_table(times, TIMED)]
        if BISECTION.exists():
            lines += ["", *radix_table(times, timings(BISECTION))]
    (RUNS / "global.md").write_text("\n".join(lines) + "\n")
    print(RUNS / "global.md")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.5))
    pg19(axes[0], frame)
    ruler(axes[1], scores, styles)
    axes[1].set_title("RULER, keep 512 rows", fontsize=9)
    if times is not None:
        timing(axes[2], times, TIMED)
    save(fig, RUNS / "global.png")


if __name__ == "__main__":
    main()
