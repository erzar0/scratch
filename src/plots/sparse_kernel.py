"""Figure and table of part 2 of the sparse decode study: halving inside fixed
blocks of keys (stage F), only the first rounds inside blocks (stage F2), and
RULER at 50 samples per task and length (stage G).

    kernel.png  top left: PG-19 loss above dense against keys per block, one line
                per keep, global halving at the right end; top right: that loss
                against the rounds run inside blocks of 1024, the rest global;
                bottom left: RULER score by context length, rows with 50 samples
                per task and length; bottom right: microseconds per layer from
                experiments/sparse_kernel.py
    kernel.md   the block and hybrid rows against SparQ with 32 dims, paired over
                books; timing and speedup over dense;
                RULER, tournament minus SparQ, paired by sample

uv run python -m plots.sparse_kernel
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import ticker

from experiments.results import paired_delta, read
from plots.sparse import (
    FAMILY,
    LONGCTX,
    RULER_TASKS,
    RUNS,
    below,
    label,
    look,
    plain,
    summary,
)
from plots.style import apply, context_axis, save

BLOCKS = (1024, 4096)
KEEPS = (256, 512)
ALL = 32768  # one block holding every slot: global halving
GLOBAL = "tournament-query-halving-drop-{}"
SPARQ = "sparq-32-128"
HYBRID_ROW = "hybrid-1024-3-512"
RULER_ROWS = ["dense", SPARQ, HYBRID_ROW, *(GLOBAL.format(k) for k in KEEPS)]
PAIRS = [  # stage G
    *((GLOBAL.format(k), SPARQ) for k in KEEPS),
    (HYBRID_ROW, SPARQ),
    (HYBRID_ROW, GLOBAL.format(512)),
]
TIMING = RUNS / "kernel.csv"
TIMED = {  # kernel.csv method -> legend
    "dense": "dense attention (best of SDPA and bmm-softmax-bmm)",
    "sparq-64-128": "SparQ, 64 dims, top-128 (PG-19 loss of the keep-512 hybrid)",
    "sparq-32-128": "SparQ, 32 dims, top-128",
    "sparq-16-128": "SparQ, 16 dims, top-128",
    "hybrid-256": "tournament hybrid, last 3 rounds global, keep 256",
    "hybrid-512": "tournament hybrid, last 3 rounds global, keep 512",
}
LENGTHS = [4096, 8192, 16384, 32768]
SAMPLES = 50
HYBRID = 1024, 512, 6  # stage F2: block, keep, rounds at 32768 tokens


def block_rows() -> list[str]:
    """Stage F rows and their references, in table order."""
    rows = [f"blocks-{b}-{k}" for k in KEEPS for b in BLOCKS]
    return [*rows, *hybrid_rows().values(), *(GLOBAL.format(k) for k in KEEPS), SPARQ]


def hybrid_rows() -> dict[int, str]:
    """Rounds inside blocks -> row, from 0 (global) to every round (all blocks)."""
    block, keep, total = HYBRID
    inner = {m: f"hybrid-{block}-{m}-{keep}" for m in range(1, total)}
    return {0: GLOBAL.format(keep), **inner, total: f"blocks-{block}-{keep}"}


def blocks(ax: plt.Axes, frame: pl.DataFrame) -> None:
    """PG-19 loss against keys per block, one line per keep."""
    points = {r: (e, s) for r, e, s in frame.select("row", "extra", "se").iter_rows()}
    for keep, style in zip(KEEPS, ("-", "--"), strict=True):
        rows = [(b, f"blocks-{b}-{keep}") for b in BLOCKS] + [
            (ALL, GLOBAL.format(keep))
        ]
        shown = [(b, *points[r]) for b, r in rows if r in points]
        if shown:
            xs, ys, es = zip(*shown, strict=True)
            ax.errorbar(
                xs,
                ys,
                es,
                fmt="o",
                ls=style,
                color=FAMILY["tournament"],
                label=f"tournament, |q| order, losers dropped, keep {keep}",
            )
    for row, style in ((SPARQ, "--"), ("exact-128", ":")):
        if row in points:
            color = FAMILY[row.split("-")[0]]
            ax.axhline(points[row][0], color=color, ls=style, lw=1, label=label(row))
    ax.set_xscale("log", base=2)
    ax.set_xticks([*BLOCKS, ALL], [*map(str, BLOCKS), "all (global)"])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_yscale("log")
    plain(ax.yaxis)
    ax.set_xlabel("keys per block (each block halves on its own)")
    ax.set_ylabel("PG-19 loss above dense (nats)")
    ax.set_title("Halving inside fixed blocks, 32768 tokens", fontsize=9)
    below(ax)


def hybrid(ax: plt.Axes, frame: pl.DataFrame) -> None:
    """PG-19 loss against the rounds run inside blocks, the rest global."""
    points = {r: (e, s) for r, e, s in frame.select("row", "extra", "se").iter_rows()}
    block, keep, total = HYBRID
    shown = [(m, *points[r]) for m, r in hybrid_rows().items() if r in points]
    if shown:
        xs, ys, es = zip(*shown, strict=True)
        ax.errorbar(
            xs,
            ys,
            es,
            fmt="o-",
            color=FAMILY["tournament"],
            label=f"tournament, |q| order, losers dropped, keep {keep}",
        )
    for row, style in ((SPARQ, "--"), ("exact-128", ":")):
        if row in points:
            color = FAMILY[row.split("-")[0]]
            ax.axhline(points[row][0], color=color, ls=style, lw=1, label=label(row))
    ax.set_xticks(range(total + 1))
    ax.set_xlabel(f"rounds halved inside blocks of {block} (of {total}; rest global)")
    ax.set_ylabel("PG-19 loss above dense (nats)")
    ax.set_title("First rounds in blocks, then global, 32768 tokens", fontsize=9)
    below(ax)


def ruler50() -> pl.DataFrame:
    """RULER samples (index < 50) of the rows that have all of them; qwen = dense."""
    scores = read(LONGCTX).filter(
        pl.col("task").is_in(RULER_TASKS)
        & (pl.col("index") < SAMPLES)
        & pl.col("row").str.contains(r"^qwen(:|$)")
    )
    scores = scores.with_columns(
        row=pl.when(pl.col("row") == "qwen")
        .then(pl.lit("dense"))
        .otherwise(pl.col("row").str.strip_prefix("qwen:"))
    )
    counts = scores.group_by("row").agg(pl.len())
    full = counts.filter(pl.col("len") == len(RULER_TASKS) * len(LENGTHS) * SAMPLES)
    return scores.filter(pl.col("row").is_in(full["row"].to_list()))


def ruler(
    ax: plt.Axes,
    scores: pl.DataFrame,
    styles: Callable[[str], tuple[str, str]] | None = None,
) -> None:
    """RULER score by context length, with its standard error; styles: row ->
    (color, line style), else the family colors."""
    rows = sorted(scores["row"].unique().to_list(), key=lambda r: (r != "dense", r))
    for row in rows:
        part = (
            scores.filter(pl.col("row") == row)
            .group_by("length")
            .agg(
                mean=100 * pl.col("score").mean(),
                se=100 * pl.col("score").std() / pl.len().sqrt(),
            )
            .sort("length")
        )
        others = [r for r in rows if r != "dense"]
        color, style = styles(row) if styles else look(row, others)
        ax.errorbar(
            part["length"],
            part["mean"],
            part["se"],
            fmt="o",
            ls=style,
            color=color,
            label=label(row),
        )
    context_axis(ax, LENGTHS)
    ax.set_ylabel(f"RULER score (%), 4 tasks, {SAMPLES} samples each")
    below(ax)


def table(frame: pl.DataFrame, scores: pl.DataFrame) -> str:
    """kernel.md: the block rows vs SparQ 32, and the RULER pair by length."""
    books = read(RUNS / "pg19.csv")

    def ce(row: str) -> np.ndarray:
        return books.filter(pl.col("row") == row).sort("book")["ce"].to_numpy()

    lines = [
        "| Method | Loss above dense (nats) | Minus SparQ, 32 dims (nats) "
        "| K and V read (share of dense) | K sectors, best layout (share of dense K) |",
        "|---|---|---|---|---|",
    ]
    for r in frame.filter(pl.col("row").is_in(block_rows())).iter_rows(named=True):
        versus = "—"
        if r["row"] != SPARQ and len(ce(r["row"])) == len(ce(SPARQ)):
            versus = "{:+.3f} ± {:.3f}".format(*paired_delta(ce(r["row"]), ce(SPARQ)))
        cells = [
            label(r["row"]),
            f"{r['extra']:+.3f} ± {r['se']:.3f}",
            versus,
            f"{r['reads']:.3f}",
            f"{r['sectors']:.3f}",
        ]
        lines.append(f"| {' | '.join(cells)} |")
    for pair in PAIRS:
        if set(pair) <= set(scores["row"].to_list()):
            lines += ["", *ruler_pair(scores, pair)]
    return "\n".join(lines) + "\n"


def ruler_pair(scores: pl.DataFrame, pair: tuple[str, str]) -> list[str]:
    """Table lines: two rows' RULER scores by length, paired by sample."""
    a, b = (
        scores.filter(pl.col("row") == r).sort("task", "length", "index") for r in pair
    )
    keys = ["task", "length", "index"]
    assert a.select(keys).equals(b.select(keys)), "RULER samples differ"
    x_all, y_all = (100 * f["score"].to_numpy() for f in (a, b))
    lengths = a["length"].to_numpy()
    lines = [
        f"| Context length | {label(pair[0])} (%) | {label(pair[1])} (%) "
        "| Difference (points, paired by sample) |",
        "|---|---|---|---|",
    ]
    for length in [*LENGTHS, None]:
        mask = lengths == length if length else np.ones(len(lengths), bool)
        x, y = x_all[mask], y_all[mask]
        diff, se = paired_delta(x, y)
        cells = [str(length or "all"), f"{x.mean():.1f}", f"{y.mean():.1f}"]
        lines.append(f"| {' | '.join(cells)} | {diff:+.1f} ± {se:.1f} |")
    return lines


def timings(path: Path = TIMING) -> pl.DataFrame:
    """kernel.csv: microseconds per method and length, dense = its best variant."""
    times = read(path).with_columns(
        method=pl.col("method").str.replace("dense-bmm", "dense")
    )
    return times.group_by("method", "n").agg(pl.col("us").min()).sort("n")


def timing(ax: plt.Axes, times: pl.DataFrame, timed: dict[str, str] = TIMED) -> None:
    """Microseconds per layer and decode step against context length."""
    rows = list(timed)
    for method, legend in timed.items():
        part = times.filter(pl.col("method") == method)
        color = FAMILY[method.split("-")[0]]
        kin = [r for r in rows if r.split("-")[0] == method.split("-")[0]]
        style = ("-", "--", ":")[kin.index(method)]
        ax.plot(part["n"], part["us"], "o", ls=style, color=color, label=legend)
    ax.set_yscale("log")
    plain(ax.yaxis)
    context_axis(ax, sorted(times["n"].unique().to_list()))
    ax.set_ylabel("microseconds per layer and decode step")
    ax.set_title("Triton kernels, batch 1, 8 KV heads, RTX 4070 Ti SUPER", fontsize=9)
    below(ax)


def timing_table(times: pl.DataFrame, timed: dict[str, str] = TIMED) -> list[str]:
    """Table lines: microseconds per method, and speedup over dense, by length."""
    wide = times.pivot("method", index="n", values="us").sort("n")
    methods = [m for m in timed if m in wide.columns and m != "dense"]
    head = " | ".join(f"{timed[m]} (µs, speedup)" for m in methods)
    lines = [
        f"| Context length | dense (µs) | {head} |",
        "|---|---|" + "---|" * len(methods),
    ]
    for r in wide.iter_rows(named=True):
        cells = [f"{r[m]:.1f}, {r['dense'] / r[m]:.1f}×" for m in methods]
        lines.append(f"| {r['n']} | {r['dense']:.1f} | {' | '.join(cells)} |")
    return lines


def main() -> None:
    """Write kernel.png and kernel.md."""
    apply()
    frame = summary()
    scores = ruler50().filter(pl.col("row").is_in(RULER_ROWS))
    times = timings() if TIMING.exists() else None
    text = table(frame, scores)
    if times is not None:
        text += "\n" + "\n".join(timing_table(times)) + "\n"
    (RUNS / "kernel.md").write_text(text)
    print(RUNS / "kernel.md")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    blocks(axes[0, 0], frame)
    hybrid(axes[0, 1], frame)
    ruler(axes[1, 0], scores)
    if times is not None:
        timing(axes[1, 1], times)
    save(fig, RUNS / "kernel.png")


if __name__ == "__main__":
    main()
