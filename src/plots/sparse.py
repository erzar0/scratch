"""Figure and table of the sparse decode attentions (experiments/sparse_qwen.py).

    sparse.png  top: PG-19 loss above dense per tournament design and loser rule
                (stage A); that loss against the share of K and V read, and against
                K sectors in each row's best layout, each method through its
                budgets (stage B). Bottom: each layer's output error; RULER score
                by context length, and over every task and length (when
                qwen.evaluate long has qwen:<row> rows)
    table.md    one line per row, plain words

uv run python -m plots.sparse
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib import ticker
from matplotlib.axis import Axis
from matplotlib.patches import Patch

from experiments.results import paired_delta, read
from plots.style import REFERENCE, SERIES, apply, context_axis, save

RUNS = Path(os.environ.get("SPARSE_OUT", "runs/sparse"))
LONGCTX = Path("runs/qwen/longctx.csv")
DESIGNS = {  # stage A: design -> its row, loser rule left open
    "random order, bracket": "tournament-random-bracket-{}-128",
    "random order, halving": "tournament-random-halving-{}-128",
    "Hadamard order, halving": "tournament-hadamard-halving-{}-128",
    "PCA order, halving": "tournament-pca-halving-{}-128",
    "|q| order, bracket": "tournament-query-bracket-{}-128",
    "|q| order, halving": "tournament-query-halving-{}-128",
    "|q| order, halving, last 16 kept": "tournament-query-halving-{}-128-8-16",
}
LOSERS = {"drop": "#2a78d6", "mean": "#8fb8ea", "approx": "#d4e3f6"}
ORIGINAL = "tournament-random-bracket-approx-128"
FAMILY = {  # row prefix -> color
    "tournament": SERIES[0],
    "sparq": SERIES[1],
    "loki": SERIES[2],
    "hybrid": SERIES[3],
    "blocks": SERIES[4],
    "first": SERIES[2],  # never beside Loki in a figure
    "global": SERIES[0],  # the global halving kernel
    "sampled": SERIES[4],  # sampled-median cutoffs; never beside blocks
    "groups": SERIES[4],  # fixed groups of 8 dims; sampled rows are timing only
    "shuffled": SERIES[2],  # keys in random slots; first rows are not beside them
    "parallel": SERIES[2],  # the shuffled blocks' kernel
    "bandit": SERIES[3],  # never beside hybrid rows
    "exact": REFERENCE,
    "dense": "#000000",
}
NAMES = {
    "tournament": "tournament",
    "sparq": "SparQ",
    "loki": "Loki",
    "exact": "exact top-k",
    "dense": "dense",
}
WORDS = {
    "random": "random order",
    "hadamard": "Hadamard order",
    "query": "|q| order",
    "pca": "PCA order",
    "bracket": "bracket",
    "halving": "halving",
    "approx": "losers keep approximate scores",
    "drop": "losers dropped",
    "mean": "losers to mean value",
    "meanv": "losers' weight on their mean value",
    "pairs": "rotary pairs",
    "contiguous": "as stored",
}
SINGLES = {  # stage B rows off the budget lines -> short tag
    "sparq-32-64": "32 dims, top-64",
    "sparq-16-256": "16 dims, top-256",
    "tournament-query-halving-drop-128-16": "16 dims per round",
    "tournament-query-halving-drop-128-8-16": "last 16 kept",
    "tournament-query-halving-approx-128": "losers keep approximate scores",
    "tournament-pca-halving-drop-128": "PCA order",
}
SHOWN = (  # layer error and RULER panels
    ORIGINAL,
    "tournament-query-halving-drop-128",
    "tournament-query-halving-drop-128-16",
)
RULER_TASKS = ["niah_single_2", "niah_multikey_2", "niah_multivalue", "ruler_vt"]
BY_LENGTH = (  # RULER by length
    "exact-128",
    "sparq-32-128",
    "sparq-16-128",
    "tournament-query-halving-drop-256",
    "tournament-query-halving-drop-128",
)
SECTORS = ("sectors_rows", "sectors_chunks", "sectors_transpose", "sectors_copies")
REFERENCES = ("exact-128", "sparq-32-128", "loki-post-32-128")
BUDGET = (  # stage B: legend, family, (row, tag) by budget
    ("exact top-k (numbers: k)", "exact", "exact-{}", (64, 128, 256)),
    ("SparQ, top-128 (numbers: dims)", "sparq", "sparq-{}-128", (16, 32, 64)),
    (
        "Loki post-rotary, top-128 (numbers: axes)",
        "loki",
        "loki-post-{}-128",
        (16, 32, 64),
    ),
    (
        "tournament, |q| order, halving (numbers: keep)",
        "tournament",
        "tournament-query-halving-drop-{}",
        (64, 128, 256, 512),
    ),
)


def label(row: str) -> str:
    """A row name in words."""
    name, *settings = row.split("-")
    if name == "dense":
        return "dense attention"
    if name == "exact":
        return f"exact top-{settings[0]}"
    if name == "sparq":
        return f"SparQ, {settings[0]} dims, top-{settings[1]}"
    if name == "loki":
        return f"Loki {settings[0]}-rotary, {settings[1]} axes, top-{settings[2]}"
    if name == "shuffled":
        inner = label("-".join(settings)).removeprefix("tournament: ")
        return f"tournament, keys in random slots: {inner}"
    if name == "blocks":
        block, keep = settings
        return (
            f"tournament: |q| order, halving in blocks of {block}, "
            f"losers dropped, keep {keep}"
        )
    if name == "bandit":
        bound, width, keep = settings
        sd = "standard deviation" + ("" if width == "1" else "s")
        radius = (
            f"exact bound, radius × {width}"
            if bound == "exact"
            else f"Gaussian bound, {width} {sd}"
        )
        return f"bandit: |q| order, {radius}, keep {keep}"
    if name == "hybrid":
        block, rounds, keep = settings
        first = "first round" if rounds == "1" else f"first {rounds} rounds"
        return (
            f"tournament: |q| order, {first} in blocks of {block}, "
            f"then global, losers dropped, keep {keep}"
        )
    if name in ("first", "sampled"):
        width, losers, keep = settings
        cut = "halving" if name == "first" else "halving with sampled-median cutoff"
        return (
            f"tournament: |q| order, first round on {width} dims, {cut}, "
            f"{WORDS[losers]}, keep {keep}"
        )
    if name == "groups":
        grouping, width, losers, keep = settings
        return (
            f"tournament: fixed groups of 8 dims ({WORDS[grouping]}) by |q|, first "
            f"round on {width} dims, halving, {WORDS[losers]}, keep {keep}"
        )
    order, rounds, losers, keep, *more = settings
    dims, local = [*more, *("8", "0")[len(more) :]]
    words = [WORDS[order], WORDS[rounds], WORDS[losers], f"keep {keep}"]
    words += [f"{dims} dims per round"] if dims != "8" else []
    words += [f"last {local} kept"] if local != "0" else []
    return f"tournament: {', '.join(words)}"


def summary() -> pl.DataFrame:
    """Per row: extra loss vs dense and its paired SE, and the mean layer costs."""
    books = read(RUNS / "pg19.csv")
    dense = books.filter(pl.col("row") == "dense").sort("book")["ce"].to_numpy()
    layers = read(RUNS / "layers.csv").group_by("row").mean().drop("layer")
    rows = []
    for row in books["row"].unique(maintain_order=True):
        ce = books.filter(pl.col("row") == row).sort("book")["ce"].to_numpy()
        if len(ce) != len(dense):
            continue
        extra, se = paired_delta(ce, dense) if row != "dense" else (0.0, 0.0)
        rows.append({"row": row, "extra": extra, "se": se})
    frame = pl.DataFrame(rows).join(layers, on="row", how="left")
    layouts = [c for c in SECTORS if c in frame.columns]
    return frame.with_columns(sectors=pl.min_horizontal(layouts) if layouts else None)


def plain(axis: Axis) -> None:
    """Log axis labelled 0.1, 0.2, 0.5, 1, ... instead of powers of ten."""
    axis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    axis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    axis.set_minor_formatter(ticker.NullFormatter())


def designs(ax: plt.Axes, frame: pl.DataFrame) -> None:
    """Stage A: extra loss per design and loser rule, references as lines."""
    extra = dict(frame.select("row", "extra").iter_rows())
    se = dict(frame.select("row", "se").iter_rows())
    for y, row in enumerate(DESIGNS.values()):
        for offset, (losers, color) in zip(
            (-0.27, 0, 0.27), LOSERS.items(), strict=True
        ):
            name = row.format(losers)
            if name in extra:
                ax.barh(y + offset, extra[name], 0.27, xerr=se[name], color=color)
                if name == ORIGINAL:
                    ax.annotate(
                        "original design",
                        (extra[name], y + offset),
                        xytext=(3, -3),
                        textcoords="offset points",
                        fontsize=7,
                    )
    for row, style in zip(REFERENCES, ("-", "--", ":"), strict=True):
        if row in extra:
            ax.axvline(extra[row], color=REFERENCE, ls=style, lw=1, label=label(row))
    ax.set_yticks(range(len(DESIGNS)), list(DESIGNS))
    ax.invert_yaxis()
    ax.set_xscale("log")
    plain(ax.xaxis)
    ax.set_xlabel("PG-19 loss above dense (nats)")
    ax.set_title("Tournament designs, keep 128, 8 dimensions per round", fontsize=9)
    lines, _ = ax.get_legend_handles_labels()
    swatches = [Patch(color=c, label=WORDS[w]) for w, c in LOSERS.items()]
    ax.legend(
        handles=[*lines, *swatches],
        fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )


def budget(ax: plt.Axes, frame: pl.DataFrame, cost: str, xlabel: str) -> None:
    """Stage B: extra loss against a cost column, each family through its budgets."""
    frame = frame.filter(pl.col(cost).is_not_null())
    points = {
        r: (x, y, e)
        for r, x, y, e in frame.select("row", cost, "extra", "se").iter_rows()
    }
    for legend, family, row, sizes in BUDGET:
        shown = [(points[row.format(n)], n) for n in sizes if row.format(n) in points]
        if not shown:
            continue
        xs, ys, es = zip(*(p for p, _ in shown), strict=True)
        ax.errorbar(xs, ys, es, fmt="o-", color=FAMILY[family], label=legend)
        for (x, y, _), n in shown:
            ax.annotate(
                str(n), (x, y), xytext=(4, 2), textcoords="offset points", fontsize=7
            )
    others = frame.filter(pl.col("row").is_in(list(SINGLES)))
    for row, x, y, e in others.select("row", cost, "extra", "se").iter_rows():
        ax.errorbar(x, y, e, fmt="s", color=FAMILY[row.split("-")[0]], mfc="none")
        ax.annotate(
            SINGLES[row], (x, y), xytext=(4, -8), textcoords="offset points", fontsize=6
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain(ax.xaxis)
    plain(ax.yaxis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PG-19 loss above dense (nats)")
    ax.set_title("Budgets (open squares: single settings)", fontsize=9)
    below(ax)


def look(row: str, rows: list[str]) -> tuple[str, str]:
    """The family's color; within a family solid, dashed, then dotted."""
    family = row.split("-")[0]
    kin = [r for r in rows if r.split("-")[0] == family]
    style = ("-", "--", ":")[kin.index(row)] if row in kin else "-"
    return FAMILY[family], style


def below(ax: plt.Axes) -> None:
    """Legend under the axes."""
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16))


def layer_errors(ax: plt.Axes, rows: list[str]) -> None:
    """Output error relative to exact attention, per layer."""
    layers = read(RUNS / "layers.csv")
    for row in rows:
        part = layers.filter(pl.col("row") == row).sort("layer")
        if part.height:
            color, style = look(row, rows)
            ax.plot(
                part["layer"], part["error"], color=color, ls=style, label=label(row)
            )
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("output error, relative to exact")
    below(ax)


def ruler_scores(rows: pl.Series) -> pl.DataFrame:
    """RULER samples of rows (and dense, the qwen row): 4 tasks, first 25 each."""
    scores = read(LONGCTX).filter(
        pl.col("task").is_in(RULER_TASKS)
        & (pl.col("index") < 25)
        & pl.col("row").str.contains(r"^qwen(:|$)")
    )
    named = scores.with_columns(
        row=pl.when(pl.col("row") == "qwen")
        .then(pl.lit("dense"))
        .otherwise(pl.col("row").str.strip_prefix("qwen:"))
    )
    return named.filter(pl.col("row").is_in(rows.to_list()))


def ruler(ax: plt.Axes, scores: pl.DataFrame) -> None:
    """Mean RULER score by context length, dense and the BY_LENGTH rows."""
    for row in ["dense", *BY_LENGTH]:
        part = (
            scores.filter(pl.col("row") == row)
            .group_by("length")
            .agg(pl.col("score").mean())
            .sort("length")
        )
        if part.height:
            color, style = look(row, list(BY_LENGTH))
            ax.plot(
                part["length"],
                100 * part["score"],
                "o",
                color=color,
                ls=style,
                label=label(row),
            )
    context_axis(ax, [4096, 8192, 16384, 32768])
    ax.set_ylabel("RULER score (%), 4 tasks")
    below(ax)


def ruler_overall(ax: plt.Axes, scores: pl.DataFrame) -> None:
    """Mean RULER score over every task and length per row, with its SE."""
    means = (
        scores.group_by("row")
        .agg(
            mean=100 * pl.col("score").mean(),
            se=100 * pl.col("score").std() / pl.len().sqrt(),
        )
        .sort("mean")
    )
    for y, (row, mean, se) in enumerate(means.iter_rows()):
        ax.barh(y, mean, xerr=se, color=FAMILY[row.split("-")[0]])
    ax.set_yticks(range(means.height), [label(r) for r in means["row"]], fontsize=6)
    ax.set_xlabel("RULER score (%), 4 tasks, 4096 to 32768 tokens")


def table(frame: pl.DataFrame) -> str:
    """Markdown table, one line per row."""
    lines = [
        "| Method | Loss above dense (nats) | K and V read (share of dense) "
        "| K sectors, rows / 8-dim chunks / transpose / two copies "
        "| Exact top-128 kept "
        "| Exact weight on kept keys | Output error |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in frame.sort("extra").iter_rows(named=True):
        sectors = "—"
        if r.get("sectors_rows") is not None:
            sectors = " / ".join(
                "—" if r.get(k) is None else f"{r[k]:.3f}" for k in SECTORS
            )
        costs = "—" if r["reads"] is None else f"{r['reads']:.3f}"
        quality = (
            ["—"] * 3
            if r["recall"] is None
            else [f"{r['recall']:.2f}", f"{r['mass']:.2f}", f"{r['error']:.3f}"]
        )
        loss = f"{r['extra']:+.3f} ± {r['se']:.3f}"
        cells = [label(r["row"]), loss, costs, sectors, *quality]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write sparse.png and table.md."""
    apply()
    frame = summary()
    (RUNS / "table.md").write_text(table(frame))
    print(RUNS / "table.md")
    shown = [*SHOWN, *REFERENCES]
    scores = ruler_scores(frame["row"]) if LONGCTX.exists() else None
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    designs(axes[0, 0], frame)
    budget(axes[0, 1], frame, "reads", "K and V read, share of dense attention")
    budget(
        axes[0, 2],
        frame,
        "sectors",
        "32-byte sectors of K read, best layout, share of dense K",
    )
    layer_errors(axes[1, 0], shown)
    if scores is not None and scores.height:
        ruler(axes[1, 1], scores)
        ruler_overall(axes[1, 2], scores)
    save(fig, RUNS / "sparse.png")


if __name__ == "__main__":
    main()
