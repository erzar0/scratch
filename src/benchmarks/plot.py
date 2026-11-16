"""Figures and tables of a suite: every version on one figure per benchmark.

    uv run python -m benchmarks.plot [src/benchmarks/suites/standard.toml]

Each figure is a dot plot: one row per version in registry order, labelled
(left-aligned) by its short name and the implementation of its fast step; thin
lines split the families; docs/methods.md explains every name. One panel per
distribution or task; one dot per context length (or budget), darker = longer
(or larger). Whiskers: latency min to max over repeats; fidelity and tasks ± one
standard error over seeds. Paper-sized, without titles: captions are in
tables.md. Every figure is at the standard budget KEEP, except `budget`:
    latency, fidelity, fidelity-harder (the harder distributions), tasks
    budget: latency and output error per budget at 32K keys, LLM-like
Writes runs/benchmarks/<suite>/<figure>.png and tables.md, from the
suite's newest finished MLflow runs.
"""

import argparse
import tomllib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory

from attention.registry import KEEP, MIN_RATIO, VERSIONS
from benchmarks import synthetic
from benchmarks.tasks import ANSWERS, MODEL
from common import style, tracking

OUT = Path("runs/benchmarks")
REFERENCE = 32768  # the budget figure's context length, where measured
WIDTH, ROW = 5.5, 0.155  # inches: NeurIPS text width, one version row
PANEL_MIN, GAP = 0.8, 0.18  # inches: narrowest panel, gap between panels
SHORT = {
    "iid": "iid",
    "outlier-channels": "Outlier channels",
    "llm-like": "LLM-like",
    "flat": "Flat",
    "sharp": "Sharp",
    "local": "Local",
    "many-needles": "Many needles",
}
DISTRIBUTIONS = {
    "iid": "iid Gaussian q, k, v",
    "outlier-channels": f"{synthetic.OUTLIERS} outlier channels in q and k",
    "llm-like": f"outlier channels, {synthetic.SINKS} sink keys and"
    f" {synthetic.NEEDLES} needle keys per query",
    "flat": "LLM-like with q × 0.5 (softer attention)",
    "sharp": "LLM-like with q × 2 (sharper attention)",
    "local": f"LLM-like with neighbouring keys alike: noise averaged over"
    f" {synthetic.LOCAL} positions, needles as runs of {synthetic.LOCAL} keys",
    "many-needles": f"LLM-like with {synthetic.MANY_NEEDLES} needle keys per query",
}
TASKS = {
    "copy": ("Copy", "the second half repeats the first"),
    "mqar": ("MQAR", f"{synthetic.PAIRS} key-value pairs, {synthetic.QUERIES} asked"),
    "needle": ("Needle", f"{synthetic.NEEDLE_PAIRS} key-value pairs in filler"),
}
FAMILIES = Counter(version.family for version in VERSIONS.values())
NAMES = (
    "docs/methods.md explains each row's version and its settings; n/a: a version"
    " this benchmark does not run."
)


class Panel(NamedTuple):
    """One panel: its rows' `panel` value, title, x label and scale."""

    key: str
    title: str
    xlabel: str
    log: bool


def length(n: int) -> str:
    """8192 -> 8K."""
    return f"{n // 1024}K"


class Dots(NamedTuple):
    """What a figure's dots stand for: the `n` column's legend title and labels."""

    title: str = "Context length (tokens)"
    label: Callable[[int], str] = length


LENGTHS, BUDGETS = Dots(), Dots("Budget (keys kept)", str)


def spread(frame: pl.DataFrame, value: str) -> pl.DataFrame:
    """version, legend, panel, n, mean, low, high: mean ± SE of value over seeds."""
    se = (pl.col(value).std() / pl.len().sqrt()).fill_null(0)  # one seed: none
    out = frame.group_by("version", "legend", "panel", "n").agg(
        mean=pl.col(value).mean(), se=se
    )
    return out.with_columns(
        low=pl.col("mean") - pl.col("se"), high=pl.col("mean") + pl.col("se")
    ).drop("se")


def standard(rows: pl.DataFrame) -> pl.DataFrame:
    """The rows at the standard budget (runs from before budgets were all at KEEP)."""
    if "keep" not in rows.columns:
        return rows
    return rows.filter(pl.col("keep").fill_null(KEEP) == KEEP)


def order(frame: pl.DataFrame) -> list[str]:
    """Versions in registry order, then any no longer registered."""
    present = set(frame["version"])
    return [v for v in VERSIONS if v in present] + sorted(present - VERSIONS.keys())


def legends(frame: pl.DataFrame) -> dict[str, str]:
    """Version id -> legend: the registry's, else the one its run was tagged with."""
    tagged = dict(frame.select("version", "legend").unique().iter_rows())
    return {v: VERSIONS[v].legend if v in VERSIONS else tagged[v] for v in tagged}


def describe(panels: list[Panel], words: dict[str, str]) -> str:
    """'Panels: iid: iid Gaussian q, k, v; ...'."""
    return "Panels: " + "; ".join(f"{p.title}: {words[p.key]}" for p in panels) + "."


def present(panels: list[Panel], frame: pl.DataFrame) -> list[Panel]:
    """The panels with data."""
    found = set(frame["panel"])
    return [p for p in panels if p.key in found]


def dots(
    frame: pl.DataFrame, panels: list[Panel], path: Path, kind: Dots = LENGTHS
) -> None:
    """The figure of frame (see the module docstring); a row per registered version
    on every figure, n/a where it has no data."""
    versions = list(VERSIONS) + [v for v in order(frame) if v not in VERSIONS]
    labels = {v: version.legend for v, version in VERSIONS.items()} | legends(frame)
    count, values = len(versions), sorted(frame["n"].unique())
    colors = dict(zip(values, style.ramp(len(values)), strict=True))
    fig = plt.figure(dpi=300)  # measure text as saved
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]

    def inches(text: str) -> float:
        t = fig.text(0, 0, text)
        w = t.get_window_extent(renderer).width / fig.dpi
        t.remove()
        return float(w)

    pad = max(inches(labels[v]) for v in versions) + 0.08  # names, then a gap
    left = 0.03 + pad
    room = (WIDTH - left - 0.1 - GAP * (len(panels) - 1)) / len(panels)
    panel = max(PANEL_MIN, room)
    span = len(panels) * (panel + GAP) - GAP
    top, bottom = 0.2, 0.42  # panel titles; x ticks, x label and legend
    width, height = left + span + 0.1, top + count * ROW + bottom
    fig.set_size_inches(width, height)
    shared = len({p.xlabel for p in panels}) == 1
    for k, p in enumerate(panels):
        box = ((left + k * (panel + GAP)) / width, bottom / height)
        ax = fig.add_axes((*box, panel / width, count * ROW / height))
        rows_y = blended_transform_factory(ax.transAxes, ax.transData)

        def note(v: str, text: str, ax: Axes = ax, rows_y: Any = rows_y) -> None:
            """Muted italic text in v's row, at the panel's left."""
            ax.text(
                0.03,
                versions.index(v),
                text,
                transform=rows_y,
                va="center",
                color=style.MUTED,
                style="italic",
            )

        ax.set_ylim(count - 0.5, -0.5)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if k == 0:  # names right of the figure edge, left-aligned
            ax.set_yticks(range(count), [labels[v] for v in versions], ha="left")
            ax.tick_params(axis="y", pad=pad * 72)
        else:
            ax.set_yticks([])
        for i in range(1, count, 2):
            ax.axhspan(i - 0.5, i + 0.5, color=style.BAND, lw=0, zorder=0)
        for i in range(1, count):
            families = {versions[i].split("/")[0], versions[i - 1].split("/")[0]}
            if len(families) == 2 and max(FAMILIES[f] for f in families) > 1:
                ax.axhline(i - 0.5, color=style.MUTED, lw=0.4)
        ax.grid(axis="x", color=style.GRID, lw=0.4)
        ax.set_axisbelow(True)
        if p.log:
            ax.set_xscale("log")
        ax.set_title(p.title, loc="left", weight="bold", pad=3)
        found = set(frame.filter(pl.col("panel") == p.key)["version"])
        for v in versions:
            if v not in found:
                note(v, "n/a")
        for j, n in enumerate(values):
            part = frame.filter((pl.col("panel") == p.key) & (pl.col("n") == n))
            if p.log:
                zero = part.filter((pl.col("mean") == 0) & (j == 0))["version"]
                for v in zero:
                    note(v, "0 (reference)")
                part = part.filter(pl.col("mean") > 0)
            mean, low, high = (part[c].to_numpy() for c in ("mean", "low", "high"))
            offset = (j - (len(values) - 1) / 2) * 0.14
            ax.errorbar(
                mean,
                [versions.index(v) + offset for v in part["version"]],
                xerr=np.stack([mean - low, high - mean]),
                fmt="o",
                ms=3,
                mfc=colors[n],
                mec=style.SURFACE,
                mew=0.3,
                ecolor=colors[n],
                elinewidth=0.6,
                capsize=0,
            )
        if not shared:
            ax.set_xlabel(p.xlabel)
        if p.log:
            lo, hi = ax.get_xlim()
            decades = np.log10(hi / lo)
            if decades < 1:  # LogLocator would fall back to crowded linear ticks
                ax.xaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10]))
            else:
                subs = (1.0, 3.0) if decades < 2.2 else (1.0,)
                ax.xaxis.set_major_locator(LogLocator(subs=subs))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
    if shared:
        x, y = (left + span / 2) / width, (bottom - 0.2) / height
        fig.text(x, y, panels[0].xlabel, ha="center", va="top")
    if len(values) > 1:
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                ms=3,
                mfc=colors[n],
                mec=style.SURFACE,
                mew=0.3,
                label=kind.label(n),
            )
            for n in values
        ]
        fig.legend(
            handles=handles,
            title=kind.title,
            alignment="left",
            loc="lower left",
            bbox_to_anchor=(0.03 / width, 0.01),
            ncol=len(values),
            handletextpad=0.1,
            columnspacing=0.8,
            borderpad=0,
            borderaxespad=0,
        )
    style.save(fig, path)


def table(
    frame: pl.DataFrame,
    panels: list[Panel],
    cell: Callable[..., str],
    kind: Dots = LENGTHS,
) -> list[str]:
    """Markdown: per panel, one row per version, one column per length (or budget)."""
    versions, values = order(frame), sorted(frame["n"].unique())
    tagged = legends(frame)
    lines = []
    for p in panels:
        part = frame.filter(pl.col("panel") == p.key)
        cells = {
            (v, n): cell(mean, low, high)
            for v, n, mean, low, high in part.select(
                "version", "n", "mean", "low", "high"
            ).iter_rows()
        }
        lines += [
            f"#### {p.title}",
            "",
            "| Name | Kernel | " + " | ".join(map(kind.label, values)) + " |",
            "|---|---|" + "---:|" * len(values),
        ]
        for v in versions:
            known = VERSIONS.get(v)
            name, kernel = (
                (known.name, known.implementation) if known else (tagged[v], "")
            )
            row = [cells.get((v, n), "not run") for n in values]
            lines.append(f"| {name} | {kernel} | " + " | ".join(row) + " |")
        lines.append("")
    return lines


def microseconds(mean: float, low: float, high: float) -> str:
    """Latency cell."""
    return "out of memory" if np.isnan(mean) else f"{mean:.0f} ({low:.0f}–{high:.0f})"


def plus_minus(mean: float, low: float, high: float) -> str:
    """Mean ± SE cell."""
    return "out of memory" if np.isnan(mean) else f"{mean:.3g} ± {high - mean:.2g}"


def shape(config: dict[str, Any]) -> str:
    """The suite's attention shape, as the benchmarks read it (defaults: standard)."""
    return (
        f"{config.get('kv_heads', 8)} KV heads × {config.get('query_heads', 2)} query"
        f" heads, head dim {config.get('dim', 128)}"
    )


def latency_frame(rows: pl.DataFrame) -> pl.DataFrame:
    """Latency rows as version, legend, panel (distribution), n, mean, low, high."""
    return rows.select(
        "version",
        "legend",
        pl.col("distribution").alias("panel"),
        "n",
        pl.col("median_us").alias("mean"),
        pl.col("min_us").alias("low"),
        pl.col("max_us").alias("high"),
    )


def latency(rows: pl.DataFrame, config: dict[str, Any], out: Path) -> list[str]:
    """latency figure and its tables."""
    frame = latency_frame(standard(rows))
    every = [
        Panel(d, SHORT[d], "µs per decode step", True) for d in config["distributions"]
    ]
    panels = present(every, frame)
    missing = (
        " A missing dot ran out of memory." if frame["mean"].is_nan().any() else ""
    )
    caption = (
        f"Decode-step latency on {rows['gpu'][0]}: median of {config['repeats']}"
        " repeats of CUDA-graph replays; whiskers min to max. Batch 1,"
        f" {shape(config)}, bf16 K and V;"
        f" every sparse version gives exact attention to about {KEEP} keys (exceptions:"
        f" docs/methods.md).{missing}"
        f" {describe(panels, DISTRIBUTIONS)} {NAMES}"
    )
    dots(frame, panels, out / "latency.png")
    return [
        f"### Latency, µs per decode step, budget {KEEP}: median (min–max)",
        "",
        f"latency.png: {caption}",
        "",
        *table(frame, panels, microseconds),
    ]


def fidelity(rows: pl.DataFrame, config: dict[str, Any], out: Path) -> list[str]:
    """fidelity and fidelity-harder figures and their tables."""
    rows = standard(rows).with_columns(panel=pl.col("distribution"))
    frame = spread(rows, "error")
    every = [
        Panel(d, SHORT[d], "error relative to dense", True)
        for d in config["distributions"]
    ]
    every = present(every, frame)
    lines = []
    for name, harder in (("fidelity", False), ("fidelity-harder", True)):
        panels = [p for p in every if (p.key in synthetic.HARDER) == harder]
        if not panels:
            continue
        flat = [p.title for p in panels if p.key in ("iid", "flat")]
        uniform = (
            f" On {' and '.join(flat)} data attention is near uniform, so no {KEEP}"
            " keys come close to dense."
            if flat
            else ""
        )
        caption = (
            "Output error against dense attention, |o − o_dense| / |o_dense| per head:"
            f" mean over {config['queries']} decode queries, ± SE over"
            f" {len(config['seeds'])} seeds. {shape(config)}; every sparse version"
            f" gives exact attention to about {KEEP} keys"
            " and runs as its float32 PyTorch emulation, which selects the same keys"
            " as its kernel (tests/attention/test_registry.py)."
            f"{uniform} {describe(panels, DISTRIBUTIONS)}"
            f" {NAMES}"
        )
        dots(frame, panels, out / f"{name}.png")
        lines += [
            f"### {name}: output error relative to dense, budget {KEEP}, mean ± SE",
            "",
            f"{name}.png: {caption}",
            "",
            *table(frame, panels, plus_minus),
        ]
    extra = {
        "mass": "Dense attention weight on the picked keys",
        "recall": "Recall of the dense top-128 keys",
        "reads": "Share of K and V elements read",
    }
    for column, words in extra.items():
        lines.append(f"### Fidelity: {words}, budget {KEEP}, mean ± SE")
        lines += ["", *table(spread(rows, column), every, plus_minus)]
    return lines


def tasks(rows: pl.DataFrame, config: dict[str, Any], out: Path) -> list[str]:
    """tasks figure and its table."""
    frame = spread(rows.with_columns(panel=pl.col("task")), "accuracy")
    panels = [Panel(t, TASKS[t][0], "answer accuracy", False) for t in config["tasks"]]
    words = {t: about for t, (_, about) in TASKS.items()}
    caption = (
        f"Task accuracy: share of the last {ANSWERS} answers predicted exactly, ± SE"
        f" over {len(config['seeds'])} seeds of {config['sequences']} sequences."
        f" Llama, {MODEL['num_hidden_layers']} layers, width {MODEL['hidden_size']},"
        f" {MODEL['num_attention_heads']} query and {MODEL['num_key_value_heads']}"
        f" KV heads of dim {MODEL['head_dim']}, trained once on all tasks with dense"
        " attention; each version replaces attention in every layer at the answers."
        f" {describe(panels, words)} {NAMES}"
    )
    dots(frame, panels, out / "tasks.png")
    return [
        "### Tasks: answer accuracy, mean ± SE",
        "",
        f"tasks.png: {caption}",
        "",
        *table(frame, panels, plus_minus),
    ]


def budget(frames: dict[str, pl.DataFrame], out: Path) -> list[str]:
    """budget figure: latency and output error per budget, at REFERENCE keys
    (else the longest) and LLM-like data (else the first distribution)."""
    setting: list[str] = []

    def at(rows: pl.DataFrame, what: str) -> pl.DataFrame:
        if "keep" not in rows.columns:
            rows = rows.with_columns(keep=pl.lit(KEEP))
        rows = rows.with_columns(pl.col("keep").fill_null(KEEP))
        many = (
            rows.group_by("n").agg(pl.col("keep").n_unique()).filter(pl.col("keep") > 1)
        )
        lengths = set(many["n"]) or set(rows["n"])
        n = REFERENCE if REFERENCE in lengths else max(lengths)
        found = set(rows["distribution"])
        where = "llm-like" if "llm-like" in found else sorted(found)[0]
        setting.append(f"{what} at {length(n)} keys, {SHORT[where]}")
        rows = rows.filter((pl.col("n") == n) & (pl.col("distribution") == where))
        return rows.with_columns(n=pl.col("keep"))

    parts, panels = [], []
    if "latency" in frames:
        rows = latency_frame(at(frames["latency"], "latency"))
        parts.append(rows.with_columns(panel=pl.lit("latency")))
        panels.append(Panel("latency", "Latency", "µs per decode step", True))
    if "fidelity" in frames:
        rows = at(frames["fidelity"], "output error").with_columns(
            panel=pl.lit("error")
        )
        parts.append(spread(rows, "error"))
        panels.append(Panel("error", "Output error", "error relative to dense", True))
    frame = pl.concat([p.cast({"n": pl.Int64}) for p in parts], how="diagonal")
    dots(frame, panels, out / "budget.png", BUDGETS)
    caption = (
        f"Every version at each budget, the number of keys attended exactly:"
        f" {'; '.join(setting)}. A budget runs where the context is at least"
        f" {MIN_RATIO} times it. Whiskers as in the per-benchmark figures. {NAMES}"
    )
    cells = {"latency": microseconds, "error": plus_minus}
    lines = ["### Budget: latency and output error per budget", ""]
    lines += [f"budget.png: {caption}", ""]
    for p in panels:
        lines += table(frame, [p], cells[p.key], BUDGETS)
    return lines


def main() -> None:
    """Parse the arguments, write the figures and tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path("src/benchmarks/suites/standard.toml")
    parser.add_argument("suite", type=Path, nargs="?", default=default)
    args = parser.parse_args()
    suite = tomllib.loads(args.suite.read_text())
    name = suite.pop("name")
    out = OUT / name
    style.apply()
    writers = {"latency": latency, "fidelity": fidelity, "tasks": tasks}
    frames, lines = {}, [f"# Benchmarks: suite {name}", ""]
    for benchmark, config in suite.items():
        rows = tracking.load(benchmark, name)
        if rows.is_empty():
            continue
        frames[benchmark] = rows
        lines += writers[benchmark](rows, config, out)
    if frames.keys() & {"latency", "fidelity"}:
        lines += budget(frames, out)
    if frames:
        (out / "tables.md").write_text("\n".join(lines))
        print(out / "tables.md")


if __name__ == "__main__":
    main()
