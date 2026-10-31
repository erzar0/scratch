"""Figures of the Qwen runs (runs/qwen); each skips results that do not exist yet.

    training.png  stage 1: each CSA2 layer's error on Qwen's inputs; stage 2: KL to
                  Qwen and cross-entropy on WikiText val, CSA2-Qwen and the control
    pg19.png      PG-19 loss by position, per row
    ruler.png     RULER and BABILong score per task and context length, per row
    short.png     lm-eval-harness scores as % of Qwen's
    bench.png     cost against context length: layer prefill, decode step, cache
    route.png     double indexing: PG-19 loss at 16k-32k against reads saved
    probe.png     one CSA2 layer trained alone at bigger sizes: error on Qwen's
                  inputs (qwen/probe.py)
    tables.md     the summary tables in plain words (main rows; double indexing)

uv run python -m plots.qwen
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from experiments.results import paired_delta, read
from plots.paper import signed, times
from plots.style import REFERENCE, SERIES, apply, context_axis, save

RUNS = Path("runs/qwen")
BUCKETS_LAST = 16384  # start of the last PG-19 position bucket
ROWS = {
    "qwen": ("Qwen (original)", "#444444", "-"),
    "dense": ("Qwen, attention fine-tuned", SERIES[1], "--"),
    "csa2": ("CSA2-Qwen", SERIES[0], "-"),
    "csa2:window": ("CSA2-Qwen, window only", SERIES[2], ":"),
    "csa2:random": ("CSA2-Qwen, random picks", SERIES[3], ":"),
    "csa2:oracle": ("CSA2-Qwen, oracle picks", SERIES[4], "-."),
}
# double indexing rows (tier D), drawn only in route.png
ROUTES = {
    "csa2": "pure CSA2",
    "csa2:mean": "mean summary",
    "csa2:bound": "bound summary",
    "csa2:learned": "learned router",
    "csa2:mixed": "mixed: learned + bound",
}


def exists(name: str) -> bool:
    """Whether runs/qwen/<name> exists and is not empty."""
    path = RUNS / name
    return path.exists() and path.stat().st_size > 0


def fig_training() -> None:
    """Stage-1 layer errors and stage-2 validation losses against tokens."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 3.2))
    if exists("csa2_transfer.csv"):
        frame = read(RUNS / "csa2_transfer.csv").drop_nulls("val_error_0")
        errors = frame.select(pl.selectors.starts_with("val_error_")).to_numpy()
        tokens = frame["tokens"].to_numpy() / 1e6
        left.plot(tokens, errors, color=SERIES[0], alpha=0.15, linewidth=0.6)
        left.plot(
            tokens, errors.mean(1), "o-", color=SERIES[0], label="mean over layers"
        )
        left.set(
            xlabel="training tokens (M)",
            ylabel="squared error / output mean square",
            title="Stage 1: layers imitate Qwen's attention",
            ylim=(0, None),
        )
        left.legend()
    for name, row in (("csa2_e2e.csv", "csa2"), ("dense_e2e.csv", "dense")):
        if not exists(name):
            continue
        frame = read(RUNS / name).drop_nulls("val_kl")
        label, color, _ = ROWS[row]
        seen = frame["tokens"] / 1e6
        right.plot(seen, frame["val_kl"], "-", color=color, label=f"{label}: KL")
        right.plot(seen, frame["val_ce"], "--", color=color, label=f"{label}: loss")
    right.set(
        xlabel="training tokens (M)",
        ylabel="nats per token",
        title="Stage 2: WikiText validation",
    )
    right.legend(fontsize=6)
    save(fig, RUNS / "training.png")


def fig_pg19() -> None:
    """Mean PG-19 loss in each position bucket, per row."""
    if not exists("pg19.csv"):
        return
    frame = read(RUNS / "pg19.csv")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for row, (label, color, style) in ROWS.items():
        part = frame.filter(pl.col("row") == row)
        if part.is_empty():
            continue
        means = part.group_by("start", "end").agg(pl.col("ce").mean()).sort("start")
        ends = means["end"].to_list()
        ax.plot(ends, means["ce"], style, color=color, marker="o", label=label)
    context_axis(ax, ends)
    ax.set(
        xlabel="position in the book, end of bucket (tokens)",
        ylabel="loss (nats per token)",
        title="PG-19 test books",
    )
    ax.legend()
    save(fig, RUNS / "pg19.png")


def fig_ruler() -> None:
    """Heatmap of the mean score per task and length, one panel per row."""
    if not exists("longctx.csv"):
        return
    frame = read(RUNS / "longctx.csv")
    rows = [r for r in ROWS if r in set(frame["row"])]
    task_names = list(dict.fromkeys(frame["task"]))
    lengths = sorted(set(frame["length"]))
    fig, axes = plt.subplots(
        1, len(rows), figsize=(2.2 * len(rows) + 1.5, 4.2), squeeze=False
    )
    for ax, row in zip(axes[0], rows, strict=True):
        means = (
            frame.filter(pl.col("row") == row)
            .group_by("task", "length")
            .agg(pl.col("score").mean())
        )
        grid = np.full((len(task_names), len(lengths)), np.nan)
        for task, length, score in means.iter_rows():
            grid[task_names.index(task), lengths.index(length)] = score
        ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for (i, j), value in np.ndenumerate(grid):
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5)
        ax.set_xticks(range(len(lengths)), [f"{t // 1024}k" for t in lengths])
        ax.set_yticks(range(len(task_names)), task_names if ax is axes[0][0] else [])
        ax.set(title=ROWS[row][0], xlabel="context (tokens)")
    save(fig, RUNS / "ruler.png")


def fig_short() -> None:
    """Each lm-eval task's main score as % of Qwen's, per row."""
    if not exists("short.csv"):
        return
    frame = read(RUNS / "short.csv").filter(
        pl.col("metric").is_in(["acc", "acc_norm", "contains", "exact_match"])
    )
    # one metric per task: normalized accuracy where lm-eval reports it
    preferred = frame.sort("metric", descending=True).unique(
        ["row", "task"], keep="first"
    )
    qwen = preferred.filter(pl.col("row") == "qwen").select(
        "task", pl.col("value").alias("qwen")
    )
    table = preferred.join(qwen, on="task").with_columns(
        (100 * pl.col("value") / pl.col("qwen")).alias("percent")
    )
    task_names = sorted(set(table["task"]))
    rows = [r for r in ROWS if r != "qwen" and r in set(table["row"])]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(task_names) + 1.2))
    height = 0.8 / max(len(rows), 1)
    for k, row in enumerate(rows):
        part = dict(table.filter(pl.col("row") == row).select("task", "percent").rows())
        y = np.arange(len(task_names)) + k * height
        x = [part.get(t, np.nan) for t in task_names]
        ax.barh(y, x, height, color=ROWS[row][1], label=ROWS[row][0])
    ax.axvline(100, color="#444444", linewidth=0.8)
    ax.set_yticks(np.arange(len(task_names)) + 0.4 - height / 2, task_names)
    ax.set(xlabel="% of Qwen's score", title="Short-context tasks (lm-eval-harness)")
    ax.legend(fontsize=6)
    save(fig, RUNS / "short.png")


def fig_bench() -> None:
    """Layer prefill time, decode-step time and cache size against context length."""
    if not exists("bench_layer.csv"):
        return
    frame = read(RUNS / "bench_layer.csv")
    panels = (
        ("prefill_ms", "prefill (ms)"),
        ("decode_us_batch1", "decode step, batch 1 (µs)"),
        ("cache_mib", "cache per sequence (MiB)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
    names = {"csa2": "CSA2 Full layer", "exact (flash)": "Qwen-sized exact attention"}
    for ax, (column, title) in zip(axes, panels, strict=True):
        for color, (layer, label) in zip(SERIES, names.items(), strict=False):
            part = frame.filter(pl.col("layer") == layer).sort("T")
            ax.plot(part["T"], part[column], marker="o", color=color, label=label)
        ax.set_yscale("log")
        context_axis(ax, sorted(set(frame["T"])))
        ax.set(title=title, xlabel="context (tokens)")
    axes[0].legend(fontsize=6)
    save(fig, RUNS / "bench.png")


def fig_route() -> None:
    """Double indexing: PG-19 loss at 16k-32k tokens against reads saved at 32k."""
    if not (exists("lm.csv") and exists("pg19.csv")):
        return
    reads = read(RUNS / "lm.csv").group_by("row").agg(pl.col("reads_32k").first())
    loss = (
        read(RUNS / "pg19.csv")
        .filter(pl.col("start") == BUCKETS_LAST)
        .group_by("row")
        .agg(pl.col("ce").mean())
    )
    table = reads.join(loss, on="row").filter(pl.col("row").is_in(list(ROUTES)))
    if "csa2" not in set(table["row"]):
        return
    flat = table.filter(pl.col("row") == "csa2")["reads_32k"].item()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for row, label in ROUTES.items():
        part = table.filter(pl.col("row") == row)
        if part.is_empty():
            continue
        x = flat / part["reads_32k"].item()
        ax.plot(x, part["ce"].item(), "o", color=SERIES[0])
        ax.annotate(
            label,
            (x, part["ce"].item()),
            fontsize=6,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set(
        xlabel="times fewer indexer reads at 32k tokens",
        ylabel="loss at tokens 16k-32k (nats)",
        title="Double indexing on CSA2-Qwen, PG-19",
    )
    save(fig, RUNS / "route.png")


PROBE_MASKS = {
    "qwen": "all earlier tokens (Qwen)",
    "window": "last 128 tokens",
    "window+sink": "last 128 + first 4 tokens",
    "window+sink+oracle": "last 128 + first 4 + best 64 blocks of 4",
    "window+sink weight": "last 128; first 4 take weight but give zeros",
    "window+sink weight+oracle": "the same + best 64 blocks of 4",
}
# variant -> (label, color, line style); dashed = queries also read the first tokens
PROBE_VARIANTS = {
    "base": ("current sizes", SERIES[0], "-"),
    "wide": ("key/value 512 wide", SERIES[1], "-"),
    "full-rank": ("full-rank query and output", SERIES[2], "-"),
    "rope": ("every channel rotated", SERIES[3], "-"),
    "no-pool": ("no pooling (blocks of 1)", SERIES[4], "-"),
    "all": ("all four", REFERENCE, "-"),
    "sinks": ("current sizes + first 4 tokens", SERIES[0], "--"),
    "sinks+all": ("all four + first 4 tokens", REFERENCE, "--"),
}


def probe_layers() -> pl.DataFrame:
    """Every probe_layer*.csv (default layers, then other layer sets) as one frame."""
    paths = sorted(RUNS.glob("probe_layer*.csv"))
    return pl.concat([read(path) for path in paths], how="diagonal")


def fig_probe() -> None:
    """Held-out error of each probe variant against training step, per layer."""
    if not any(RUNS.glob("probe_layer*.csv")):
        return
    frame = probe_layers()
    layers = sorted(set(frame["layer"]))
    fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 3.2))
    for ax, layer in zip(np.atleast_1d(axes), layers, strict=True):
        for variant, (label, color, style) in PROBE_VARIANTS.items():
            part = frame.filter(
                (pl.col("layer") == layer) & (pl.col("variant") == variant)
            ).sort("step")
            if not part.is_empty():
                ax.plot(part["step"], part["val_mix"], style, color=color, label=label)
        ax.set(
            xlabel="training step (8192 tokens each)",
            ylabel="squared error / output mean square",
            title=f"Layer {layer}, held-out web + books",
            ylim=(0, None),
        )
    np.atleast_1d(axes)[0].legend(fontsize=6)
    save(fig, RUNS / "probe.png")


def probe_tables() -> str:
    """Markdown: Qwen under each attention mask; final probe errors per variant."""
    text = ""
    if exists("probe_mask.csv"):
        loss = read(RUNS / "probe_mask.csv").group_by("row").agg(pl.col("ce").mean())
        qwen = loss.filter(pl.col("row") == "qwen")["ce"].item()
        rows = [
            {
                "tokens each query may read": label,
                "WikiText loss (nats)": f"{ce:.2f}",
                "extra loss vs all tokens": signed(ce - qwen),
            }
            for row, label in PROBE_MASKS.items()
            for ce in loss.filter(pl.col("row") == row)["ce"]
        ]
        text += "\n\n## Qwen's own attention, masked like CSA2\n\n" + markdown(rows)
    if any(RUNS.glob("probe_layer*.csv")):
        frame = probe_layers()
        last = frame.filter(pl.col("step") == frame["step"].max())
        rows = []
        for variant, (label, _, _) in PROBE_VARIANTS.items():
            entry = {"sizes": label}
            for part in (
                last.filter(pl.col("variant") == variant)
                .sort("layer")
                .iter_rows(named=True)
            ):
                entry["parameters (M)"] = f"{part['params'] / 1e6:.1f}"
                for name, where in (
                    ("val_mix", "web + books"),
                    ("val_wikitext", "WikiText"),
                ):
                    entry[f"layer {part['layer']} error, {where}"] = f"{part[name]:.2f}"
            if len(entry) > 1:
                rows.append(entry)
        text += "\n\n## One CSA2 layer trained alone (error on Qwen's inputs)\n\n"
        text += markdown(rows)
    return text


FAMILIES = ("RULER", "BABILong", "many-shot ICL")


def task_family(task: str) -> str:
    """RULER, BABILong or many-shot ICL."""
    if task.startswith("babilong"):
        return "BABILong"
    return "many-shot ICL" if task.startswith("icl_") else "RULER"


def summary(names: dict[str, str]) -> list[dict[str, str]]:
    """Per row (row -> name): WikiText loss and extra loss vs Qwen (paired over
    windows), KL to Qwen, PG-19 loss at 16k-32k, long-context scores at 4k and 32k,
    short-task retention."""
    lm = read(RUNS / "lm.csv")
    base = lm.filter(pl.col("row") == "qwen").select("window", "ce")
    pg19 = read(RUNS / "pg19.csv") if exists("pg19.csv") else None
    longctx = read(RUNS / "longctx.csv") if exists("longctx.csv") else None
    retained = short_retention()
    table = []
    for row, name in names.items():
        part = lm.filter(pl.col("row") == row)
        if part.is_empty():
            continue
        paired = part.join(base, on="window", suffix="_qwen")
        mean, se = paired_delta(paired["ce"].to_numpy(), paired["ce_qwen"].to_numpy())
        entry = {
            "model": name,
            "WikiText loss (nats)": f"{part['ce'].to_numpy().mean():.3f}",
            "extra loss vs Qwen (nats)": "—"
            if row == "qwen"
            else f"{signed(mean)} ± {se:.3f}",
            "KL to Qwen (nats)": "—"
            if row == "qwen"
            else f"{part['kl'].to_numpy().mean():.3f}",
        }
        if pg19 is not None:
            last = pg19.filter(
                (pl.col("row") == row) & (pl.col("start") == BUCKETS_LAST)
            )
            if not last.is_empty():
                entry["PG-19 loss, tokens 16k-32k"] = (
                    f"{last['ce'].to_numpy().mean():.3f}"
                )
        if longctx is not None:
            scores = (
                longctx.filter(pl.col("row") == row)
                .with_columns(pl.col("task").map_elements(task_family, pl.String))
                .group_by("task", "length")
                .agg(pl.col("score").mean())
            )
            found = {(family, length): v for family, length, v in scores.rows()}
            for family in FAMILIES:
                for length in (4096, 32768):
                    if (family, length) in found:
                        score = 100 * found[family, length]
                        entry[f"{family}, {length // 1024}k (%)"] = f"{score:.1f}"
        if row in retained:
            entry["short tasks, % of Qwen"] = f"{retained[row]:.1f}"
        table.append(entry)
    return table


def short_retention() -> dict[str, float]:
    """Row -> mean over lm-eval tasks of its score as % of Qwen's."""
    if not exists("short.csv"):
        return {}
    frame = read(RUNS / "short.csv").filter(
        pl.col("metric").is_in(["acc", "acc_norm", "contains", "exact_match"])
    )
    preferred = frame.sort("metric", descending=True).unique(
        ["row", "task"], keep="first"
    )
    qwen = preferred.filter(pl.col("row") == "qwen").select(
        "task", pl.col("value").alias("qwen")
    )
    percent = (
        preferred.join(qwen, on="task")
        .group_by("row")
        .agg((100 * pl.col("value") / pl.col("qwen")).mean())
    )
    return dict(percent.rows())


def route_table() -> list[dict[str, str]]:
    """Double indexing rows: summary columns and indexer reads saved vs pure CSA2."""
    lm = read(RUNS / "lm.csv")
    flat = lm.filter(pl.col("row") == "csa2")
    if flat.is_empty():
        return []
    table = summary(ROUTES)
    for entry, row in zip(
        table, [r for r in ROUTES if r in set(lm["row"])], strict=True
    ):
        part = lm.filter(pl.col("row") == row)
        for length in ("4k", "32k"):
            ratio = flat[f"reads_{length}"][0] / part[f"reads_{length}"][0]
            entry[f"times fewer indexer reads, {length}"] = times(ratio)
    return table


def markdown(rows: list[dict[str, str]]) -> str:
    """A Markdown table; cells a row lacks are blank."""
    columns = list(dict.fromkeys(k for row in rows for k in row))
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(r.get(c, "") for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def tables() -> None:
    """runs/qwen/tables.md: the main rows, then double indexing."""
    if not exists("lm.csv"):
        return
    main_rows = summary({row: label for row, (label, _, _) in ROWS.items()})
    text = "## Qwen, CSA2-Qwen and controls\n\n" + markdown(main_rows)
    text += "\n\n## Double indexing on CSA2-Qwen\n\n" + markdown(route_table())
    text += probe_tables()
    (RUNS / "tables.md").write_text(text + "\n")


def main() -> None:
    """Every figure whose results exist."""
    apply()
    fig_training()
    fig_pg19()
    fig_ruler()
    fig_short()
    fig_bench()
    fig_route()
    fig_probe()
    tables()


if __name__ == "__main__":
    main()
