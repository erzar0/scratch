"""Paper figures (report/figs/*.svg) and the numbers its tables show
(report/numbers.json, formatted here so no table number is typed by hand); "pdf"
also compiles report/paper.typ (NeurIPS template, matplotlib's STIX fonts as Times).

    uv run --with typst python -m plots.paper [pdf]
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from experiments.eval_csa import OUT as CSA_LOSS
from experiments.eval_csa import TOP_K, WINDOW, flat_reference
from experiments.results import read
from model.train import TrainConfig
from plots import layer, methods, route, training
from plots.style import GRID as GRID_LINE
from plots.style import REFERENCE, SERIES, TARGET, apply

FIGS = Path("report/figs")
NUMBERS = Path("report/numbers.json")
PAPER = Path("report/paper.typ")
PAIRED = Path("runs/paper/paired_se.csv")
PAIRED_SEED1 = Path("runs/paper/paired_se_s1.csv")
COPY = Path("runs/paper/copy_test.csv")
PICKS = Path("runs/paper/pick_recall.csv")
WIDTH = 5.5  # inches: NeurIPS text width
LENGTHS = (2048, 8192)
BIG_T = 131072
# method names in the CSVs -> names in the paper
DISPLAY = {"pure CSA2": "pure CSA2", "random": "random entries", "oracle": "oracle"}
DISPLAY |= {
    f"double indexing {m}": f"double indexing: {m}"
    for m in ("mean", "bound", "learned", "mixed")
}
# indexer bench rows -> legend names
BENCH_NAMES = {
    "flat": "pure CSA2 (scores every entry)",
    "learned S=16 s=4": "double indexing: learned, superblocks of 16, keep 4",
    "bound S=16 s=32": "double indexing: bound, superblocks of 16, keep 32",
    "learned S=64 s=4": "double indexing: learned, superblocks of 64, keep 4",
}
LAYER_NAMES = {"csa2": "CSA2 layer", "exact (flash)": "full attention (FlashAttention)"}
WORK = "selection work (index keys read)"
EXTRA = "extra loss vs pure CSA2 (nats)"
# route.quality legend -> paper names
ROUTE_NAMES = {"flat": "pure CSA2", "learned + bound": "mixed: learned + bound"}
# a selecting layer of model 1 at T=2048 (Full or Reindex: 0, 2, 4, 6); 4 is a needle
STRIP_LAYER = 4  # layer in every eval_route calibration (both T, both models)
ROUTE_LABELS = {"mean": "mean", "bound": "bound", "learned": "learned_r1"}
ROUTE_LABELS |= {"mixed": "mixed_r1"}
TRAINING = {  # log name -> (label, T)
    "csa2_seq2048": ("model 1", 2048),
    "csa2_seq2048_s1": ("model 2 (another seed)", 2048),
    "csa2_seq8192": ("8192-token model", 8192),
}


def style() -> None:
    """Shared style at paper size: STIX (Times, as the paper body), 8 pt."""
    apply()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.5,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.5,
        }
    )


def save_svg(fig: Figure, name: str, bottom: float = 0.0) -> None:
    """Write report/figs/{name}.svg; bottom: figure fraction kept for a legend."""
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(FIGS / f"{name}.svg")
    plt.close(fig)
    print(FIGS / f"{name}.svg")


def fig_route() -> None:
    """Double indexing per coarse mode at both T."""
    frame = read(route.OUT / "route_loss.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    keys = mticker.FuncFormatter(
        lambda x, _: f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k"
    )
    for ax, seq_len in zip(axes, LENGTHS, strict=True):
        route.quality(ax, frame, seq_len)
        ax.set(xlabel=WORK, ylabel=EXTRA, ylim=(-0.003, None))
        flat_loss = flat_reference(TrainConfig(seq_len=seq_len))[0]
        ax.set_title(f"{seq_len}-token sequences (pure CSA2: {flat_loss:.4f})")
        ax.xaxis.set_minor_locator(mticker.LogLocator(subs=(2, 4)))
        ax.xaxis.set_major_formatter(keys)
        ax.xaxis.set_minor_formatter(keys)
        relabel(ax, ROUTE_NAMES, 6)
    legend = axes[0].get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    labels = ["pure CSA2" if t.startswith("pure CSA2") else t for t in labels]
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(legend.legend_handles, labels, loc="lower center", ncol=3, fontsize=6.5)
    save_svg(fig, "route", bottom=0.14)


def fig_cost() -> None:
    """Layer bench (CSA2 vs flash) and indexer bench (pure CSA2 vs double indexing),
    with plain-language titles and legend names."""
    layers, indexer = read(layer.BENCH), read(route.BENCH)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6))
    layer.panel(
        axes[0, 0], layers, "decode_us_batch8", "time per new token, whole layer (µs)"
    )
    layer.panel(axes[0, 1], layers, "cache_mib", "memory for past tokens (MiB / seq.)")
    route.bench(
        axes[1, 0], indexer, "decode_us_batch64", "time to pick entries, new token (µs)"
    )
    route.bench(
        axes[1, 1],
        indexer,
        "prefill_index_ms",
        "time to pick entries, whole prompt (ms)",
    )
    for ax, names in ((axes[0, 0], LAYER_NAMES), (axes[0, 1], LAYER_NAMES)):
        relabel(ax, names, 6)
    for ax in axes[1]:
        relabel(ax, BENCH_NAMES, 5.5)
    legend = axes[1, 0].get_legend()  # long names: one shared legend under the panels
    labels = [t.get_text() for t in legend.get_texts()]
    for ax in axes[1]:
        ax.get_legend().remove()
    fig.legend(legend.legend_handles, labels, loc="lower center", ncol=2, fontsize=6)
    save_svg(fig, "cost", bottom=0.07)


def relabel(ax: plt.Axes, names: dict[str, str], size: float) -> None:
    """Rename legend entries by their prefix (keeps suffixes like "(OOM from 64k)")."""
    handles, labels = ax.get_legend_handles_labels()
    for i, text in enumerate(labels):
        prefix = next((k for k in names if text.startswith(k)), None)
        if prefix is not None:
            labels[i] = names[prefix] + text[len(prefix) :]
    ax.legend(handles, labels, fontsize=size)


def fig_picks() -> None:
    """One real query's picks per method (first T) and pick recall per method."""
    recall = read(PICKS).filter(pl.col("seed") == 0)
    recall = recall.with_columns(pl.col("pick_recall").cast(pl.Float64))
    order = [m for m in DISPLAY if m != "pure CSA2"]
    layer_index = STRIP_LAYER
    strip = np.load(PICKS.with_name(f"pick_strip_T{LENGTHS[0]}.npz"))
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    score = strip[f"score|{layer_index}"]
    target = strip[f"pure CSA2|{layer_index}"]
    rows = ["pure CSA2", *order]
    for block in target:
        left.axvline(block, color=GRID_LINE, lw=3, zorder=0)
    top = len(rows)
    left.fill_between(
        np.arange(len(score)),
        top,
        top + 1.6 * (score - score.min()) / np.ptp(score),  # head weights can be < 0
        color=REFERENCE,
        alpha=0.5,
        lw=0,
    )
    for y, method in enumerate(reversed(rows)):
        blocks = strip[f"{method}|{layer_index}"]
        hit = np.isin(blocks, target)
        left.plot(blocks[hit], np.full(hit.sum(), y), "o", ms=3, color=SERIES[0])
        left.plot(blocks[~hit], np.full((~hit).sum(), y), "x", ms=3.5, color=SERIES[1])
    left.set_yticks(
        [*range(top), top + 0.5],
        [*(DISPLAY[m] for m in reversed(rows)), "indexer score"],
    )
    left.set(
        xlim=(0, len(score)),
        xlabel="entry (0 = start of the text; right = most recent)",
        title=f"one query's 8 picks (layer {layer_index}, last token of {LENGTHS[0]})",
    )
    left.grid(False)
    for color, seq_len in zip(SERIES, LENGTHS, strict=False):
        rows_t = recall.filter((pl.col("T") == seq_len) & (pl.col("layer") == "all"))
        values = [rows_t.filter(pl.col("method") == m)["pick_recall"][0] for m in order]
        offset = 0.2 if seq_len == LENGTHS[0] else -0.2
        right.barh(
            np.arange(len(order))[::-1] + offset,
            values,
            0.4,
            color=color,
            label=f"{seq_len} tokens",
        )
    right.set_yticks(range(len(order)), [DISPLAY[m] for m in order][::-1])
    right.set(xlim=(0, 1), xlabel="pick recall", title="all layers")
    right.legend(loc="upper right")
    save_svg(fig, "picks")


def fig_training() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    training.draw([ax])
    save_svg(fig, "training")


# --- numbers.json -----------------------------------------------------------------


def signed(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:+.{digits}f}"


def setting(config: str) -> str:
    """A CSV config in words: "S=16, s=4" -> "superblocks of 16, keep 4"; mixed
    "S=8/8, s=1/8" adds the needle layers' setting; "top_k=8" -> "8 entries"."""
    if config.startswith("top_k="):
        return f"{config.removeprefix('top_k=')} entries"
    sizes, kept = (part.split("=")[1].split("/") for part in config.split(", "))
    words = f"superblocks of {sizes[0]}, keep {kept[0]}"
    if len(sizes) == 2:
        size = "" if sizes[1] == sizes[0] else f"superblocks of {sizes[1]}, "
        words += f" (needle layers: {size}keep {kept[1]})"
    return words


def times(x: float | None) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}×"


def selector_fronts(seq_len: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Double-indexing mode -> (reads_ratio, Δ vs flat) of every config."""
    fronts = {}
    routes = read(route.OUT / "route_loss.csv").filter(pl.col("T") == seq_len)
    for mode, label in ROUTE_LABELS.items():
        rows = routes.filter(pl.col("coarse") == label)
        fronts[f"double indexing {mode}"] = (rows["reads_ratio"], rows["delta_vs_flat"])
    return {k: (np.asarray(x), np.asarray(y)) for k, (x, y) in fronts.items()}


def selector_table(seq_len: int) -> list[dict[str, Any]]:
    """Rows: method, headline config, reads ratio, Δ ± SE, whether it can match."""
    fronts = selector_fronts(seq_len)
    paired = read(PAIRED).filter(pl.col("T") == seq_len)
    flat_loss, flat_reads = flat_reference(TrainConfig(seq_len=seq_len))
    baselines = read(CSA_LOSS).filter(
        (pl.col("T") == seq_len)
        & (pl.col("seed") == 0)
        & (pl.col("top_k") == TOP_K)
        & (pl.col("window") == WINDOW)
    )
    rows = []
    for row in paired.iter_rows(named=True):
        method = row["method"]
        entry: dict[str, Any] = {
            "method": DISPLAY[method],
            "config": setting(row["config"]),
        }
        entry["delta"] = f"{row['delta']:+.4f} ± {row['se']:.4f}"
        if method in fronts:
            ratio, delta = fronts[method]
            matched = ratio[delta <= TARGET]
            entry["ratio_at_target"] = times(matched.max() if len(matched) else None)
            entry["ratio"] = times(row["reads_ratio"])
        else:  # random / oracle at the trained top_k
            reads = baselines.filter(pl.col("selector") == method)["reads_per_seq"][0]
            ratio_value = flat_reads / reads if reads else None
            entry["ratio"] = "no reads" if ratio_value is None else times(ratio_value)
            entry["ratio_at_target"] = "—"
        entry["matched"] = entry["ratio_at_target"] != "—"
        rows.append(entry)
    return rows


def training_table() -> list[dict[str, str]]:
    """Final val / test loss and wall time from each training log."""
    rows = []
    for name, (label, seq_len) in TRAINING.items():
        text = Path(f"runs/train/{name}.log").read_text()
        found = re.findall(r"final val ([\d.]+) .*test ([\d.]+) .*?, (\d+)s", text)
        val, test, seconds = found[-1]
        rows.append(
            {
                "model": label,
                "T": str(seq_len),
                "val": val,
                "test": test,
                "minutes": f"{int(seconds) / 60:.1f}",
            }
        )
    return rows


def vanilla_table() -> list[dict[str, str]]:
    """E0: pure CSA2 off its training config, and random / oracle entries."""
    frame = read(CSA_LOSS).filter(pl.col("seed") == 0)

    def loss(seq_len: int, selector: str, top_k: int, window: int) -> float:
        return frame.filter(
            (pl.col("T") == seq_len)
            & (pl.col("selector") == selector)
            & (pl.col("top_k") == top_k)
            & (pl.col("window") == window)
        )["val_loss"][0]

    settings = [("flat", k, WINDOW) for k in (2, 4, 16, 32)]
    settings += [("flat", TOP_K, w) for w in (1, 16, 64)]
    settings += [(s, k, WINDOW) for s in ("random", "oracle") for k in (2, TOP_K)]
    rows = []
    for selector, top_k, window in settings:
        name = DISPLAY["pure CSA2" if selector == "flat" else selector]
        entry = {"selector": name, "top_k": str(top_k), "window": str(window)}
        for seq_len in LENGTHS:
            # flat: vs the trained config; random / oracle: vs flat at the same top_k
            base = loss(seq_len, "flat", TOP_K if selector == "flat" else top_k, WINDOW)
            entry[str(seq_len)] = signed(loss(seq_len, selector, top_k, window) - base)
        rows.append(entry)
    return rows


def cost_table() -> dict[str, list[dict[str, str]]]:
    """Layer and indexer benches at BIG_T (prefill at 32k, where CSA2 still fits)."""
    layers = read(layer.BENCH)
    layer_rows = []
    for name in layers["layer"].unique(maintain_order=True):
        rows = layers.filter(pl.col("layer") == name)
        big, mid = rows.filter(pl.col("T") == BIG_T), rows.filter(pl.col("T") == 32768)
        layer_rows.append(
            {
                "layer": LAYER_NAMES[name],
                "decode1": f"{big['decode_us_batch1'][0]:.0f}",
                "decode8": f"{big['decode_us_batch8'][0]:.0f}",
                "cache": f"{big['cache_mib'][0]:.1f}",
                "prefill32k": f"{mid['prefill_ms'][0]:.1f}",
            }
        )
    indexer = read(route.BENCH).filter(pl.col("T") == BIG_T)
    indexer_rows = [
        {
            "method": BENCH_NAMES[row["method"]],
            "decode": f"{row['decode_us_batch64']:.0f}",
            "prefill": f"{row['prefill_index_ms']:.1f}",
            "kib": f"{row['indexer_kib_per_token']:.0f}",
        }
        for row in indexer.iter_rows(named=True)
    ]
    return {"layer": layer_rows, "indexer": indexer_rows}


def paired_rows(path: Path, seq_len: int) -> dict[str, dict[str, Any]]:
    """Method -> its paired_se row at seq_len."""
    rows = read(path).filter(pl.col("T") == seq_len).iter_rows(named=True)
    return {r["method"]: r for r in rows}


def pm(row: dict[str, Any] | None) -> str:
    """ "+0.0093 ± 0.0009" from a row's delta and se; "—" without a row."""
    return (
        "—" if row is None else f"{float(row['delta']):+.4f} ± {float(row['se']):.4f}"
    )


def seed_table() -> list[dict[str, Any]]:
    """Headlines picked on model 1 (seed 0), Δ on model 1 and on model 2 (seed 1)."""
    first, second = (paired_rows(path, LENGTHS[0]) for path in (PAIRED, PAIRED_SEED1))
    return [
        {
            "method": DISPLAY[method],
            "config": setting(row["config"]),
            "seed0": pm(row),
            "seed1": pm(second.get(method)),
            "matched0": float(row["delta"]) <= TARGET,
            "matched1": method in second and float(second[method]["delta"]) <= TARGET,
        }
        for method, row in first.items()
    ]


def copy_table() -> list[dict[str, Any]]:
    """Copy stress test: Δ on the repeated half vs pure CSA2, per T and model."""
    frame = read(COPY)
    runs = {
        f"{t}_s{seed}": {
            r["method"]: r
            for r in frame.filter(
                (pl.col("T") == t) & (pl.col("seed") == seed)
            ).iter_rows(named=True)
        }
        for t, seed in ((LENGTHS[0], 0), (LENGTHS[1], 0), (LENGTHS[0], 1))
    }
    normal = {t: paired_rows(PAIRED, t) for t in LENGTHS}
    rows = []
    for method in runs[f"{LENGTHS[0]}_s0"]:
        entry = {"method": DISPLAY[method]}
        for key, rows_t in runs.items():
            entry[key] = (
                pm(rows_t.get(method))
                if method != "pure CSA2"
                else (
                    f"{float(rows_t[method]['copy_loss']):.3f}"
                    if method in rows_t
                    else "—"
                )
            )
        for t in LENGTHS:  # the same method's Δ on normal text, for comparison
            entry[f"normal_{t}"] = pm(normal[t].get(method))
        rows.append(entry)
    return rows


def pick_table() -> list[dict[str, str]]:
    """Pick and mass recall over all layers per method and T (model 1)."""
    frame = read(PICKS).filter((pl.col("seed") == 0) & (pl.col("layer") == "all"))
    rows = []
    for method in DISPLAY:
        entry = {"method": DISPLAY[method]}
        for t in LENGTHS:
            row = frame.filter((pl.col("T") == t) & (pl.col("method") == method))
            entry[f"pick_{t}"] = f"{float(row['pick_recall'][0]):.2f}"
            entry[f"mass_{t}"] = f"{float(row['mass_recall'][0]):.2f}"
        rows.append(entry)
    return rows


def numbers() -> dict[str, Any]:
    flat = {str(t): f"{flat_reference(TrainConfig(seq_len=t))[0]:.4f}" for t in LENGTHS}
    return {
        "flat_val": flat,
        "training": training_table(),
        "vanilla": vanilla_table(),
        "selectors": {str(t): selector_table(t) for t in LENGTHS},
        "cost": cost_table(),
        "seeds": seed_table(),
        "copy": copy_table(),
        "picks": pick_table(),
    }


def main() -> None:
    style()
    for fig in (fig_route, fig_cost, fig_training, fig_picks):
        fig()
    methods.main()
    NUMBERS.write_text(json.dumps(numbers(), ensure_ascii=False, indent=1))
    print(NUMBERS)
    if "pdf" in sys.argv:
        pdf()


def pdf() -> None:
    """Compile report/paper.typ next to itself; STIXGeneral stands in for Times."""
    import typst  # type: ignore[import-not-found]  # only under uv run --with typst

    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    output = PAPER.with_suffix(".pdf")
    typst.compile(str(PAPER), output=str(output), font_paths=[str(fonts)])
    print(output)


if __name__ == "__main__":
    main()
