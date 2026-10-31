"""Kernel timing on real Qwen3-0.6B keys against random keys, 32768 tokens.

    kernel_real.png  left: microseconds per method on random keys
                     (experiments/sparse_kernel.py) and on real keys
                     (experiments/sparse_kernel_real.py, mean over layers, bar
                     from the fastest to the slowest layer); right: real keys
                     per layer
    kernel_real.md   the same numbers

uv run python -m plots.sparse_real
"""

import matplotlib.pyplot as plt
import polars as pl

from experiments.results import read
from plots.sparse import FAMILY, RUNS, below
from plots.sparse_global import TIMED
from plots.sparse_kernel import timings
from plots.style import apply, save

N = 32768
REAL = RUNS / "kernel_real.csv"
PER_LAYER = ("sparq-32-128", "sparq-64-128", "hybrid-512", "sampled-8-drop-512")


def per_layer() -> pl.DataFrame:
    """Real keys: microseconds per method and layer, dense = its best variant."""
    times = read(REAL).with_columns(
        method=pl.col("method").str.replace("dense-bmm", "dense")
    )
    return times.group_by("method", "layer").agg(pl.col("us").min()).sort("layer")


def stats(layers: pl.DataFrame) -> pl.DataFrame:
    """Per method in TIMED order: real-key mean, fastest, slowest; random keys."""
    real = layers.group_by("method").agg(
        mean=pl.col("us").mean(), low=pl.col("us").min(), high=pl.col("us").max()
    )
    random = timings().filter(pl.col("n") == N).select("method", random="us")
    order = pl.DataFrame({"method": [m for m in TIMED if m in real["method"]]})
    joined = order.with_row_index().join(real, on="method").join(random, on="method")
    return joined.sort("index")


def color(method: str) -> str:
    """The method's family color."""
    return FAMILY[method.split("-")[0]]


def main() -> None:
    """Write kernel_real.png and kernel_real.md."""
    apply()
    layers = per_layer()
    frame = stats(layers)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5), width_ratios=(1.3, 1))
    for y, r in enumerate(frame.iter_rows(named=True)):
        spread = [[r["mean"] - r["low"]], [r["high"] - r["mean"]]]
        left.errorbar(r["mean"], y, xerr=spread, fmt="o", color=color(r["method"]))
        left.plot(r["random"], y, "o", mfc="none", color=color(r["method"]))
    left.plot([], [], "ko", label="real keys, mean over layers (bar: layer range)")
    left.plot([], [], "ko", mfc="none", label="random keys")
    left.set_yticks(range(len(frame)), [TIMED[m] for m in frame["method"]])
    left.invert_yaxis()
    left.set_xlabel("microseconds per layer and decode step")
    left.set_title(f"Triton kernels, {N} tokens, RTX 4070 Ti SUPER", fontsize=9)
    below(left)
    for method in PER_LAYER:
        part = layers.filter(pl.col("method") == method)
        style = "--" if method == "sparq-32-128" else "-"
        right.plot(part["layer"], part["us"], ls=style, color=color(method))
        right.plot([], [], ls=style, color=color(method), label=TIMED[method])
    right.set_xlabel("layer")
    right.set_ylabel("microseconds, real keys")
    right.set_title("Real keys per layer", fontsize=9)
    below(right)
    save(fig, RUNS / "kernel_real.png")

    lines = [
        "| Method | Random keys (µs) | Real keys, mean over layers (µs) "
        "| Real keys, fastest and slowest layer (µs) | Real ÷ random |",
        "|---|---|---|---|---|",
    ]
    for r in frame.iter_rows(named=True):
        lines.append(
            f"| {TIMED[r['method']]} | {r['random']:.1f} | {r['mean']:.1f} "
            f"| {r['low']:.1f}–{r['high']:.1f} | {r['mean'] / r['random']:.2f} |"
        )
    (RUNS / "kernel_real.md").write_text("\n".join(lines) + "\n")
    print(RUNS / "kernel_real.md")


if __name__ == "__main__":
    main()
