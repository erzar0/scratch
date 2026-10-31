"""PG-19 loss against kernel time: SparQ settings beside the tournament kernels.

    pareto.png  PG-19 loss above dense at 32768 tokens (experiments/sparse_qwen.py)
                against microseconds per layer and decode step on random keys
                (experiments/sparse_kernel.py), at 32768 and 131072 tokens;
                SparQ settings joined per dims, tournament kernels as points
    pareto.md   the same numbers

Loss is measured at 32768 tokens only; the 131072 panel pairs it with the
time at 131072 (assumed: loss there unmeasured). The hybrid kernel runs 3
rounds in blocks at 32768, the measured setting, and more beyond, so it is
shown at 32768 only.

uv run python -m plots.sparse_pareto
"""

import matplotlib.pyplot as plt
import polars as pl

from plots.sparse import FAMILY, RUNS, below, label, plain, summary
from plots.sparse_global import TIMED
from plots.sparse_kernel import timings
from plots.style import apply, save

LENGTHS = (32768, 131072)
SPARQ_DIMS = {64: "-", 32: "--", 16: ":"}
TOURNAMENT = {  # kernel.csv method -> pg19.csv row with its loss
    "global-512": "tournament-query-halving-drop-512",
    "sampled-8-drop-512": "sampled-8-drop-512",
    "sampled-16-drop-512": "sampled-16-drop-512",
    "sampled-16-approx-512": "sampled-16-approx-512",
    "hybrid-512": "hybrid-1024-3-512",
    "parallel-32-512": "shuffled-blocks-1024-512",  # 32 blocks of 1024 at 32768
}
MARKERS = dict(zip(TOURNAMENT, "osD^vP", strict=True))
SHORT = 32768  # the hybrid's measured setting


def points() -> pl.DataFrame:
    """Rows with both a loss and a time: method, row, extra, se, n, us."""
    rows = {m: m for m in timings()["method"].unique() if m.startswith("sparq")}
    pairs = pl.DataFrame(
        {"method": [*rows, *TOURNAMENT], "row": [*rows.values(), *TOURNAMENT.values()]}
    )
    times = timings().filter(pl.col("n").is_in(LENGTHS))
    joined = pairs.join(summary(), on="row").join(times, on="method")
    return joined.filter((pl.col("method") != "hybrid-512") | (pl.col("n") == SHORT))


def main() -> None:
    """Write pareto.png and pareto.md."""
    apply()
    frame = points()
    dense = timings().filter(pl.col("method") == "dense")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, n in zip(axes, LENGTHS, strict=True):
        at = frame.filter(pl.col("n") == n)
        for dims, style in SPARQ_DIMS.items():
            line = at.filter(pl.col("method").str.starts_with(f"sparq-{dims}-"))
            line = line.with_columns(
                top=pl.col("method").str.split("-").list.get(2).cast(int)
            ).sort("top")
            if line.is_empty():
                continue
            ax.plot(
                line["us"],
                line["extra"],
                "o",
                ls=style,
                color=FAMILY["sparq"],
                label=f"SparQ, {dims} dims, top-"
                + " / ".join(str(t) for t in line["top"]),
            )
            for r in line.iter_rows(named=True):
                ax.annotate(
                    str(r["top"]),
                    (r["us"], r["extra"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=FAMILY["sparq"],
                )
        for method, marker in MARKERS.items():
            hit = at.filter(pl.col("method") == method)
            if hit.is_empty():
                continue
            family = method.split("-")[0]
            ax.errorbar(
                hit["us"][0],
                hit["extra"][0],
                hit["se"][0],
                fmt=marker,
                color=FAMILY[family],
                label=TIMED[method],
            )
        us = dense.filter(pl.col("n") == n)["us"]
        if len(us):
            ax.axvline(us[0], color=FAMILY["dense"], lw=1, ls=":", label="dense time")
        ax.set_yscale("log")
        plain(ax.yaxis)
        ax.set_xlabel(f"microseconds per layer and decode step, {n} tokens")
        ax.set_ylabel("PG-19 loss above dense at 32768 tokens (nats)")
        ax.set_title(f"Loss against time, {n} tokens", fontsize=9)
        below(ax)
    save(fig, RUNS / "pareto.png")

    wide = frame.pivot("n", index=["row", "extra", "se"], values="us")
    lines = [
        "| Method | PG-19 loss above dense at 32768 tokens (nats) "
        "| µs, 32768 tokens | µs, 131072 tokens |",
        "|---|---|---|---|",
    ]
    for r in wide.sort("extra").iter_rows(named=True):
        times = [r.get(str(n)) for n in LENGTHS]
        cells = [f"{t:.1f}" if t is not None else "—" for t in times]
        loss = f"{r['extra']:+.3f} ± {r['se']:.3f}"
        lines.append(f"| {label(r['row'])} | {loss} | {' | '.join(cells)} |")
    (RUNS / "pareto.md").write_text("\n".join(lines) + "\n")
    print(RUNS / "pareto.md")


if __name__ == "__main__":
    main()
