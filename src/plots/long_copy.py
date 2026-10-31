"""Copy test at 8K-128K tokens (experiments/long_copy.py).

    long_copy.png  top: pure CSA2's text and copy loss against context length,
                   and the share of repeat queries whose picks hold the entry to
                   copy (mean over selecting layers); bottom: each double-indexing
                   setting's and the oracle's extra loss over pure CSA2 on the
                   first half (text) and on the repeat (copy), ± paired SE
    long_copy.md   the same per length, with extra text loss and reads saved

uv run python -m plots.long_copy
"""

from typing import Any

import matplotlib.pyplot as plt
import polars as pl

from experiments.long_copy import OUT
from experiments.results import read
from plots.paper import DISPLAY, setting
from plots.style import INK, REFERENCE, SERIES, apply, context_axis, save, target_line

STYLE = {"pure CSA2": (REFERENCE, "-"), "random": (REFERENCE, ":")}
STYLE["oracle"] = (INK, "--")
for mode, color in zip(("mean", "bound", "learned", "mixed"), SERIES, strict=False):
    STYLE[f"double indexing {mode}"] = (color, "-")


def label(row: dict[str, Any]) -> str:
    """Plain name: the method, and its setting where it has one."""
    if row["method"] == "pure CSA2":
        return "pure CSA2"
    return f"{DISPLAY[row['method']]}, {setting(row['config'])}"


def main() -> None:
    """Write long_copy.png and long_copy.md."""
    apply()
    frame = read(OUT).sort("T")
    lengths = frame["T"].unique().sort().to_list()
    fig, ((left, right), (text, copy)) = plt.subplots(2, 2, figsize=(12, 9.5))
    pure = frame.filter(pl.col("method") == "pure CSA2")
    left.plot(pure["T"], pure["text_loss"], "o-", color=REFERENCE, label="first half")
    left.plot(pure["T"], pure["copy_loss"], "o--", color=INK, label="repeat")
    for method, (color, ls) in STYLE.items():
        rows = frame.filter(pl.col("method") == method)
        if rows.is_empty():
            continue
        name = label(rows.row(0, named=True))
        style = {"color": color, "ls": ls, "marker": "o"}
        right.plot(rows["T"], rows["hit_rate"], label=name, **style)
        if method not in ("pure CSA2", "random"):
            for ax, part in ((text, "text"), (copy, "copy")):
                ax.errorbar(
                    rows["T"], rows[f"{part}_delta"], rows[f"{part}_se"], **style
                )
    for ax in (text, copy):
        ax.axhline(0, color=REFERENCE, lw=0.8)
        target_line(ax)
    left.set_ylabel("pure CSA2 loss (nats)")
    left.set_title("Does the 8192-token model still copy?", fontsize=9)
    left.legend()
    text.set_ylabel("first-half loss above pure CSA2 (nats)")
    text.set_title("Extra loss on text (random entries: see table)", fontsize=9)
    copy.set_ylabel("repeat loss above pure CSA2 (nats)")
    copy.set_title("Extra loss on the repeat (random entries: see table)", fontsize=9)
    right.set_ylabel("repeat queries whose picks hold the entry to copy")
    right.set_title("Copy hit rate, mean over selecting layers", fontsize=9)
    for ax in (left, right, text, copy):
        context_axis(ax, lengths)
    right.set_ylim(top=right.get_ylim()[1] * 1.7)  # headroom for the legend
    right.legend(loc="upper left")
    save(fig, OUT.with_suffix(".png"))

    lines = []
    for t in lengths:
        lines += [
            f"### {t} tokens",
            "",
            "| Method | Copy loss above pure CSA2 (nats) "
            "| Text loss above pure CSA2 (nats) | Copy hit rate "
            "| Index reads saved vs pure CSA2 |",
            "|---|---|---|---|---|",
        ]
        for row in frame.filter(pl.col("T") == t).iter_rows(named=True):
            if row["method"] == "pure CSA2":
                copy = f"{row['copy_loss']:.3f} (loss)"
                text = f"{row['text_loss']:.3f} (loss)"
            else:
                copy = f"{row['copy_delta']:+.3f} ± {row['copy_se']:.3f}"
                text = f"{row['text_delta']:+.3f} ± {row['text_se']:.3f}"
            ratio = row.get("reads_ratio")
            saved = "—" if ratio is None else f"{ratio:.2f}×"
            lines.append(
                f"| {label(row)} | {copy} | {text} | {row['hit_rate']:.3f} | {saved} |"
            )
        lines.append("")
    OUT.with_suffix(".md").write_text("\n".join(lines))
    print(OUT.with_suffix(".md"))


if __name__ == "__main__":
    main()
