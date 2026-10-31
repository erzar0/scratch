"""Bound bandit (sparse/bound_bandit.py) against the tournament and SparQ.

    bandit.png  left: PG-19 loss above dense at 32768 tokens against the share of K
                and V read, each method through its settings; right: recall of the
                exact top-128 against the same share
    bandit.md   per bandit row: loss above dense, minus the tournament at keep 512
                and minus SparQ 32 dims top-512 (paired over books), share read,
                recall; then the references

uv run python -m plots.sparse_bandit
"""

import matplotlib.pyplot as plt

from plots.sparse import FAMILY, RUNS, below, label, plain, summary
from plots.sparse_shuffled import GLOBAL, versus
from plots.style import REFERENCE, apply, save

SPARQ = "sparq-32-512"
LINES = (  # legend, family, line style, row pattern, settings
    (
        "bandit, exact bound (numbers: radius scale)",
        "bandit",
        "-",
        "bandit-exact-{}-512",
    ),
    (
        "bandit, Gaussian bound (numbers: standard deviations)",
        "bandit",
        "--",
        "bandit-gauss-{}-512",
    ),
    (
        "tournament, |q| order, halving (numbers: keep)",
        "tournament",
        "-",
        "tournament-query-halving-drop-{}",
    ),
    ("SparQ, 16 dims (numbers: top-k)", "sparq", "--", "sparq-16-{}"),
    ("SparQ, 32 dims (numbers: top-k)", "sparq", "-", "sparq-32-{}"),
    ("exact top-k (numbers: k)", "exact", "-", "exact-{}"),
)


def setting(pattern: str, row: str) -> str | None:
    """The {} part of row if row fits pattern."""
    head, tail = pattern.split("{}")
    fits = row.startswith(head) and row.endswith(tail) and len(row) > len(head + tail)
    part = row[len(head) : len(row) - len(tail)]
    return part if fits and "-" not in part else None


def main() -> None:
    """Write bandit.png and bandit.md."""
    apply()
    stats = {r["row"]: r for r in summary().iter_rows(named=True)}
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))
    for legend, family, ls, pattern in LINES:
        shown = sorted(
            (stats[r]["reads"], s, stats[r])
            for r in stats
            if (s := setting(pattern, r)) is not None and stats[r]["reads"] is not None
        )
        if not shown:
            continue
        xs = [x for x, _, _ in shown]
        style = {"color": FAMILY[family], "ls": ls, "marker": "o"}
        es = [p["se"] for _, _, p in shown]
        left.errorbar(xs, [p["extra"] for _, _, p in shown], es, label=legend, **style)
        right.plot(xs, [p["recall"] for _, _, p in shown], **style)
        for x, s, p in shown:
            for ax, y in ((left, p["extra"]), (right, p["recall"])):
                ax.annotate(
                    s, (x, y), xytext=(4, 2), textcoords="offset points", fontsize=7
                )
    for ax in (left, right):
        ax.set_xscale("log")
        plain(ax.xaxis)
        ax.set_xlabel("share of K and V read, mean over layers")
    left.set_yscale("log")
    plain(left.yaxis)
    left.set_ylabel("PG-19 loss above dense at 32768 tokens (nats)")
    left.set_title("Loss against reads", fontsize=9)
    right.set_ylabel("recall of the exact top-128 keys, mean over layers")
    right.set_title("Recall", fontsize=9)
    right.axhline(1, color=REFERENCE, lw=0.5)
    below(left)
    save(fig, RUNS / "bandit.png")

    lines = [
        "| Bandit, keep 512 | Loss above dense (nats) "
        f"| Minus {label(GLOBAL)} (nats) | Minus {label(SPARQ)} (nats) "
        "| Share of K and V read | Recall of the exact top-128 |",
        "|---|---|---|---|---|---|",
    ]
    for row, p in stats.items():
        if row.startswith("bandit-"):
            lines.append(
                f"| {label(row)} | {p['extra']:+.3f} ± {p['se']:.3f} "
                f"| {versus(row, GLOBAL)} | {versus(row, SPARQ)} "
                f"| {p['reads']:.3f} | {p['recall']:.3f} |"
            )
    lines += [
        "",
        "| Reference | Loss above dense (nats) | Share of K and V read | Recall |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {label(row)} | {p['extra']:+.3f} ± {p['se']:.3f} | {p['reads']:.3f} "
        f"| {p['recall']:.3f} |"
        for _, _, _, pattern in LINES[2:]
        for row, p in stats.items()
        if setting(pattern, row) is not None and p["reads"] is not None
    ]
    (RUNS / "bandit.md").write_text("\n".join(lines) + "\n")
    print(RUNS / "bandit.md")


if __name__ == "__main__":
    main()
