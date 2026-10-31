"""Cut from plots/paper.py when the bandits were archived (frozen, not run).

Teaser (selectors.svg), dense-control figure and tables, bandit seed spread and the
chunk bench table. They used KEY_KNOBS, CHUNK_ROWS, model.heads.SEQ_LEN,
plots.csa_bandit and plots.chunk_bandit.SELECTORS (all archived next to this file).
"""

def fig_selectors() -> None:
    """Every CSA block picker vs pure CSA at both T, one legend (teaser)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    for ax, seq_len in zip(axes, LENGTHS, strict=True):
        csa_bandit.frontier(ax, seq_len)
        ax.set(xlabel=WORK, ylabel=EXTRA, title=f"{seq_len}-token sequences")
        ax.get_legend().remove()
        for label in ax.texts if seq_len == LENGTHS[1] else ():  # clear of the oracle
            label.xyann = (0, -3)
            label.set_va("top")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    # values differ per T (they are in the tables): "pure CSA (loss 4.03)" -> "pure CSA"
    labels = [re.sub(r" \(loss [\d.]+\)$| [+][\d.]+(?= \()", "", s) for s in labels]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    save_svg(fig, "selectors", bottom=0.12)



def key_bandit_front() -> pl.DataFrame:
    """key_loss rows with reads / exact from the error sweep (same configs)."""
    errors = (
        read("runs/key_bandit/key_error.csv")
        .group_by(KEY_KNOBS)
        .agg(
            (pl.col("reads").mean() / ((SEQ_LEN + 1) / 2)).alias("reads_ratio"),
            pl.col("rel_err").mean(),
        )
    )
    losses = read("runs/key_bandit/key_loss.csv").filter(pl.col("method") != "exact")
    losses = losses.with_columns(pl.col(k).cast(pl.Float64) for k in KEY_KNOBS)
    errors = errors.with_columns(pl.col(k).cast(pl.Float64) for k in KEY_KNOBS)
    return losses.join(errors, on=KEY_KNOBS).sort("reads_ratio")


def chunk_rows(seq_len: int) -> pl.DataFrame:
    return read("runs/chunk_bandit/chunk_loss.csv").filter(pl.col("T") == seq_len)


def fig_dense() -> None:
    """Bandits replacing full softmax: Δ val vs reads at T=1024, KL at T=4096."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    ax = axes[0]
    keys = key_bandit_front()
    front_series(
        ax,
        keys["reads_ratio"].to_numpy(),
        keys["delta_vs_exact"].to_numpy(),
        SERIES[3],
        "per-key bandit",
    )
    rows = chunk_rows(1024)
    for selector, color in CHUNK_COLORS.items():
        line = rows.filter(pl.col("selector") == selector)
        x, y = line["reads_ratio"].to_numpy(), line["delta_vs_exact"].to_numpy()
        front_series(ax, x, y, color, f"chunk {selector}")
    target_line(ax)
    ax.set(
        yscale="symlog",
        xlabel="keys read / keys full attention reads",
        ylabel="extra loss vs full attention (nats)",
        title="1024-token sequences: loss",
    )
    ax.set_yscale("symlog", linthresh=TARGET)
    ax.set_ylim(-0.002, 0.3)
    ax.legend()
    ax = axes[1]
    rows = chunk_rows(4096)
    for selector, color in CHUNK_COLORS.items():
        line = rows.filter(pl.col("selector") == selector)
        x, y = line["reads_ratio"].to_numpy(), line["kl"].to_numpy()
        front_series(ax, x, y, color, f"chunk {selector}")
    windows = rows.filter(pl.col("selector") == "window").sort("reads_ratio")
    ax.plot(
        windows["reads_ratio"],
        windows["kl"],
        "x--",
        color=REFERENCE,
        label="recent keys only",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="keys read / keys full attention reads",
        ylabel="output change vs full attention (KL)",
        title="4096-token sequences: output change",
    )
    ax.legend()
    save_svg(fig, "dense")



def seed_spread(seq_len: int) -> dict[str, float]:
    """Bandit -> max over front configs of the Δ range across seeds 0-2."""
    spread = {}
    for name in csa_bandit.STUDIES:
        rows = read(csa_bandit.OUT / f"{name}.csv").filter(pl.col("T") == seq_len)
        ranges = csa_bandit.seed_spread(rows)
        if not ranges.is_empty():
            spread[name] = float(np.max((ranges["high"] - ranges["low"]).to_numpy()))
    return spread


def channel_all_seeds(seq_len: int) -> str:
    """Fewer reads (worst seed) of channel configs whose every seed has Δ ≤ TARGET."""
    rows = read(csa_bandit.OUT / "channel.csv").filter(pl.col("T") == seq_len)
    knobs = ["ucb_width", "tolerance", "dims_per_round"]
    seeds = rows.group_by(knobs).agg(
        pl.col("delta_vs_flat").max().alias("worst"),
        pl.col("reads_ratio").min(),
        pl.col("seed").n_unique().alias("seeds"),
    )
    robust = seeds.filter((pl.col("seeds") == 3) & (pl.col("worst") <= TARGET))
    return times(
        float(np.max(robust["reads_ratio"].to_numpy())) if len(robust) else None
    )



def dense_table() -> dict[str, Any]:
    """Per-key bandit configs; CHUNK_ROWS of the chunk study."""
    keys = [
        {
            "config": f"z={r['ucb_width']:g}, tol={r['tolerance']:g}, "
            f"d={r['dims_per_round']:g}",
            "reads": f"{r['reads_ratio']:.2f}",
            "rel_err": f"{r['rel_err']:.3f}",
            "delta": f"{r['delta_vs_exact']:+.4f} ± {r['delta_se']:.4f}",
        }
        for r in key_bandit_front().iter_rows(named=True)
    ]
    chunks = [
        {
            "T": str(seq_len),
            "config": r["name"],
            "reads": f"{r['reads_ratio']:.2f}",
            "delta": f"{r['delta_vs_exact']:+.4f} ± {r['delta_se']:.4f}",
            "kl": f"{r['kl']:.4f}",
        }
        for seq_len, names in CHUNK_ROWS.items()
        for r in chunk_rows(seq_len)
        .filter(pl.col("name").is_in(names))
        .sort(pl.col("name").replace_strict({n: i for i, n in enumerate(names)}))
        .iter_rows(named=True)
    ]
    return {"key": keys, "chunk": chunks}



# from cost_table():
    chunk = read("runs/bench/chunk_bandit.csv").filter(pl.col("T") == 32768)
    chunk_rows_ = [
        {
            "config": row["config"],
            "ms": f"{row['ms']:.2f}",
            "gib": f"{row['temp_gib']:.2f}",
        }
        for row in chunk.iter_rows(named=True)
    ]
