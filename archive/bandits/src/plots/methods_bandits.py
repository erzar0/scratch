"""Cut from plots/methods.py when the bandits were archived (frozen, not run).

Bandit rows of methods_csa.svg and the whole methods_dense.svg; they used row(),
frame(), label(), top(), in_superblocks(), RNG, SCORE and PULL from plots/methods.py
(row() had a pulls: dict[int, int] | None argument drawing PULL-coloured dots).
"""

def superblock_bandit(ax: Axes) -> None:
    order = RNG.permuted(np.tile(np.arange(SIZE), (SUPERBLOCKS, 1)), axis=1)
    first = order[:, 0] + SIZE * np.arange(SUPERBLOCKS)  # one pulled block each
    survivors = top(SCORE[first], 4)  # ponytail: UCB with 1 pull = the pull itself
    pulls = {int(b): 1 for b in first}
    best = np.full(SUPERBLOCKS, -np.inf)
    best[:] = SCORE[first]
    for group in survivors:
        extra = order[group, 1:3] + SIZE * group
        pulls |= {int(b): 1 for b in extra}
        best[group] = max(best[group], SCORE[extra].max())
    kept = top(
        np.where(np.isin(np.arange(SUPERBLOCKS), list(survivors)), best, -1), KEPT
    )
    picked = top(SCORE, TOP_K, in_superblocks(kept))
    pulled = np.isin(np.arange(N_BLOCKS), list(pulls))
    row(ax, 1, SCORE, pulled, pulls=pulls)
    label(ax, 1, "1. sample blocks")
    row(ax, 0, SCORE, in_superblocks(kept), picked)
    label(ax, 0, "2. score inside")
    frame(
        ax,
        "Superblock bandit: sample a few blocks per group, keep the groups with "
        "the best sample",
        2,
        f"{len(pulls)} + {KEPT * SIZE}",
        picked,
    )


def channel_bandit(ax: Axes) -> None:
    channels, per_round = 16, 4
    parts = SCORE[:, None] / channels + 0.03 * RNG.standard_normal((N_BLOCKS, channels))
    active = np.ones(N_BLOCKS, bool)
    reads = 0.0
    rounds = channels // per_round
    for r in range(1, rounds + 1):
        seen = r * per_round
        reads += active.sum() * per_round / channels
        estimate = parts[:, :seen].sum(1) * channels / seen
        radius = 0.25 * np.sqrt((channels - seen) / seen)
        threshold = np.sort(np.where(active, estimate - radius, -np.inf))[-TOP_K]
        y = rounds - r
        last = r == rounds
        picked = top(SCORE, TOP_K, active) if last else set()
        row(ax, y, estimate, active, picked)
        label(ax, y, f"{seen}/{channels} channels")
        active &= estimate + radius >= threshold
    frame(
        ax,
        "Channel bandit: read the score a few channels at a time; drop blocks "
        "that cannot reach the top 8",
        rounds,
        f"{reads:.1f}",
        picked,
    )


def token_bandit(ax: Axes) -> None:
    logits = SCORE[:, None] + 0.25 * RNG.standard_normal((N_BLOCKS, SIZE))
    pulls = {b: 1 for b in range(N_BLOCKS)}
    estimate = logits[:, 0]
    survivors = top(estimate, 16)
    for block in survivors:
        pulls[block] = 2
        estimate[block] = logits[block, :2].mean()
    picked = top(estimate, TOP_K)
    row(ax, 0, estimate, np.ones(N_BLOCKS, bool), picked, pulls)
    label(ax, 0, "sample raw tokens")
    n_tokens = sum(pulls.values())
    frame(
        ax,
        "Token bandit: no indexer; the main query reads a few raw tokens per block",
        1,
        f"{n_tokens}×4 = {4 * n_tokens}",
        picked,
    )


def dense_panel(ax: Axes, read: np.ndarray, title: str, note: str) -> None:
    """(T, T) share of each (query, key) pair read, future keys hatched grey."""
    n = read.shape[0]
    future = np.triu(np.ones((n, n), bool), 1)
    ax.imshow(np.where(future, np.nan, read), cmap="Blues", vmin=-0.15, vmax=1)
    ax.imshow(np.where(future, 1.0, np.nan), cmap="Greys", vmin=0, vmax=6)
    ax.set_title(title, fontsize=8, pad=9)
    ax.set_xlabel(note, fontsize=6.5)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_dense() -> plt.Figure:
    n, chunk, window, refined = 48, 4, 8, 2
    causal = np.tril(np.ones((n, n)))
    rng = np.random.default_rng(5)
    chunk_score = rng.random(n // chunk)
    chunk_score[[1, 5]] += 1  # two chunks the queries need
    chunks = np.zeros((n, n))
    for q in range(0, n, chunk):
        old = q // chunk - window // chunk  # chunks before the window
        best = top(chunk_score[: max(old, 0)], refined) if old > 0 else set()
        for c in range(max(old, 0)):
            chunks[q : q + chunk, c * chunk : (c + 1) * chunk] = 1 if c in best else 0.3
        chunks[q : q + chunk, max(old, 0) * chunk : q + chunk] = 1
    chunks *= causal
    logits = rng.standard_normal((n, n)) + 2.5 * (rng.random((n, n)) < 0.08)
    logits = np.where(causal > 0, logits, -np.inf)
    gap = logits.max(1, keepdims=True) - logits
    rounds = 8
    eliminated = np.ceil(rounds * (1.2 / np.maximum(gap, 1e-9)) ** 2)
    keys = np.clip(eliminated, 1, rounds) / rounds * causal
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.3))
    dense_panel(axes[0], causal, "Full softmax", "every past key, exactly")
    dense_panel(
        axes[1],
        chunks,
        "Chunk bandit",
        f"window + {refined} chunks exactly;\nother chunks as one average key",
    )
    dense_panel(
        axes[2],
        keys,
        "Per-key bandit",
        "shade: share of q·k channels\nread before a key is dropped",
    )
    for ax in axes:
        ax.text(
            -1.5, n / 2, "query →", rotation=90, va="center", ha="right", fontsize=6
        )
        ax.text(n / 2, -1.2, "key →", ha="center", fontsize=6)
    return fig


