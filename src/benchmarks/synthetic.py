"""Synthetic inputs: generated Q/K/V (fidelity, latency) and task sequences (tasks).

Q/K/V distributions, per KV head:
    iid               every entry N(0, 1)
    outlier-channels  OUTLIERS random channels of q and k (the same ones) scaled
                      by OUTLIER_SCALE, as LLM keys have a few large channels; q
                      and k then rescaled to the iid norm sqrt(D)
    llm-like          outlier channels, a shared query component with SINKS sink
                      keys along it (the first positions), and NEEDLES keys per
                      query along that query's group direction (norms SINK and
                      NEEDLE times sqrt(D), so both get far more weight than a
                      random key)
    flat              llm-like with q scaled by 0.5: softer attention over more keys
    sharp             llm-like with q scaled by 2: attention peaked on a few keys
    local             llm-like whose keys each average the noise of the LOCAL
                      positions up to theirs (rescaled to unit variance), and whose
                      needles are runs of LOCAL adjacent keys, the needle added to
                      each key's noise (alike, not copies): the locality of text
                      that block methods assume
    many-needles      llm-like with MANY_NEEDLES needle keys per query

Tasks (token ids; FILLER, KEYS and VALUES are disjoint vocabulary ranges):
    copy    random filler, then the same tokens again: every second-half token
            is an answer (an exact lookup at a distance of n / 2)
    mqar    multi-query associative recall (Zoology, arXiv 2312.04927): PAIRS
            key-value pairs at the start, filler, then QUERIES of the keys, each
            followed by its value (the answer)
    needle  NEEDLE_PAIRS key-value pairs at random places in filler, then every
            key again, each followed by its value (the answer)
"""

import torch
from torch.nn.functional import avg_pool1d, pad

TASKS = ("copy", "mqar", "needle")
OUTLIERS, OUTLIER_SCALE = 4, 4.0
SINKS, NEEDLES = 4, 4
SINK, NEEDLE = 2.0, 1.5
LOCAL, MANY_NEEDLES = 16, 64
HARDER = {  # llm-like variants: q scale, needles per query, neighbour window
    "flat": (0.5, NEEDLES, 1),
    "sharp": (2.0, NEEDLES, 1),
    "local": (1.0, NEEDLES, LOCAL),
    "many-needles": (1.0, MANY_NEEDLES, 1),
}
DISTRIBUTIONS = ("iid", "outlier-channels", "llm-like", *HARDER)
FILLER, KEYS, VALUES = range(16, 256), range(256, 768), range(768, 1024)
VOCAB = 1024
PAIRS, QUERIES, NEEDLE_PAIRS = 64, 64, 16
SHORTEST = {"copy": 2, "mqar": 2 * (PAIRS + QUERIES), "needle": 4 * NEEDLE_PAIRS}


def qkv(
    distribution: str,
    n: int,
    queries: int,
    seed: int,
    groups: int = 8,
    size: int = 2,
    dim: int = 128,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(G, size, Q, D) queries at the last Q positions; (G, n, D) keys, values."""
    gen = torch.Generator(device).manual_seed(seed)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device)

    q, k, v = (
        normal(groups, size, queries, dim),
        normal(groups, n, dim),
        normal(groups, n, dim),
    )
    if distribution == "iid":
        return q, k, v
    q_scale, needles, window = HARDER.get(distribution, (1.0, NEEDLES, 1))
    if window > 1:  # key t averages the noise of positions t - window + 1 .. t
        k = avg_pool1d(pad(k.mT, (window - 1, 0)), window, stride=1).mT * window**0.5
    scale = torch.ones(groups, 1, dim, device=device)
    outliers = normal(groups, dim).argsort(-1)[:, None, :OUTLIERS]
    scale.scatter_(-1, outliers, OUTLIER_SCALE)
    scale = scale / scale.square().mean(-1, keepdim=True).sqrt()
    q, k = q * scale[:, None], k * scale
    if distribution == "outlier-channels":
        return q, k, v
    assert distribution == "llm-like" or distribution in HARDER, distribution
    shared = normal(groups, 1, 1, dim) * scale[:, None]
    q = q + 0.5 * shared
    unit = shared[:, 0] / shared[:, 0].norm(dim=-1, keepdim=True)  # (G, 1, D)
    k[:, :SINKS] = SINK * dim**0.5 * unit
    group = q.sum(1)  # (G, Q, D)
    group = group / group.norm(dim=-1, keepdim=True)
    at = torch.randint(  # needle starts; window 1 draws as before the runs
        SINKS,
        n - queries - window + 1,
        (groups, queries * needles),
        generator=gen,
        device=device,
    )
    needle = (NEEDLE * dim**0.5 * group).repeat_interleave(needles * window, 1)
    if window == 1:
        k.scatter_(1, at[..., None].expand(-1, -1, dim), needle)
    else:  # a run of `window` keys from each start, the needle added to their noise
        at = (at[..., None] + torch.arange(window, device=device)).flatten(1)
        k.scatter_add_(1, at[..., None].expand(-1, -1, dim), needle)
    return q * q_scale, k, v


def sequence(
    task: str, n: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """(n,) token ids and (n,) whether each token is an answer."""

    def draw(vocab: range, count: int) -> torch.Tensor:
        return torch.randint(vocab.start, vocab.stop, (count,), generator=gen)

    def distinct(vocab: range, count: int) -> torch.Tensor:
        return vocab.start + torch.randperm(len(vocab), generator=gen)[:count]

    answer = torch.zeros(n, dtype=torch.bool)
    if task == "copy":
        half = draw(FILLER, n // 2)
        ids = torch.cat([half, half, draw(FILLER, n % 2)])
        answer[n // 2 : 2 * (n // 2)] = True
        return ids, answer
    pairs = PAIRS if task == "mqar" else NEEDLE_PAIRS
    asked = QUERIES if task == "mqar" else NEEDLE_PAIRS
    keys, values = distinct(KEYS, pairs), draw(VALUES, pairs)
    kv = torch.stack([keys, values], 1)  # (pairs, 2)
    ids = draw(FILLER, n)
    if task == "mqar":
        ids[: 2 * pairs] = kv.flatten()
    else:
        assert task == "needle"
        slots = torch.randperm((n - 2 * asked) // 2, generator=gen)[:pairs] * 2
        ids[slots], ids[slots + 1] = keys, values
    pick = torch.randperm(pairs, generator=gen)[:asked]
    ids[n - 2 * asked :] = kv[pick].flatten()
    answer[n - 2 * asked + 1 :: 2] = True
    return ids, answer
