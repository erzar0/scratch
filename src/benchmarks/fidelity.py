"""Fidelity: each version's emulation against dense attention on generated Q/K/V.

Per distribution, length and seed: `queries` decode queries at the last positions,
each over its causal past, measured as the HF hook measures (harness.measure):
output error relative to dense, dense attention mass on the picked keys, recall of
the dense top-128, share of K and V elements read, and 32-byte K sectors read per
layout relative to reading all of K (NaN for versions that do not count them).
Each budget in `budgets` (default KEEP) runs where the length is at least
MIN_RATIO times it; a run out of GPU memory logs NaN.
"""

import math
from typing import Any

import torch

from attention import harness
from attention.registry import KEEP, MIN_RATIO, Version, versions
from benchmarks import synthetic

SECTORS = (*harness.LAYOUTS, "sectors_copies")


def measures(sums: dict[str, float]) -> dict[str, float]:
    """Per-query means of the harness sums."""
    count = sums["count"]
    out = {name: sums[name] / count for name in ("error", "mass", "recall")}
    out["reads"] = sums["reads"] / sums["dense_reads"]
    for name in SECTORS:
        out[name] = sums.get(name, float("nan")) / sums["dense_sectors"]
    return out


def measure(
    emulation: harness.Method, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> dict[str, float]:
    """The harness measures of emulation on queries q at the last positions."""
    queries, n = q.shape[2], k.shape[1]
    scaling = q.shape[-1] ** -0.5
    state = harness.Installed(emulation, queries)
    step = max(1, min(queries, 2**21 // n))  # bounds (G, g, Q, n) memory
    positions = torch.arange(n, device=q.device)
    for s in range(0, queries, step):
        e = min(s + step, queries)
        last = n - queries + e
        valid = positions[None, :last] <= positions[n - queries + s : last, None]
        logits = (q[:, :, s:e] @ k[:, None, :last].mT) * scaling
        logits = logits.masked_fill(~valid, float("-inf"))
        chunk = harness.Chunk(
            q[:, :, s:e], k[:, :last], v[:, :last], logits, valid, 0, scaling
        )
        state.measure(chunk, emulation(chunk))
    return measures(state.sums[0])


@torch.no_grad()
def run(
    version: Version, config: dict[str, Any], device: str = "cuda"
) -> list[dict[str, Any]]:
    """One row per distribution, length, seed and budget; out of memory: NaN."""
    budgets = {
        keep: versions(keep)[version.id] for keep in config.get("budgets", [KEEP])
    }
    rows = []
    for distribution in config["distributions"]:
        for n in config["lengths"]:
            for seed in config["seeds"]:
                q, k, v = synthetic.qkv(
                    distribution, n, config["queries"], seed, device=device
                )
                for keep, at in budgets.items():
                    if n < MIN_RATIO * keep:
                        continue
                    assert at.emulation is not None, "latency only"
                    try:
                        values = measure(at.emulation, q, k, v)
                    except torch.OutOfMemoryError:
                        values = dict.fromkeys(("error", "mass", "recall"), math.nan)
                        torch.cuda.empty_cache()
                    where = {"distribution": distribution, "n": n, "seed": seed}
                    rows.append(where | {"keep": keep} | values)
                    print(
                        f"  {distribution} {n} seed {seed} keep {keep}:"
                        f" error {values['error']:.4f}",
                        flush=True,
                    )
                del q, k, v
    return rows
