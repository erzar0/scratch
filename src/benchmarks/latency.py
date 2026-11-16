"""Latency: one decode step of each version's fast implementation on the GPU.

Batch 1, one new token; `kv_heads` KV heads of `query_heads` (default 2) query
heads, head dim `dim` (default 128), bf16 K and V from synthetic.qkv. Each budget
in `budgets` (default KEEP) runs where the length is at least MIN_RATIO times it.
Timed by common.timing (CUDA-graph replays, medians of `repeats`); state kept at
cache-write time is built before the timing.
"""

from functools import partial
from typing import Any

import torch

from attention.fast_pytorch import KV
from attention.registry import KEEP, MIN_RATIO, Version, versions
from benchmarks import synthetic
from common.timing import time_gpu


@torch.no_grad()
def run(
    version: Version, config: dict[str, Any], device: str = "cuda"
) -> list[dict[str, Any]]:
    """One row per distribution, length and budget: µs (median, min, max), peak GiB."""
    budgets = {
        keep: versions(keep)[version.id] for keep in config.get("budgets", [KEEP])
    }
    rows = []
    for distribution in config["distributions"]:
        for n in config["lengths"]:
            q, k, v = synthetic.qkv(
                distribution,
                n,
                1,
                0,
                groups=config["kv_heads"],
                size=config.get("query_heads", 2),
                dim=config.get("dim", 128),
                device=device,
            )
            q = q[:, :, 0].bfloat16()
            kv = KV(k.bfloat16(), v.bfloat16())
            del k, v
            for keep, at in budgets.items():
                if n < MIN_RATIO * keep:
                    continue
                timing = time_gpu(partial(at.fast, q, kv), config["repeats"])
                where = {"distribution": distribution, "n": n, "keep": keep}
                rows.append(where | timing._asdict())
                print(f"  {distribution} {n} keep {keep}: {timing.median_us:.1f} µs")
            del kv
            torch.cuda.empty_cache()
    return rows
