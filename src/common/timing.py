"""GPU timing: triton.testing.do_bench_cudagraph repeats, OOM -> NaN, peak memory.

Under WSL the idle GPU clocks down and its clock wanders, so a 0.3 s warm-up runs
first and the timing is repeated: each repeat is the median of CUDA-graph replays
(one decode step per replay, as serving stacks run them).
"""

import time
from collections.abc import Callable
from typing import NamedTuple

import torch
import triton.testing


class Timing(NamedTuple):
    """µs per call over repeats (median, min, max of the repeats' medians), and peak
    GiB above what was allocated before; NaN µs on OOM and when the peak exceeds GPU
    memory (WSL then pages to host memory instead of failing)."""

    median_us: float
    min_us: float
    max_us: float
    peak_gib: float


def time_gpu(fn: Callable[[], object], repeats: int = 3, rep_ms: int = 100) -> Timing:
    """Time fn() on the GPU; its first call (compilation) is not timed."""
    nan = float("nan")
    try:
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        if peak * 2**30 > torch.cuda.get_device_properties().total_memory:
            return Timing(nan, nan, nan, peak)
        end = time.perf_counter() + 0.3
        while time.perf_counter() < end:
            fn()
            torch.cuda.synchronize()
        runs = [
            1000
            * triton.testing.do_bench_cudagraph(fn, rep=rep_ms, return_mode="median")
            for _ in range(repeats)
        ]
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return Timing(nan, nan, nan, nan)
    runs.sort()
    return Timing(runs[len(runs) // 2], runs[0], runs[-1], peak)
