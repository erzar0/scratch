"""Shared GPU timer.

Under WSL the idle GPU clocks down to 210 MHz and a sync costs ~1 ms, so: warm up
~0.3 s, queue ~50 ms of calls per timing with one sync at the end, and report the
minimum over repeats (the least disturbed run). Decode steps are timed as one CUDA
graph replay (graphed), as serving stacks run them: a per-kernel launch costs ~10 µs
under WSL and would hide the GPU work.
"""

import time
from collections.abc import Callable
from typing import NamedTuple

import torch


class Timing(NamedTuple):
    """ms per call and peak memory above what was allocated before; ms is NaN on OOM
    and when the peak exceeds GPU memory (WSL then pages to host memory, ~100x
    slower, instead of failing)."""

    ms: float
    peak_gib: float


def graphed(fn: Callable[..., object], *args: object) -> Callable[[], object]:
    """fn(*args) captured as one CUDA graph on these very tensors; call the result to
    replay it. Warm-up calls (compilation) run on a side stream, as capture needs."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args)
    return graph.replay


def measure(fn: Callable[..., object], *args: object, reps: int = 7) -> Timing:
    """Time fn(*args) on the GPU; the first call (compilation) is not timed.

    Args:
        fn: Function to time, e.g. a torch.compile'd one.
        *args: Its arguments.
        reps: Timing repeats; the minimum is reported.

    Returns:
        ms per call and peak GiB; NaN ms if it does not fit in GPU memory.
    """
    try:
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn(*args)
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        if peak * 2**30 > torch.cuda.get_device_properties().total_memory:
            return Timing(float("nan"), peak)
        start = time.perf_counter()
        while time.perf_counter() - start < 0.3:
            fn(*args)
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        calls = max(20, int(0.05 / (time.perf_counter() - start)))
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            for _ in range(calls):
                fn(*args)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) / calls)
    except (torch.OutOfMemoryError, torch._dynamo.exc.BackendCompilerFailed) as error:
        # torch.compile wraps an OOM while it benchmarks kernels
        inner = getattr(error, "inner_exception", error)
        if not isinstance(inner, torch.OutOfMemoryError):
            raise
        torch.cuda.empty_cache()
        return Timing(float("nan"), float("nan"))
    return Timing(min(times) * 1e3, peak)
