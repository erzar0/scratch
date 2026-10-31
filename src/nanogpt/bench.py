"""Training cost of record 91's 10 attention layers: CSA2 as train_csa2.py builds them
(and with transformers' dense masked attention, nanogpt/csa2_library.py)
against FlashAttention 2 layers (qkv and output projections around the varlen kernel
record 91 calls, without its extras: a lower bound for dense). One step is forward and
backward of the stack on one packed row, bf16, compiled as training compiles the model.

speed [STACK]  forward and forward+backward ms and peak memory of each stack (or
         only STACK, replacing its rows): training rows
         (2048/4096/6144 tokens of FineWeb val, documents cut at 2048) and single
         documents of 8K-32K tokens -> runs/nanogpt/bench.csv
profile  where CSA2's time goes at a 6144-token training row: eager, by component
         (backward kernels attributed to the forward op that made them, by autograd
         sequence number); compiled, by kernel kind -> runs/nanogpt/bench_profile.csv

PYTHONPATH=src ~/repos/modded-nanogpt-r91/.venv/bin/python -m nanogpt.bench STUDY
"""

import re
import sys
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import torch
from flash_attn.flash_attn_interface import (  # type: ignore[import-not-found]
    flash_attn_varlen_func,
)
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from bench.timing import measure
from deepseek import csa2
from experiments.results import Results
from nanogpt import csa2_varlen as cv
from nanogpt import model as nm
from nanogpt.csa2_library import LibraryCsa2Attention
from nanogpt.probe import SIZES

RUNS = Path("runs/nanogpt")
D_MODEL, HEADS, HEAD_DIM, LAYERS = 768, 6, 128, 10
BOS, CUT = 50256, 2048  # record 91's document separator and training document cut
TRAINING_ROWS = (2048, 4096, 6144)
# 64K: out of memory inside torch.compile (not caught by measure); CSA2 training
# steps already fail at 16K within 70% of the 16 GB card
LONG_ROWS = (8192, 16384, 32768)
# record 91's final windows: ws_short 6 and ws_long 13 blocks of 128, in its layer order
WINDOWS = (768, 768, 768, 1664, 768, 768, 768, 768, 768, 1664)
# stage 3 ms per step, CSA2 minus stripped dense (8 rows of 6144 per step)
STEP_GAP_MS, ROWS_PER_STEP = 1505 - 838, 8


class Csa2Stack(nn.Module):
    """train_csa2.py's 10 CSA2 layers on a residual stream."""

    def __init__(self, layer: type[cv.VarlenCsa2Attention]) -> None:
        super().__init__()
        modes = ((csa2.FULL, csa2.REUSE, csa2.REINDEX, csa2.REUSE) * 3)[:LAYERS]
        config = csa2.small_config(modes, **SIZES)
        self.layers = nn.ModuleList(layer(config, i) for i in range(LAYERS))

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        shared = csa2.SharedState()
        for layer in self.layers:
            with torch.autocast("cuda", torch.bfloat16):
                x = x + layer(x, cu_seqlens, shared)
        return x


class FlashStack(nn.Module):
    """10 layers of qkv projection, FA2 varlen causal attention, output projection;
    window None = full causal."""

    def __init__(self, windows: tuple[int | None, ...]) -> None:
        super().__init__()
        self.windows = windows
        width = HEADS * HEAD_DIM
        self.qkv = nn.ModuleList(nn.Linear(D_MODEL, 3 * width, False) for _ in windows)
        self.out = nn.ModuleList(nn.Linear(width, D_MODEL, False) for _ in windows)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        longest = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        for qkv, out, window in zip(self.qkv, self.out, self.windows, strict=True):
            q, k, v = qkv(x[0]).view(length, 3, HEADS, HEAD_DIM).unbind(1)
            y = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                longest,
                longest,
                causal=True,
                window_size=(-1 if window is None else window, 0),
            )
            x = x + out(y.reshape(length, -1))[None]
        return x


def stacks() -> dict[str, nn.Module]:
    """Every timed stack, on the GPU, Linear weights bf16 as training casts them."""
    built: dict[str, nn.Module] = {
        "CSA2": Csa2Stack(cv.VarlenCsa2Attention),
        "CSA2, transformers attention": Csa2Stack(LibraryCsa2Attention),
        "flash, record 91 windows": FlashStack(WINDOWS),
        "flash, full causal": FlashStack((None,) * LAYERS),
    }
    for stack in built.values():
        stack.cuda()
        for module in stack.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.bfloat16()
    return built


def training_row(length: int) -> torch.Tensor:
    """cu_seqlens of the first `length` FineWeb val tokens, documents cut at CUT."""
    shard = sorted(nm.FINEWEB.glob("fineweb_val_*.bin"))[0]
    tokens = np.fromfile(shard, dtype=np.uint16, offset=256 * 4, count=length)
    bos = np.flatnonzero(tokens == BOS)
    assert bos[0] == 0, bos[:3]
    ends = np.append(bos[1:], length)
    starts = [s for b, e in zip(bos, ends, strict=True) for s in range(b, e, CUT)]
    return torch.tensor([*starts, length], dtype=torch.int32, device="cuda")


def rows() -> list[tuple[str, int, torch.Tensor]]:
    """(packing, length, cu_seqlens) of every timed row."""
    single = [
        ("one document", t, torch.tensor([0, t], dtype=torch.int32, device="cuda"))
        for t in LONG_ROWS
    ]
    return [("training", t, training_row(t)) for t in TRAINING_ROWS] + single


def inputs(length: int) -> torch.Tensor:
    """A random bf16 residual stream (1, length, D_MODEL) that takes gradients."""
    x = torch.randn(1, length, D_MODEL, device="cuda", dtype=torch.bfloat16)
    return x.requires_grad_()


def train_step(stack: Callable, x: torch.Tensor, cu: torch.Tensor) -> None:
    """Forward and backward, as one training micro-batch runs the attention layers."""
    stack(x, cu).float().square().mean().backward()


def speed(only: str = "") -> None:
    """Every stack at every row; compiled CSA2 also eager. only: just this stack,
    replacing its rows."""
    torch._dynamo.config.cache_size_limit = 64
    # allocations past physical memory fail (NaN) instead of paging to host under WSL
    torch.cuda.set_per_process_memory_fraction(0.7)
    built = stacks()
    timed = {
        name: torch.compile(s, dynamic=False, fullgraph=True)
        for name, s in built.items()
    }
    timed["CSA2, not compiled"] = built["CSA2"]
    if only:
        timed = {only: timed[only]}
    results = Results(RUNS / "bench.csv", {"stack": only} if only else {})
    gap: dict[str, float] = {}
    for packing, length, cu in rows():
        x = inputs(length)
        for name, stack in timed.items():
            forward = measure(stack, x, cu, reps=5)
            step = measure(train_step, stack, x, cu, reps=5)
            row = {"stack": name, "packing": packing, "T": length}
            row |= {"forward_ms": forward.ms, "train_step_ms": step.ms}
            results.add(row | {"peak_gib": step.peak_gib})
            print(row, f"{step.peak_gib:.2f} GiB", flush=True)
            if packing == "training" and length == TRAINING_ROWS[-1]:
                gap[name] = step.ms
            torch.cuda.empty_cache()
    if only:
        return
    extra = (gap["CSA2"] - gap["flash, record 91 windows"]) * ROWS_PER_STEP
    print(
        f"CSA2 minus flash with windows, {ROWS_PER_STEP} rows of {TRAINING_ROWS[-1]}:"
        f" {extra:.0f} ms per step; measured training gap {STEP_GAP_MS} ms"
    )


class Labels:
    """record_function scopes around CSA2's parts, added from outside: forward hooks on
    its submodules, wrappers on csa2_varlen's module-level functions."""

    PARTS = {
        "q projection": ("q_a_proj", "q_a_norm", "q_b_proj", "q_b_norm"),
        "kv projection": ("kv_proj", "kv_norm"),
        "compressor": ("compressor",),
        "indexer": ("indexer",),
        "output projection": ("o_a_proj", "o_b_proj"),
        "RoPE": ("rotary_emb",),
    }
    FUNCTIONS = {
        "sparse attention": (cv, "sparse_attention"),
        "RoPE": (cv, "apply_rotary_pos_emb"),
        "indexer loss": (csa2, "indexer_loss"),
    }

    def __init__(self, stack: Csa2Stack) -> None:
        self.open: list[AbstractContextManager] = []
        for layer in stack.layers:
            for label, names in self.PARTS.items():
                for name in names:
                    module = getattr(layer, name, None)
                    if module is not None:
                        module.register_forward_pre_hook(self.enter(label))
                        module.register_forward_hook(self.leave)
        for label, (owner, name) in self.FUNCTIONS.items():
            setattr(owner, name, self.wrap(label, getattr(owner, name)))

    def enter(self, label: str) -> Callable:
        def hook(*_: object) -> None:
            scope = record_function(label)
            scope.__enter__()
            self.open.append(scope)

        return hook

    def leave(self, *_: object) -> None:
        self.open.pop().__exit__(None, None, None)

    @staticmethod
    def wrap(label: str, fn: Callable) -> Callable:
        def labeled(*args: object, **kwargs: object) -> object:
            with record_function(label):
                return fn(*args, **kwargs)

        return labeled


def profiled(fn: Callable[[], object]) -> profile:
    """fn under the profiler, CPU ops and CUDA kernels."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return prof


def kernel_ms(prof: profile) -> dict[str, float]:
    """Device ms per CUDA kernel name; GPU-side annotation ranges (record_function
    scopes, compiled graph calls) overlap the kernels and are left out."""
    return {
        e.key: e.self_device_time_total / 1e3
        for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA and not e.is_user_annotation
    }


def by_component(stack: Csa2Stack, x: torch.Tensor, cu: torch.Tensor) -> list[dict]:
    """Eager forward and backward device ms per Labels part; the rest is "other"."""
    Labels(stack)
    for _ in range(3):
        train_step(stack, x, cu)
    holder: list[torch.Tensor] = []
    forward = profiled(lambda: holder.append(stack(x, cu).float().square().mean()))
    backward = profiled(lambda: holder[0].backward())
    labels = set(Labels.PARTS) | set(Labels.FUNCTIONS)
    fwd: dict[str, float] = defaultdict(float)
    part_of: dict[int, str] = {}
    for event in forward.events():
        if event.name not in labels:
            continue
        # the scope's own device time also counts its GPU-side annotation range; its
        # child ops hold the kernels (a nested scope counts under its own label)
        children = [c for c in event.cpu_children if c.name not in labels]
        fwd[event.name] += sum(c.device_time_total for c in children) / 1e3
        todo = list(event.cpu_children)
        while todo:
            child = todo.pop()
            if child.sequence_nr >= 0:
                part_of[child.sequence_nr] = event.name
            todo += child.cpu_children
    bwd: dict[str, float] = defaultdict(float)
    for event in backward.events():
        if event.name.startswith("autograd::engine::evaluate_function"):
            part = part_of.get(event.sequence_nr, "other")
            bwd[part] += event.device_time_total / 1e3
    totals = {"forward": sum(kernel_ms(forward).values())}
    totals["backward"] = sum(kernel_ms(backward).values())
    # no kernel counted twice: labeled scopes and backward nodes do not nest
    for name, parts in (("forward", fwd), ("backward", bwd)):
        assert sum(parts.values()) <= 1.02 * totals[name], (name, parts, totals)
    fwd["other"] = totals["forward"] - sum(fwd.values())
    bwd["other"] += totals["backward"] - sum(bwd.values())
    return [
        {"mode": "eager", "pass": name, "part": part, "ms": ms}
        for name, parts in (("forward", fwd), ("backward", bwd))
        for part, ms in parts.items()
    ]


KINDS = (  # first match wins
    ("matmul", r"gemm|gemv|cutlass|xmma|s16816|wgmma|splitK"),
    ("top-k and sort", r"topk|sort|radix|select"),
    # window keys are an unfold of the row, picks an index
    ("gather and scatter", r"index|gather|scatter|embedding|unfold"),
    ("fused reduction (softmax, norms, sums)", r"triton_(red|per)_|softmax|reduce"),
    ("fused elementwise", r"triton_poi_|elementwise|vectorized|copy"),
)


def by_kernel_kind(stack: Callable, x: torch.Tensor, cu: torch.Tensor) -> list[dict]:
    """Compiled forward and backward device ms per KINDS entry."""
    for _ in range(3):
        train_step(stack, x, cu)
    holder: list[torch.Tensor] = []
    passes = {
        "forward": profiled(
            lambda: holder.append(stack(x, cu).float().square().mean())
        ),
        "backward": profiled(lambda: holder[0].backward()),
    }
    out = []
    for name, prof in passes.items():
        kinds: dict[str, float] = defaultdict(float)
        for kernel, ms in kernel_ms(prof).items():
            kind = next((k for k, p in KINDS if re.search(p, kernel)), "other")
            kinds[kind] += ms
        out += [
            {"mode": "compiled", "pass": name, "part": k, "ms": ms}
            for k, ms in kinds.items()
        ]
    return out


def profile_csa2() -> None:
    """Both breakdowns at the last training row."""
    torch._dynamo.config.cache_size_limit = 64
    length = TRAINING_ROWS[-1]
    cu, x = training_row(length), inputs(length)
    rows_out = by_kernel_kind(
        torch.compile(stacks()["CSA2"], dynamic=False, fullgraph=True), x, cu
    )
    eager = stacks()["CSA2"]
    assert isinstance(eager, Csa2Stack)
    rows_out += by_component(eager, x, cu)
    results = Results(RUNS / "bench_profile.csv", {})
    for row in rows_out:
        results.add(row | {"T": length})
    for mode in ("compiled", "eager"):
        part = [r for r in rows_out if r["mode"] == mode]
        total = sum(r["ms"] for r in part)
        print(f"{mode}: {total:.1f} ms forward+backward")
        for r in sorted(part, key=lambda r: -r["ms"]):
            share = r["ms"] / total
            print(f"  {r['pass']:8s} {r['part']:40s} {r['ms']:7.2f} ms {share:5.1%}")


if __name__ == "__main__":
    {"speed": speed, "profile": profile_csa2}[sys.argv[1]](*sys.argv[2:])
