"""Training cost of CSA2's K=V window (csa2_window) vs record 91's own window K, V:
one layer's window build + csa2_flex.attention, forward and backward, compiled
(fullgraph), random bf16 tensors at stage 1 and 3 row lengths (16K, 48K tokens per
row with SWEEP_BATCH_DIV=8), 2048-token documents, exact top 64 picks made once.
Own K, V: the k half of the qk projection + the v projection (768 -> 6 x 128);
K=V windows: shared_window's G x 128 projection (the unused k half of the qk
projection is not timed in either); GQA: the 1 window head read by every query head
in place (csa2_flex_gqa) instead of copied to 6 heads. ms and peak MiB per option
-> runs/nanogpt/training/train_window.csv (stage, version, ms, peak_mib).
    PYTHONPATH=src python experiments/csa2/bench/train_window.py
"""

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from nanogpt import csa2_flex, csa2_flex_gqa, csa2_select, csa2_window

H, D, C, M, DOC = 6, 128, 768, 4, 2048
BF = torch.bfloat16
OUT = Path("runs/nanogpt/training/train_window.csv")


def tables(length: int, width: int = 96) -> tuple[torch.Tensor, torch.Tensor]:
    """record 91's Yarn factor1, factor2 (half-truncated RoPE)."""
    freq = (1 / 1024) ** torch.linspace(0, 1, width // 4)
    freq = torch.cat([freq.repeat_interleave(2), torch.zeros(width // 2)])
    theta = torch.outer(torch.arange(length).float(), freq)
    f2 = theta.sin()
    f2[..., 1::2] *= -1
    return theta.cos().cuda().to(BF), f2.cuda().to(BF)


def own(x, w, q, k, keys, values, picks, cu, f1, f2):
    v = F.linear(x, w[0]).view(-1, H, D)
    return csa2_flex.attention(q, k, v, keys, values, *picks, cu, D**-0.5, M)


def shared(x, w, q, k, keys, values, picks, cu, f1, f2):
    kw, vw = csa2_window.shared_window(x, w, H, f1, f2)
    return csa2_flex.attention(q, kw, vw, keys, values, *picks, cu, D**-0.5, M)


def gqa(x, w, q, k, keys, values, picks, cu, f1, f2):
    kw, vw = csa2_window.shared_window(x, w, 1, f1, f2)
    return csa2_flex_gqa.attention(q, kw, vw, keys, values, *picks, cu, D**-0.5, M)


OPTIONS = {
    "record 91 own K V (6 heads)": (own, 6),
    "1 shared K=V window head": (shared, 1),
    "1 shared K=V window head, read in place (GQA)": (gqa, 1),
    "2 shared K=V window heads": (shared, 2),
    "per-head K=V window": (shared, 6),
}

if __name__ == "__main__":
    rows = []
    for stage, length in (("stage 1", 16384), ("stage 3", 49152)):
        torch.manual_seed(0)
        f1, f2 = tables(length)
        x = torch.randn(length, C, device="cuda", dtype=BF, requires_grad=True)
        q, k = (
            torch.randn(length, H, D, device="cuda", dtype=BF, requires_grad=True)
            for _ in range(2)
        )
        values = torch.randn(length // M, D, device="cuda", dtype=BF)
        keys = csa2_flex.rotate(values, M, f1.float(), f2.float()).to(BF)
        cu = torch.arange(0, length + 1, DOC, device="cuda", dtype=torch.int32)
        with torch.no_grad():
            picks = csa2_select.exact(q, keys, csa2_flex.starts(cu, length), M, 64)
        base = None
        for name, (fn, groups) in OPTIONS.items():
            dim, count = (H * D, 1) if fn is own else (D, groups)
            init = (torch.randn(dim, C, device="cuda", dtype=BF) for _ in range(count))
            w = tuple(t.div(C**0.5).requires_grad_() for t in init)
            run = torch.compile(fn, fullgraph=True, dynamic=False)
            grad = torch.randn(length, H, D, device="cuda", dtype=BF)

            def step() -> None:
                out = run(x, w, q, k, keys, values, picks, cu, f1, f2)  # noqa: B023
                out.backward(grad)  # noqa: B023

            step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            step()
            peak = (torch.cuda.max_memory_allocated() - before) / 2**20
            bench = triton.testing.do_bench
            ms = min(bench(step, return_mode="median") for _ in range(3))
            base = base or ms
            print(f"{stage} | {name} | {ms:.2f} ms | x{ms / base:.2f} | {peak:.0f} MiB",
                  flush=True)  # fmt: skip
            rows.append((stage, name, round(ms, 2), round(peak)))
            for t in (x, q, k, *w):
                t.grad = None
    with OUT.open("w", newline="") as f:
        csv.writer(f).writerows([("stage", "version", "ms", "peak_mib"), *rows])
    print("wrote", OUT)
