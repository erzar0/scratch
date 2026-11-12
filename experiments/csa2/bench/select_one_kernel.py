# ruff: noqa: N803, B023  (Triton constexpr names; lambdas run inside their loop)
"""One-kernel top 64 by score bins, v2 of the scratchpad select_hist.py (correct but
slower than csa2_decode past 8K entries: per-score atomics into hot bins, and
listing in 64-row steps). Here each program bins its block into a local histogram
(one atomic per nonzero bin), then after a grid barrier (at most 64 programs, all
resident on 66 SMs) lists its whole block at once; the prefilter rescores its kept
rows exactly in chunks of RESCORE. The last program takes the top 64 of the list.
Checked against csa2_decode.top64 / prefilter64 (same sets), timed against them.

    PYTHONPATH=src python experiments/csa2/bench/select_one_kernel.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent))
from decode_csa2 import graphed_ms  # noqa: E402

from nanogpt import csa2_decode as cd  # noqa: E402
from nanogpt.csa2_decode import LEAST, _bin, _pack  # noqa: E402

MOST = 64  # programs: all resident at once (66 SMs), or the barrier deadlocks
BINS = 2048
CHUNK = 512  # list entries per final topk step
RESCORE = 32  # prefilter: kept rows rescored exactly per step


@triton.jit
def _score(q_ptr, w_ptr, k_ptr, rows, ok, scale, cl, qb, D: tl.constexpr,
           C: tl.constexpr, H: tl.constexpr, L: tl.constexpr):  # fmt: skip
    """Scores of rows: L > 0 the prefilter (q's block cl, qb), else exact (H == 0)
    or CSA2's indexer over H heads."""
    if L > 0:
        at = k_ptr + rows[:, None].to(tl.int64) * D + cl[None, :]
        k = tl.load(at, ok[:, None], other=0.0).to(tl.float32)
        score = tl.sum(k * qb[None, :], 1)
    else:
        c = tl.arange(0, D)
        at = k_ptr + rows[:, None].to(tl.int64) * D + c[None, :]
        if C < D:
            k = tl.load(at, ok[:, None] & (c < C)[None, :], other=0.0)
        else:
            k = tl.load(at, ok[:, None], other=0.0)
        k = k.to(tl.float32)
        if H == 0:
            score = tl.sum(k * tl.load(q_ptr + c).to(tl.float32)[None, :], 1)
        else:
            score = tl.zeros(rows.shape, tl.float32)
            for h in tl.static_range(H):
                qh = tl.load(q_ptr + h * D + c).to(tl.float32)
                dot = tl.sum(k * qh[None, :], 1)
                score += tl.maximum(dot * scale, 0.0) * tl.load(w_ptr + h)
    return score


@triton.jit
def _select(q_ptr, w_ptr, k_ptr, score_ptr, hist_ptr, sync_ptr, list_ptr, rows_ptr,
            pick_ptr, N, KEEP, scale, BLOCK: tl.constexpr, D: tl.constexpr,
            C: tl.constexpr, H: tl.constexpr, L: tl.constexpr, SUB: tl.constexpr,
            BINS: tl.constexpr, TOP: tl.constexpr, CHUNK: tl.constexpr,
            RESCORE: tl.constexpr):  # fmt: skip
    """sync: arrivals, finished programs, list length, unused."""
    p = tl.program_id(0)
    programs = tl.num_programs(0)
    cl = tl.arange(0, 16)
    qb = tl.zeros((16,), tl.float32)
    if L > 0:
        c = tl.arange(0, D)
        q = tl.abs(tl.load(q_ptr + c).to(tl.float32))
        cl = tl.argmax(tl.sum(tl.reshape(q, (D // L, L)), 1), 0) * L + tl.arange(0, L)
        qb = tl.load(q_ptr + cl).to(tl.float32)
    local = tl.zeros((BINS,), tl.int32)
    for s in range(0, BLOCK, SUB):
        rows = p * BLOCK + s + tl.arange(0, SUB)
        ok = rows < N
        score = _score(q_ptr, w_ptr, k_ptr, rows, ok, scale, cl, qb, D, C, H, L)
        tl.store(score_ptr + rows, score, ok)
        local += tl.histogram(_bin(score), BINS, mask=ok)
    tl.atomic_add(hist_ptr + tl.arange(0, BINS), local, mask=local > 0, sem="relaxed")
    # grid barrier: sync[0] counts arrivals
    tl.debug_barrier()
    seen = tl.atomic_add(sync_ptr, 1, sem="acq_rel") + 1
    while seen < programs:
        seen = tl.atomic_add(sync_ptr, 0, sem="acquire")
    hist = tl.load(hist_ptr + tl.arange(0, BINS), cache_modifier=".cg")
    at_least = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
    cut = tl.max(tl.where(at_least >= KEEP, tl.arange(0, BINS), -1), 0)
    rows = p * BLOCK + tl.arange(0, BLOCK)
    ok = rows < N
    score = tl.load(score_ptr + rows, ok, other=0.0)
    kept = ok & (_bin(score) >= cut)
    n = tl.sum(kept.to(tl.int32), 0)
    if n > 0:
        base = tl.atomic_add(sync_ptr + 2, n, sem="relaxed")
        slot = base + tl.cumsum(kept.to(tl.int32), 0) - 1
        if L > 0:  # kept rows listed, then rescored exactly: only they are read
            tl.store(rows_ptr + slot, rows, kept)
            tl.debug_barrier()
            for s in range(0, n, RESCORE):
                j = s + tl.arange(0, RESCORE)
                okj = j < n
                r = tl.load(rows_ptr + base + j, okj, other=0)
                sc = _score(q_ptr, w_ptr, k_ptr, r, okj, scale, cl, qb, D, C, 0, 0)
                tl.store(list_ptr + base + j, _pack(sc, r), okj)
        else:
            tl.store(list_ptr + slot, _pack(score, rows), kept)
    tl.debug_barrier()
    if tl.atomic_add(sync_ptr + 1, 1, sem="acq_rel") == programs - 1:
        total = tl.atomic_add(sync_ptr + 2, 0, sem="acquire")
        best = tl.full((TOP,), LEAST, tl.int64)
        for s in range(0, total, CHUNK):
            j = s + tl.arange(0, CHUNK)
            x = tl.load(list_ptr + j, j < total, other=LEAST, cache_modifier=".cg")
            if CHUNK > TOP:
                x = tl.topk(x, TOP)
            best = tl.topk(tl.cat(best, x, can_reorder=True), TOP)
        tl.store(pick_ptr + tl.arange(0, TOP), 0x7FFFFFFF - (best & 0xFFFFFFFF))
        tl.store(sync_ptr + tl.arange(0, 4), tl.zeros((4,), tl.int32))
        tl.store(hist_ptr + tl.arange(0, BINS), tl.zeros((BINS,), tl.int32))


_state: dict = {}


def select(q, keys, weights=None, width=0, keep=64, dims=0, most=MOST, tile=8192):
    """Top 64 entries: exact (keep 64), indexer (weights), or prefilter (dims > 0,
    keep > 64; n <= keep: exact)."""
    n, dim = keys.shape
    if dims and n <= keep:
        dims, keep = 0, 64
    if keys.device not in _state:
        _state[keys.device] = (
            torch.zeros(4, dtype=torch.int32, device=keys.device),
            torch.zeros(BINS, dtype=torch.int32, device=keys.device),
        )
    sync, hist = _state[keys.device]
    block = max(16, triton.next_power_of_2(triton.cdiv(n, most)))
    sub = min(block, max(16, tile // (dims or dim)))
    programs = triton.cdiv(n, block)
    assert programs <= MOST
    score = torch.empty(programs * block, dtype=torch.float32, device=keys.device)
    listed = torch.empty(n, dtype=torch.int64, device=keys.device)
    rows = torch.empty(n if dims else 1, dtype=torch.int32, device=keys.device)
    picks = torch.empty(64, dtype=torch.int64, device=keys.device)
    _select[(programs,)](q, q if weights is None else weights, keys, score, hist, sync,
                         listed, rows, picks, n, keep, dim**-0.5, BLOCK=block, D=dim,
                         C=width or dim, H=0 if weights is None else q.shape[0],
                         L=dims, SUB=sub, BINS=BINS, TOP=64, CHUNK=CHUNK,
                         RESCORE=RESCORE, num_warps=8)  # fmt: skip
    return picks


if __name__ == "__main__":
    torch.manual_seed(0)
    for n in (2048, 8192, 32768, 131072):
        keys = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16)
        ik = torch.randn(n, 32, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(128, device="cuda")
        q96 = q * (torch.arange(128, device="cuda") < 96)
        qi = torch.randn(6, 32, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(6, device="cuda")
        cases = {
            "B exact (96 ch)": (lambda: cd.top64(q96, keys, width=96),
                                lambda m, t: select(q96, keys, width=96, most=m,
                                                    tile=t)),
            "A exact (128 ch)": (lambda: cd.top64(q, keys),
                                 lambda m, t: select(q, keys, most=m, tile=t)),
            "prefilter 512": (lambda: cd.prefilter64(q, keys, 512),
                              lambda m, t: select(q, keys, keep=512, dims=16,
                                                  most=m, tile=t)),
            "CSA2 indexer": (lambda: cd.top64(qi, ik, w),
                             lambda m, t: select(qi, ik, w, most=m, tile=t)),
        }  # fmt: skip
        for name, (old, new) in cases.items():
            want = set(old().tolist())
            for _ in range(3):  # buffers left zero: repeat calls agree
                got = set(new(MOST, 8192).tolist())
                assert got == want, (name, n, len(got & want))
            assert not _state[keys.device][0].any() and not _state[keys.device][1].any()
            line = f"{n:>6} {name:<17} current {graphed_ms(old) * 1e3:6.1f} us |"
            for m, t in ((32, 8192), (64, 8192), (64, 16384)):
                ms = graphed_ms(lambda: new(m, t))  # noqa: B023
                line += f" {m} prog tile {t} {ms * 1e3:6.1f}"
            print(line, flush=True)
