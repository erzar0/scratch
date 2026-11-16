"""csa2_attend_split.csa2_attend on tensor cores: the query heads share their one
K = V head, so a program scores HG heads at once (tl.dot) instead of one head per
program re-reading every row (as FlashMLA's MQA decode).

Program (s, g, p) of grid (row chunks, head groups, output slices): R rows (a window
chunk, s < W / R, or a chunk of the picks), heads g HG .. g HG + HG - 1, output
channels p DV .. p DV + DV - 1. Logits over all D channels (the slices' programs
repeat them: FlashMLA's 2 x 256 split of 512), rounded to bf16 as there; the
probabilities are rounded to bf16 for the value product (tl.dot), unlike there.
It stores its partial (row max, exp sum, unnormalized output); the last program of
(g, p) to arrive (atomic counter, left zero) combines the S partials, adds the sink
and turns the last ROPE channels back (inverse RoPE, as csa2_attend_split).
FP8 rows are converted to bf16 before tl.dot (this GPU's Triton: no FP8 x bf16 dot).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

ROWS, HEADS, SPLIT, WARPS = 32, 16, 2, 4  # rows, heads, slices per program; warps
_counts: dict[torch.device, Tensor] = {}


@triton.jit
def _partial(o_ptr, part, hh, cv, DV: tl.constexpr, HG: tl.constexpr):
    """(HG, DV) unnormalized output of partial `part` at slice channels cv."""
    return tl.load(o_ptr + (part * HG + hh)[:, None] * DV + cv[None, :],
                   cache_modifier=".cg")  # fmt: skip


@triton.jit
def _csa2_mqa(
    q_ptr,
    win_ptr,
    kv_ptr,
    pick_ptr,
    sink_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    m_ptr,
    l_ptr,
    o_ptr,
    count_ptr,
    scale,
    D: tl.constexpr,
    W: tl.constexpr,
    ROPE: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    SP: tl.constexpr,
    HG: tl.constexpr,
    DV: tl.constexpr,
    BD: tl.constexpr,
):
    s, g, p = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hh = tl.arange(0, HG)
    heads = g * HG + hh
    r = tl.arange(0, R)
    if s < W // R:
        rows = win_ptr + (s * R + r).to(tl.int64)[:, None] * D
    else:
        picks = tl.load(pick_ptr + (s - W // R) * R + r)
        rows = kv_ptr + picks.to(tl.int64)[:, None] * D
    acc = tl.zeros((HG, R), dtype=tl.float32)
    for d0 in tl.static_range(0, D, BD):
        cd = d0 + tl.arange(0, BD)
        qt = tl.load(q_ptr + heads[:, None] * D + cd[None, :])
        kt = tl.load(rows + cd[None, :]).to(tl.float32).to(tl.bfloat16)
        acc = tl.dot(qt, tl.trans(kt), acc)
    logit = acc.to(tl.bfloat16).to(tl.float32) * scale
    m = tl.max(logit, 1)
    e = tl.exp(logit - m[:, None])
    cv = p * DV + tl.arange(0, DV)
    vt = tl.load(rows + cv[None, :]).to(tl.float32).to(tl.bfloat16)
    o = tl.dot(e.to(tl.bfloat16), vt)
    slot = g * (D // DV) + p
    part = slot * S + s
    tl.store(o_ptr + (part * HG + hh)[:, None] * DV + tl.arange(0, DV)[None, :], o)
    tl.store(m_ptr + part * HG + hh, m)
    tl.store(l_ptr + part * HG + hh, tl.sum(e, 1))
    tl.debug_barrier()
    if tl.atomic_add(count_ptr + slot, 1) == S - 1:
        j = tl.arange(0, SP)
        at = (slot * S + j)[:, None] * HG + hh[None, :]
        ok = (j < S)[:, None]
        ms = tl.load(m_ptr + at, ok, other=float("-inf"), cache_modifier=".cg")
        sink = tl.load(sink_ptr + heads)
        top = tl.maximum(tl.max(ms, 0), sink)
        total = tl.exp(sink - top)
        rot = cv >= D - ROPE
        half = ROPE // 2
        k = cv - (D - ROPE)
        partner = tl.where(rot, D - ROPE + (k + half) % ROPE, cv) - p * DV
        own = tl.arange(0, DV)
        out = tl.zeros((HG, DV), dtype=tl.float32)
        turned = tl.zeros((HG, DV), dtype=tl.float32)
        for i in range(S):
            mi = tl.load(m_ptr + (slot * S + i) * HG + hh, cache_modifier=".cg")
            li = tl.load(l_ptr + (slot * S + i) * HG + hh, cache_modifier=".cg")
            w = tl.exp(mi - top)
            total += li * w
            out += w[:, None] * _partial(o_ptr, slot * S + i, hh, own, DV, HG)
            turned += w[:, None] * _partial(o_ptr, slot * S + i, hh, partner, DV, HG)
        tl.store(count_ptr + slot, 0)
        out = (out / total[:, None]).to(tl.bfloat16).to(tl.float32)
        turned = (turned / total[:, None]).to(tl.bfloat16).to(tl.float32)
        cos = tl.load(cos_ptr + k, rot, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + k, rot, other=0.0).to(tl.float32)
        sign = tl.where(k < half, -1.0, 1.0)
        out = tl.where(rot[None, :], out * cos - sign * turned * sin, out)
        tl.store(out_ptr + heads[:, None] * D + cv[None, :], out.to(tl.bfloat16))


def csa2_attend(
    q: Tensor,
    window_kv: Tensor,
    main_kv: Tensor,
    picks: Tensor,
    sinks: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """csa2_attend_split.csa2_attend (q (H, D) bf16, rows bf16 or FP8), ROWS rows,
    HEADS heads and D / SPLIT output channels per program."""
    heads, dim = q.shape
    w, top = window_kv.shape[0], picks.shape[0]
    assert w % ROWS == 0 and top % ROWS == 0 and heads % HEADS == 0
    chunks = (w + top) // ROWS
    slots = heads // HEADS * SPLIT
    device = q.device
    if device not in _counts:
        _counts[device] = torch.zeros(64, dtype=torch.int32, device=device)
    assert slots <= 64
    m = torch.empty(slots * chunks * HEADS, dtype=torch.float32, device=device)
    lse = torch.empty_like(m)
    o = torch.empty(chunks * heads * dim, dtype=torch.float32, device=device)
    out = torch.empty_like(q)
    _csa2_mqa[(chunks, heads // HEADS, SPLIT)](
        q, window_kv, main_kv, picks, sinks, cos, sin, out, m, lse, o,
        _counts[device], dim**-0.5, D=dim, W=w, ROPE=cos.shape[0], R=ROWS,
        S=chunks, SP=triton.next_power_of_2(chunks), HG=HEADS, DV=dim // SPLIT,
        BD=64, num_warps=WARPS,
    )  # fmt: skip
    return out
