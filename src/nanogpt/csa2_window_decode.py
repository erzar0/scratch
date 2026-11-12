"""Decode attention with CSA2's window (csa2_window): G shared K=V vectors per window
token instead of each head's own K and V; head h reads vector h // (H / G). The
window is cached like the entries: keys stored rotated, a value is its key turned
back on the first 64 channels in pairs by its row of a pair_table (csa2_decode).
Split as csa2_attend_split (flash-decoding, 1 warp): program (h, s) takes R window
rows or R picks, the head's last program combines.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from nanogpt import csa2_attend_split as split
from nanogpt.csa2_attend_split import _arrive, _combined, _weights

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)


@triton.jit
def _kv_split(
    q_ptr,
    qp_ptr,
    win_ptr,
    wtab_ptr,
    keys_ptr,
    pick_ptr,
    table_ptr,
    out_ptr,
    m_ptr,
    l_ptr,
    o_ptr,
    count_ptr,
    key_row,
    key_block,
    scale,
    G: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    SP: tl.constexpr,
):
    """Window chunks (q, the head's group of win) or pick chunks (q_pick, keys);
    both K=V, values turned back by their table rows, rounded to bf16."""
    h, s = tl.program_id(0), tl.program_id(1)
    a = tl.arange(0, 64)
    p = tl.arange(0, 32)
    r = tl.arange(0, R)
    if s < W // R:
        rows = (s * R + r).to(tl.int64)
        row = win_ptr + rows[:, None] * (G * 128) + (h // (H // G)) * 128 + a[None, :]
        lo = tl.load(row).to(tl.float32)  # window and entries may differ in dtype
        hi = tl.load(row + 64).to(tl.float32)
        tab = wtab_ptr + rows[:, None] * 64 + p[None, :]
        qv = q_ptr + h * 128
    else:
        picks = tl.load(pick_ptr + (s - W // R) * R + r)
        key = keys_ptr + picks[:, None] * key_row + (a % 16)[None, :]
        lo = tl.load(key + (a // 16 * key_block)[None, :]).to(tl.float32)
        hi = tl.load(key + ((a // 16 + 4) * key_block)[None, :]).to(tl.float32)
        tab = table_ptr + picks[:, None] * 64 + p[None, :]
        qv = qp_ptr + h * 128
    cos = tl.load(tab).to(tl.float32)
    sin = tl.load(tab + 32).to(tl.float32)
    q_lo = tl.load(qv + a).to(tl.float32)
    q_hi = tl.load(qv + 64 + a).to(tl.float32)
    logit = (tl.sum(lo * q_lo[None, :], 1) + tl.sum(hi * q_hi[None, :], 1)) * scale
    m = tl.max(logit, 0)
    e = tl.exp(logit - m)
    x0, x1 = tl.split(tl.reshape(lo, (R, 32, 2)))
    v0 = (cos * x0 - sin * x1).to(tl.bfloat16).to(tl.float32)
    v1 = (cos * x1 + sin * x0).to(tl.bfloat16).to(tl.float32)
    o0 = tl.sum(e[:, None] * v0, 0)
    o1 = tl.sum(e[:, None] * v1, 0)
    o = o_ptr + (h * S + s) * 128
    tl.store(o + a, tl.reshape(tl.join(o0, o1), (64,)))
    tl.store(o + 64 + a, tl.sum(e[:, None] * hi, 0))
    if _arrive(m, tl.sum(e, 0), m_ptr, l_ptr, count_ptr, S):
        w, total, _ = _weights(m_ptr, l_ptr, count_ptr, float("-inf"), S, SP)
        c = tl.arange(0, 128)
        out = _combined(o_ptr, w, c, 128, S, SP) / total
        tl.store(out_ptr + h * 128 + c, out.to(tl.bfloat16))


def kv_attend(
    q: Tensor,
    win: Tensor,
    win_table: Tensor,
    keys: Tensor,
    picks: Tensor,
    table: Tensor,
    scale: float,
    q_pick: Tensor | None = None,
    rows: int = 16,
) -> Tensor:
    """One token's attention, q (H, 128), over its window win (W, G, 128) (rotated
    keys, rows win_table (W, 64) of pair_table) and keys[picks] (keys (N, 128) or
    block-major (8, N, 16), rows table (N, 64)); q_pick: the picks' query."""
    heads, dim = q.shape
    w, groups = win.shape[0], win.shape[1]
    assert dim == 128 and win.shape[2] == 128 and heads % groups == 0
    assert win.is_contiguous() and w % rows == 0 and picks.shape[0] % rows == 0
    key_row, key_block = (16, keys.shape[1] * 16) if keys.dim() == 3 else (128, 16)
    chunks = (w + picks.shape[0]) // rows
    x = split._scratch(heads, chunks, dim, q.device)
    out = torch.empty_like(q)
    _kv_split[(heads, chunks)](
        q, q if q_pick is None else q_pick, win, win_table, keys, picks, table, out,
        x["m"], x["l"], x["o"], x["count"], key_row, key_block, scale,
        G=groups, H=heads, W=w, R=rows, S=chunks,
        SP=triton.next_power_of_2(chunks), num_warps=split.WARPS,
    )  # fmt: skip
    return out
