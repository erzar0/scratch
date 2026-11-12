"""csa2_decode's two decode attentions with each head's rows split over programs
(flash-decoding), 1 warp per program: 10 layers at one token, CSA2's 42 -> 31 us,
ours 55 -> 38 (6 programs of 4 warps leave the GPU idle; 4 warps per small chunk
cost more than they add).

Program (h, s) of grid (heads, S): head h over R rows, a window chunk (s < W / R)
or a chunk of the picks. It stores its partial (row max, exp sum, unnormalized
output); the head's last program to arrive (atomic counter, left zero) combines
the S partials.
csa2_attend: = csa2_decode.csa2_attend (sink and inverse RoPE in the combine); the
probabilities are not rounded to bf16 before the values, as there.
b_attend: = csa2_decode.b_attend (row- or block-major keys, q_pick).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

WARPS = 1  # per program; rows per program: csa2_attend 32, b_attend 16 (fastest)
_counts: dict[torch.device, Tensor] = {}


@triton.jit
def _arrive(m, total, m_ptr, l_ptr, count_ptr, S: tl.constexpr):
    """Store program (h, s)'s max and exp sum (its output stored before); True for
    the head's last program."""
    h, s = tl.program_id(0), tl.program_id(1)
    tl.store(m_ptr + h * S + s, m)
    tl.store(l_ptr + h * S + s, total)
    tl.debug_barrier()
    return tl.atomic_add(count_ptr + h, 1) == S - 1


@triton.jit
def _weights(m_ptr, l_ptr, count_ptr, least, S: tl.constexpr, SP: tl.constexpr):
    """The last program's view of head h: each partial's weight exp(m_s - top), the
    softmax denominator and top (at least `least`); the counter reset."""
    h = tl.program_id(0)
    j = tl.arange(0, SP)
    ok = j < S
    ms = tl.load(m_ptr + h * S + j, ok, other=float("-inf"), cache_modifier=".cg")
    ls = tl.load(l_ptr + h * S + j, ok, other=0.0, cache_modifier=".cg")
    top = tl.maximum(tl.max(ms, 0), least)
    w = tl.exp(ms - top)
    tl.store(count_ptr + h, 0)
    return w, tl.sum(ls * w, 0), top


@triton.jit
def _combined(o_ptr, w, c, D: tl.constexpr, S: tl.constexpr, SP: tl.constexpr):
    """sum_s w_s o_s[c] over head h's partial outputs."""
    h = tl.program_id(0)
    j = tl.arange(0, SP)
    at = o_ptr + (h * S + j)[:, None] * D + c[None, :]
    o = tl.load(at, (j < S)[:, None], other=0.0, cache_modifier=".cg")
    return tl.sum(o * w[:, None], 0)


@triton.jit
def _csa2_split(
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
):
    """csa2_decode._csa2_attend split: K = V rows, a sink logit per head, the
    inverse RoPE on the last ROPE channels in the combine."""
    h, s = tl.program_id(0), tl.program_id(1)
    c = tl.arange(0, D)
    r = tl.arange(0, R)
    q = tl.load(q_ptr + h * D + c).to(tl.float32)
    if s < W // R:
        k = tl.load(win_ptr + (s * R + r)[:, None] * D + c[None, :]).to(tl.float32)
    else:
        picks = tl.load(pick_ptr + (s - W // R) * R + r)
        k = tl.load(kv_ptr + picks[:, None] * D + c[None, :]).to(tl.float32)
    logit = tl.sum(k * q[None, :], 1).to(tl.bfloat16).to(tl.float32) * scale
    m = tl.max(logit, 0)
    e = tl.exp(logit - m)
    tl.store(o_ptr + (h * S + s) * D + c, tl.sum(e[:, None] * k, 0))
    if _arrive(m, tl.sum(e, 0), m_ptr, l_ptr, count_ptr, S):
        sink = tl.load(sink_ptr + h)
        w, total, top = _weights(m_ptr, l_ptr, count_ptr, sink, S, SP)
        total += tl.exp(sink - top)
        rot = c >= D - ROPE
        half = ROPE // 2
        j = c - (D - ROPE)
        partner = tl.where(rot, D - ROPE + (j + half) % ROPE, c)
        sign = tl.where(j < half, -1.0, 1.0)
        out = (_combined(o_ptr, w, c, D, S, SP) / total).to(tl.bfloat16)
        turned = (_combined(o_ptr, w, partner, D, S, SP) / total).to(tl.bfloat16)
        cos = tl.load(cos_ptr + j, rot, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + j, rot, other=0.0).to(tl.float32)
        out = out.to(tl.float32)
        out = tl.where(rot, out * cos - sign * turned.to(tl.float32) * sin, out)
        tl.store(out_ptr + h * D + c, out.to(tl.bfloat16))


@triton.jit
def _b_split(
    q_ptr,
    qp_ptr,
    kw_ptr,
    vw_ptr,
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
    k_stride,
    k_head,
    v_stride,
    v_head,
    scale,
    QB: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    SP: tl.constexpr,
):
    """csa2_decode._b_attend split: window chunks (keys 64 + QB channels, values
    128) or pick chunks (keys 128, values the keys turned back on the first 64
    channels in pairs, rounded to bf16)."""
    h, s = tl.program_id(0), tl.program_id(1)
    a = tl.arange(0, 64)
    r = tl.arange(0, R)
    if s < W // R:
        b = 64 + tl.arange(0, QB)
        rows = (s * R + r)[:, None]
        q = q_ptr + h * 128
        k_at = kw_ptr + rows * k_stride + h * k_head
        ka = tl.load(k_at + a[None, :]).to(tl.float32)
        kb = tl.load(k_at + b[None, :]).to(tl.float32)
        qa = tl.load(q + a).to(tl.float32)
        qb = tl.load(q + b).to(tl.float32)
        logit = (tl.sum(ka * qa[None, :], 1) + tl.sum(kb * qb[None, :], 1)) * scale
        m = tl.max(logit, 0)
        e = tl.exp(logit - m)
        v_at = vw_ptr + rows * v_stride + h * v_head
        o_lo = tl.sum(e[:, None] * tl.load(v_at + a[None, :]).to(tl.float32), 0)
        o_hi = tl.sum(e[:, None] * tl.load(v_at + 64 + a[None, :]).to(tl.float32), 0)
    else:
        p = tl.arange(0, 32)
        picks = tl.load(pick_ptr + (s - W // R) * R + r)
        qp = qp_ptr + h * 128
        key = keys_ptr + picks[:, None] * key_row + (a % 16)[None, :]
        lo = tl.load(key + (a // 16 * key_block)[None, :]).to(tl.float32)
        hi = tl.load(key + ((a // 16 + 4) * key_block)[None, :]).to(tl.float32)
        cos = tl.load(table_ptr + picks[:, None] * 64 + p[None, :]).to(tl.float32)
        sin = tl.load(table_ptr + picks[:, None] * 64 + 32 + p[None, :]).to(tl.float32)
        q_lo = tl.load(qp + a).to(tl.float32)
        q_hi = tl.load(qp + 64 + a).to(tl.float32)
        logit = (tl.sum(lo * q_lo[None, :], 1) + tl.sum(hi * q_hi[None, :], 1)) * scale
        m = tl.max(logit, 0)
        e = tl.exp(logit - m)
        x0, x1 = tl.split(tl.reshape(lo, (R, 32, 2)))
        v0 = (cos * x0 - sin * x1).to(tl.bfloat16).to(tl.float32)
        v1 = (cos * x1 + sin * x0).to(tl.bfloat16).to(tl.float32)
        o0 = tl.sum(e[:, None] * v0, 0)
        o1 = tl.sum(e[:, None] * v1, 0)
        o_lo = tl.reshape(tl.join(o0, o1), (64,))
        o_hi = tl.sum(e[:, None] * hi, 0)
    o = o_ptr + (h * S + s) * 128
    tl.store(o + a, o_lo)
    tl.store(o + 64 + a, o_hi)
    if _arrive(m, tl.sum(e, 0), m_ptr, l_ptr, count_ptr, S):
        w, total, _ = _weights(m_ptr, l_ptr, count_ptr, float("-inf"), S, SP)
        c = tl.arange(0, 128)
        out = _combined(o_ptr, w, c, 128, S, SP) / total
        tl.store(out_ptr + h * 128 + c, out.to(tl.bfloat16))


def _scratch(heads: int, chunks: int, dim: int, device: torch.device) -> dict:
    """Partials (max, exp sum, output) and the per-device zeroed counters."""
    if device not in _counts:
        _counts[device] = torch.zeros(64, dtype=torch.int32, device=device)
    assert heads <= 64
    return {
        "m": torch.empty(heads * chunks, dtype=torch.float32, device=device),
        "l": torch.empty(heads * chunks, dtype=torch.float32, device=device),
        "o": torch.empty(heads * chunks * dim, dtype=torch.float32, device=device),
        "count": _counts[device],
    }


def csa2_attend(
    q: Tensor,
    window_kv: Tensor,
    main_kv: Tensor,
    picks: Tensor,
    sinks: Tensor,
    cos: Tensor,
    sin: Tensor,
    rows: int = 32,
) -> Tensor:
    """csa2_decode.csa2_attend, `rows` rows per program."""
    heads, dim = q.shape
    w, top = window_kv.shape[0], picks.shape[0]
    assert w % rows == 0 and top % rows == 0
    chunks = (w + top) // rows
    x = _scratch(heads, chunks, dim, q.device)
    out = torch.empty_like(q)
    _csa2_split[(heads, chunks)](
        q, window_kv, main_kv, picks, sinks, cos, sin, out,
        x["m"], x["l"], x["o"], x["count"], dim**-0.5,
        D=dim, W=w, ROPE=cos.shape[0], R=rows, S=chunks,
        SP=triton.next_power_of_2(chunks), num_warps=WARPS,
    )  # fmt: skip
    return out


def b_attend(
    q: Tensor,
    k_win: Tensor,
    v_win: Tensor,
    keys: Tensor,
    picks: Tensor,
    table: Tensor,
    scale: float,
    q_pick: Tensor | None = None,
    rows: int = 16,
) -> Tensor:
    """csa2_decode.b_attend, `rows` rows per program."""
    heads, dim = q.shape
    w, top = k_win.shape[0], picks.shape[0]
    assert dim == 128 and k_win.shape[-1] in (96, 128) and table.shape[1] == 64
    assert k_win.stride(-1) == v_win.stride(-1) == 1
    assert w % rows == 0 and top % rows == 0
    key_row, key_block = (16, keys.shape[1] * 16) if keys.dim() == 3 else (128, 16)
    chunks = (w + top) // rows
    x = _scratch(heads, chunks, dim, q.device)
    out = torch.empty_like(q)
    _b_split[(heads, chunks)](
        q, q if q_pick is None else q_pick, k_win, v_win, keys, picks, table, out,
        x["m"], x["l"], x["o"], x["count"], key_row, key_block,
        k_win.stride(0), k_win.stride(1), v_win.stride(0), v_win.stride(1), scale,
        QB=k_win.shape[-1] - 64, W=w, R=rows, S=chunks,
        SP=triton.next_power_of_2(chunks), num_warps=WARPS,
    )  # fmt: skip
    return out
