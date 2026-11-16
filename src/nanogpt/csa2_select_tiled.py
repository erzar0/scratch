"""csa2_select_fast.exact (same picks, arguments and outputs) in two Triton kernels:

    _scores  one program per BQ queries: (sum_h q_h) . kv with every entry of their
             documents' prefixes, float32 dots (IEEE) of the tile's queries with BE
             entries at a time, stored at (query, entry - the query's first entry)
    _top     one program per query: csa2_select_fast's top_k by bisection on them

csa2_select_fast reads each query's entries in its own program, 2 channels per row
at 2048 entries (its register budget): training's ratio-1 layers (one entry per
token) read every entry 2048 times, 8x the bytes used. Here each entry is read once
per BQ queries, whole rows. Documents over WIDEST entries fall back to csa2_select.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from nanogpt import csa2_select
from nanogpt.block_topk_fast import _floor
from nanogpt.csa2_select_fast import WIDEST, _sortable

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

BQ, BE = 64, 32  # queries per _scores program, entries per dot


@triton.jit
def _scores(
    q_ptr,
    kv_ptr,
    start_ptr,
    s_ptr,
    T,
    N,
    M: tl.constexpr,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
    BQ: tl.constexpr,
    BE: tl.constexpr,
):
    """Queries t in [p BQ, (p + 1) BQ): q_t . kv_e for e in [ceil(start_t / M),
    min((t + 1) // M, N)) at s[t, e - ceil(start_t / M)]."""
    t = tl.program_id(0) * BQ + tl.arange(0, BQ)
    live = t < T
    lo = tl.cdiv(tl.load(start_ptr + t, live, other=0), M)
    hi = tl.where(live, tl.minimum((t + 1) // M, N), lo)
    c = tl.arange(0, D)
    q = tl.load(q_ptr + t[:, None] * D + c[None, :], live[:, None], other=0.0)
    out = s_ptr + t[:, None].to(tl.int64) * WIDTH - lo[:, None]
    for e0 in range(tl.min(tl.where(live, lo, N), 0), tl.max(hi, 0), BE):
        e = e0 + tl.arange(0, BE)
        at = kv_ptr + e[:, None].to(tl.int64) * D + c[None, :]
        k = tl.load(at, (e < N)[:, None], other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k), input_precision="ieee")
        mine = (e[None, :] >= lo[:, None]) & (e[None, :] < hi[:, None])
        tl.store(out + e[None, :], s, mask=mine)


@triton.jit
def _top(
    s_ptr,
    start_ptr,
    entry_ptr,
    valid_ptr,
    N,
    M: tl.constexpr,
    TOP: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Query t: the TOP best of its scores, the lowest entries filling ties
    (csa2_select_fast._picks' tail)."""
    t = tl.program_id(0)
    lo = tl.cdiv(tl.load(start_ptr + t), M)
    n = tl.minimum((t + 1) // M, N) - lo
    slot = tl.arange(0, WIDTH)
    alive = slot < n
    key = _sortable(tl.load(s_ptr + t.to(tl.int64) * WIDTH + slot, alive, other=0.0))
    floor, few = _floor(key, alive, TOP)
    above = alive & (key > floor)
    tied = alive & (key == floor)
    need = TOP - tl.sum(above.to(tl.int32), 0)
    keep = (alive & few) | above | (tied & (tl.cumsum(tied.to(tl.int32), 0) <= need))
    rank = tl.cumsum(keep.to(tl.int32), 0) - 1
    out = t.to(tl.int64) * TOP + rank
    tl.store(entry_ptr + out, (lo + slot).to(tl.int64), mask=keep)
    tl.store(valid_ptr + out, 1, mask=keep)


@torch.library.custom_op("csa2_select_tiled::picks", mutates_args=())
def picks(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int
) -> tuple[Tensor, Tensor]:
    """csa2_select_fast.picks with dims 0; opaque to torch.compile."""
    length, dim, n = q.shape[0], q.shape[-1], kv.shape[0]
    t = torch.arange(length, device=q.device)
    most = int(((t + 1) // m - (starts + m - 1) // m).max().clamp(min=1))
    width = max(16, triton.next_power_of_2(min(most, n)))
    if width > WIDEST:  # ponytail: long documents (eval) take csa2_select's path
        return csa2_select._picks(q, kv, starts, m, top_k, 0)
    entry = torch.zeros(length, top_k, dtype=torch.long, device=q.device)
    valid = torch.zeros(length, top_k, dtype=torch.int8, device=q.device)
    qs = q.float().sum(1).contiguous()
    kv, starts = kv.contiguous(), starts.contiguous()
    # ponytail: one buffer, 400 MB at training's 48K-token rows; parts if it grows
    score = torch.empty(length, width, device=q.device)
    _scores[(triton.cdiv(length, BQ),)](
        qs, kv, starts, score, length, n, M=m, D=dim, WIDTH=width, BQ=BQ, BE=BE
    )
    _top[(length,)](
        score,
        starts,
        entry,
        valid,
        n,
        M=m,
        TOP=top_k,
        WIDTH=width,
        num_warps=4 if width <= 512 else 8,
    )
    return entry, valid.bool()


@picks.register_fake
def _(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int
) -> tuple[Tensor, Tensor]:
    shape = (q.shape[0], top_k)
    return (
        q.new_empty(shape, dtype=torch.long),
        q.new_empty(shape, dtype=torch.bool),
    )


def exact(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int = 64
) -> tuple[Tensor, Tensor]:
    """csa2_select_fast.exact (same arguments and outputs)."""
    with torch.no_grad():
        return picks(q.detach(), kv.detach(), starts, m, top_k)
