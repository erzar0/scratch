"""csa2_select's picks (same rules, arguments and outputs) in one Triton kernel: one
program per query holds its allowed entries' scores in registers.

exact: the score (sum_h q_h) . kv over all channels, then the top_k by bisection on
the sortable score (block_topk_fast._floor), the lowest entries filling ties.
tournament: each round reads only the surviving entries' window of `dims` channels
(windows()), adds it to their scores and keeps the round's budget, ties kept, by
the same bisection; then the top_k as exact.

csa2_select sums every round its 2048-query chunk runs into the final ranking and
breaks ties in torch.topk's order; here each query ranks by its own rounds and the
lowest entries win ties. Picks agree except where scores tie. Valid picks come in
entry order (csa2_select: score order); invalid slots hold entry 0. Documents with
over WIDEST entries fall back to csa2_select.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from nanogpt import csa2_select
from nanogpt.block_topk_fast import _floor
from nanogpt.tournament_channel_blocks import windows
from nanogpt.tournament_fast import Settings

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

WIDEST = 2048  # most entries a query's program holds in registers


@triton.jit
def _sortable(score):
    """float32 -> int32 with the same order."""
    bits = score.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _picks(
    q_ptr,
    kv_ptr,
    start_ptr,
    value_ptr,
    order_ptr,
    entry_ptr,
    valid_ptr,
    N,
    M: tl.constexpr,
    TOP: tl.constexpr,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
    BD: tl.constexpr,
    L: tl.constexpr,
    R: tl.constexpr,
):
    """One program per query t: entries [ceil(start / M), min((t + 1) // M, N)),
    exact (R == 0) or R tournament rounds of L channels."""
    t = tl.program_id(0)
    lo = tl.cdiv(tl.load(start_ptr + t), M)
    n = tl.minimum((t + 1) // M, N) - lo
    slot = tl.arange(0, WIDTH)
    alive = slot < n
    rows = kv_ptr + (lo + slot)[:, None].to(tl.int64) * D
    score = tl.zeros((WIDTH,), tl.float32)
    if R == 0:
        for c0 in tl.static_range(0, D, BD):
            c = c0 + tl.arange(0, BD)
            q = tl.load(q_ptr + t * D + c)
            k = tl.load(rows + c[None, :], mask=alive[:, None], other=0.0)
            score += tl.sum(k.to(tl.float32) * q[None, :], 1)
    else:
        doubles = 0
        for r in tl.static_range(R):
            doubles += (n > (TOP << r)).to(tl.int32)
        for r in tl.static_range(R):
            if r < doubles:
                for c0 in tl.static_range(0, L, BD):
                    at = (t * R + r) * L + c0 + tl.arange(0, BD)
                    c = tl.load(order_ptr + at)
                    q = tl.load(value_ptr + at)
                    k = tl.load(rows + c[None, :], mask=alive[:, None], other=0.0)
                    score += tl.sum(k.to(tl.float32) * q[None, :], 1)
                key = _sortable(score)
                budget = tl.full((), TOP, tl.int32) << (doubles - r - 1)
                floor, few = _floor(key, alive, budget)
                alive = alive & (few | (key >= floor))
    key = _sortable(score)
    floor, few = _floor(key, alive, TOP)
    above = alive & (key > floor)
    tied = alive & (key == floor)
    need = TOP - tl.sum(above.to(tl.int32), 0)
    keep = (alive & few) | above | (tied & (tl.cumsum(tied.to(tl.int32), 0) <= need))
    rank = tl.cumsum(keep.to(tl.int32), 0) - 1
    out = t.to(tl.int64) * TOP + rank
    tl.store(entry_ptr + out, (lo + slot).to(tl.int64), mask=keep)
    tl.store(valid_ptr + out, 1, mask=keep)


def _run(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int, dims: int
) -> tuple[Tensor, Tensor]:
    length, dim = q.shape[0], q.shape[-1]
    t = torch.arange(length, device=q.device)
    most = int(((t + 1) // m - (starts + m - 1) // m).max().clamp(min=1))
    width = max(16, triton.next_power_of_2(min(most, kv.shape[0])))
    if width > WIDEST:  # ponytail: long documents (eval) take csa2_select's path
        return csa2_select._picks(q, kv, starts, m, top_k, dims)
    entry = torch.zeros(length, top_k, dtype=torch.long, device=q.device)
    valid = torch.zeros(length, top_k, dtype=torch.int8, device=q.device)
    qs = q.float().sum(1).contiguous()
    rounds = dim // dims if dims else 0
    values, order = qs, qs  # unused by exact
    if dims:
        values, order = windows(qs[:, None], Settings(top_k << rounds, top_k, dims))
    _picks[(length,)](
        qs,
        kv.contiguous(),
        starts.contiguous(),
        values,
        order,
        entry,
        valid,
        kv.shape[0],
        M=m,
        TOP=top_k,
        D=dim,
        WIDTH=width,
        BD=min(dims or 16, max(1, 4096 // width)),
        L=dims,
        R=rounds,
        num_warps=4 if width <= 512 else 8,
    )
    return entry, valid.bool()


@torch.library.custom_op("csa2_select_fast::picks", mutates_args=())
def picks(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int, dims: int
) -> tuple[Tensor, Tensor]:
    """exact (dims 0) or tournament (dims > 0) picks; opaque to torch.compile."""
    with torch.no_grad():
        return _run(q, kv, starts, m, top_k, dims)


@picks.register_fake
def _(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int, dims: int
) -> tuple[Tensor, Tensor]:
    shape = (q.shape[0], top_k)
    return (
        q.new_empty(shape, dtype=torch.long),
        q.new_empty(shape, dtype=torch.bool),
    )


def exact(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int = 64
) -> tuple[Tensor, Tensor]:
    """csa2_select.exact: each query's top_k allowed entries by (sum_h q_h) . kv.

    Args:
        q: (S, H, D) queries of one packed row.
        kv: (N, D) entries, entry e = tokens [e m, (e + 1) m).
        starts: (S,) int64 document start of each query.
        m: tokens per entry.
        top_k: picks per query.

    Returns:
        entry (S, top_k) int64 and valid (S, top_k) bool.
    """
    return picks(q.detach(), kv.detach(), starts, m, top_k, 0)


def tournament(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int = 64, dims: int = 16
) -> tuple[Tensor, Tensor]:
    """csa2_select.tournament: halving rounds of `dims` channels (as exact, plus
    dims: channels per round)."""
    return picks(q.detach(), kv.detach(), starts, m, top_k, dims)
