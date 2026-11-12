"""Exact top-k attention per block (block_topk.py's rule) in two Triton kernels:

    _scores   every query's full q.k (all channels) with each key of its document
              prefix, on tensor cores (bf16 operands, float32 accumulation; float32
              operands in IEEE float32), written in position order to a score buffer
    _select   per (query, block of tournament_fast's partition): the block's slot
              scores (in order with one block, through layouts()' table with
              several), the winners-th largest by bisection on the sortable key,
              then the scores above it and the lowest slots tied at it: exactly
              min(winners, alive) winners

Queries go through the buffer in parts of at most BUDGET floats. No grad: selection
only, as tournament_fast. Shapes, settings and mask as tournament_fast.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from nanogpt.tournament_fast import (
    DEFAULT,
    TILE,
    Settings,
    _bits,
    attend,
    bit_mask,
    doc_tiles,
    layouts,
    window_words,
)

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

BQ, BK = 64, 64  # queries and keys per score tile
SPAN = 1024  # keys per _scores program (a third grid axis: long prefixes in parallel)
BUDGET = 1 << 27  # score buffer floats (512 MiB)


@triton.jit
def _scores(
    q_ptr,
    k_ptr,
    start_ptr,
    s_ptr,
    Q0,
    C,
    W,
    sq_t,
    sq_h,
    sk_t,
    sk_h,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    SPAN: tl.constexpr,
    PRECISION: tl.constexpr,
):
    """One program per (tile of BQ queries, head, SPAN keys): q.k for the span's
    keys from the tile's lowest document start to its last query, stored at (head,
    query, key - the query's document start) where the key is in the query's
    prefix."""
    rows = tl.program_id(0) * BQ + tl.arange(0, BQ)  # queries of this part
    head = tl.program_id(1)
    real = rows < C
    start = tl.load(start_ptr + rows, mask=real, other=0)
    t = Q0 + rows  # row positions
    d = tl.arange(0, D)
    q = tl.load(
        q_ptr + rows[:, None] * sq_t + head * sq_h + d[None, :],
        mask=real[:, None],
        other=0.0,
    )
    lo = tl.min(tl.where(real, start, 2147483647), 0)
    hi = tl.max(tl.where(real, t, 0), 0)
    out = s_ptr + head * C * W + rows[:, None] * W - start[:, None]
    begin = lo // BK * BK + tl.program_id(2) * SPAN
    for k0 in range(begin, tl.minimum(hi + 1, begin + SPAN), BK):
        pos = k0 + tl.arange(0, BK)
        k = tl.load(
            k_ptr + pos[:, None] * sk_t + head * sk_h + d[None, :],
            mask=(pos <= hi)[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k), input_precision=PRECISION)
        mine = (pos[None, :] >= start[:, None]) & (pos[None, :] <= t[:, None])
        tl.store(out + pos[None, :], s, mask=real[:, None] & mine)


@triton.jit
def _floor(key, alive, K):
    """The K-th largest alive key, or a value with exactly K alive keys at or above
    it (bisection between the alive min and max); and whether K or fewer are alive."""
    lo = tl.min(tl.where(alive, key, 2147483647), 0)
    hi = tl.max(tl.where(alive, key, -2147483648), 0)
    few = tl.sum(alive.to(tl.int32), 0) <= K
    active = (~few) & (lo < hi)
    while active:
        mid = (lo >> 1) + (hi >> 1) + ((lo | hi) & 1)  # ceil((lo + hi) / 2)
        count = tl.sum(((key >= mid) & alive).to(tl.int32), 0)
        lo = tl.where(count >= K, mid, lo)
        hi = tl.where(count >= K, hi, mid - 1)
        hi = tl.where(count == K, mid, hi)
        active = lo < hi
    return lo, few


@triton.jit
def _select(
    s_ptr,
    start_ptr,
    layout_ptr,
    won_ptr,
    Q0,
    C,
    C0,
    TOTAL,
    W,
    NB,
    most_blocks,
    BLOCK: tl.constexpr,
    QUOTA: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """One program per (head and query, block), the blocks of a row adjacent: the
    block's QUOTA winners if its slots fit WIDTH (one block: WIDTH the power of 2
    at or above n, at least QUOTA; several: WIDTH = BLOCK), into won (H, TOTAL,
    most_blocks * QUOTA) at query C0 + its index in this part."""
    blk = tl.program_id(0) % NB
    row = tl.program_id(0) // NB  # head * C + query of this part
    start = tl.load(start_ptr + row % C)
    n = Q0 + row % C - start + 1  # keys in the query's prefix
    blocks = tl.cdiv(n, BLOCK)
    fits = (n <= WIDTH) & ((2 * n > WIDTH) | (WIDTH == QUOTA))
    if (blk < blocks) & (((blocks > 1) & (WIDTH == BLOCK)) | ((blocks == 1) & fits)):
        slot = tl.arange(0, WIDTH)
        pos = slot
        if WIDTH == BLOCK:  # several blocks: layouts()' table
            table = layout_ptr + BLOCK * ((blocks - 1) * blocks // 2 - 1)
            shuffled = tl.load(table + blk * BLOCK + slot, mask=blocks > 1, other=0)
            pos = tl.where(blocks > 1, shuffled, slot)
        alive = pos < n
        score = tl.load(s_ptr + row * W + pos, mask=alive, other=float("-inf"))
        bits = score.to(tl.int32, bitcast=True)
        key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)  # float order as int order
        floor, few = _floor(key, alive, QUOTA)
        above = alive & (key > floor)
        tied = alive & (key == floor)
        need = QUOTA - tl.sum(above.to(tl.int32), 0)
        lowest = tl.cumsum(tied.to(tl.int32), 0) <= need
        keep = (alive & few) | above | (tied & lowest)
        rank = tl.cumsum(keep.to(tl.int32), 0) - 1
        head = row // C
        out = won_ptr + ((head * TOTAL + C0 + row % C) * most_blocks + blk) * QUOTA
        tl.store(out + rank, start + pos, mask=keep)


@torch.no_grad()
def select(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    longest: int,
    settings: Settings = DEFAULT,
    first: int = 0,
) -> Tensor:
    """Each query's winners, row positions, -1 none (tournament_fast.select's
    arguments and shape)."""
    length, heads, dim = q.shape
    assert q.stride(-1) == 1 and k.stride(-1) == 1, "channels must be contiguous"
    block, quota = settings.block, settings.winners
    blocks = triton.cdiv(longest, block)
    width = blocks * block  # score columns per query
    queries = torch.arange(
        first, first + length, device=q.device, dtype=cu_seqlens.dtype
    )
    starts = cu_seqlens[torch.searchsorted(cu_seqlens, queries, right=True) - 1]
    starts = starts.int()
    won = torch.full(
        (heads, length, blocks * quota), -1, dtype=torch.int32, device=q.device
    )
    table = layouts(blocks, block, str(q.device), True)
    step = max(BQ, BUDGET // (heads * width) // BQ * BQ)
    scores = torch.empty(heads * min(step, length) * width, device=q.device)
    exact = q.dtype == torch.float32
    for c in range(0, length, step):
        part = min(step, length - c)
        spans = triton.cdiv(width + 2 * BK, SPAN)  # a tile's keys: <= width + 2 BK
        _scores[(triton.cdiv(part, BQ), heads, spans)](
            q[c:],
            k,
            starts[c:],
            scores,
            first + c,
            part,
            width,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            D=dim,
            BQ=BQ,
            BK=BK,
            SPAN=SPAN,
            PRECISION="ieee" if exact else "tf32",
            num_warps=4,
            num_stages=1 if exact else 2,
        )
        widths = [quota << i for i in range((block // quota).bit_length())]
        for w in widths:
            nb = blocks if w == block else 1
            _select[(heads * part * nb,)](
                scores,
                starts[c:],
                table,
                won,
                first + c,
                part,
                c,
                length,
                width,
                nb,
                blocks,
                BLOCK=block,
                QUOTA=quota,
                WIDTH=w,
                num_warps=4 if w >= 2048 else 1,  # tuned: topk_select_tune.csv
            )
    return won


@torch.library.custom_op("block_topk_fast::picked", mutates_args=())
def picked(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    low: Tensor,
    longest: int,
    block: int,
    winners: int,
    dims: int,
    first: int = 0,
) -> Tensor:
    """tournament_fast.picked with this module's select()."""
    won = select(q, k, cu_seqlens, longest, Settings(block, winners, dims), first)
    heads, length, most = won.shape
    words = window_words(k.shape[0], longest)
    assert heads * length * words < 2**31, "int32 offsets overflow: pass a chunk"
    out = torch.zeros(heads, length, words, dtype=torch.int32, device=q.device)
    _bits[(heads * length,)](
        won, low, out, length, most, words, TILE=TILE, MP=triton.next_power_of_2(most)
    )
    return out


@picked.register_fake
def _(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    low: Tensor,
    longest: int,
    block: int,
    winners: int,
    dims: int,
    first: int = 0,
) -> Tensor:
    length, heads, _ = q.shape
    words = window_words(k.shape[0], longest)
    return q.new_empty(heads, length, words, dtype=torch.int32)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    longest: int | None = None,
    settings: Settings = DEFAULT,
    chunk: int | None = None,
) -> Tensor:
    """tournament_fast.attention on this module's winners."""
    length, heads, _ = q.shape
    step = chunk or length
    assert length % TILE == 0 and step % TILE == 0, (length, step)
    bound = length if longest is None else min(longest, length)
    low, count, index = doc_tiles(cu_seqlens, length, heads)
    outs = []
    for first in reversed(range(0, length, step)):  # as tournament_fast.attention
        end = min(first + step, length)
        tiles = slice(first // TILE, end // TILE)
        words = picked(
            q[first:end].detach(),
            k.detach(),
            cu_seqlens,
            low[tiles],
            bound,
            settings.block,
            settings.winners,
            settings.dims,
            first,
        )
        mask_mod = bit_mask(words, low[tiles])
        part = q[first:end], k, v, scale, count[..., tiles], index[..., tiles, :]
        outs.append(attend(*part, mask_mod))
    return outs[0] if len(outs) == 1 else torch.cat(outs[::-1])
