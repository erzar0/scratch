"""Tournament attention for training with halving inside groups: tournament_fast's
method where, at round widths of GROUP_FROM (512) slots or more, each group of
`group` slots keeps its own top half instead of the block keeping its top half.

A round of width W splits its slots into groups of every (W / group)-th slot (a
short prefix's dead tail spreads evenly over the groups); each group keeps its
top group / 2 partial logits by a bitonic top-k (tl.topk) on 64-bit keys (score,
then the lower slot), which also packs the survivors, so there are no thresholds,
no tie lists and no reruns. Narrower rounds keep the exact top half the same way
(one group). group = 2 is a knockout (slot i against slot i + W / 2); group >= W
is the exact top half (tournament_fast's winners when no two scores tie). Shapes,
settings and mask as tournament_fast; fixed launch (warps = width / 256).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from nanogpt.tournament import _growing_rounds, _slot_positions
from nanogpt.tournament_fast import (
    DEFAULT,
    TILE,
    Settings,
    _bits,
    _order,
    attend,
    bit_mask,
    doc_tiles,
    key_copies,
    layouts,
    window_words,
)

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

GROUP_FROM = tl.constexpr(512)  # narrower rounds keep the exact top half


@triton.jit
def _halve(score, alive, pos, dot, WIDTH: tl.constexpr, G: tl.constexpr):
    """Each group of G slots (every WIDTH / G-th slot) keeps its G / 2 highest
    scores, ties to the lower slot: the survivors' alive, pos and dot, group after
    group, each group's in descending score."""
    slot = tl.arange(0, WIDTH)
    bits = score.to(tl.int32, bitcast=True)
    order = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)  # float order as int order
    key = (order.to(tl.int64) << 32) | (WIDTH - 1 - slot).to(tl.int64)
    grouped = tl.trans(tl.reshape(key, (G, WIDTH // G)))  # row b: b, b + WIDTH / G..
    top = tl.reshape(tl.topk(grouped, G // 2), (WIDTH // 2,))
    at = WIDTH - 1 - (top & 0xFFFFFFFF).to(tl.int32)
    kept = tl.gather(alive.to(tl.int32), at, 0) > 0
    return kept, tl.gather(pos, at, 0), tl.gather(dot, at, 0)


@triton.jit
def _rounds(
    q_ptr,
    order_ptr,
    kt_ptr,
    off_ptr,
    start_ptr,
    layout_ptr,
    won_ptr,
    T,
    Q0,
    C,
    most_blocks,
    D: tl.constexpr,
    CHANNELS: tl.constexpr,
    DIMS: tl.constexpr,
    ROUNDS: tl.constexpr,
    BLOCK: tl.constexpr,
    QUOTA: tl.constexpr,
    WIDTH: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
):
    """tournament_fast._rounds with _halve as each round's cut: one program per
    (head and query, block) whose block has WIDTH slots."""
    row = tl.program_id(0)  # head * C + query - Q0
    blk = tl.program_id(1)
    head = row // C
    start = tl.load(start_ptr + row % C)
    n = Q0 + row % C - start + 1  # keys in the query's cache
    blocks = tl.cdiv(n, BLOCK)
    rounds = _growing_rounds(n, blocks, QUOTA, ROUNDS)
    if (blk < blocks) & ((QUOTA << rounds) == WIDTH):
        pos = _slot_positions(layout_ptr, blocks, blk, WIDTH, BLOCK)
        alive = pos < n
        copy = blocks > 1
        here = start
        if WIDTH == BLOCK:  # pos: slots in the copy until the rounds end
            here = tl.where(copy, tl.load(off_ptr + row % C, mask=copy, other=0), start)
            pos = tl.where(copy, blk * BLOCK + tl.arange(0, WIDTH), pos)
        keys = kt_ptr + head * D * T + here
        dot = tl.zeros((WIDTH,), tl.float32)  # partial logits
        for r in tl.static_range(R):
            for j in tl.static_range(DIMS):
                i = r * DIMS + j
                d = tl.load(order_ptr + row * CHANNELS + i)
                k = tl.load(keys + pos + d * T, mask=alive, other=0.0).to(tl.float32)
                dot += tl.load(q_ptr + row * CHANNELS + i) * k
            score = tl.where(alive, dot, float("-inf"))
            if (WIDTH >> r) < GROUP_FROM:
                alive, pos, dot = _halve(score, alive, pos, dot, WIDTH >> r, WIDTH >> r)
            elif G >= (WIDTH >> r):
                alive, pos, dot = _halve(score, alive, pos, dot, WIDTH >> r, WIDTH >> r)
            else:
                alive, pos, dot = _halve(score, alive, pos, dot, WIDTH >> r, G)
        if WIDTH == BLOCK:  # slots in the copy back to document positions
            table = layout_ptr + BLOCK * ((blocks - 1) * blocks // 2 - 1)
            pos = tl.where(copy, tl.load(table + pos, mask=alive & copy, other=0), pos)
        out = won_ptr + (row * most_blocks + blk) * QUOTA + tl.arange(0, QUOTA)
        tl.store(out, start + pos, mask=alive)


def select(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    longest: int,
    settings: Settings = DEFAULT,
    first: int = 0,
    group: int = 64,
) -> Tensor:
    """tournament_fast.select (arguments, shape) with groups of `group` slots (a
    power of 2, at least 2)."""
    assert group >= 2 and group & (group - 1) == 0, group
    length, heads, dim = q.shape
    assert settings.channels <= dim, settings
    rows = heads * length
    blocks = triton.cdiv(longest, settings.block)
    query = torch.empty(heads, length, settings.channels, device=q.device)
    order = torch.empty_like(query, dtype=torch.int32)
    _order[(rows,)](
        q.contiguous(), query, order, length, heads, D=dim, CHANNELS=settings.channels
    )
    queries = torch.arange(
        first, first + length, device=q.device, dtype=cu_seqlens.dtype
    )
    starts = cu_seqlens[torch.searchsorted(cu_seqlens, queries, right=True) - 1]
    won = torch.full(
        (heads, length, blocks * settings.winners),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    table = layouts(blocks, settings.block, str(q.device), True)
    kt = k.permute(1, 2, 0).contiguous()  # (H, D, T)
    off = torch.zeros(1, dtype=torch.int32, device=q.device)
    if longest > settings.block:  # some prefix may span several blocks
        kt, off = key_copies(kt, cu_seqlens, first, length, settings.block, table)
    assert kt.numel() < 2**31, "int32 offsets overflow: pass a chunk"
    for r in range(settings.rounds + 1):
        width = settings.winners << r
        _rounds[(rows, blocks if width == settings.block else 1)](
            query,
            order,
            kt,
            off,
            starts.int(),
            table,
            won,
            kt.shape[-1],
            first,
            length,
            blocks,
            D=dim,
            CHANNELS=settings.channels,
            DIMS=settings.dims,
            ROUNDS=settings.rounds,
            BLOCK=settings.block,
            QUOTA=settings.winners,
            WIDTH=width,
            R=r,
            G=group,
            num_warps=max(1, min(8, width // 256)),
        )
    return won


@torch.library.custom_op("tournament_groups::picked", mutates_args=())
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
    group: int = 64,
) -> Tensor:
    """tournament_fast.picked with this module's select()."""
    settings = Settings(block, winners, dims)
    won = select(q, k, cu_seqlens, longest, settings, first, group)
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
    group: int = 64,
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
    group: int = 64,
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
            group,
        )
        mask_mod = bit_mask(words, low[tiles])
        part = q[first:end], k, v, scale, count[..., tiles], index[..., tiles, :]
        outs.append(attend(*part, mask_mod))
    return outs[0] if len(outs) == 1 else torch.cat(outs[::-1])
