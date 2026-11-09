"""Tournament attention for training with approximate cutoffs: tournament_fast's
method whose rounds keep about half of every block instead of exactly half.

A round's cutoff (its budget-th largest partial logit) costs tournament_fast four
histogram passes (radix select on 8-bit digits) or a sort of the round's slots. At
widths of SORT_BELOW (512) slots or more this module uses one of (CUTOFFS):
- radix16: the radix select's first 2 passes only (the top 16 bits);
- linear: one 256-bin histogram between the block's lowest and highest score;
- sample: the matching quantile of 128 evenly spaced slots, by a sort of those.
Every slot at or above the cutoff goes on. More than the vector holds (a bin's
ties) are cut in slot order, and slots are in the emulation's random order, so the
cut is random; the sample's cutoff can also keep fewer than the budget. Narrower
widths sort exactly. Fixed launch settings, no autotuning (it would move the
switch to sorting, so the winners). No tie lists and no reruns, so winners differ from
tournament.py near the cutoffs. Shapes, settings and mask as tournament_fast.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from attention.triton.common import _byte, _pick_bin, _sortable
from nanogpt.tournament import _growing_rounds, _slot_positions
from nanogpt.tournament_fast import (
    DEFAULT,
    TILE,
    Settings,
    _bits,
    _order,
    _pack,
    _top_sort,
    attend,
    bit_mask,
    doc_tiles,
    layouts,
    window_words,
)

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

CUTOFFS = ("radix16", "linear", "sample")
SAMPLE = tl.constexpr(128)  # slots the sample cutoff sorts
SORT_BELOW = 512  # narrower round widths sort exactly


@triton.jit
def _cutoff(score, alive, budget, WIDTH: tl.constexpr, CUTOFF: tl.constexpr):
    """Alive entries at or above an approximate budget-th largest alive score (see
    the module docstring); every alive entry if budget or fewer are."""
    if CUTOFF == "radix16":
        key = _sortable(score)
        floor = tl.full((), 0, tl.uint32)
        match, need = alive, budget
        for t in tl.static_range(2):
            digit = _byte(key, t)
            b, need = _pick_bin(tl.histogram(digit, 256, mask=match), need)
            match = match & (digit == b)
            floor = floor | (b.to(tl.uint32) << (24 - 8 * t))
        keep = alive & (key >= floor)
    elif CUTOFF == "linear":
        lo = tl.min(tl.where(alive, score, float("inf")), 0)
        hi = tl.max(tl.where(alive, score, float("-inf")), 0)
        step = tl.maximum(hi - lo, 1e-30) / 256
        level = tl.where(alive, tl.minimum(((score - lo) / step).to(tl.int32), 255), 0)
        b, _ = _pick_bin(tl.histogram(level, 256, mask=alive), budget)
        keep = alive & (level >= b)
    else:  # sample
        s = tl.gather(
            tl.where(alive, score, float("-inf")),
            tl.arange(0, SAMPLE) * (WIDTH // SAMPLE),
            0,
        )
        taken = tl.sum((s > float("-inf")).to(tl.int32), 0)
        total = tl.sum(alive.to(tl.int32), 0)
        want = tl.maximum((budget * taken + total - 1) // tl.maximum(total, 1), 1)
        ranked = tl.sort(s, descending=True)
        floor = tl.sum(tl.where(tl.arange(0, SAMPLE) == want - 1, ranked, 0.0), 0)
        keep = alive & (score >= floor)
    few = tl.sum(alive.to(tl.int32), 0) <= budget
    return alive & (keep | few)


@triton.jit
def _rounds(
    q_ptr,
    order_ptr,
    kt_ptr,
    start_ptr,
    layout_ptr,
    won_ptr,
    T,
    most_blocks,
    D: tl.constexpr,
    CHANNELS: tl.constexpr,
    DIMS: tl.constexpr,
    ROUNDS: tl.constexpr,
    BLOCK: tl.constexpr,
    QUOTA: tl.constexpr,
    WIDTH: tl.constexpr,
    R: tl.constexpr,
    LOG: tl.constexpr,
    CUTOFF: tl.constexpr,
    SORT_BELOW: tl.constexpr,
):
    """tournament_fast._rounds with _cutoff at SORT_BELOW slots and up and no tie
    lists: one program per (head and query, block) whose block has WIDTH slots."""
    row = tl.program_id(0)  # head * T + query
    blk = tl.program_id(1)
    head = row // T
    start = tl.load(start_ptr + row % T)
    n = row % T - start + 1  # keys in the query's cache
    blocks = tl.cdiv(n, BLOCK)
    rounds = _growing_rounds(n, blocks, QUOTA, ROUNDS)
    if (blk < blocks) & ((QUOTA << rounds) == WIDTH):
        pos = _slot_positions(layout_ptr, blocks, blk, WIDTH, BLOCK)
        alive = pos < n
        keys = kt_ptr + head * D * T + start
        dot = tl.zeros((WIDTH,), tl.float32)  # partial logits
        for r in tl.static_range(R):
            for j in tl.static_range(DIMS):
                i = r * DIMS + j
                d = tl.load(order_ptr + row * CHANNELS + i)
                k = tl.load(keys + pos + d * T, mask=alive, other=0.0).to(tl.float32)
                dot += tl.load(q_ptr + row * CHANNELS + i) * k
            score = tl.where(alive, dot, float("-inf"))
            if (WIDTH >> r) >= SORT_BELOW:
                keep = _cutoff(score, alive, WIDTH >> (r + 1), WIDTH >> r, CUTOFF)
            else:
                keep = _top_sort(score, alive, WIDTH >> (r + 1), WIDTH >> r)
            alive, pos, dot, _ = _pack(keep, pos, dot, WIDTH >> r, LOG - r)
        out = won_ptr + (row * most_blocks + blk) * QUOTA + tl.arange(0, QUOTA)
        tl.store(out, start + pos, mask=alive)


def select(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    longest: int,
    settings: Settings = DEFAULT,
    cutoff: str = "radix16",
) -> Tensor:
    """Each query's winners, row positions, -1 none (tournament_fast.select's
    arguments and shape, plus the cutoff method)."""
    assert cutoff in CUTOFFS, cutoff
    length, heads, dim = q.shape
    assert settings.channels <= dim, settings
    rows = heads * length
    blocks = triton.cdiv(longest, settings.block)
    query = torch.empty(heads, length, settings.channels, device=q.device)
    order = torch.empty_like(query, dtype=torch.int32)
    _order[(rows,)](
        q.contiguous(), query, order, length, heads, D=dim, CHANNELS=settings.channels
    )
    queries = torch.arange(length, device=q.device, dtype=cu_seqlens.dtype)
    starts = cu_seqlens[torch.searchsorted(cu_seqlens, queries, right=True) - 1]
    won = torch.full(
        (heads, length, blocks * settings.winners),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    slots = layouts(blocks, settings.block, str(q.device), False)  # random order
    kt = k.permute(1, 2, 0).contiguous()  # (H, D, T)
    for r in range(settings.rounds + 1):
        width = settings.winners << r
        _rounds[(rows, blocks if width == settings.block else 1)](
            query,
            order,
            kt,
            starts.int(),
            slots,
            won,
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
            LOG=width.bit_length() - 1,
            CUTOFF=cutoff,
            SORT_BELOW=SORT_BELOW,
            num_warps=max(1, min(8, width // 256)),  # 1 to 8 slots per thread
        )
    return won


@torch.library.custom_op("tournament_cutoff::picked", mutates_args=())
def picked(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    low: Tensor,
    longest: int,
    block: int,
    winners: int,
    dims: int,
    cutoff: str,
) -> Tensor:
    """tournament_fast.picked with this module's select()."""
    won = select(q, k, cu_seqlens, longest, Settings(block, winners, dims), cutoff)
    heads, length, most = won.shape
    words = window_words(length, longest)
    out = torch.zeros(heads, length, words, dtype=torch.int32, device=q.device)
    _bits[(heads * length,)](
        won,
        low,
        out,
        length,
        most,
        words,
        TILE=TILE,
        MP=triton.next_power_of_2(most),
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
    cutoff: str,
) -> Tensor:
    length, heads, _ = q.shape
    return q.new_empty(heads, length, window_words(length, longest), dtype=torch.int32)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    longest: int | None = None,
    settings: Settings = DEFAULT,
    cutoff: str = "radix16",
) -> Tensor:
    """tournament_fast.attention on this module's winners."""
    length, heads, _ = q.shape
    assert length % TILE == 0, length
    bound = length if longest is None else min(longest, length)
    low, count, index = doc_tiles(cu_seqlens, length, heads)
    words = picked(
        q.detach(),
        k.detach(),
        cu_seqlens,
        low,
        bound,
        settings.block,
        settings.winners,
        settings.dims,
        cutoff,
    )
    return attend(q, k, v, scale, count, index, bit_mask(words, low))
