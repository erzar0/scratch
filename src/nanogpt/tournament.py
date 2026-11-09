"""Tournament: shuffled blocks, growing budget, for training on packed documents.

Every query runs the decode tournament of attention/tournament/
shuffled_blocks_growing.py (budget 512) with its own document's prefix as the
cache: n keys in ceil(n / 1024) blocks of 1024 key slots, keys in the emulation's
random slot order, 16 winners per block. A query 3000 tokens into its document
attends 48 keys (3 blocks); a query in its document's first 16 tokens attends all.

select() is a Triton kernel, one program per (head, query, block), running the
decode kernel's rounds (attention/triton/common.py): each round adds the query's
next 8 channels by |q| and each block keeps its top half. attention() then runs
exact attention on the winners in FlexAttention, the winners as its mask, so
gradients reach q, k and v through the winners only.

Shapes: q, k, v (T, H, D), one packed row; cu_seqlens holds the documents' starts
and may repeat T at its end (record 91 pads it so).
"""

from functools import cache
from typing import cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn.attention.flex_attention import (
    FlexKernelOptions,
    create_block_mask,
    flex_attention,
)

from attention.triton.common import DIMS, _slots, _top

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

BLOCK = 1024  # key slots per block
QUOTA = 16  # winners per block: budget 512 at 32768 keys
ROUNDS = 6  # log2(BLOCK / QUOTA): the rounds of a full block
# the typed options lack fwd_ keys, which flex reads
FORWARD_STAGES = cast(FlexKernelOptions, {"fwd_num_stages": 2})


@cache
def layouts(blocks: int, device: str) -> Tensor:
    """Slot -> document position tables for 2 to `blocks` blocks, concatenated: the
    emulation's random order of 1024 b slots (tournament/shuffled_blocks.shuffle)."""
    tables = [
        torch.randperm(BLOCK * b, generator=torch.Generator().manual_seed(0))
        for b in range(2, max(blocks, 2) + 1)
    ]
    return torch.cat(tables).int().to(device)


@triton.jit
def _growing_rounds(n, blocks, QUOTA: tl.constexpr, ROUNDS: tl.constexpr):
    """Halving rounds from n keys down to QUOTA per block: the doublings of
    QUOTA * blocks still below n, at most ROUNDS."""
    rounds = 0
    for doubling in tl.static_range(ROUNDS):
        rounds += ((QUOTA * blocks) << doubling < n).to(tl.int32)
    return rounds


@triton.jit
def _slot_positions(layout_ptr, blocks, blk, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    """Document position of each of block blk's WIDTH slots: layouts()'s table
    for `blocks` full blocks, or in order in a single block."""
    slot = tl.arange(0, WIDTH)
    if WIDTH == BLOCK:  # maybe several blocks: the shuffled layout
        table = layout_ptr + BLOCK * ((blocks - 1) * blocks // 2 - 1)
        shuffled = tl.load(table + blk * BLOCK + slot, mask=blocks > 1, other=0)
        pos = tl.where(blocks > 1, shuffled, slot)
    else:
        pos = slot
    return pos


@triton.jit
def _query_rounds(
    q_ptr,
    order_ptr,
    kt_ptr,
    start_ptr,
    layout_ptr,
    won_ptr,
    T,
    most_blocks,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    ROUNDS: tl.constexpr,
    BLOCK: tl.constexpr,
    QUOTA: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """One program per (head and query, block): the block's rounds, then its
    QUOTA winners' row positions. Runs only blocks of WIDTH key slots: shorter
    prefixes use narrower blocks."""
    row = tl.program_id(0)  # head * T + query
    blk = tl.program_id(1)
    head = row // T
    start = tl.load(start_ptr + row % T)
    n = row % T - start + 1  # keys in the query's cache
    blocks = tl.cdiv(n, BLOCK)
    rounds = _growing_rounds(n, blocks, QUOTA, ROUNDS)
    # slots per block: BLOCK, or fewer in a single short block
    if (blk < blocks) & ((QUOTA << rounds) == WIDTH):
        pos = _slot_positions(layout_ptr, blocks, blk, WIDTH, BLOCK)
        alive = pos < n
        keys = kt_ptr + head * D * T + start + pos
        dot = tl.zeros((WIDTH,), tl.float32)  # partial logits
        for r in range(rounds):
            for j in tl.static_range(DIMS):
                i = r * DIMS + j
                d = tl.load(order_ptr + row * D + i)
                k = tl.load(keys + d * T, mask=alive, other=0.0).to(tl.float32)
                dot += tl.load(q_ptr + row * D + i) * k
            score = tl.where(alive, dot, float("-inf"))
            alive = _top(score, alive, WIDTH >> (r + 1))
        rank = _slots(alive)
        out = won_ptr + (row * most_blocks + blk) * QUOTA + rank
        tl.store(out, start + pos, mask=alive & (rank < QUOTA))


def select(q: Tensor, k: Tensor, cu_seqlens: Tensor) -> Tensor:
    """(H, T, 16 ceil(T / 1024)) each query's winners, row positions; -1 is none."""
    length, heads, dim = q.shape
    blocks = triton.cdiv(length, BLOCK)
    query = q.float().transpose(0, 1)  # (H, T, D)
    order = query.abs().argsort(-1, descending=True)
    queries = torch.arange(length, device=q.device, dtype=cu_seqlens.dtype)
    starts = cu_seqlens[torch.searchsorted(cu_seqlens, queries, right=True) - 1]
    won = torch.full(
        (heads, length, blocks * QUOTA), -1, dtype=torch.int32, device=q.device
    )
    args = {
        "q_ptr": query.gather(-1, order).contiguous(),
        "order_ptr": order.int().contiguous(),
        "kt_ptr": k.permute(1, 2, 0).contiguous(),  # (H, D, T)
        "start_ptr": starts.int(),
        "layout_ptr": layouts(blocks, str(q.device)),
        "won_ptr": won,
        "T": length,
        "most_blocks": blocks,
        "D": dim,
        "DIMS": DIMS,
        "ROUNDS": ROUNDS,
        "BLOCK": BLOCK,
        "QUOTA": QUOTA,
    }
    for width in (QUOTA << r for r in range(ROUNDS + 1)):
        grid = (heads * length, blocks if width == BLOCK else 1)
        # measured: 4 warps at 1024 slots beat 8 by 1.4x
        _query_rounds[grid](**args, WIDTH=width, num_warps=max(1, width // 256))
    return won


def bit_words(won: Tensor, length: int) -> Tensor:
    """(H, T, ceil(length / 32)) int32 words: bit s % 32 of word s // 32 is set when
    key s is in won (H, T, M) (-1: none)."""
    at = won.clamp(min=0)
    bits = torch.where(won < 0, 0, 1 << (at % 32))  # distinct bits: add = or
    words = torch.zeros(
        *won.shape[:2], triton.cdiv(length, 32), dtype=torch.int32, device=won.device
    )
    return words.scatter_add_(2, (at // 32).long(), bits)


# An opaque op: built inside a compiled graph, a scattered mask reached flex's
# kernel all False (torch 2.14)
@torch.library.custom_op("tournament::picked", mutates_args=())
def picked(q: Tensor, k: Tensor, cu_seqlens: Tensor) -> Tensor:
    """(H, T, ceil(T / 32)) int32 words: bit s % 32 of word s // 32 is set when key s
    is one of query t's select() winners (an eighth of a bool mask's memory)."""
    return bit_words(select(q, k, cu_seqlens), q.shape[0])


@picked.register_fake
def _(q: Tensor, k: Tensor, cu_seqlens: Tensor) -> Tensor:
    length, heads, _ = q.shape
    return q.new_empty(heads, length, triton.cdiv(length, 32), dtype=torch.int32)


def attention(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, scale: float
) -> Tensor:
    """(T, H, D) each query's exact attention on its select() winners."""
    length, heads, _ = q.shape
    words = picked(q.detach(), k.detach(), cu_seqlens)

    def winner(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return ((words[h, q_idx, kv_idx // 32] >> (kv_idx % 32)) & 1) != 0

    mask = create_block_mask(winner, None, heads, length, length, device=q.device)
    q, k, v = (x.transpose(0, 1)[None] for x in (q, k, v))
    # 2 stages: torch 2.10's 3-stage forward tile needs 112 KB, Ada has 99 KB
    out = flex_attention(
        q, k, v, block_mask=mask, scale=scale, kernel_options=FORWARD_STAGES
    )
    return cast(Tensor, out)[0].transpose(0, 1)
