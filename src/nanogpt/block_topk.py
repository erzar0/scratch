"""Exact top-k attention per block, for screening against the tournament: each
query's document prefix in tournament_fast's blocks (the same partition into
blocks of `block` key slots), each block sending its `winners` keys with the
highest full q.k (all channels, float32) to attention instead of the halving
rounds' survivors. Unfused torch (one matmul per query chunk, torch.topk), no
grad: for checkpoint evaluation and as the quality reference the rounds
approximate. Shapes, settings and mask as tournament_fast.
"""

import torch
import triton
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

CHUNK = 1024  # queries per matmul


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
    length, heads, _ = q.shape
    block, winners = settings.block, settings.winners
    blocks = triton.cdiv(longest, block)
    queries = torch.arange(first, first + length, device=q.device)
    starts = cu_seqlens[torch.searchsorted(cu_seqlens, queries.int(), right=True) - 1]
    n = queries - starts + 1  # keys in each query's prefix
    count = torch.div(n + block - 1, block, rounding_mode="floor")
    won = torch.full(
        (heads, length, blocks * winners), -1, dtype=torch.int32, device=q.device
    )
    table = layouts(blocks, block, str(q.device), True)
    kf = k.float().transpose(0, 1)  # (H, T, D)
    for c in range(0, length, CHUNK):
        part = slice(c, c + CHUNK)
        logits = torch.einsum("thd,hsd->hts", q[part].float(), kf)  # (H, C, T)
        for b in range(1, blocks + 1):
            mine = (count[part] == b).nonzero().squeeze(1)
            if not len(mine):
                continue
            if b == 1:  # slots in position order
                offset = torch.arange(block, device=q.device)[None]
            else:  # layouts()' table for b blocks
                offset = table[block * ((b - 1) * b // 2 - 1) :][: b * block][None]
            at = starts[part][mine, None] + offset  # (Q, b * block)
            alive = offset < n[part][mine, None]
            got = logits[:, mine[:, None], at.clamp(max=k.shape[0] - 1)]  # (H, Q, S)
            got = got.masked_fill(~alive, float("-inf"))
            got = got.view(heads, len(mine), b, block)
            top, index = got.topk(min(winners, block), dim=-1)
            pos = at.view(len(mine), b, block)[None].expand(heads, -1, -1, -1)
            pos = pos.gather(-1, index)
            pos = torch.where(top > float("-inf"), pos, -1).flatten(2)
            won[:, c + mine, : b * winners] = pos.int()
    return won


@torch.library.custom_op("block_topk::picked", mutates_args=())
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
