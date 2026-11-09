"""Tile selection inside a sliding window: tile_select, but each query attends only
keys at most `window` positions back (FlashAttention's window_size=(window, 0)),
and each 128-query tile picks its key blocks among those its queries can see.

    blocks   `block` keys each, from max(document start, tile start - window) to
             the tile; the tile's own blocks always
    budget   `keys` per tile

For a model trained with sliding windows (record 91's dense attention): the same
reach as in training, fewer keys. Selection has no grad.
"""

from typing import cast

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from nanogpt.tile_select import DEFAULT, Settings, options
from nanogpt.tournament_fast import TILE
from nanogpt.tournament_local import starts


# Opaque, as tile_select.picks: flex's lowering needs fixed-layout mask inputs
@torch.library.custom_op("tile_window::picks", mutates_args=())
def picks(
    q: Tensor, k: Tensor, cu_seqlens: Tensor, block: int, keys: int, window: int
) -> tuple[Tensor, Tensor]:
    """Flex's KV block counts (1, H, tiles) and indices (1, H, tiles, key blocks):
    each tile's picks first (its own blocks, then the best earlier by score)."""
    length, heads, dim = q.shape
    n, nb, own = length // TILE, length // block, TILE // block
    tile_q = q.view(n, TILE, heads, dim).mean(1, dtype=torch.float32)
    block_k = k.view(nb, block, heads, dim).mean(1, dtype=torch.float32)
    score = torch.einsum("ihd,jhd->hij", tile_q, block_k)
    tile = torch.arange(n, device=q.device)
    first = TILE * tile
    found = torch.searchsorted(cu_seqlens, first.to(cu_seqlens.dtype), right=True)
    start = torch.maximum(cu_seqlens[found - 1].long(), first - window)
    j = torch.arange(nb, device=q.device)
    mine = j[None] // own == tile[:, None]
    earlier = ((j[None] + 1) * block > start[:, None]) & (
        j[None] // own < tile[:, None]
    )
    score = score.masked_fill(~earlier, float("-inf")).masked_fill(mine, float("inf"))
    index = score.argsort(-1, descending=True).int()
    extra = earlier.sum(1).clamp(max=keys // block - own)
    count = (own + extra).int().expand(heads, n).contiguous()
    return count[None], index[None]


@picks.register_fake
def _(
    q: Tensor, k: Tensor, cu_seqlens: Tensor, block: int, keys: int, window: int
) -> tuple[Tensor, Tensor]:
    length, heads, _ = q.shape
    n, nb = length // TILE, length // block
    count = q.new_empty(1, heads, n, dtype=torch.int32)
    return count, q.new_empty(1, heads, n, nb, dtype=torch.int32)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    window: int,
    settings: Settings = DEFAULT,
) -> Tensor:
    """Exact attention of each query on its document prefix's last window + 1 keys
    inside its tile's picked key blocks; q, k, v (T, H, D) one packed row, T a
    multiple of 128."""
    length, _, _ = q.shape
    s = settings
    assert length % TILE == 0 and TILE % s.block == 0, (length, s)
    assert s.keys >= TILE and not s.grow, s
    count, index = picks(q.detach(), k.detach(), cu_seqlens, s.block, s.keys, window)
    start = starts(cu_seqlens, length)

    def prefix(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        near = q_idx - kv_idx <= window
        return (kv_idx <= q_idx) & (kv_idx >= start[q_idx]) & near

    mask = BlockMask.from_kv_blocks(
        count,
        index,
        BLOCK_SIZE=(TILE, s.block),
        mask_mod=prefix,
        seq_lengths=(length, length),
    )
    q, k, v = (x.transpose(0, 1)[None] for x in (q, k, v))
    out = flex_attention(
        q, k, v, block_mask=mask, scale=scale, kernel_options=options(s.block)
    )
    return cast(Tensor, out)[0].transpose(0, 1)
