"""Tile keys: each tile of `tile` queries attends the keys at its own positions plus
its best earlier single keys by score (mean q over the tile . k), `keys` keys in
all. Each tile's picked keys and values are gathered into its own slots of one
buffer; flex attends each tile only to its slots, so every query costs `keys` keys
and the matmuls keep tensor cores (`tile` queries x picked keys).

    candidates   keys from the tile's first query's document start to the tile
    slots        `keys` per tile, its own positions first; unused slots masked

Attention is exact over each query's document prefix inside its tile's picks.
Selection has no grad; gradients reach k and v through the gather.
"""

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    FlexKernelOptions,
    flex_attention,
)

from nanogpt import tile_select  # noqa: F401  (torch 2.10: small backward configs)
from nanogpt.tournament_fast import TILE, flex_options
from nanogpt.tournament_local import starts

SLOTS = 128  # gathered slots per flex key block


@dataclass(frozen=True)
class Settings:
    """Queries per tile, keys per tile (its own included)."""

    tile: int = 128
    keys: int = 512


DEFAULT = Settings()


# Opaque, as tile_select.picks: flex's lowering needs fixed-layout mask inputs
@torch.library.custom_op("tile_keys::picks", mutates_args=())
def picks(
    q: Tensor, k: Tensor, cu_seqlens: Tensor, first: int, tile: int, keys: int
) -> tuple[Tensor, Tensor, Tensor]:
    """For the queries q (from position `first`) against all keys k: each slot's
    key position (H, tiles x keys), unused slots = len(k); flex's KV block counts
    (1, H, tiles) and indices (1, H, tiles, slot blocks), each tile's own first."""
    chunk, heads, dim = q.shape
    length = k.shape[0]
    n, per = chunk // tile, keys // SLOTS
    tile_q = q.view(n, tile, heads, dim).mean(1, dtype=torch.float32)
    score = torch.einsum("ihd,jhd->hij", tile_q, k.float())
    tiles = torch.arange(n, device=q.device)
    begin = first + tile * tiles
    found = torch.searchsorted(cu_seqlens, begin.to(cu_seqlens.dtype), right=True)
    start = cu_seqlens[found - 1].long()
    j = torch.arange(length, device=q.device)
    earlier = (j[None] >= start[:, None]) & (j[None] < begin[:, None])
    top = score.masked_fill(~earlier, float("-inf")).topk(keys - tile, dim=-1)
    best = top.indices.masked_fill(top.values == float("-inf"), length)
    own = begin[:, None] + torch.arange(tile, device=q.device)
    pos = torch.cat([own.expand(heads, n, tile), best], -1).int().view(heads, -1)
    used = tile + earlier.sum(1).clamp(max=keys - tile)
    count = ((used + SLOTS - 1) // SLOTS).int().expand(heads, n).contiguous()
    blocks = torch.arange(n * per, device=q.device)
    index = (blocks[None] + per * tiles[:, None]) % (n * per)
    return pos, count[None], index.int().expand(heads, n, -1).contiguous()[None]


@picks.register_fake
def _(
    q: Tensor, k: Tensor, cu_seqlens: Tensor, first: int, tile: int, keys: int
) -> tuple[Tensor, Tensor, Tensor]:
    chunk, heads, _ = q.shape
    n = chunk // tile
    pos = q.new_empty(heads, n * keys, dtype=torch.int32)
    count = q.new_empty(1, heads, n, dtype=torch.int32)
    return pos, count, q.new_empty(1, heads, n, n * keys // SLOTS, dtype=torch.int32)


def options(tile: int) -> FlexKernelOptions:
    """tournament_fast.flex_options() with query tiles at most `tile`: flex needs
    its kernels' query tiles to divide the mask's query block (and the dq kernel's
    key tile to divide its query tile)."""
    out = cast(dict[str, int], dict(flex_options()))
    if tile < TILE:
        for key in ("fwd_BLOCK_M", "bwd_BLOCK_M1", "bwd_BLOCK_M2", "bwd_BLOCK_N2"):
            out[key] = min(out.get(key, tile), tile)
    return cast(FlexKernelOptions, out)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    settings: Settings = DEFAULT,
    chunk: int | None = None,
) -> Tensor:
    """Exact attention of each query on its document prefix inside its tile's
    picked keys; q, k, v (T, H, D) one packed row, T a multiple of the tile. chunk:
    queries per pass (a multiple of the tile; bounds memory, no grad use)."""
    length, _, dim = q.shape
    s, size = settings, chunk or length
    assert length % s.tile == 0 and size % s.tile == 0, (length, size, s)
    assert s.keys % SLOTS == 0 and s.tile < s.keys <= length, (length, s)
    start = starts(cu_seqlens, length)
    kh, vh = k.transpose(0, 1), v.transpose(0, 1)
    out = []
    for first in range(0, length, size):
        qc = q[first : first + size]
        pos, count, index = picks(
            qc.detach(), k.detach(), cu_seqlens, first, s.tile, s.keys
        )
        rows = pos.clamp(max=length - 1).long()[..., None].expand(-1, -1, dim)
        # float32: the gather's backward scatter-adds; bf16 atomics there are slow
        kg, vg = (x.float().gather(1, rows).to(x.dtype)[None] for x in (kh, vh))

        def prefix(
            b: Tensor,
            h: Tensor,
            q_idx: Tensor,
            kv_idx: Tensor,
            pos: Tensor = pos,
            first: int = first,
        ) -> Tensor:
            p = pos[h, kv_idx]
            return (p <= q_idx + first) & (p >= start[q_idx + first])

        mask = BlockMask.from_kv_blocks(
            count,
            index,
            BLOCK_SIZE=(s.tile, SLOTS),
            mask_mod=prefix,
            seq_lengths=(len(qc), pos.shape[1]),
        )
        o = flex_attention(
            qc.transpose(0, 1)[None],
            kg,
            vg,
            block_mask=mask,
            scale=scale,
            kernel_options=options(s.tile),
        )
        out.append(cast(Tensor, o)[0].transpose(0, 1))
    return torch.cat(out) if len(out) > 1 else out[0]
