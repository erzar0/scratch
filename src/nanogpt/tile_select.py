"""Tile selection: each tile of 128 queries attends the key blocks covering its own
positions plus its best earlier key blocks by pooled score (mean q over the tile .
mean k over the block), up to a key budget. Flex computes only those blocks.

    blocks   `block` keys each (16, 32, 64 or 128), from the tile's first query's
             document start to the tile; the tile's own blocks always
    budget   fixed: `keys` per tile; grow: `keys` per 2048 keys of the tile's
             longest prefix (rounded up)

Attention is exact over each query's document prefix inside the picked blocks
(tournament_local.dense's mask). Selection has no grad.
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

from nanogpt.tournament_fast import TILE, flex_options
from nanogpt.tournament_local import starts


def _small_bwd_config() -> None:
    """Torch 2.10 drops flex backward configs whose default tiles (64 on sm8x) do
    not divide the mask's blocks before kernel_options apply, leaving none for
    blocks of 16 or 32: add a 16-tile config (its tiles and warps then come from
    kernel_options; torch 2.14 shrinks the default itself)."""
    from torch._inductor.choices import InductorChoices
    from torch._inductor.template_heuristics.triton import (  # type: ignore[import-not-found]
        FlexBwDConfig,
    )

    configs = InductorChoices.get_flex_attention_bwd_configs
    small = FlexBwDConfig(16, 16, 16, 16, 1, 4)

    def with_small(
        self: InductorChoices,
        head_dim: int,
        dtype: torch.dtype,
        device_type: str | None = "cuda",
    ) -> list:
        found = configs(self, head_dim, dtype, device_type)
        return found if small in found else [*found, small]

    InductorChoices.get_flex_attention_bwd_configs = with_small  # type: ignore[method-assign]


if tuple(map(int, torch.__version__.split(".")[:2])) < (2, 11):  # fork, r91 venvs
    _small_bwd_config()


@dataclass(frozen=True)
class Settings:
    """Keys per block, key budget per query tile, budget growing with the prefix."""

    block: int = 16
    keys: int = 512
    grow: bool = False


DEFAULT = Settings()


# Opaque, as tournament_fast.tiles: flex's lowering needs fixed-layout mask inputs
@torch.library.custom_op("tile_select::picks", mutates_args=())
def picks(
    q: Tensor, k: Tensor, cu_seqlens: Tensor, block: int, keys: int, grow: bool
) -> tuple[Tensor, Tensor]:
    """Flex's KV block counts (1, H, tiles) and indices (1, H, tiles, key blocks):
    each tile's picks first (its own blocks, then the best earlier by score)."""
    length, heads, dim = q.shape
    n, nb, own = length // TILE, length // block, TILE // block
    tile_q = q.view(n, TILE, heads, dim).mean(1, dtype=torch.float32)
    block_k = k.view(nb, block, heads, dim).mean(1, dtype=torch.float32)
    score = torch.einsum("ihd,jhd->hij", tile_q, block_k)
    tile = torch.arange(n, device=q.device)
    first = (TILE * tile).to(cu_seqlens.dtype)
    start = cu_seqlens[torch.searchsorted(cu_seqlens, first, right=True) - 1].long()
    j = torch.arange(nb, device=q.device)
    mine = j[None] // own == tile[:, None]
    earlier = ((j[None] + 1) * block > start[:, None]) & (
        j[None] // own < tile[:, None]
    )
    score = score.masked_fill(~earlier, float("-inf")).masked_fill(mine, float("inf"))
    index = score.argsort(-1, descending=True).int()
    budget = keys * ((TILE * (tile + 1) - start + 2047) // 2048) if grow else keys
    extra = earlier.sum(1).clamp(max=budget // block - own)  # keys >= TILE
    count = (own + extra).int().expand(heads, n).contiguous()
    return count[None], index[None]


@picks.register_fake
def _(
    q: Tensor, k: Tensor, cu_seqlens: Tensor, block: int, keys: int, grow: bool
) -> tuple[Tensor, Tensor]:
    length, heads, _ = q.shape
    n, nb = length // TILE, length // block
    count = q.new_empty(1, heads, n, dtype=torch.int32)
    return count, q.new_empty(1, heads, n, nb, dtype=torch.int32)


def options(block: int) -> FlexKernelOptions:
    """tournament_fast.flex_options() with key tiles, and the dk/dv kernel's query
    tile, at most `block`: flex needs its kernels' key tiles to divide the mask's
    key block, and that query tile to divide its key tile."""
    out = cast(dict[str, int], dict(flex_options()))
    for key in ("fwd_BLOCK_N", "bwd_BLOCK_N1", "bwd_BLOCK_N2", "bwd_BLOCK_M1"):
        if key in out or block < TILE:
            out[key] = min(out.get(key, block), block)
    return cast(FlexKernelOptions, out)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    settings: Settings = DEFAULT,
) -> Tensor:
    """Exact attention of each query on its document prefix inside its tile's
    picked key blocks; q, k, v (T, H, D) one packed row, T a multiple of 128."""
    length, _, _ = q.shape
    s = settings
    assert length % TILE == 0 and TILE % s.block == 0 and s.keys >= TILE, (length, s)
    count, index = picks(q.detach(), k.detach(), cu_seqlens, s.block, s.keys, s.grow)
    start = starts(cu_seqlens, length)

    def prefix(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return (kv_idx <= q_idx) & (kv_idx >= start[q_idx])

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
