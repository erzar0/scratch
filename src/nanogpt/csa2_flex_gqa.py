"""csa2_flex.attention with one shared K=V window head (csa2_window.shared_window
with heads=1): flex attention's GQA mapping lets every query head read the one window
head and the entries in place, instead of 6 copies of each. Same output and gradients
as csa2_flex.attention with the window expanded to every head
(tests/nanogpt/test_csa2_flex_gqa.py).
"""

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from nanogpt.csa2_flex import mask, mask_inputs, tiles
from nanogpt.tournament_fast import TILE, MaskMod, flex_options


def attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    count: Tensor,
    index: Tensor,
    mask_mod: MaskMod,
) -> Tensor:
    """tournament_fast.attend with k, v (T, 1, D) shared by every query head."""
    lengths = q.shape[0], k.shape[0]
    block_mask = BlockMask.from_kv_blocks(
        count, index, BLOCK_SIZE=TILE, mask_mod=mask_mod, seq_lengths=lengths
    )
    q, k, v = (x.transpose(0, 1)[None] for x in (q, k, v))
    out = flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=True,
        kernel_options=flex_options(),
    )
    return cast(Tensor, out)[0].transpose(0, 1)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    keys: Tensor,
    values: Tensor,
    entry: Tensor,
    valid: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    m: int = 4,
) -> Tensor:
    """csa2_flex.attention with the window k, v (T, 1, D): one head for all queries."""
    length, n = q.shape[0], keys.shape[0]
    pad = -n % TILE
    keys, values = (F.pad(x, (0, 0, 0, pad)) for x in (keys, values))
    begin, bits = mask_inputs(entry, valid, cu_seqlens, length, n + pad)
    count, index = tiles(begin, length, n, m)
    k = torch.cat([k, keys[:, None]])
    v = torch.cat([v, values[:, None]])
    return attend(q, k, v, scale, count, index, mask(begin, bits, length))
