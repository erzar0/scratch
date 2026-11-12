"""Routed channels for CSA2's picked entries (MoE over attention channels): a router
per attention layer gives each token a softmax over `blocks` blocks of q's first
`width` channels and keeps the top `keep`. Picks (csa2_select on q_routed) and the
logits on picked entries use only those channels, gate-weighted; the window keeps
every channel. One route per token, shared by all heads, so a decode pick reads
`keep` blocks of each entry (contiguous in a block-major cache).

Gate of a kept block b: sqrt(blocks / keep) * keep * p_b / (sum of kept p). Equal
gates scale by sqrt(blocks / keep), so the routed logit's spread matches the full
one; the router learns through the gates, as an MoE router.

attention() is csa2_flex.attention with q and k twice as wide: [q, q_routed] against
[k, 0] (window) and [0, entry keys], values unchanged.
Shapes: q, k, v (T, H, D) for one packed row, T a multiple of 128; x (T, C); router
w (blocks, C); entries (N, D).
"""

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    FlexKernelOptions,
    flex_attention,
)

from nanogpt import csa2_flex
from nanogpt.tournament_fast import TILE

# q, k 256 wide: tournament_fast's swept tiles (128 x 32, 3 stages) exceed 101 KB;
# bwd M1 32 fits torch 2.10 but needs 102400 B in 2.14 (limit 101376), so 16
OPTIONS = {"fwd_BLOCK_M": 64, "fwd_BLOCK_N": 32, "fwd_num_stages": 2,
           "fwd_num_warps": 4, "bwd_BLOCK_M1": 16, "bwd_BLOCK_N1": 32,
           "bwd_BLOCK_M2": 32, "bwd_BLOCK_N2": 32, "bwd_num_stages": 1,
           "bwd_num_warps": 4}  # fmt: skip


def route(
    x: Tensor, w: Tensor, q: Tensor, width: int, blocks: int = 6, keep: int = 2
) -> Tensor:
    """q zeroed outside each token's kept blocks of its first `width` channels, kept
    channels times their block's gate."""
    p = F.linear(x, w).float().softmax(-1)
    top, kept = p.topk(keep, -1)
    gate = top / top.sum(-1, keepdim=True) * (keep * math.sqrt(blocks / keep))
    g = torch.zeros_like(p).scatter(-1, kept, gate)
    g = g.repeat_interleave(width // blocks, -1)
    return q * F.pad(g, (0, q.shape[-1] - width))[:, None].type_as(q)


def attention(
    q: Tensor,
    q_routed: Tensor,
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
    """(T, H, D) softmax over each query's window keys (logits q . k) and picked
    entries (logits q_routed . keys); arguments as csa2_flex.attention."""
    length, heads, dim = q.shape
    n = keys.shape[0]
    pad = -n % TILE
    keys, values = (F.pad(x, (0, 0, 0, pad)) for x in (keys, values))
    begin, bits = csa2_flex.mask_inputs(entry, valid, cu_seqlens, length, n + pad)
    count, index = csa2_flex.tiles(begin, length, n, m)
    q = torch.cat([q, q_routed], -1)
    entries = F.pad(keys, (dim, 0))[:, None].expand(-1, heads, -1)
    k = torch.cat([F.pad(k, (0, dim)), entries])
    v = torch.cat([v, values[:, None].expand(-1, heads, -1)])
    mask = BlockMask.from_kv_blocks(
        count,
        index,
        BLOCK_SIZE=TILE,
        mask_mod=csa2_flex.mask(begin, bits, length),
        seq_lengths=(length, k.shape[0]),
    )
    q, k, v = (x.transpose(0, 1)[None] for x in (q, k, v))
    options = cast(FlexKernelOptions, OPTIONS)  # the typed options lack fwd_ keys
    out = flex_attention(q, k, v, block_mask=mask, scale=scale, kernel_options=options)
    return cast(Tensor, out)[0].transpose(0, 1)
