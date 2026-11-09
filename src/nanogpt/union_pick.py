"""Union picks, exact attention: union_dims' picks (each query's best `keys` keys of
its document prefix, spread over the shuffled windows, scored on the union of the
query's and each key's top `dims` |channels|), then softmax(scale * q.k) over the
picks on every channel. Separates picking quality from union_dims' approximate
final logits. One pass (union_dims' rounds tied it). Torch, no grad: for checkpoint
evaluation (loss only, no speed). Shapes as union_dims.
"""

import math

import torch
from torch import Tensor

from nanogpt.union_dims import CHUNK, DEFAULT, Settings, picks, union_logits


@torch.library.custom_op("union_pick::attention", mutates_args=())
def _attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    block: int,
    keys: int,
    dims: int,
) -> Tensor:
    s = Settings(block, keys, dims, "pass")
    out = torch.empty_like(q)
    for first in range(0, q.shape[0], CHUNK):
        part = slice(first, first + CHUNK)
        keep = picks({dims: union_logits(q[part], k, dims)}, cu_seqlens, first, s)
        logits = torch.einsum("qhd,shd->hqs", q[part].float(), k.float())
        weights = (scale * logits).masked_fill(~keep, -math.inf).softmax(-1)
        out[part] = torch.einsum("hqs,shd->qhd", weights, v.float()).to(q.dtype)
    return out


@_attention.register_fake
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    block: int,
    keys: int,
    dims: int,
) -> Tensor:
    return torch.empty_like(q)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    settings: Settings = DEFAULT,
) -> Tensor:
    """softmax(scale * q.k) over each query's union picks, times v; (T, H, D)."""
    s = settings
    return _attention(q, k, v, cu_seqlens, scale, s.block, s.keys, s.dims)
