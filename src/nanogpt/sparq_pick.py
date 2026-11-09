"""SparQ-rule picks, exact attention: as union_pick, but keys are scored on the
query's top `dims` |q| channels only (SparQ, Ribar et al. 2023, step 1, per head):

    s(q, k) = sum over d in top(|q|) of q_d k_d

then each query's best `keys` keys over the shuffled windows (union_dims.picks) and
softmax(scale * q.k) over the picks on every channel. In decode this reads `dims`
channels per key instead of union_pick's `dims` channels plus each key's own top
`dims` values and indices. Torch, no grad: loss only. Shapes as union_dims.
"""

import math

import torch
from torch import Tensor

from nanogpt.union_dims import CHUNK, DEFAULT, Settings, picks, top


def sparq_logits(q: Tensor, k: Tensor, dims: int) -> Tensor:
    """(H, Q, S) float32 s(q, k) for q (Q, H, D) and k (S, H, D)."""
    q = q.float()
    return torch.einsum("qhd,shd->hqs", q * top(q, dims), k.float())


@torch.library.custom_op("sparq_pick::attention", mutates_args=())
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
        keep = picks({dims: sparq_logits(q[part], k, dims)}, cu_seqlens, first, s)
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
    """softmax(scale * q.k) over each query's SparQ-rule picks, times v; (T, H, D)."""
    s = settings
    return _attention(q, k, v, cu_seqlens, scale, s.block, s.keys, s.dims)
