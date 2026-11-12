"""Decode (one token, batch 1) of csa2_ref's candidate blocks, as TensorRT-LLM's CSA2
backend runs DeepSeek V4.1-Flash-1017: the candidate source scores every entry once
and keeps the `blocks` aligned blocks of SIZE entries with the largest maximum
score, the newest block pinned; its own top 64 and each later index layer's are
taken inside those blocks (exact for the source: its top 64 lie in its top 64
blocks). csa2_decode.top64 takes the top 64s.

block_max: csa2_decode.top64's indexer score sum_h w_h relu(q_h . k) / sqrt(dim) of
every entry, max per block -> (N / SIZE,) fp32, one Triton kernel.
candidates: the kept blocks' entries (blocks * SIZE,) int64.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

B = 32  # blocks per program


@triton.jit
def _block_max(
    q_ptr,
    w_ptr,
    k_ptr,
    out_ptr,
    n_blocks,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    SIZE: tl.constexpr,
    B: tl.constexpr,
):
    b = tl.program_id(0) * B + tl.arange(0, B)
    e = b[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
    c = tl.arange(0, D)
    ok = b < n_blocks
    k = tl.load(
        k_ptr + e[:, :, None] * D + c[None, None, :], mask=ok[:, None, None], other=0.0
    ).to(tl.float32)
    score = tl.zeros((B, SIZE), dtype=tl.float32)
    for h in tl.static_range(H):
        q = tl.load(q_ptr + h * D + c).to(tl.float32)
        logit = tl.sum(k * q[None, None, :], 2)
        score += tl.maximum(logit, 0.0) * tl.load(w_ptr + h).to(tl.float32)
    tl.store(out_ptr + b, tl.max(score, 1) * scale, mask=ok)


def block_max(q: Tensor, weights: Tensor, keys: Tensor, size: int = 8) -> Tensor:
    """(N / size,) fp32 max indexer score per block of keys (N, D); q (H, D),
    weights (H,)."""
    n, dim = keys.shape
    assert n % size == 0  # ponytail: full blocks only (decode bench lengths)
    n_blocks = n // size
    out = torch.empty(n_blocks, dtype=torch.float32, device=keys.device)
    _block_max[(triton.cdiv(n_blocks, B),)](
        q, weights, keys, out, n_blocks, dim**-0.5,
        H=q.shape[0], D=dim, SIZE=size, B=B, num_warps=4,
    )  # fmt: skip
    return out


def candidates(
    q: Tensor, weights: Tensor, keys: Tensor, blocks: int = 256, size: int = 8
) -> Tensor:
    """The entries (blocks * size,) of the `blocks` best blocks by block_max, the
    newest (last) block pinned, as model.py's select_candidate_blocks."""
    best = block_max(q, weights, keys, size)
    best[-1:].fill_(float("inf"))  # a kernel, not a host copy: CUDA graph safe
    top = best.topk(blocks).indices
    return (top[:, None] * size + torch.arange(size, device=keys.device)).flatten()
