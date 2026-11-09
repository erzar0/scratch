"""Triton SparQ decode, the rule of attention/sparq.py: _sparq_scores (r rows of
K_T, every key, each head), each head's softmax summed over the group and the
local window in torch, torch.topk, then common.attend. Shapes as in
attention/triton/common.py."""

import torch
import triton
import triton.language as tl

from attention.triton.common import BLOCK, attend

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)


@triton.jit
def _sparq_scores(
    q_ptr,
    dims_ptr,
    scale_ptr,
    kt_ptr,
    out_ptr,
    n,
    s_kt_g,
    s_kt_d,
    R: tl.constexpr,
    B: tl.constexpr,
):
    """One program per (KV head, B keys): each head's scaled partial logits over
    the R dims dims_ptr lists."""
    g = tl.program_id(0)
    pos = tl.program_id(1) * B + tl.arange(0, B)
    alive = pos < n
    a0 = tl.zeros((B,), tl.float32)
    a1 = tl.zeros((B,), tl.float32)
    for j in tl.static_range(R):
        d = tl.load(dims_ptr + g * R + j)
        k = tl.load(kt_ptr + g * s_kt_g + d * s_kt_d + pos, mask=alive, other=0.0)
        k = k.to(tl.float32)
        a0 += tl.load(q_ptr + (g * 2) * R + j) * k
        a1 += tl.load(q_ptr + (g * 2 + 1) * R + j) * k
    s0 = a0 * tl.load(scale_ptr + g * 2)
    s1 = a1 * tl.load(scale_ptr + g * 2 + 1)
    tl.store(out_ptr + (g * 2) * n + pos, s0, mask=alive)
    tl.store(out_ptr + (g * 2 + 1) * n + pos, s1, mask=alive)


def select(
    q: torch.Tensor, k_t: torch.Tensor, r: int, top: int, local: int = 16
) -> torch.Tensor:
    """(G, top) SparQ's picks for one decode token."""
    groups, dim, n = k_t.shape
    qf = q.float()
    q_abs = qf.abs()
    dims = q_abs.sum(1).topk(r, -1).indices
    picked = dims[:, None].expand(-1, 2, -1)
    q_sel = qf.gather(-1, picked).contiguous()
    frac = q_abs.gather(-1, picked).sum(-1) / q_abs.sum(-1)
    scale = (dim**-0.5 / frac.sqrt()).contiguous()
    score = torch.empty(groups, 2, n, device=q.device)
    _sparq_scores[(groups, triton.cdiv(n, BLOCK))](
        q_sel,
        dims.int().contiguous(),
        scale,
        k_t,
        score,
        n,
        s_kt_g=k_t.stride(0),
        s_kt_d=k_t.stride(1),
        R=r,
        B=BLOCK,
    )
    score = score.softmax(-1).sum(1)  # SparQ's GQA rule
    score[:, n - local :] = float("inf")  # the local window always wins
    return score.topk(top, -1).indices.int()


def sparq(
    q: torch.Tensor,
    k: torch.Tensor,
    k_t: torch.Tensor,
    v: torch.Tensor,
    r: int,
    top: int,
    local: int = 16,
) -> torch.Tensor:
    """(G, 2, D) SparQ attention for one decode token."""
    return attend(q, k, v, select(q, k_t, r, top, local))
