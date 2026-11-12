"""A decode-friendly pick rule for CSA2's compressed KV, applied to models trained with
csa2_select (an inference rule: eval only, never trained).

prefilter: each query scores every allowed entry on one aligned block of `dims`
channels (the block with the largest sum of |q| over the heads' summed query; one
32-byte sector per bf16 entry when dims = 16), keeps every entry whose score falls
in or above the histogram bin of its keep-th best (bins: the top 11 bits of the
score's order-preserving int32, the bins csa2_decode's kernels count), then picks
the top_k of those by the full score (sum_h q_h) . kv. Queries with keep or fewer
allowed entries get exact picks.

Arguments and outputs as csa2_select.exact; CHUNK queries at a time.
"""

import torch
from torch import Tensor

from nanogpt.csa2_select import CHUNK, _allowed

BIN_SHIFT = 21  # 32 - 11 bits


def bins(score: Tensor) -> Tensor:
    """Histogram bin of each fp32 score: top 11 bits of its sortable int32."""
    bits = score.view(torch.int32)
    return torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits) >> BIN_SHIFT


def block_of(qs: Tensor, dims: int) -> Tensor:
    """(S,) index of each query's aligned `dims` block with the largest sum |q|."""
    return qs.abs().unflatten(-1, (-1, dims)).sum(-1).argmax(-1)


@torch.library.custom_op("csa2_select_decode::prefilter", mutates_args=())
def picks(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int, keep: int, dims: int
) -> tuple[Tensor, Tensor]:
    """prefilter's picks; opaque to torch.compile."""
    with torch.no_grad():
        return _prefilter(q, kv, starts, m, top_k, keep, dims)


@picks.register_fake
def _(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int, keep: int, dims: int
) -> tuple[Tensor, Tensor]:
    shape = (q.shape[0], top_k)
    return (
        q.new_empty(shape, dtype=torch.long),
        q.new_empty(shape, dtype=torch.bool),
    )


def prefilter(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int = 64,
    keep: int = 1024,
    dims: int = 16,
) -> tuple[Tensor, Tensor]:
    """top_k picks among the entries a one-block prefilter keeps (module docstring)."""
    return picks(q.detach(), kv.detach(), starts, m, top_k, keep, dims)


def _prefilter(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int, keep: int, dims: int
) -> tuple[Tensor, Tensor]:
    length, dim = q.shape[0], q.shape[-1]
    entry = torch.zeros(length, top_k, dtype=torch.long, device=q.device)
    valid = torch.zeros(length, top_k, dtype=torch.bool, device=q.device)
    qs = q.float().sum(1)
    block = block_of(qs, dims)
    channel = torch.arange(dim, device=q.device) // dims
    q_block = qs * (channel == block[:, None])
    for a in range(0, length, CHUNK):
        b = min(a + CHUNK, length)
        lo, hi = -(-int(starts[a]) // m), min(b // m, kv.shape[0])
        if hi <= lo:
            continue
        keys = kv[lo:hi].float()
        allowed = _allowed(starts, m, a, b, lo, hi)
        first = (q_block[a:b] @ keys.T).masked_fill(~allowed, float("-inf"))
        bin_ = bins(first).masked_fill(~allowed, torch.iinfo(torch.int32).min)
        rank = min(keep, hi - lo)
        cut = bin_.topk(rank, -1).values[:, -1:]
        kept = allowed & ((bin_ >= cut) | (allowed.sum(-1, keepdim=True) <= keep))
        score = (qs[a:b] @ keys.T).masked_fill(~kept, float("-inf"))
        top, index = score.topk(min(top_k, hi - lo), -1)
        entry[a:b, : top.shape[1]] = index + lo
        valid[a:b, : top.shape[1]] = top > float("-inf")
    return entry, valid
