"""Exact picks by the largest head logit, parameter-free (an inference rule: eval
only, never trained): each query's top_k allowed entries by max_h q_h . kv instead of
csa2_select.exact's (sum_h q_h) . kv, so an entry one head needs is not outvoted by
heads that ignore it (Kascade's head-aware picks). Arguments and outputs as
csa2_select.exact; CHUNK queries at a time.
"""

import torch
from torch import Tensor

from nanogpt.csa2_select import CHUNK, _allowed


@torch.library.custom_op("csa2_select_head_max::head_max", mutates_args=())
def picks(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int
) -> tuple[Tensor, Tensor]:
    """head_max's picks; opaque to torch.compile."""
    with torch.no_grad():
        return _head_max(q, kv, starts, m, top_k)


@picks.register_fake
def _(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int
) -> tuple[Tensor, Tensor]:
    shape = (q.shape[0], top_k)
    return (
        q.new_empty(shape, dtype=torch.long),
        q.new_empty(shape, dtype=torch.bool),
    )


def head_max(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int = 64
) -> tuple[Tensor, Tensor]:
    """top_k picks by the largest head logit (module docstring)."""
    return picks(q.detach(), kv.detach(), starts, m, top_k)


def _head_max(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int
) -> tuple[Tensor, Tensor]:
    length = q.shape[0]
    entry = torch.zeros(length, top_k, dtype=torch.long, device=q.device)
    valid = torch.zeros(length, top_k, dtype=torch.bool, device=q.device)
    for a in range(0, length, CHUNK):
        b = min(a + CHUNK, length)
        lo, hi = -(-int(starts[a]) // m), min(b // m, kv.shape[0])
        if hi <= lo:
            continue
        score = torch.einsum("qhd,ed->qhe", q[a:b].float(), kv[lo:hi].float()).amax(1)
        score = score.masked_fill(~_allowed(starts, m, a, b, lo, hi), float("-inf"))
        top, index = score.topk(min(top_k, hi - lo), -1)
        entry[a:b, : top.shape[1]] = index + lo
        valid[a:b, : top.shape[1]] = top > float("-inf")
    return entry, valid
