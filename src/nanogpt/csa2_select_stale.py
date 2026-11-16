"""Block picks with the block step refreshed every `every` queries, parameter-free
(an inference rule: eval only, never trained): a decoder that scores the block means
only every `every` tokens (LServe, ReTopK) and rescores the kept entries exactly at
each token. Query t keeps the `keep` best full blocks by the block means as query
r = t - t % every ranked them (among the blocks r could read), r's partly readable
blocks, and every entry finished since r; its picks are the top_k of those by its
own exact score. every 1 is csa2_select_block.block with the "mean" summary.
Arguments and outputs as csa2_select_block.block.
"""

import torch
from torch import Tensor

from nanogpt.csa2_select import CHUNK, _allowed
from nanogpt.csa2_select_block import kept_entries


@torch.library.custom_op("csa2_select_stale::stale", mutates_args=())
def picks(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int,
    size: int,
    keep: int,
    every: int,
) -> tuple[Tensor, Tensor]:
    """stale's picks; opaque to torch.compile."""
    with torch.no_grad():
        return _stale(q, kv, starts, m, top_k, size, keep, every)


@picks.register_fake
def _(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int,
    size: int,
    keep: int,
    every: int,
) -> tuple[Tensor, Tensor]:
    shape = (q.shape[0], top_k)
    return (
        q.new_empty(shape, dtype=torch.long),
        q.new_empty(shape, dtype=torch.bool),
    )


def stale(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int = 64,
    size: int = 8,
    keep: int = 256,
    every: int = 8,
) -> tuple[Tensor, Tensor]:
    """top_k picks with blocks ranked every `every` queries (module docstring)."""
    return picks(q.detach(), kv.detach(), starts, m, top_k, size, keep, every)


def _stale(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int,
    size: int,
    keep: int,
    every: int,
) -> tuple[Tensor, Tensor]:
    length = q.shape[0]
    entry = torch.zeros(length, top_k, dtype=torch.long, device=q.device)
    valid = torch.zeros(length, top_k, dtype=torch.bool, device=q.device)
    qs = q.float().sum(1)
    for a in range(0, length, CHUNK):
        b = min(a + CHUNK, length)
        lo, hi = -(-int(starts[a]) // m), min(b // m, kv.shape[0])
        if hi <= lo:
            continue
        lo, hi = lo // size * size, -(-hi // size) * size  # whole blocks
        keys = kv[lo:hi].float()
        keys = torch.cat([keys, keys.new_zeros(hi - lo - len(keys), keys.shape[1])])
        allowed = _allowed(starts, m, a, b, lo, hi)
        t = torch.arange(a, b, device=q.device)
        r = t - t % every
        e = torch.arange(lo, hi, device=q.device)
        then = (e * m >= starts[a:b, None]) & (e < (r[:, None] + 1) // m)
        kept = kept_entries(qs[r], keys, then, size, keep, "mean")
        kept = (kept | allowed & ~then) & allowed
        exact = (qs[a:b] @ keys.T).masked_fill(~kept, float("-inf"))
        top, index = exact.topk(min(top_k, hi - lo), -1)
        entry[a:b, : top.shape[1]] = index + lo
        valid[a:b, : top.shape[1]] = top > float("-inf")
    return entry, valid
