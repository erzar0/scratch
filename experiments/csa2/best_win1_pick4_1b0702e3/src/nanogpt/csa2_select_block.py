"""Block prefilter picks for CSA2's compressed KV, parameter-free (an inference rule:
eval only, never trained). Entries are grouped in aligned blocks of `size`; each
query scores every block it may read in full by one summary of the block's keys,
keeps the `keep` best blocks plus the blocks it may read only in part (its document's
first and its own last: scored exactly, so no later or other-document entry steers a
pick), then picks the top_k kept entries by the full score (sum_h q_h) . kv.

Summaries (qs: the heads' summed query; k over the block's entries, per channel):
    mean     qs . mean(k)                           (Qwen3.8-Next QSA, arXiv:2608.30320)
    max      qs . max(k)
    minmax   relu(qs) . max(k) + min(qs, 0) . min(k): an upper bound on every
             entry's score in the block (Quest)
    meanstd  qs . mean(k) + |qs| . std(k)          (UNIQUE, arXiv:2605.27740)
Arguments and outputs as csa2_select.exact; CHUNK queries at a time.
"""

import torch
from torch import Tensor

from nanogpt.csa2_select import CHUNK, _allowed

SUMMARIES = ("mean", "max", "minmax", "meanstd")


def block_scores(qs: Tensor, blocks: Tensor, summary: str) -> Tensor:
    """(Q, B) score of queries qs (Q, D) on blocks (B, size, D) by summary."""
    if summary == "mean":
        return qs @ blocks.mean(1).T
    if summary == "max":
        return qs @ blocks.amax(1).T
    if summary == "minmax":
        return qs.clamp(min=0) @ blocks.amax(1).T + qs.clamp(max=0) @ blocks.amin(1).T
    assert summary == "meanstd", summary
    return qs @ blocks.mean(1).T + qs.abs() @ blocks.std(1, correction=0).T


@torch.library.custom_op("csa2_select_block::block", mutates_args=())
def picks(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int,
    size: int,
    keep: int,
    summary: str,
) -> tuple[Tensor, Tensor]:
    """block's picks; opaque to torch.compile."""
    with torch.no_grad():
        return _block(q, kv, starts, m, top_k, size, keep, summary)


@picks.register_fake
def _(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int,
    size: int,
    keep: int,
    summary: str,
) -> tuple[Tensor, Tensor]:
    shape = (q.shape[0], top_k)
    return (
        q.new_empty(shape, dtype=torch.long),
        q.new_empty(shape, dtype=torch.bool),
    )


def block(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int = 64,
    size: int = 16,
    keep: int = 64,
    summary: str = "mean",
) -> tuple[Tensor, Tensor]:
    """top_k picks among the entries of each query's kept blocks (module docstring)."""
    return picks(q.detach(), kv.detach(), starts, m, top_k, size, keep, summary)


def _block(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int,
    size: int,
    keep: int,
    summary: str,
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
        inside = allowed.unflatten(-1, (-1, size))
        full, part = inside.all(-1), inside.any(-1) & ~inside.all(-1)
        score = block_scores(qs[a:b], keys.unflatten(0, (-1, size)), summary)
        score = score.masked_fill(~full, float("-inf"))
        best = score.topk(min(keep, score.shape[1]), -1).indices
        kept = torch.zeros_like(full).scatter(1, best, True) & full | part
        kept = kept.repeat_interleave(size, 1) & allowed
        exact = (qs[a:b] @ keys.T).masked_fill(~kept, float("-inf"))
        top, index = exact.topk(min(top_k, hi - lo), -1)
        entry[a:b, : top.shape[1]] = index + lo
        valid[a:b, : top.shape[1]] = top > float("-inf")
    return entry, valid
