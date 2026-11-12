"""Parameter-free picks for CSA2's compressed KV (DeepSeek-V4.1 CSA2): each query's
top_k entries of the shared K=V head, scored by the logit summed over the query
heads, (sum_h q_h) . kv. Replaces CSA2's learned indexer: no indexer weights, no
indexer loss, no indexer K in the cache.

exact: the top_k entries by that score.
tournament: run 3's rule (tournament_channel_blocks) on the entries. A prefix of n
entries runs the doublings of top_k below n (at most D // dims rounds). Each round
adds the next window of `dims` adjacent channels of the summed q (windows()) to the
partial score and keeps the best top_k << (rounds left) entries, ties kept, so the
last round leaves top_k. One pool per query, not run 3's shuffled key blocks.

Entry e holds tokens [e m, (e + 1) m). Query t may pick it once finished,
e < (t + 1) // m, and only inside its own document, e m >= start (as the archived
csa2.top_picks and VarlenIndexer). No grad.

Shapes: q (S, H, D) for one packed row, kv (N, D), starts (S,) int64 document start
of each query. Returns entry (S, top_k) int64 and valid (S, top_k) bool; a short
prefix leaves invalid extras at entry 0.
"""

import torch
from torch import Tensor

from nanogpt.tournament_channel_blocks import windows
from nanogpt.tournament_fast import Settings

CHUNK = 2048  # queries per pass


def _allowed(starts: Tensor, m: int, a: int, b: int, lo: int, hi: int) -> Tensor:
    """(b - a, hi - lo) bool: queries a..b-1 may pick entries lo..hi-1."""
    t = torch.arange(a, b, device=starts.device)[:, None]
    e = torch.arange(lo, hi, device=starts.device)
    return (e * m >= starts[a:b, None]) & (e < (t + 1) // m)


def _halve(
    partials: list[Tensor], allowed: Tensor, doubles: Tensor, top_k: int
) -> Tensor:
    """(Q, E) summed partial scores, -inf for entries out after each query's rounds
    (`doubles` per query; partials holds the most rounds any query runs)."""
    score = torch.zeros(allowed.shape, device=allowed.device)
    alive = allowed
    for r, partial in enumerate(partials):
        score = score + partial
        live = score.masked_fill(~alive, float("-inf"))
        budget = top_k * 2 ** (doubles - r - 1).clamp(min=0)
        floor = live.sort(-1, descending=True).values.gather(
            -1, (budget - 1).clamp(max=live.shape[-1] - 1)[:, None]
        )
        alive = alive & ((r >= doubles)[:, None] | (live >= floor))
    return score.masked_fill(~alive, float("-inf"))


def _picks(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int, dims: int
) -> tuple[Tensor, Tensor]:
    """exact (dims == 0) or tournament picks, CHUNK queries at a time."""
    length, dim = q.shape[0], q.shape[-1]
    entry = torch.zeros(length, top_k, dtype=torch.long, device=q.device)
    valid = torch.zeros(length, top_k, dtype=torch.bool, device=q.device)
    qs = q.float().sum(1)
    if dims:
        rounds = dim // dims
        values, order = windows(qs[:, None], Settings(top_k << rounds, top_k, dims))
    for a in range(0, length, CHUNK):
        b = min(a + CHUNK, length)
        lo, hi = -(-int(starts[a]) // m), min(b // m, kv.shape[0])  # starts ascend
        if hi <= lo:
            continue
        keys = kv[lo:hi].float()
        allowed = _allowed(starts, m, a, b, lo, hi)
        if dims:
            n = allowed.sum(-1)
            doubles = (
                n[:, None] > top_k * 2 ** torch.arange(rounds, device=n.device)
            ).sum(-1)
            window = [slice(r * dims, (r + 1) * dims) for r in range(rounds)]
            partials = [
                torch.zeros_like(qs[a:b]).scatter(
                    -1, order[0, a:b, w].long(), values[0, a:b, w]
                )
                @ keys.T
                for w in window[: int(doubles.max())]
            ]
            score = _halve(partials, allowed, doubles, top_k)
        else:
            score = (qs[a:b] @ keys.T).masked_fill(~allowed, float("-inf"))
        top, index = score.topk(min(top_k, hi - lo), -1)
        entry[a:b, : top.shape[1]] = index + lo
        valid[a:b, : top.shape[1]] = top > float("-inf")
    return entry, valid


@torch.library.custom_op("csa2_select::picks", mutates_args=())
def picks(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int, dims: int
) -> tuple[Tensor, Tensor]:
    """exact (dims 0) or tournament (dims > 0) picks; opaque to torch.compile."""
    with torch.no_grad():
        return _picks(q, kv, starts, m, top_k, dims)


@picks.register_fake
def _(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int, dims: int
) -> tuple[Tensor, Tensor]:
    shape = (q.shape[0], top_k)
    return (
        q.new_empty(shape, dtype=torch.long),
        q.new_empty(shape, dtype=torch.bool),
    )


def exact(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int = 64
) -> tuple[Tensor, Tensor]:
    """Each query's top_k allowed entries by (sum_h q_h) . kv."""
    return picks(q.detach(), kv.detach(), starts, m, top_k, 0)


def tournament(
    q: Tensor, kv: Tensor, starts: Tensor, m: int, top_k: int = 64, dims: int = 16
) -> tuple[Tensor, Tensor]:
    """Each query's top_k allowed entries by the halving rounds (module docstring)."""
    return picks(q.detach(), kv.detach(), starts, m, top_k, dims)
