"""Tournament attention plus a local window: each query attends its tournament_fast
winners and also the last `window` keys of its own document, exactly.

The window keys lie in the key tiles flex already visits (from the query tile's
document start to the diagonal), so the window adds no tiles. Selection is
unchanged: a window key can also be a winner, so the window does not free budget.
"""

import torch
from torch import Tensor

from nanogpt.tournament_fast import (
    DEFAULT,
    TILE,
    Settings,
    attend,
    bit_mask,
    doc_tiles,
    picked,
)


# Opaque, as tournament_fast.tiles: computed in the graph, a mask buffer reaches
# flex's lowering without a fixed layout in no-grad graphs (torch 2.10)
@torch.library.custom_op("tournament_local::starts", mutates_args=())
def starts(cu_seqlens: Tensor, length: int) -> Tensor:
    """(T,) each query's document start."""
    query = torch.arange(length, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    return cu_seqlens[torch.searchsorted(cu_seqlens, query, right=True) - 1]


@starts.register_fake
def _(cu_seqlens: Tensor, length: int) -> Tensor:
    return cu_seqlens.new_empty(length)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    longest: int | None = None,
    settings: Settings = DEFAULT,
    window: int = 128,
) -> Tensor:
    """tournament_fast.attention, the last `window` keys of each query's document
    (itself included) attended too."""
    length, heads, _ = q.shape
    assert length % TILE == 0, length
    bound = length if longest is None else min(longest, length)
    low, count, index = doc_tiles(cu_seqlens, length, heads)
    words = picked(
        q.detach(),
        k.detach(),
        cu_seqlens,
        low,
        bound,
        settings.block,
        settings.winners,
        settings.dims,
    )
    start = starts(cu_seqlens, length)
    winner = bit_mask(words, low)

    def winner_or_local(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        recent = (kv_idx <= q_idx) & (q_idx - kv_idx < window)
        return winner(b, h, q_idx, kv_idx) | (recent & (kv_idx >= start[q_idx]))

    return attend(q, k, v, scale, count, index, winner_or_local)


def dense(q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, scale: float) -> Tensor:
    """Exact attention over each query's document prefix (an unbounded window, no
    selection), on the same tiles."""
    length, heads, _ = q.shape
    assert length % TILE == 0, length
    _, count, index = doc_tiles(cu_seqlens, length, heads)
    start = starts(cu_seqlens, length)

    def prefix(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return (kv_idx <= q_idx) & (kv_idx >= start[q_idx])

    return attend(q, k, v, scale, count, index, prefix)
