"""Tournament plus a local window (src/nanogpt/tournament_local.py): window 0 is
tournament_fast; otherwise attention is the softmax over the winners and the last
`window` keys of each query's document, forward and backward; a window as long
as the documents is dense attention in each document, as is dense(). Needs CUDA:
    uv run python -m tests.nanogpt.test_tournament_local
"""

import torch
from torch import Tensor

from nanogpt import tournament_fast
from nanogpt.tournament_fast import DEFAULT, select
from nanogpt.tournament_local import attention, dense
from tests.nanogpt.test_tournament import packed
from tests.nanogpt.test_tournament_fast import (
    first_config,
    matches_masked_softmax,
    winner_mask,
)


def local_mask(q: Tensor, cu: Tensor, window: int) -> Tensor:
    """(1, T, T) key in the query's document and among its last `window` keys."""
    query = torch.arange(q.shape[0], device=q.device)
    start = cu[torch.searchsorted(cu, query.int(), right=True) - 1]
    gap = query[:, None] - query[None]
    return ((gap >= 0) & (gap < window) & (query[None] >= start[:, None]))[None]


def window_0_is_tournament_fast() -> None:
    q, k, v, cu = packed((300, 1200, 548), torch.bfloat16)
    got = attention(q, k, v, cu, 0.12, 2048, DEFAULT, 0)
    want = tournament_fast.attention(q, k, v, cu, 0.12, 2048, DEFAULT)
    assert torch.equal(got, want)


def is_the_softmax_over_winners_and_window(window: int) -> None:
    compiled = torch.compile(attention)

    def mask_of(q: Tensor, k: Tensor, cu: Tensor) -> Tensor:
        won = winner_mask(select(q, k, cu, 2048, DEFAULT), q.shape[0])
        return won | local_mask(q, cu, window)

    matches_masked_softmax(
        lambda q, k, v, cu: compiled(q, k, v, cu, 0.12, 2048, DEFAULT, window),
        mask_of,
        window,
    )


def dense_is_the_softmax_over_each_prefix() -> None:
    compiled = torch.compile(dense)
    matches_masked_softmax(
        lambda q, k, v, cu: compiled(q, k, v, cu, 0.12),
        lambda q, k, cu: local_mask(q, cu, q.shape[0]),
        "dense",
    )


if __name__ == "__main__":
    tournament_fast._rounds.early_config_prune = first_config
    window_0_is_tournament_fast()
    print("window 0: tournament_fast's output")
    for window in (1, 128, 2048):  # 2048: dense in every document
        is_the_softmax_over_winners_and_window(window)
    print("windows 1, 128, 2048: softmax over winners and window, forward, backward")
    dense_is_the_softmax_over_each_prefix()
    print("dense: softmax over each document prefix, forward, backward")
