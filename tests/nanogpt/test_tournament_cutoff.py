"""Approximate-cutoff training tournament (src/nanogpt/tournament_cutoff.py): each
cutoff keeps winners only in the query's own prefix, at most the budget, mostly
the exact tournament's (tournament_fast); attention() is the softmax over its
winners, forward and backward. Needs CUDA:
    uv run python -m tests.nanogpt.test_tournament_cutoff
"""

import torch

from nanogpt import tournament_fast
from nanogpt.tournament_cutoff import CUTOFFS, attention, select
from nanogpt.tournament_fast import DEFAULT, Settings
from tests.nanogpt.test_tournament import packed
from tests.nanogpt.test_tournament_fast import (
    first_config,
    matches_masked_softmax,
    winner_mask,
)

SETTINGS = (DEFAULT, Settings(2048, 32, 16))  # tournament.py's; training run 1's
DOCS = (5, 700, 1500, 2600)


def overlap_with_exact(settings: Settings, cutoff: str) -> float:
    """Share of the exact winners kept; asserts prefix-only and within budget."""
    q, k, _, cu = packed(DOCS)
    length, longest = q.shape[0], max(DOCS)
    got = winner_mask(select(q, k, cu, longest, settings, cutoff), length)
    want = winner_mask(tournament_fast.select(q, k, cu, longest, settings), length)
    query = torch.arange(length, device="cuda")
    start = cu[torch.searchsorted(cu, query.int(), right=True) - 1]
    key = torch.arange(length, device="cuda")
    prefix = (key[None] >= start[:, None]) & (key[None] <= query[:, None])
    assert not bool((got & ~prefix).any()), (settings, cutoff)
    budget = settings.winners * -(-(query - start + 1) // settings.block)
    assert bool((got.sum(-1) <= budget.clamp(max=length)).all()), (settings, cutoff)
    return float((got & want).sum() / want.sum())


def attention_is_the_softmax_over_the_winners(cutoff: str) -> None:
    """bf16 forward and gradients within 1% of float32 math on the same winners."""
    compiled = torch.compile(attention)
    matches_masked_softmax(
        lambda q, k, v, cu: compiled(q, k, v, cu, 0.12, 2048, DEFAULT, cutoff),
        lambda q, k, cu: winner_mask(
            select(q, k, cu, 2048, DEFAULT, cutoff), q.shape[0]
        ),
        cutoff,
    )


if __name__ == "__main__":
    tournament_fast._rounds.early_config_prune = first_config
    for cutoff in CUTOFFS:
        for settings in SETTINGS:
            share = overlap_with_exact(settings, cutoff)
            print(f"{cutoff}, {settings}: {share:.4f} of the exact winners")
            assert share > 0.9, (cutoff, settings, share)
        attention_is_the_softmax_over_the_winners(cutoff)
    print("prefix only, within budget; attention = softmax over the winners")
