"""The general tournament (src/attention/tournament/general.py): its rounds."""

import torch

from attention.tournament.general import Tournament
from tests.attention.test_harness import make_chunk


def test_bracket_and_halving_survivors() -> None:
    score = torch.tensor([[[3.0, 1, 0, 5, 2, 2.5, 9, 4]]])
    alive = torch.ones_like(score, dtype=torch.bool)

    bracket = Tournament(rounds="bracket").survivors(score, alive, 1, 4)
    halving = Tournament(rounds="halving").survivors(score, alive, 1, 4)

    assert bracket[0, 0].nonzero().flatten().tolist() == [0, 3, 5, 6]
    assert halving[0, 0].nonzero().flatten().tolist() == [0, 3, 6, 7]


def test_local_window_always_wins() -> None:
    chunk = make_chunk(groups=1, queries=2, keys=16)
    n = chunk.valid.sum(-1)

    selected = Tournament("random", "halving", "drop", keep=4, local=3)(chunk).selected

    for q in range(2):
        assert selected[0, q, n[q] - 3 : n[q]].all()


def test_bracket_leaves_one_key_per_block_and_counts_reads() -> None:
    chunk = make_chunk(groups=1, queries=1, keys=8)

    outcome = Tournament(keep=2)(chunk)  # 2 rounds of 8 dimensions

    assert outcome.selected.unflatten(-1, (2, 4)).sum(-1).tolist() == [[[1, 1]]]
    # K: 4 keys out after 8 dims, 2 after 16, 2 winners whole; V: every key
    assert outcome.reads.item() == 4 * 8 + 2 * 16 + 2 * 16 + 8 * 16
