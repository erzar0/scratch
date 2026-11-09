"""Global halving with sampled-median cutoffs.

attention/tournament/sampled_cutoffs.py
"""

import torch

from attention.tournament.sampled_cutoffs import SampledHalving
from attention.tournament.wide_first_round import FirstRound
from tests.attention.test_harness import make_chunk


def test_a_sample_of_every_key_is_first_round() -> None:
    # keep 4 of 64: budgets 32, 16 (sampled, the sample holds every key), 8, 4;
    # keep 4 of 16: 2 rounds, both exact
    for keys, first in ((64, 4), (16, 8)):
        chunk = make_chunk(groups=2, queries=3, keys=keys)  # 16 dims
        for losers in ("drop", "approx"):
            want = FirstRound(losers=losers, keep=4, first=first, dims=4)(chunk)
            got = SampledHalving(losers=losers, keep=4, first=first, dims=4)(chunk)
            assert torch.equal(got.selected, want.selected)
            torch.testing.assert_close(got.output, want.output)
            assert torch.equal(got.reads, want.reads)

    chunk = make_chunk(groups=2, queries=3, keys=64)
    few = SampledHalving(keep=4, first=4, dims=4, sample=16)(chunk)  # a real sample
    assert bool(few.selected.sum(-1).eq(4).all())


def test_the_sample_spans_every_position() -> None:
    # scores rising with position, as recent keys tend to: a sample of the
    # oldest keys alone sets the cutoff too low, and the budget cap then
    # drops the newest keys
    n = 8191
    score = torch.arange(n, dtype=torch.float)[None]
    alive = torch.ones(1, n, dtype=torch.bool)
    won = SampledHalving(keep=4).survivors(score, alive, 1, 4096)

    kept = int(won.sum())
    assert abs(kept - 4096) < 300 and kept <= 4096 + 4096 // 8
    assert bool(won[:, -kept:].all())  # the highest scores
