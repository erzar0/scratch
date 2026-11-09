"""Global halving whose losers share their mean value.

attention/tournament/mean_value_losers.py
"""

from dataclasses import replace

import torch

from attention.tournament.mean_value_losers import MeanLosers
from attention.tournament.wide_first_round import FirstRound
from tests.attention.test_harness import make_chunk


def test_mean_losers_are_approximate_losers_with_equal_loser_values() -> None:
    chunk = make_chunk(groups=2, queries=1, keys=12)  # all keys valid
    got = MeanLosers(keep=4, first=8)(chunk)
    lost = ~got.selected[:, 0, :, None]  # (G, N, 1)
    mean = (chunk.value * lost).sum(1, keepdim=True) / lost.sum(1, keepdim=True)
    same = replace(chunk, value=torch.where(lost, mean, chunk.value))

    # every loser holding the losers' mean value: approximate losers agree
    want = FirstRound(losers="approx", keep=4, first=8)(same)

    torch.testing.assert_close(got.output, want.output)
    dropped = FirstRound(losers="drop", keep=4, first=8)(chunk)
    assert torch.equal(got.reads, dropped.reads)  # no loser value read
