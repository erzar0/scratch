"""Global halving whose losers share their mean value (src/sparse/mean_losers.py)."""

from dataclasses import replace

import torch

from sparse.first_round import FirstRound
from sparse.mean_losers import MeanLosers
from tests.test_sparse import make_chunk


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
