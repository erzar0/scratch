"""Global halving with a wider first round (src/sparse/first_round.py)."""

import torch

from sparse.first_round import FirstRound
from sparse.tournament import Tournament
from tests.test_sparse import make_chunk


def test_first_round_reads_first_dims_then_dims_per_round() -> None:
    chunk = make_chunk(groups=2, queries=1, keys=16)  # 16 dims, all keys valid
    for losers in ("drop", "approx"):  # first = dims: plain global halving
        want = Tournament("query", "halving", losers, keep=4)(chunk)
        got = FirstRound(losers=losers, keep=4, first=8)(chunk)
        assert torch.equal(got.selected, want.selected)
        torch.testing.assert_close(got.output, want.output)
        assert torch.equal(got.reads, want.reads)
        assert all(torch.equal(got.extra[k], v) for k, v in want.extra.items())

    one = FirstRound(keep=8, first=8, dims=4)(chunk)  # 1 round on the top 8 dims
    two = FirstRound(keep=4, first=8, dims=4)(chunk)  # then 4 more

    assert torch.equal(
        one.selected, Tournament("query", "halving", keep=8)(chunk).selected
    )
    # 8 keys lose on 8 dims, 4 on 12; 4 winners read 16 dims of K and of V
    assert bool(two.reads.eq(8 * 8 + 4 * 12 + 4 * 16 + 4 * 16).all())
