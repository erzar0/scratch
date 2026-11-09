"""Global halving with dims read in fixed groups.

attention/tournament/channel_groups.py
"""

from dataclasses import replace

import torch

from attention.tournament.channel_groups import GroupOrder, permutation
from attention.tournament.wide_first_round import FirstRound
from tests.attention.test_harness import make_chunk


def test_groups_in_q_order_are_first_round() -> None:
    # |q| equal within each stored group and falling across groups: the |q|
    # order then reads whole groups, so both pick the same keys; with pairs the
    # groups straddle 8-dim chunks of the original order, so FirstRound reads
    # more sectors
    chunk = make_chunk(groups=2, queries=3, keys=64, dim=32)  # 4 rounds of 8 dims
    for grouping in ("pairs", "contiguous"):
        perm = permutation(grouping, 32, "cpu")
        level = torch.tensor([4.0, 3.0, 2.0, 1.0]).repeat_interleave(8)
        stored = chunk.query.sign() * level  # (G, 2, Q, D), in the stored order
        query = stored @ perm.T
        logits = (query @ chunk.key[:, None].mT * chunk.scaling).masked_fill(
            ~chunk.valid, float("-inf")
        )
        case = replace(chunk, query=query, logits=logits)
        for losers in ("drop", "approx"):
            want = FirstRound(losers=losers, keep=4, first=8)(case)
            got = GroupOrder(grouping=grouping, losers=losers, keep=4, first=8)(case)
            assert torch.equal(got.selected, want.selected)
            torch.testing.assert_close(got.output, want.output)
            assert torch.equal(got.reads, want.reads)
            ours, theirs = got.extra["sectors_chunks"], want.extra["sectors_chunks"]
            fewer = ours < theirs if grouping == "pairs" else ours == theirs
            assert bool(fewer.all())
