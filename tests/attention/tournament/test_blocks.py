"""Block-local halving (src/attention/tournament/blocks.py)."""

import torch

from attention.tournament.blocks import BlockTournament
from attention.tournament.general import Tournament
from tests.attention.test_harness import make_chunk


def test_one_block_is_global_halving() -> None:
    chunk = make_chunk(queries=3, keys=16)
    glob = Tournament("query", "halving", "drop", keep=4)(chunk)

    for block in (16, 64):  # whole range, and longer than it
        local = BlockTournament(keep=4, block=block)(chunk)
        assert torch.equal(local.selected, glob.selected)
        torch.testing.assert_close(local.output, glob.output)


def test_every_block_keeps_the_same_share() -> None:
    score = torch.tensor(
        [[[9.0, 8, 7, 6, 0, 1, 2, 3, 5, 4, -1, -2, 1.5, 0.5, 2.5, 3.5]]]
    )
    alive = torch.ones_like(score, dtype=torch.bool)

    kept = BlockTournament(block=4).survivors(score, alive, 1, 8)

    assert kept[0, 0].nonzero().flatten().tolist() == [0, 1, 6, 7, 8, 9, 14, 15]
    chunk = make_chunk(groups=1, queries=1, keys=16)
    won = BlockTournament(keep=4, block=8)(chunk).selected  # 2 rounds: 8 -> 4 -> 2
    assert won.unflatten(-1, (2, 8)).sum(-1).tolist() == [[[2, 2]]]
