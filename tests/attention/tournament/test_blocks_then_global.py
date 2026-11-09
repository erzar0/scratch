"""Block rounds, then global rounds (src/attention/tournament/blocks_then_global.py)."""

import torch

from attention.tournament.blocks import BlockTournament
from attention.tournament.blocks_then_global import HybridTournament
from attention.tournament.general import Tournament
from tests.attention.test_harness import make_chunk


def test_no_block_rounds_is_global_and_all_is_blocks() -> None:
    chunk = make_chunk(groups=1, queries=1, keys=16)  # keep 4: 2 rounds
    glob = Tournament("query", "halving", "drop", keep=4)(chunk).selected
    blocks = BlockTournament(keep=4, block=8)(chunk).selected

    none = HybridTournament(keep=4, block=8, block_rounds=0)(chunk).selected
    every = HybridTournament(keep=4, block=8, block_rounds=2)(chunk).selected

    assert torch.equal(none, glob)
    assert torch.equal(every, blocks)
