"""Block rounds, then global rounds (src/sparse/hybrid_tournament.py)."""

import torch

from sparse.block_tournament import BlockTournament
from sparse.hybrid_tournament import HybridTournament
from sparse.tournament import Tournament
from tests.test_sparse import make_chunk


def test_no_block_rounds_is_global_and_all_is_blocks() -> None:
    chunk = make_chunk(groups=1, queries=1, keys=16)  # keep 4: 2 rounds
    glob = Tournament("query", "halving", "drop", keep=4)(chunk).selected
    blocks = BlockTournament(keep=4, block=8)(chunk).selected

    none = HybridTournament(keep=4, block=8, block_rounds=0)(chunk).selected
    every = HybridTournament(keep=4, block=8, block_rounds=2)(chunk).selected

    assert torch.equal(none, glob)
    assert torch.equal(every, blocks)
