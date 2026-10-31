"""Tournaments on keys stored in a random order (src/sparse/shuffled.py)."""

import torch

from sparse.block_tournament import BlockTournament
from sparse.shuffled import ShuffledBlocks, ShuffledTournament
from sparse.tournament import Tournament
from tests.test_sparse import make_chunk


def test_order_free_halving_is_unchanged_and_blocks_change() -> None:
    # global halving ignores key order: same keys, same output; halving inside
    # blocks of 16 depends on which keys share a block
    chunk = make_chunk(keys=64, dim=32)  # 3 rounds of 8 dims at keep 8
    method = Tournament("query", "halving", "drop", 8)
    want, got = method(chunk), ShuffledTournament("query", "halving", "drop", 8)(chunk)
    assert torch.equal(got.selected, want.selected)
    torch.testing.assert_close(got.output, want.output)
    blocks = BlockTournament(keep=8, block=16)(chunk).selected
    mixed = ShuffledBlocks(keep=8, block=16)(chunk).selected
    assert torch.equal(blocks.sum(-1), mixed.sum(-1))
    assert not torch.equal(blocks, mixed)
