"""csa2_block_mean: a pool of every block that starts before the window picks as
csa2_exact_picks does."""

import torch

from attention.csa2_block_mean import Csa2BlockMean
from attention.csa2_exact_picks import Csa2ExactPicks
from tests.attention.test_harness import make_chunk


def test_full_pool_is_exact_picks() -> None:
    chunk = make_chunk(keys=203)  # 24 blocks of 8 start before each query's window
    pooled = Csa2BlockMean(keep=8, window=16, pool=3)(chunk)
    exact = Csa2ExactPicks(keep=8, window=16)(chunk)
    assert torch.equal(pooled.selected, exact.selected)
