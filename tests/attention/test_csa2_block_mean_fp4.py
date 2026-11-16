"""csa2_block_mean_fp4: block means are V4.1's FP4 main-cache rows of the means."""

import torch

from attention.csa2_block_mean import Csa2BlockMean
from attention.csa2_block_mean_fp4 import Csa2BlockMeanFp4
from nanogpt.csa2_ref import fake_quant


def test_means_are_fp4_rows() -> None:
    torch.manual_seed(0)
    key = torch.randn(2, 203, 32)
    want = fake_quant(Csa2BlockMean(block=8).means(key), "main")
    got = Csa2BlockMeanFp4(block=8).means(key)
    assert got.shape == (2, 25, 32)
    assert torch.equal(got, want)
    assert not torch.equal(got, Csa2BlockMean(block=8).means(key))
