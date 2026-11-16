"""DeepSeek-V4.1 CSA2's layout, block-mean picks on FP4 block means.

As csa2_block_mean, but each block's mean key is stored as V4.1's main cache row:
FP4 E2M1 values with an E4M3 scale per 16 channels (nanogpt.csa2_ref.fake_quant),
0.56 bytes per channel instead of 1 (FP8) or 2 (bf16). The keys rescored inside the
kept blocks stay exact.
"""

from dataclasses import dataclass

import torch

from attention.csa2_block_mean import Csa2BlockMean
from nanogpt.csa2_ref import fake_quant


@dataclass(frozen=True)
class Csa2BlockMeanFp4(Csa2BlockMean):
    """Csa2BlockMean with its block means rounded to FP4."""

    def means(self, key: torch.Tensor) -> torch.Tensor:
        """(G, n // block, D) FP4-rounded mean key of each full block."""
        return fake_quant(super().means(key), "main")
