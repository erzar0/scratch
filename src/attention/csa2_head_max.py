"""DeepSeek-V4.1 CSA2's layout, exact picks by the largest head logit.

As csa2_exact_picks, but a key's pick score is the maximum of its logits over the
query heads instead of their sum: a key one head needs is not outvoted by heads
that ignore it (Kascade's head-aware picks). Same reads as exact picks: the heads'
scores come from one matrix product over every older key.
"""

from dataclasses import dataclass

import torch

from attention.csa2_exact_picks import Csa2ExactPicks
from attention.harness import Chunk


@dataclass(frozen=True)
class Csa2HeadMax(Csa2ExactPicks):
    """The window plus the top `keep` older keys by their largest head logit."""

    def score(self, chunk: Chunk) -> torch.Tensor:
        """(G, Q, N) each key's largest logit over the heads."""
        return chunk.logits.amax(1)
