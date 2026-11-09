"""SparQ attention (Ribar et al., arXiv 2312.04985), Algorithm 1 with its GQA rule.

1. The r dimensions of largest |q|, summed over the group, score every key:
   softmax(q[I] K[I]^T / sqrt(d f)), f = |q[I]|_1 / |q|_1.
2. The k keys of highest score summed over the group, the last `local` positions
   always among them, get exact attention.
Step 3 (the rest of the weight to the mean value) is left out, as the paper does
for GQA models. Reads per query and KV head: n r + 2 k d; K sectors per layout
(attend.k_sectors).
"""

from dataclasses import dataclass

import torch

from attention.harness import (
    Chunk,
    Outcome,
    attend,
    k_sectors,
    query_rank,
    recent_keys,
    temperature,
    top_mask,
)


@dataclass(frozen=True)
class SparQ:
    """SparQ attention.

    Attributes:
        r: Dimensions of the approximate scores.
        k: Keys that get exact attention.
        local: Most recent positions always kept.
    """

    r: int = 32
    k: int = 128
    local: int = 16

    def __call__(self, chunk: Chunk) -> Outcome:
        """SparQ on one chunk."""
        q, valid = chunk.query, chunk.valid
        dim = q.shape[-1]
        q_abs = q.abs()
        dims = q_abs.sum(1).topk(self.r, -1).indices
        read = torch.zeros_like(q_abs[:, 0]).scatter_(-1, dims, 1.0)  # (G, Q, D)
        scale = temperature(q_abs, read, chunk.scaling)
        approx = ((q * read[:, None]) @ chunk.key[:, None].mT) * scale[..., None]
        score = approx.masked_fill(~valid, float("-inf")).softmax(-1).sum(1)
        recent = recent_keys(valid, self.local)
        selected = top_mask(score.masked_fill(recent, float("inf")), self.k, valid)
        reads = valid.sum(-1) * self.r + 2 * dim * selected.sum(-1)
        levels = torch.tensor([0, self.r, dim], device=q.device)
        level = valid.long() + selected.long()  # 1: r dims read, 2: in full
        sectors = k_sectors(query_rank(q), levels, level)
        return Outcome(attend(chunk, selected), selected, reads, sectors)
