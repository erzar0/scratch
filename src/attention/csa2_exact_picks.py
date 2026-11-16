"""DeepSeek-V4.1 CSA2's layout with our parameter-free picks: exact picks.

V4.1-Flash-1017 attends to its 128 newest keys plus the top `index_topk` entries
of a learned indexer, on top of the window. Here the indexer is replaced by the
summed query (our record 91 rule): picks are the top `keep` keys by (q1 + q2) . k,
scored exactly, among the keys before the window. The learned parts are gone: no
indexer, no sink, no compressor (one entry per key), values are the suite's v.
Picks never fall inside the window: on raw keys such a pick would be a second copy
of a window key (in V4.1 it is a separate compressed row). So it reads exactly
keep + window keys, more than the budget of the other rows.
"""

from dataclasses import dataclass

import torch

from attention.harness import Chunk, Outcome, attend, recent_keys, top_mask


@dataclass(frozen=True)
class Csa2ExactPicks:
    """The window plus the top `keep` older keys by summed-query score."""

    keep: int = 512  # picks, on top of the window (V4.1: index_topk 512)
    window: int = 128  # newest keys, always attended (V4.1: window_size 128)

    def __call__(self, chunk: Chunk) -> Outcome:
        """Exact attention over the window and the picks; every key's K read."""
        recent = recent_keys(chunk.valid, self.window)
        older = chunk.valid & ~recent
        score = self.score(chunk).masked_fill(~older, float("-inf"))
        selected = top_mask(score, self.keep, older) | recent
        reads = chunk.query.shape[-1] * (chunk.valid.sum(-1) + selected.sum(-1))
        return Outcome(attend(chunk, selected), selected, reads)

    def score(self, chunk: Chunk) -> torch.Tensor:
        """(G, Q, N) pick score of every key: the summed query's logit."""
        return chunk.logits.sum(1)
