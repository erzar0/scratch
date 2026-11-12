"""Exact top-k by the summed query: the pick rule of our CSA2-layout record 91 runs
(src/nanogpt/csa2_decode.top64 without weights).

One score per key: q . k with q the sum of the KV head's query heads, so the heads'
logits summed. Exact top-k sums their softmax instead, which lets one head's peak
outvote the other head's spread. Every key is read; values of the top k only.
"""

from dataclasses import dataclass

from attention.harness import Chunk, Outcome, attend, top_mask


@dataclass(frozen=True)
class SummedQueryTopK:
    """The k keys of highest summed-query score: every key read, values of k only."""

    k: int = 512

    def __call__(self, chunk: Chunk) -> Outcome:
        """Exact attention over each group's top k by its heads' summed logits."""
        dim = chunk.query.shape[-1]
        selected = top_mask(chunk.logits.sum(1), self.k, chunk.valid)
        reads = dim * (chunk.valid.sum(-1) + selected.sum(-1))
        return Outcome(attend(chunk, selected), selected, reads)
