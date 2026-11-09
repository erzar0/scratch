"""Exact top-k: every key scored exactly, values of the top k only (an oracle)."""

from dataclasses import dataclass

from attention.harness import Chunk, Outcome, attend, group_score, top_mask


@dataclass(frozen=True)
class ExactTopK:
    """The k keys of highest exact score: every key read, values of k only."""

    k: int = 128

    def __call__(self, chunk: Chunk) -> Outcome:
        """Exact attention over the exact top-k of each group."""
        dim = chunk.query.shape[-1]
        selected = top_mask(group_score(chunk.logits), self.k, chunk.valid)
        reads = dim * (chunk.valid.sum(-1) + selected.sum(-1))
        return Outcome(attend(chunk, selected), selected, reads)
