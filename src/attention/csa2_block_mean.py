"""DeepSeek-V4.1 CSA2's layout with our parameter-free picks: block-mean picks.

V4.1-Flash-1017 picks in two levels: one layer keeps the best 2048 blocks of 8
entries by its learned indexer's block maximum, then each later layer takes its
top 512 inside them. Here both levels use the summed query (our record 91 rule):
blocks of `block` keys that start before the window are ranked by (q1 + q2) . (the
block's mean key, kept as keys are written); the best `pool` x keep blocks are kept;
the top `keep` of their keys before the window by exact summed-query score are the
picks. pool 1 is the 64 x 8 rule that passed our nanogpt eval gate; pool 4 is
V4.1's ratio (2048 blocks for 512 picks). As csa2_exact_picks: no indexer, sink or
compressor; values are the suite's v; picks avoid the window, so it reads exactly
keep + window keys (V4.1's newest-block pin is moot: that block is in the window).
"""

from dataclasses import dataclass

import torch

from attention.harness import Chunk, Outcome, attend, recent_keys, top_mask


@dataclass(frozen=True)
class Csa2BlockMean:
    """The window plus the top `keep` older keys inside the best blocks by mean."""

    keep: int = 512  # picks, on top of the window
    window: int = 128  # newest keys, always attended
    block: int = 8  # keys per block (V4.1: candidate_block_size 8)
    pool: int = 1  # blocks kept per pick (V4.1: 2048 / 512 = 4)

    def __call__(self, chunk: Chunk) -> Outcome:
        """Exact attention over the window and the picks."""
        q, valid = chunk.query, chunk.valid
        n = chunk.key.shape[1]
        n_blocks = n // self.block
        means = self.means(chunk.key)
        first = (valid.sum(-1) - self.window).clamp(min=0)  # (Q,) window's first key
        ids = torch.arange(n_blocks, device=q.device)
        allowed = ids * self.block < first[:, None]  # blocks starting before it
        score = (q.sum(1) @ means.mT).masked_fill(~allowed, float("-inf"))
        kept = top_mask(score, self.pool * self.keep, allowed)
        kept = kept.repeat_interleave(self.block, -1)
        kept = torch.nn.functional.pad(kept, (0, n - kept.shape[-1]))
        recent = recent_keys(valid, self.window)
        candidates = kept & valid & ~recent
        picks = chunk.logits.sum(1).masked_fill(~candidates, float("-inf"))
        selected = top_mask(picks, self.keep, candidates) | recent
        scored = allowed.sum(-1) + candidates.sum(-1) + recent.sum(-1)
        reads = q.shape[-1] * (scored + selected.sum(-1))
        return Outcome(attend(chunk, selected), selected, reads)

    def means(self, key: torch.Tensor) -> torch.Tensor:
        """(G, n // block, D) mean key of each full block of key (G, n, D)."""
        n_blocks = key.shape[1] // self.block
        return key[:, : n_blocks * self.block].unflatten(1, (-1, self.block)).mean(2)
