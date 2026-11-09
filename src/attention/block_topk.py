"""Block top-k: CSA2-style block selection without training (DeepSeek V4.1 CSA2).

CSA2's indexer scores one compressed entry per block of consecutive keys and
attends to the top-k entries. Training-free here: a block's summary key is the mean
of its keys, and the kept blocks' own keys get exact attention, plus the most
recent `window` keys (CSA2's sliding window).

    flat     score the mean key of every block that starts before the recent
             window (none lies inside it; one may straddle its edge), keep the
             best (keep - window) / block blocks
    double   double indexing: first score one summary per complete superblock
             (`superblock` consecutive blocks) and keep the best `kept_superblocks`;
             then score only the blocks of those and of the query's own superblock
             summary  mean   the superblock's mean key
                      bound  per-channel midpoint and half-range: q·mid + |q|·half
                             is an upper bound on each head's best logit in it
                             (Quest, arXiv 2406.10774)

Group rule: a block's score is the logsumexp of its scaled logits over the group's
query heads (the tournament's rule). Reads per query and KV head: D per summary key
scored, plus K and V of every attended key.
"""

from dataclasses import dataclass

import torch

from attention.harness import Chunk, Outcome, attend


def summary_keys(blocks: torch.Tensor, coarse: str) -> torch.Tensor:
    """(..., S, c) keys of each superblock -> (..., r, c) the keys the coarse step
    scores. mean: the mean key (r = 1). bound: per-channel midpoint and half-range
    (r = 2)."""
    if coarse == "bound":
        high, low = blocks.amax(-2), blocks.amin(-2)
        return torch.stack([(high + low) / 2, (high - low) / 2], -2)
    assert coarse == "mean"
    return blocks.mean(-2, keepdim=True)


def group_logits(q: torch.Tensor, keys: torch.Tensor, scaling: float) -> torch.Tensor:
    """(G, Q, M) logsumexp over the heads of q (G, g, Q, D) · keys (G, M, D)."""
    return ((q @ keys[:, None].mT) * scaling).logsumexp(1)


@dataclass(frozen=True)
class BlockTopK:
    """Block top-k (see the module docstring).

    Attributes:
        block: Consecutive keys per block.
        keep: Keys attended exactly: the window plus (keep - window) / block blocks.
        window: Most recent keys, always attended.
        superblock: Blocks per superblock; 0 is the flat indexer.
        kept_superblocks: Superblocks the coarse step keeps.
        coarse: mean or bound, the coarse step's summary.
    """

    block: int = 4
    keep: int = 512
    window: int = 32
    superblock: int = 0
    kept_superblocks: int = 32
    coarse: str = "mean"

    def __call__(self, chunk: Chunk) -> Outcome:
        """The selection on one chunk."""
        q, valid = chunk.query, chunk.valid
        groups, _, queries, dim = q.shape
        n = chunk.key.shape[1]
        n_blocks = n // self.block
        blocks = chunk.key[:, : n_blocks * self.block].unflatten(1, (-1, self.block))
        count = valid.sum(-1)  # (Q,) keys each query sees
        done = count // self.block  # complete blocks per query
        start = (count - self.window + self.block - 1).clamp(min=0) // self.block
        ids = torch.arange(n_blocks, device=q.device)  # start: blocks before the window
        allowed = (ids < start[:, None]).expand(groups, -1, -1)
        mean = summary_keys(blocks, "mean")[:, :, 0]
        score = group_logits(q, mean, chunk.scaling)  # (G, Q, n_blocks)
        reads = (start * dim).expand(groups, -1)
        if self.superblock:
            kept, reads = self.coarse_step(chunk, blocks, done)
            allowed = allowed & kept
        top = (self.keep - self.window) // self.block
        best = score.masked_fill(~allowed, float("-inf"))
        best = best.topk(min(top, n_blocks), -1).indices
        chosen = torch.zeros_like(allowed).scatter_(-1, best, True) & allowed
        keys = chosen.repeat_interleave(self.block, -1)
        keys = torch.nn.functional.pad(keys, (0, n - keys.shape[-1]))
        recent = torch.arange(n, device=q.device) >= (count - self.window)[:, None]
        selected = (keys | recent) & valid
        reads = reads + 2 * dim * selected.sum(-1)
        return Outcome(attend(chunk, selected), selected, reads)

    def coarse_step(
        self, chunk: Chunk, blocks: torch.Tensor, done: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(G, Q, n_blocks) blocks in a kept or the query's own superblock, and
        (G, Q) the reads of both steps' summary keys.

        Args:
            chunk: The chunk.
            blocks: (G, n_blocks, block, D) keys of the complete blocks.
            done: (Q,) complete blocks per query.
        """
        q, size = chunk.query, self.superblock
        groups, n_blocks = blocks.shape[:2]
        n_super = n_blocks // size
        supers = blocks[:, : n_super * size].unflatten(1, (-1, size)).flatten(2, 3)
        keys = summary_keys(supers, self.coarse)  # (G, n_super, r, D)
        if self.coarse == "bound":
            mid, half = keys.unbind(-2)
            upper = q @ mid[:, None].mT + q.abs() @ half[:, None].mT
            score = (upper * chunk.scaling).logsumexp(1)
        else:
            score = group_logits(q, keys[:, :, 0], chunk.scaling)
        finished = done // size  # complete superblocks per query
        ids = torch.arange(n_super, device=q.device)
        score = score.masked_fill(ids >= finished[:, None], float("-inf"))
        kept = score.topk(min(self.kept_superblocks, n_super), -1).indices
        # unfinished picks are padding; they and the own superblock get flag n_super
        kept = torch.where(kept < finished[:, None], kept, n_super)
        flag = torch.zeros(groups, q.shape[2], n_super + 1, dtype=torch.bool)
        flag = flag.to(q.device).scatter_(-1, kept, True)
        owner = (torch.arange(n_blocks, device=q.device) // size).clamp(max=n_super)
        in_kept = flag[..., owner] | (owner >= finished[:, None])
        # blocks scored: those of the kept superblocks and the own one's complete ones
        scored = finished.clamp(max=self.kept_superblocks) * size + done % size
        reads = (finished * keys.shape[-2] + scored) * q.shape[-1]
        return in_kept, reads.expand(groups, -1)
