"""DeepSeek-V4.1 CSA2 with no parameters of its own (src/nanogpt/csa2_pf.py).

V4.1-Flash-1017 attends to its 128 newest keys plus `index_topk` compressed entries
picked by a learned indexer; a learned compressor makes each entry from `ratio`
tokens. Here both are weight-free: an entry is the mean key and mean value of an
aligned block of `ratio` keys (kept as keys are written), and the picks are the top
`keep` entries by summed-query score (q1 + q2) . (mean key), among the entries that
end before the window. One softmax over the window's keys and the picked entries;
no sink. The suite's keys are post-rotary, so an entry is the mean of rotated keys
(the model rotates the mean at the block's first position). Selected = the window
plus every key inside a picked entry, so it covers keep x ratio + window keys while
reading keep + window rows.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from attention.harness import Chunk, Outcome, recent_keys, top_mask


def pool(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """(G, n // ratio, D) means of the aligned blocks of ratio rows of x (G, n, D)."""
    blocks = x.shape[1] // ratio
    return x[:, : blocks * ratio].unflatten(1, (blocks, ratio)).mean(2)


@dataclass(frozen=True)
class Csa2ParameterFree:
    """The window plus the top `keep` older block means by summed-query score."""

    keep: int = 512  # picked entries, on top of the window (V4.1: index_topk 512)
    window: int = 128  # newest keys, always attended (V4.1: window_size 128)
    ratio: int = 4  # keys per entry (V4.1: compress_ratio 4)

    def __call__(self, chunk: Chunk) -> Outcome:
        """Softmax over the window's keys and the picked entries."""
        q, valid, r = chunk.query, chunk.valid, self.ratio
        n = chunk.key.shape[1]
        k_mean, v_mean = pool(chunk.key, r), pool(chunk.value, r)
        first = (valid.sum(-1) - self.window).clamp(min=0)  # (Q,) window's first key
        ids = torch.arange(k_mean.shape[1], device=q.device)
        allowed = ids < (first // r)[:, None]  # entries ending before the window
        score = (q.sum(1) @ k_mean.mT).masked_fill(~allowed, float("-inf"))
        picked = top_mask(score, self.keep, allowed)  # (G, Q, entries)
        recent = recent_keys(valid, self.window)
        entry_logits = (q @ k_mean[:, None].mT) * chunk.scaling
        logits = torch.cat(
            [
                chunk.logits.masked_fill(~recent, float("-inf")),
                entry_logits.masked_fill(~picked[:, None], float("-inf")),
            ],
            -1,
        )
        out = logits.softmax(-1) @ torch.cat([chunk.value, v_mean], 1)[:, None]
        covered = F.pad(picked.repeat_interleave(r, -1), (0, n - k_mean.shape[1] * r))
        reads = q.shape[-1] * (allowed.sum(-1) + 2 * recent.sum(-1) + picked.sum(-1))
        return Outcome(out, covered | recent, reads)
