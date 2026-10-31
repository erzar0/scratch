"""Global halving whose cutoff is the median of a sample, not of every key.

Halving only needs about half of the keys to go on, so each early round takes
its cutoff from about SAMPLE alive keys and aims at the round's budget; the
slots hold an eighth more (HEADROOM), so they rarely overflow. The sample is
picked by a hash of position, so it spans every position (recent keys score
higher) and does not line up with RoPE periods, as a fixed stride does. The
last TAIL rounds stay exact, as in the kernel (src/sparse/kernel_sampled.py),
whose rounds then need one pass each. Otherwise FirstRound (first-round width,
losers dropped or approximate).
"""

from dataclasses import dataclass

import torch

from sparse.first_round import FirstRound

SAMPLE = 4096  # keys sampled for round 1; later rounds see about half survive
TAIL = 2  # exact last rounds
HEADROOM = 8  # slots for budget + budget // HEADROOM survivors
MASK = (1 << 31) - 1  # 31-bit hash: a product of two stays below 2**62


def position_hash(n: int, r: int, device: torch.device) -> torch.Tensor:
    """(n,) a hash in [0, 2**31) of each position, new in every round."""
    h = (torch.arange(n, device=device) + r * 0x61C88647) & MASK
    for _ in range(2):
        h = ((h ^ (h >> 15)) * 0x5BD1E995) & MASK
    return h ^ (h >> 15)


@dataclass(frozen=True)
class SampledHalving(FirstRound):
    """FirstRound with sampled-median cutoffs before the last TAIL rounds."""

    sample: int = SAMPLE

    def survivors(
        self, score: torch.Tensor, alive: torch.Tensor, r: int, budget: int
    ) -> torch.Tensor:
        """Alive keys at or above the sample's cutoff; exact in the last rounds."""
        if budget <= self.keep << (TAIL - 1):
            return super().survivors(score, alive, r, budget)
        m = self.sample if r == 1 else self.sample // 2  # the kernel's survivors
        count = alive.sum(-1, keepdim=True)
        # each alive key with probability m / count (all of them if count <= m);
        # later rounds divide by the last round's budget instead, since the
        # kernel draws their sample before it knows how many keys survive
        expected = count if r == 1 else torch.full_like(count, budget * 2)
        rate = (m << 31) // expected.clamp(min=1)
        sampled = alive & (position_hash(alive.shape[-1], r, alive.device) < rate)
        n_sampled = sampled.sum(-1, keepdim=True)
        need = (n_sampled * budget + count - 1) // count  # ceil
        top = score.masked_fill(~sampled, float("-inf")).topk(int(need.max()), -1)
        cut = top.values.gather(-1, (need - 1).clamp(min=0))
        won = alive & ((score >= cut) | (count <= budget))
        # ponytail: past the slots, the last by position drop (the kernel: any);
        # about 6 standard deviations away at SAMPLE 4096
        return won & (won.long().cumsum(-1) <= budget + budget // HEADROOM)
