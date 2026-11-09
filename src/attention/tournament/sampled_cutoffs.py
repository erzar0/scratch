"""Global halving whose cutoff is the median of a sample, not of every key.

Halving only needs about half of the keys to go on, so each early round takes
its cutoff from about SAMPLE alive keys and aims at the round's budget; the
slots hold an eighth more (HEADROOM), so they rarely overflow. The sample is
picked by a hash of position, so it spans every position (recent keys score
higher) and does not line up with RoPE periods, as a fixed stride does. The
last TAIL rounds stay exact, as in the kernel
(attention/triton/tournament_sampled_cutoffs.py),
whose rounds then need one pass each. Otherwise FirstRound (first-round width,
losers dropped or approximate).
"""

from dataclasses import dataclass

import torch

from attention.tournament.wide_first_round import FirstRound

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


def sample_keys(
    alive: torch.Tensor, r: int, m: int, expected: torch.Tensor
) -> torch.Tensor:
    """(G, Q, M) about m of the alive keys, each with chance m / expected (all of
    them if expected <= m), picked by round r's position hash."""
    rate = (m << 31) // expected.clamp(min=1)
    return alive & (position_hash(alive.shape[-1], r, alive.device) < rate)


def sample_cutoff(
    score: torch.Tensor, sampled: torch.Tensor, budget: int, count: torch.Tensor
) -> torch.Tensor:
    """(G, Q, 1) the cutoff that keeps budget of count keys, read off the sample:
    its need-th best score, need = ceil(sample size * budget / count)."""
    n_sampled = sampled.sum(-1, keepdim=True)
    need = (n_sampled * budget + count - 1) // count  # ceil
    top = score.masked_fill(~sampled, float("-inf")).topk(int(need.max()), -1)
    return top.values.gather(-1, (need - 1).clamp(min=0))


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
        # later rounds divide by the last round's budget instead of the count,
        # since the kernel draws their sample before it knows how many survive
        expected = count if r == 1 else torch.full_like(count, budget * 2)
        cut = sample_cutoff(score, sample_keys(alive, r, m, expected), budget, count)
        won = alive & ((score >= cut) | (count <= budget))
        # ponytail: past the slots, the last by position drop (the kernel: any);
        # about 6 standard deviations away at SAMPLE 4096
        return won & (won.long().cumsum(-1) <= budget + budget // HEADROOM)
