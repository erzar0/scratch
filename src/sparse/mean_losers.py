"""Global halving whose losers share one mean value vector.

Weights as with approximate losers: winners exact, each loser its last partial
logit capped at the lowest winner logit, one softmax over all keys. The losers'
total weight then goes to the mean of their value vectors, which is the
running sum of all values minus the winners' (O(d) per decode step), so no
loser value is read. Otherwise FirstRound.
"""

from dataclasses import dataclass

import torch

from sparse.attend import Chunk
from sparse.first_round import FirstRound


@dataclass(frozen=True)
class MeanLosers(FirstRound):
    """FirstRound whose losers' weight goes to their mean value."""

    losers: str = "meanv"

    def attend(
        self,
        chunk: Chunk,
        won: torch.Tensor,
        current: torch.Tensor,
        approx: torch.Tensor,
    ) -> torch.Tensor:
        """(G, g, Q, D) winners exact; losers' weight times their mean value."""
        winner = won[:, None]
        floor = chunk.logits.masked_fill(~winner, float("inf")).amin(-1, keepdim=True)
        weight = torch.where(winner, chunk.logits, approx.minimum(floor)).softmax(-1)
        lost = (chunk.valid & ~won).float()  # (G, Q, N)
        mean = (lost @ chunk.value) / lost.sum(-1, keepdim=True).clamp(min=1)
        exact = weight.masked_fill(~winner, 0) @ chunk.value[:, None]
        share = weight.masked_fill(winner, 0).sum(-1, keepdim=True)
        return exact + share * mean[:, None]
