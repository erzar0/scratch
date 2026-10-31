"""Tournaments on keys stored in a random order (the parallel tournament's layout).

Block tournaments (block_tournament.py) give every block of consecutive positions
the same quota, and the keys attention needs cluster (recent keys, sinks), so
crowded blocks lose them. Stored in a random order, each block is a random sample
of the context, and its quota matches its share of the global top keys up to
random crowding. The bracket's pairs become random pairs. Attention ignores key
order, so only the per-round selection sees the shuffle; the rest (causal masks,
padding) stays in position order. K sectors are still counted in position order,
not the shuffled layout (whose scattered survivors touch more sectors).
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import torch

from sparse.block_tournament import BlockTournament
from sparse.tournament import Tournament

type Pick = Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor]


@cache
def shuffle(n: int, device: str) -> torch.Tensor:
    """(n,) a fixed random order of n key slots."""
    # ponytail: one random order per slot count; a growing cache would fix each
    # token's slot by a position hash at write time
    return torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(device)


def shuffled(
    pick: Pick, score: torch.Tensor, alive: torch.Tensor, r: int, budget: int
) -> torch.Tensor:
    """pick's survivors with the slots in the random order, back in position order."""
    perm = shuffle(score.shape[-1], str(score.device))
    return pick(score[..., perm], alive[..., perm], r, budget)[..., perm.argsort()]


@dataclass(frozen=True)
class ShuffledTournament(Tournament):
    """Tournament (bracket or halving) on shuffled key slots."""

    def survivors(
        self, score: torch.Tensor, alive: torch.Tensor, r: int, budget: int
    ) -> torch.Tensor:
        """(G, Q, M) keys that go on from round r."""
        return shuffled(super().survivors, score, alive, r, budget)


@dataclass(frozen=True)
class ShuffledBlocks(BlockTournament):
    """BlockTournament on shuffled key slots: each block a random sample."""

    def survivors(
        self, score: torch.Tensor, alive: torch.Tensor, r: int, budget: int
    ) -> torch.Tensor:
        """(G, Q, M) keys that go on from round r."""
        return shuffled(super().survivors, score, alive, r, budget)
