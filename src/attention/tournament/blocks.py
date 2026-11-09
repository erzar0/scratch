"""Tournament attention with halving inside fixed blocks of keys (kernel-friendly).

Global halving needs a k-th value over every survivor each round, a global sync
per round in a kernel. Here each block of `block` key slots (aligned at position 0)
keeps its own top half every round, so one GPU program per block can run all the
rounds alone. Price: every block ends with the same quota, block / 2^R winners.
Defaults are part 1's best tournament: |q| order, halving, losers dropped.
"""

from dataclasses import dataclass

import torch

from attention.tournament.general import Tournament


@dataclass(frozen=True)
class BlockTournament(Tournament):
    """Tournament whose halving runs inside each block of `block` key slots."""

    order: str = "query"
    rounds: str = "halving"
    losers: str = "drop"
    keep: int = 256
    block: int = 1024

    def in_blocks(self, r: int) -> bool:
        """Every round halves inside the blocks."""
        return True

    def survivors(
        self, score: torch.Tensor, alive: torch.Tensor, r: int, budget: int
    ) -> torch.Tensor:
        """(G, Q, M) keys that go on from round r: in a block round the same share
        of every block, else global halving."""
        if not self.in_blocks(r):
            return Tournament.survivors(self, score, alive, r, budget)
        slots = score.shape[-1]
        block = min(self.block, slots)
        share = budget * block // slots  # = block >> r
        assert share >= 1, "block shorter than 2^rounds"
        blocks = score.unflatten(-1, (-1, block))
        floor = blocks.kthvalue(block - share + 1, -1, keepdim=True).values
        return alive & (blocks >= floor).flatten(-2)
