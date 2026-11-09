"""Global halving that reads K in fixed groups of 8 dims, ordered per query.

From the transposed K (d, n) that the halving kernels read, one 32-byte sector
is one dim of 16 neighbouring keys, so a round over scattered survivors
touches about every sector until fewer than 1 in 16 keys survive. Stored in
groups, (d / 8, n, 8), one group of a key is 16 contiguous bytes, so a survivor
costs one sector, shared with at most one neighbour (attend.k_sectors counts
this layout as sectors_chunks). The groups are fixed when K is written; only
their order follows the query: by |q| summed over the group's dims and query
heads, where FirstRound ranks single dims.

Groupings:
    pairs       4 adjacent rotary frequencies with both of their dims (Qwen's
                RoPE turns dims i and i + d/2 together)
    contiguous  dims 8j .. 8j + 7 as stored
Otherwise FirstRound (first-round width, losers dropped or approximate).
"""

from dataclasses import dataclass
from functools import cache

import torch

from attention.harness import Chunk
from attention.tournament.wide_first_round import FirstRound

GROUP = 8  # dims per group: 16 bytes of bf16, half a sector


@cache
def permutation(grouping: str, dim: int, device: str) -> torch.Tensor:
    """(D, D) map of q and k into the stored order, each group contiguous."""
    order = torch.arange(dim)
    if grouping == "pairs":  # [half, group, frequency] -> [group, half, frequency]
        order = order.view(2, dim // GROUP, GROUP // 2).transpose(0, 1).flatten()
    else:
        assert grouping == "contiguous"
    return torch.eye(dim)[:, order].to(device)


@dataclass(frozen=True)
class GroupOrder(FirstRound):
    """FirstRound reading whole groups of GROUP dims, groups by |q|."""

    grouping: str = "pairs"

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.first % GROUP == 0 and self.dims == GROUP

    def rotate(self, chunk: Chunk) -> tuple[torch.Tensor, torch.Tensor]:
        """q and k in the stored order: each group's dims contiguous."""
        perm = permutation(self.grouping, chunk.query.shape[-1], str(chunk.key.device))
        return chunk.query @ perm, chunk.key @ perm

    def reading_order(self, q: torch.Tensor) -> torch.Tensor:
        """(G, Q, D) groups by |q| summed over their dims and the group's heads;
        the dims of a group in stored order."""
        energy = q.abs().sum(1).unflatten(-1, (-1, GROUP)).sum(-1)  # (G, Q, D / 8)
        place = energy.argsort(-1, descending=True).argsort(-1)
        within = torch.arange(GROUP, device=q.device)
        return (place[..., None] * GROUP + within).flatten(-2)
