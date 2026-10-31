"""Global halving with a wider first round (the user's sort-based variant).

Every key is scored on the top `first` dimensions of the |q| order; the better
half goes on and adds `dims` more per round, until about keep keys remain for
exact attention. Keeping the sorted upper half is the same set as halving's
kthvalue cutoff, so only the first round's width differs from
Tournament("query", "halving", losers, keep). Losers drop or keep approximate
scores, as in sparse/tournament.py.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sparse.attend import Chunk, Outcome, k_sectors
from sparse.tournament import Tournament, rotation


@dataclass(frozen=True)
class FirstRound(Tournament):
    """Tournament whose first round reads `first` dimensions, later ones `dims`."""

    order: str = "query"
    rounds: str = "halving"
    losers: str = "drop"
    keep: int = 256
    first: int = 16

    def __call__(self, chunk: Chunk) -> Outcome:
        """Tournament.__call__ with round r reading first + dims (r - 1) dims."""
        groups, size, queries, dim = chunk.query.shape
        n_keys = chunk.key.shape[1]
        total = math.ceil(math.log2(n_keys / self.keep)) if n_keys > self.keep else 0
        assert self.first + (total - 1) * self.dims <= dim, "too few dimensions"
        assert self.order == "query" and self.rounds == "halving"
        rot = rotation(self.order, chunk.layer, groups, dim, str(chunk.key.device))
        q = chunk.query @ rot[:, None]
        slots = self.keep * 2**total if total else n_keys
        k = F.pad(chunk.key @ rot, (0, 0, 0, slots - n_keys))
        alive = F.pad(chunk.valid, (0, slots - n_keys)).expand(groups, -1, -1)
        last = chunk.valid.sum(-1, keepdim=True) - self.local
        recent = alive & (torch.arange(slots, device=alive.device) >= last)
        rank = q.abs().sum(1).argsort(-1, descending=True).argsort(-1)
        step = (rank - (self.first - self.dims)).clamp(min=0)  # round 1: top first
        q_abs = q.abs()
        flat = torch.zeros(groups, size * queries, slots, device=q.device)
        partial = flat.view(groups, size, queries, slots)
        current = approx = torch.full_like(partial, float("-inf"))
        lost_in = torch.zeros(alive.shape, dtype=torch.uint8, device=q.device)
        for r in range(1, total + 1):
            in_round = (step >= (r - 1) * self.dims) & (step < r * self.dims)
            flat.baddbmm_((q * in_round[:, None]).flatten(1, 2), k.mT)
            seen = (q_abs * (step < r * self.dims)[:, None]).sum(-1)
            scale = chunk.scaling / (seen / q_abs.sum(-1)).sqrt()
            current = partial * scale[..., None]
            score = current.logsumexp(1).masked_fill(~alive, float("-inf"))
            score = score.masked_fill(recent, float("inf"))
            survive = self.survivors(score, alive, r, self.keep * 2 ** (total - r))
            lost = alive & ~survive
            if self.losers != "drop":
                approx = torch.where(lost[:, None], current, approx)
            lost_in = torch.where(lost, r, lost_in)
            alive = survive
        won = alive[..., :n_keys]
        lost_in = torch.where(won, total + 1, lost_in[..., :n_keys])
        output = self.attend(chunk, won, current[..., :n_keys], approx[..., :n_keys])
        levels = [0] + [self.first + self.dims * (r - 1) for r in range(1, total + 1)]
        dims_read = torch.tensor(levels + [dim], device=q.device)
        reads = dims_read[lost_in.long()].sum(-1) + dim * self.value_reads(chunk, won)
        return Outcome(output, won, reads, k_sectors(rank, dims_read, lost_in))
