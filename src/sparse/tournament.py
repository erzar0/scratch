"""Tournament attention: keys compete on a few dimensions at a time.

Every key is scored on the first DIMS dimensions of an order; keys then compete,
and each round the survivors add DIMS more. After R = ceil(log2(n / keep)) rounds,
about keep keys remain and get exact attention. Variants:

    order    random    a fixed random permutation per layer and KV head
             hadamard  random signs, then a Hadamard rotation, of q and k
             query     dimensions by |q| summed over the group, per query
             pca       Loki's post-rotary PCA basis, by variance (sparse/loki.py)
    rounds   bracket   adjacent survivors in position order meet; one key per block
                       of 2^R positions is left
             halving   the best keep 2^(R - r) keys by score go on from round r
                       (as many as the bracket keeps)
    losers   approx    keep the scaled partial logit they lost with, capped at the
                       lowest winner logit (so every value is read)
             drop      leave the softmax
             mean      their share goes to the mean value (SparQ step 3)
    local    the last `local` positions win every round (SparQ's local window;
             halving only)

Scores are partial logits q[I] k[I] / sqrt(d f), f = |q[I]|_1 / |q|_1 (SparQ's
temperature); keys compete on the log of the group's summed exp score.
Costs: elements read, and 32-byte sectors of K per layout (attend.k_sectors).
"""

import math
from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F

from sparse import loki
from sparse.attend import Chunk, Outcome, k_sectors


def hadamard(dim: int) -> torch.Tensor:
    """Sylvester's (dim, dim) Hadamard matrix, scaled to be orthogonal."""
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)]
        )
    return matrix / math.sqrt(dim)


@cache
def rotation(
    order: str, layer: int, groups: int, dim: int, device: str
) -> torch.Tensor:
    """(G, D, D) orthogonal map of q and k into the order's basis, in order."""
    if order == "pca":
        return loki.basis("post", device)[layer]
    if order == "query":
        return torch.eye(dim, device=device).expand(groups, dim, dim)
    generator = torch.Generator().manual_seed(layer)
    if order == "random":
        perms = [torch.randperm(dim, generator=generator) for _ in range(groups)]
        return torch.stack([torch.eye(dim)[:, p] for p in perms]).to(device)
    assert order == "hadamard"
    signs = torch.randint(0, 2, (groups, dim, 1), generator=generator) * 2 - 1
    return (signs * hadamard(dim)).to(device)


@dataclass(frozen=True)
class Tournament:
    """Tournament attention (see the module docstring).

    Attributes:
        order: random, hadamard, query or pca.
        rounds: bracket or halving.
        losers: approx, drop or mean.
        keep: Keys left for exact attention.
        dims: Dimensions added per round.
        local: Most recent positions that always win (halving only).
    """

    order: str = "random"
    rounds: str = "bracket"
    losers: str = "approx"
    keep: int = 128
    dims: int = 8
    local: int = 0

    def survivors(
        self, score: torch.Tensor, alive: torch.Tensor, r: int, budget: int
    ) -> torch.Tensor:
        """(G, Q, M) keys that go on from round r, at most budget per query."""
        if self.rounds == "bracket":
            blocks = score.unflatten(-1, (-1, 2**r))
            best = blocks.argmax(-1, keepdim=True)
            won = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, best, True)
            return (won & alive.unflatten(-1, (-1, 2**r))).flatten(-2)
        assert self.rounds == "halving"
        rank = score.shape[-1] - budget + 1  # the budget-th largest
        return alive & (score >= score.kthvalue(rank, -1, keepdim=True).values)

    def __call__(self, chunk: Chunk) -> Outcome:
        """The tournament on one chunk; extra holds K sectors per layout."""
        groups, size, queries, dim = chunk.query.shape
        n_keys = chunk.key.shape[1]
        total = math.ceil(math.log2(n_keys / self.keep)) if n_keys > self.keep else 0
        assert total * self.dims <= dim, "too few dimensions for the rounds"
        assert not self.local or self.rounds == "halving"
        rot = rotation(self.order, chunk.layer, groups, dim, str(chunk.key.device))
        q = chunk.query @ rot[:, None]
        slots = self.keep * 2**total if total else n_keys
        k = F.pad(chunk.key @ rot, (0, 0, 0, slots - n_keys))
        alive = F.pad(chunk.valid, (0, slots - n_keys)).expand(groups, -1, -1)
        last = chunk.valid.sum(-1, keepdim=True) - self.local
        recent = alive & (torch.arange(slots, device=alive.device) >= last)
        rank = torch.arange(dim, device=q.device).expand(groups, queries, dim)
        if self.order == "query":
            rank = q.abs().sum(1).argsort(-1, descending=True).argsort(-1)
        q_abs = q.abs()
        flat = torch.zeros(groups, size * queries, slots, device=q.device)
        partial = flat.view(groups, size, queries, slots)
        current = approx = torch.full_like(partial, float("-inf"))
        lost_in = torch.zeros(alive.shape, dtype=torch.uint8, device=q.device)
        for r in range(1, total + 1):
            in_round = (rank >= (r - 1) * self.dims) & (rank < r * self.dims)
            flat.baddbmm_((q * in_round[:, None]).flatten(1, 2), k.mT)
            seen = (q_abs * (rank < r * self.dims)[:, None]).sum(-1)
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
        dims_read = torch.tensor(
            [self.dims * r for r in range(total + 1)] + [dim], device=q.device
        )
        reads = dims_read[lost_in.long()].sum(-1) + dim * self.value_reads(chunk, won)
        return Outcome(output, won, reads, k_sectors(rank, dims_read, lost_in))

    def attend(
        self,
        chunk: Chunk,
        won: torch.Tensor,
        current: torch.Tensor,
        approx: torch.Tensor,
    ) -> torch.Tensor:
        """(G, g, Q, D) exact attention on the winners, losers per self.losers."""
        winner = won[:, None]
        exact = chunk.logits.masked_fill(~winner, float("-inf"))
        if self.losers == "approx":
            floor = chunk.logits.masked_fill(~winner, float("inf")).amin(
                -1, keepdim=True
            )
            logits = torch.where(winner, chunk.logits, approx.minimum(floor))
            return logits.softmax(-1) @ chunk.value[:, None]
        output = exact.softmax(-1) @ chunk.value[:, None]
        if self.losers == "drop" or bool(won.eq(chunk.valid).all()):
            return output
        assert self.losers == "mean"
        share = torch.where(winner, current, approx).softmax(-1)
        alpha = (share * winner).sum(-1, keepdim=True)
        valid = chunk.valid.float()
        mean_value = (valid @ chunk.value) / valid.sum(-1, keepdim=True)
        return alpha * output + (1 - alpha) * mean_value[:, None]

    def value_reads(self, chunk: Chunk, won: torch.Tensor) -> torch.Tensor:
        """(G, Q) values read: all for approx, winners (and the mean) otherwise."""
        if self.losers == "approx":
            return chunk.valid.sum(-1).expand(won.shape[0], -1)
        return won.sum(-1) + (self.losers == "mean")
