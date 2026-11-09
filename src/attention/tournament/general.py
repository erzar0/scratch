"""Tournament attention: keys compete on a few dimensions at a time.

Every key is scored on the first DIMS dimensions of an order; keys then compete,
and each round the survivors add DIMS more. After R = ceil(log2(n / keep)) rounds,
about keep keys remain and get exact attention. Variants:

    order    random    a fixed random permutation per layer and KV head
             hadamard  random signs, then a Hadamard rotation, of q and k
             query     dimensions by |q| summed over the group, per query
             pca       Loki's PCA basis of the context's keys, by variance (loki.py)
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
from typing import NamedTuple

import torch
import torch.nn.functional as F

from attention import loki
from attention.harness import (
    Chunk,
    Outcome,
    capped_logits,
    k_sectors,
    query_rank,
    recent_keys,
    temperature,
)


def halving_rounds(n: int, keep: int) -> int:
    """Rounds that halve n keys down to about keep: ceil(log2(n / keep)), none if
    n <= keep."""
    return math.ceil(math.log2(n / keep)) if n > keep else 0


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
    if order == "query":
        return torch.eye(dim, device=device).expand(groups, dim, dim)
    generator = torch.Generator().manual_seed(layer)
    if order == "random":
        perms = [torch.randperm(dim, generator=generator) for _ in range(groups)]
        return torch.stack([torch.eye(dim)[:, p] for p in perms]).to(device)
    assert order == "hadamard"
    signs = torch.randint(0, 2, (groups, dim, 1), generator=generator) * 2 - 1
    return (signs * hadamard(dim)).to(device)


class Rounds(NamedTuple):
    """Where the rounds left every key slot.

    Attributes:
        alive: (G, Q, M) the winners.
        lost_in: (G, Q, M) uint8 round each loser went out in (0: never played).
        current: (G, g, Q, M) scaled partial logits after the last round.
        approx: (G, g, Q, M) each loser's scaled partial logit when it lost
            (-inf if losers are dropped or the key won).
    """

    alive: torch.Tensor
    lost_in: torch.Tensor
    current: torch.Tensor
    approx: torch.Tensor


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

    def read_by(self, r: int) -> int:
        """Dimensions every key still playing has read after round r (0: none)."""
        return self.dims * r

    def in_blocks(self, r: int) -> bool:
        """Whether round r halves inside fixed blocks of key slots."""
        return False

    def rotate(self, chunk: Chunk) -> tuple[torch.Tensor, torch.Tensor]:
        """q (G, g, Q, D) and k (G, N, D) in the basis whose dimensions keys read."""
        groups, _, _, dim = chunk.query.shape
        rot = (
            loki.basis(chunk.key)
            if self.order == "pca"
            else rotation(self.order, chunk.layer, groups, dim, str(chunk.key.device))
        )
        return chunk.query @ rot[:, None], chunk.key @ rot

    def reading_order(self, q: torch.Tensor) -> torch.Tensor:
        """(G, Q, D) each dimension's place in the order keys read them."""
        if self.order == "query":
            return query_rank(q)
        groups, _, queries, dim = q.shape
        return torch.arange(dim, device=q.device).expand(groups, queries, dim)

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

    def compete(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rank: torch.Tensor,
        alive: torch.Tensor,
        recent: torch.Tensor,
        total: int,
        scaling: float,
    ) -> Rounds:
        """The `total` rounds on the key slots of k (G, M, D): each round every
        alive key adds the next dims to its partial logits, and the survivors go on.

        Args:
            q: (G, g, Q, D) queries in the reading basis.
            rank: (G, Q, D) each dimension's place in the reading order.
            alive: (G, Q, M) valid key slots.
            recent: (Q, M) slots that win every round.
        """
        groups, size, queries, _ = q.shape
        q_abs = q.abs()
        flat = torch.zeros(groups, size * queries, k.shape[1], device=q.device)
        partial = flat.view(groups, size, queries, -1)  # unscaled partial logits
        current = approx = torch.full_like(partial, float("-inf"))
        lost_in = torch.zeros(alive.shape, dtype=torch.uint8, device=q.device)
        for r in range(1, total + 1):
            now = (rank >= self.read_by(r - 1)) & (rank < self.read_by(r))
            flat.baddbmm_((q * now[:, None]).flatten(1, 2), k.mT)  # add round r's dims
            read = rank < self.read_by(r)
            current = partial * temperature(q_abs, read, scaling)[..., None]
            score = current.logsumexp(1).masked_fill(~alive, float("-inf"))
            score = score.masked_fill(recent, float("inf"))
            survive = self.survivors(score, alive, r, self.keep << (total - r))
            lost = alive & ~survive
            if self.losers != "drop":
                approx = torch.where(lost[:, None], current, approx)
            lost_in = torch.where(lost, r, lost_in)
            alive = survive
        return Rounds(alive, lost_in, current, approx)

    def __call__(self, chunk: Chunk) -> Outcome:
        """The tournament on one chunk; extra holds K sectors per layout."""
        groups, _, _, dim = chunk.query.shape
        n_keys = chunk.key.shape[1]
        total = halving_rounds(n_keys, self.keep)
        assert self.read_by(total) <= dim, "too few dimensions for the rounds"
        assert not self.local or self.rounds == "halving"
        q, k = self.rotate(chunk)
        rank = self.reading_order(q)
        pad = (self.keep << total if total else n_keys) - n_keys  # slots past n
        played = self.compete(
            q,
            F.pad(k, (0, 0, 0, pad)),
            rank,
            F.pad(chunk.valid, (0, pad)).expand(groups, -1, -1),
            F.pad(recent_keys(chunk.valid, self.local), (0, pad)),
            total,
            chunk.scaling,
        )
        won = played.alive[..., :n_keys]
        output = self.attend(
            chunk, won, played.current[..., :n_keys], played.approx[..., :n_keys]
        )
        # dims each key read: the level it went out in, total + 1 for winners (all)
        level = torch.where(won, total + 1, played.lost_in[..., :n_keys])
        levels = [self.read_by(r) for r in range(total + 1)] + [dim]
        dims_read = torch.tensor(levels, device=q.device)
        reads = dims_read[level.long()].sum(-1) + dim * self.value_reads(chunk, won)
        return Outcome(output, won, reads, k_sectors(rank, dims_read, level))

    def attend(
        self,
        chunk: Chunk,
        won: torch.Tensor,
        current: torch.Tensor,
        approx: torch.Tensor,
    ) -> torch.Tensor:
        """(G, g, Q, D) exact attention on the winners, losers per self.losers."""
        if self.losers == "approx":
            return capped_logits(chunk, won, approx).softmax(-1) @ chunk.value[:, None]
        winner = won[:, None]
        exact = chunk.logits.masked_fill(~winner, float("-inf"))
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
