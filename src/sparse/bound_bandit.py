"""Bound bandit: per-key successive elimination, channels read in |q| order.

The archived per-key bandit read channels in a random order and bounded the unread
part statistically, which needs the channels read to be a random sample. Here the
channels go by |q| summed over the group (SparQ's and the tournament's order), and
the unread part of q_h · c_j, with c_j = k_j - mu (mu: the head's mean key), is
bounded by width times
    exact   |q_h unread| |c_j unread|, |c_j unread|² = |c_j|² - |c_j read|²
            (Cauchy-Schwarz: at width 1 no key of the exact top-keep is dropped)
    gauss   sqrt(sum over unread channels of q_hc² var_c), var_c the keys' variance
            per channel (a typical size, not a bound)
Each round reads `dims` more channels of every alive key and drops the keys whose
upper bound falls below the keep-th largest lower bound. Scores are logsumexp over
the group's heads (the tournament's GQA rule). The last round reads every channel,
so exactly the top keep keys by that score get exact attention. Reads: a key
dropped in round r reads r · dims elements of K; winners read K and V.
"""

from dataclasses import dataclass

import torch

from sparse.attend import Chunk, Outcome, attend, k_sectors


@dataclass(frozen=True)
class BoundBandit:
    """Bound bandit (see the module docstring).

    Attributes:
        bound: exact or gauss.
        width: Scale of the radius; below 1 drops sooner, without a guarantee.
        keep: Keys that get exact attention.
        dims: Channels read per round.
    """

    bound: str = "exact"
    width: float = 1.0
    keep: int = 512
    dims: int = 8

    def __call__(self, chunk: Chunk) -> Outcome:
        """The elimination on one chunk; extra holds K sectors per layout."""
        q, valid = chunk.query, chunk.valid
        groups, size, queries, dim = q.shape
        n = chunk.key.shape[1]
        assert dim % self.dims == 0 and self.bound in ("exact", "gauss")
        rounds = dim // self.dims
        # ponytail: the chunk's mean key; a decoder fixes mu per head once (the
        # bound holds for any mu)
        mu = chunk.key.mean(1, keepdim=True)  # (G, 1, D)
        c = chunk.key - mu
        offset = q @ mu.mT[:, None]  # (G, g, Q, 1): q_h · mu
        c_sq = c.square()
        total_sq = c_sq.sum(-1)[:, None, None]  # (G, 1, 1, N)
        var = c_sq.mean(1)[:, None, :, None]  # (G, 1, D, 1)
        rank = q.abs().sum(1).argsort(-1, descending=True).argsort(-1)  # (G, Q, D)
        partial = torch.zeros(groups, size, queries, n, device=q.device)
        read_sq = torch.zeros(groups, queries, n, device=q.device)
        alive = valid.expand(groups, -1, -1)
        level = torch.zeros(alive.shape, dtype=torch.long, device=q.device)
        for r in range(1, rounds + 1):
            now = (rank >= (r - 1) * self.dims) & (rank < r * self.dims)
            partial += (q * now[:, None]) @ c[:, None].mT
            q_left = q * (rank >= r * self.dims)[:, None]
            if self.bound == "exact":
                read_sq += now.float() @ c_sq.mT
                c_left = (total_sq - read_sq[:, None]).clamp(min=0).sqrt()
                radius = q_left.norm(dim=-1, keepdim=True) * c_left
            else:
                radius = (q_left.square() @ var).sqrt()
            est = (offset + partial) * chunk.scaling
            radius = radius * (self.width * chunk.scaling)
            lower = (est - radius).logsumexp(1).masked_fill(~alive, float("-inf"))
            upper = (est + radius).logsumexp(1)
            cutoff = lower.topk(min(self.keep, n), -1).values[..., -1:]
            survive = alive & (upper >= cutoff)
            level = torch.where(alive & ~survive, r, level)
            alive = survive
            if bool((alive.sum(-1) <= self.keep).all()):
                break  # upper >= lower >= cutoff: no later round drops a key
        level = torch.where(alive, rounds + 1, level)
        levels = [self.dims * r for r in range(rounds + 1)] + [dim]
        dims_read = torch.tensor(levels, device=q.device)
        reads = dims_read[level].sum(-1) + dim * alive.sum(-1)
        sectors = k_sectors(rank, dims_read, level)
        return Outcome(attend(chunk, alive), alive, reads, sectors)
