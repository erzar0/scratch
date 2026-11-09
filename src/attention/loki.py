"""Loki (Singhania et al., arXiv 2406.02542): approximate scores in a PCA basis.

basis() fits, per KV head, the principal axes of the scored context's keys. Loki
rotates q and k into that basis (any orthogonal map keeps q k^T), scores every key
on the first d axes, and gives the top k exact attention. Reads per query and KV
head: n d + 2 k d_head.
"""

from dataclasses import dataclass

import torch

from attention.harness import Chunk, Outcome, attend, group_score, top_mask


def basis(key: torch.Tensor) -> torch.Tensor:
    """(G, D, D) principal axes of key (G, N, D), by variance.

    ponytail: fitted on the context's own keys, not on an offline calibration set
    as in the paper; on one synthetic distribution the two agree.
    """
    # uncentered second moments: the axes that best reconstruct the keys (Lemma 4.2)
    moments = key.double().mT @ key.double()
    return torch.linalg.eigh(moments)[1].flip(-1).float()


@dataclass(frozen=True)
class Loki:
    """Loki attention.

    Attributes:
        d: PCA axes of the approximate scores.
        k: Keys that get exact attention.
    """

    d: int = 32
    k: int = 128

    def __call__(self, chunk: Chunk) -> Outcome:
        """Loki on one chunk."""
        axes = basis(chunk.key)[..., : self.d]
        q = chunk.query @ axes[:, None]
        k = chunk.key @ axes
        approx = (q @ k[:, None].mT) * chunk.scaling
        score = group_score(approx.masked_fill(~chunk.valid, float("-inf")))
        selected = top_mask(score, self.k, chunk.valid)
        dim = chunk.query.shape[-1]
        reads = chunk.valid.sum(-1) * self.d + 2 * dim * selected.sum(-1)
        return Outcome(attend(chunk, selected), selected, reads)
