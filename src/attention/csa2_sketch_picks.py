"""DeepSeek-V4.1 CSA2's layout with our parameter-free picks, three tiers cheaper:
a small per-key sketch scores every key before the window, the best pool x keep are
rescored exactly, the top keep are the picks (Csa2ExactPicks' rule inside them).

A sketch is kept as keys are written, so stage 1 reads fewer bytes per key than the
exact score (V4.1-Flash-1017's indexer reads 128 FP4 channels per entry):
- sign: 1 bit per channel of the key turned by a fixed random rotation, times one
  scale per key (the mean size), as RaBitQ (arXiv 2405.12497);
- pca: the key's first D / fraction principal coordinates, as Loki;
- random: D / fraction Gaussian projections (Johnson-Lindenstrauss);
- channels: no extra cache; the summed query's top 16-channel blocks (by size) read
  from the key, as SparQ.
"""

from dataclasses import dataclass

import torch

from attention import loki
from attention.csa2_exact_picks import Csa2ExactPicks
from attention.harness import Chunk, Outcome, attend, recent_keys, top_mask

CHANNEL_BLOCK = 16  # channels read together by the channels sketch


@dataclass(frozen=True)
class Csa2SketchPicks(Csa2ExactPicks):
    """The window plus the top `keep` older keys among the best by a key sketch."""

    sketch: str = "sign"  # sign | pca | random | channels
    fraction: int = 1  # sketch width: D // fraction channels (sign: bits)
    pool: int = 4  # candidates per pick, rescored exactly

    def projection(self, key: torch.Tensor) -> torch.Tensor | None:
        """(..., D, r) axes the keys of key (G, n, D) are sketched on; None: raw."""
        dim = key.shape[-1]
        r = dim // self.fraction
        if self.sketch == "channels":
            return None
        if self.sketch == "pca":
            return loki.basis(key.float())[..., :r].to(key.dtype)
        axes = torch.randn(dim, dim, generator=torch.Generator().manual_seed(0))
        if self.sketch == "sign":
            axes = torch.linalg.qr(axes)[0]
        return axes[:, :r].to(key)

    def sketch_keys(self, key: torch.Tensor, axes: torch.Tensor | None) -> torch.Tensor:
        """(G, n, r) what is kept per key."""
        if axes is None:
            return key
        out = key @ axes
        if self.sketch == "sign":
            out = out.sign() * out.abs().mean(-1, keepdim=True)
        return out

    def sketch_query(self, qs: torch.Tensor, axes: torch.Tensor | None) -> torch.Tensor:
        """(G, Q, r) the summed query qs (G, Q, D) in the sketch's coordinates."""
        if axes is not None:
            return qs @ axes
        size = qs.abs().unflatten(-1, (-1, CHANNEL_BLOCK)).sum(-1)
        top = size.topk(qs.shape[-1] // self.fraction // CHANNEL_BLOCK, -1).indices
        keep = torch.zeros_like(size, dtype=torch.bool).scatter_(-1, top, True)
        return qs * keep.repeat_interleave(CHANNEL_BLOCK, -1)

    def read_per_key(self, dim: int) -> float:
        """K elements stage 1 reads per key (a sign bit: 1/16 of a bf16 value)."""
        r = dim // self.fraction
        return r / 16 if self.sketch == "sign" else r

    def __call__(self, chunk: Chunk) -> Outcome:
        """Exact attention over the window and the picks."""
        recent = recent_keys(chunk.valid, self.window)
        older = chunk.valid & ~recent
        axes = self.projection(chunk.key)
        keys = self.sketch_keys(chunk.key, axes)
        approx = self.sketch_query(chunk.query.sum(1), axes) @ keys.mT
        approx = approx.masked_fill(~older, float("-inf"))
        candidates = top_mask(approx, self.pool * self.keep, older)
        score = self.score(chunk).masked_fill(~candidates, float("-inf"))
        selected = top_mask(score, self.keep, candidates) | recent
        dim = chunk.query.shape[-1]
        stage1 = self.read_per_key(dim) * older.sum(-1)
        reads = stage1 + dim * (candidates.sum(-1) + selected.sum(-1))
        return Outcome(attend(chunk, selected), selected, reads)
