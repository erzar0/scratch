"""DeepSeek-V4.1 CSA2, one decode token as DeepSeek's V4.1-Flash-1017 code runs it
(src/nanogpt/csa2_ref.py ports that code for training): latency only.

CSA2 picks entries with a learned indexer over index keys of its own, so on the
suite's keys its picks mean nothing (as deepseek_csa): the index keys, index query,
head weights and sinks are random, seeded. One layer of ratio 1 (V4.1's last
layers: one entry per token); the suite's keys are its entries and its window rows,
K = V (CSA2 stores one row per token, so the suite's values are not read).
Per sequence:
    indexer    sum_h w_h relu(q_h . k_index) / sqrt(128) for every entry; top keep;
    attention  over the keep picks and the newest 128 rows, a sink logit per head,
               then the output turned back by RoPE on its last 16 channels.
Batch: one sequence per KV head of the suite, 2 query heads each, as
deepseek_csa.LibraryCsa; index 2 heads of 128, also as LibraryCsa. Not timed: the
projections, cache writes, FP4 caches, and V4.1's candidate blocks (its later
index layers search only 256 blocks of 8 that an earlier layer chose).
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from attention.fast_pytorch import KV


@dataclass(frozen=True)
class Csa2Reference:
    """CSA2's decode step at the suite's shapes; q, kv give shapes and K = V rows."""

    keep: int = 512  # entries picked by the indexer
    window: int = 128  # newest rows, always read
    index_heads: int = 2
    rope: int = 16  # channels at the end that carry RoPE

    def weights(self, q: Tensor, n: int) -> dict[str, Tensor]:
        """Random index keys (G, n, D), index query (G, H, D), head weights (G, H),
        sinks (G, g) and RoPE angles (rope / 2,), seeded: the same at every keep."""
        groups, heads, dim = q.shape
        gen = torch.Generator(q.device).manual_seed(0)

        def normal(*shape: int) -> Tensor:
            return torch.randn(*shape, generator=gen, device=q.device)

        angle = normal(self.rope // 2)
        return {
            "index_k": normal(groups, n, dim).to(q.dtype),
            "qi": normal(groups, self.index_heads, dim).to(q.dtype),
            "w": normal(groups, self.index_heads),
            "sinks": normal(groups, heads),
            "cos": angle.cos(),
            "sin": angle.sin(),
        }

    def __call__(self, q: Tensor, kv: KV) -> tuple[Tensor, Tensor]:
        """(G, g, D) output of one new token per sequence, and the picks (G, keep)."""
        dim = q.shape[-1]
        s = kv.once("csa2_reference", lambda: self.weights(q, kv.k.shape[1]))
        score = (s["index_k"] @ s["qi"].mT).float().relu()  # (G, n, H)
        picks = (score @ s["w"][..., None])[..., 0].topk(self.keep, -1).indices
        rows = torch.cat(
            [
                kv.k.gather(1, picks[..., None].expand(-1, -1, dim)),
                kv.k[:, -self.window :],
            ],
            1,
        )  # (G, keep + window, D), K = V
        logits = (q @ rows.mT).float() * dim**-0.5
        logits = torch.cat([logits, s["sinks"][..., None]], -1)
        out = (logits.softmax(-1)[..., :-1].to(rows.dtype) @ rows).float()
        x1, x2 = out[..., -self.rope :].chunk(2, -1)  # RoPE pairs: (i, i + rope / 2)
        cos, sin = s["cos"], s["sin"]
        turned = torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], -1)
        return torch.cat([out[..., : -self.rope], turned], -1).to(q.dtype), picks
