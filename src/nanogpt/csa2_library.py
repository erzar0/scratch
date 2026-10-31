"""CSA2 attending the way transformers' DeepSeek-V4 does: dense, then masked.

transformers 5.19 has DeepSeek-V4 CSA, not V4.1 CSA2. VarlenCsa2Attention attends
with csa2_varlen.sparse_attention: each query gathers its own k + window keys. This
layer instead runs HF's eager_attention_forward, as DeepseekV4Attention does: every
query against all S raw entries and all n main KV entries, an additive mask keeping
the sliding window and the picks. Same math (tests/test_nanogpt.py); O(S (S + n))
memory. Masks are document-cut, as csa2_varlen's.
"""

import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    eager_attention_forward,
)

from deepseek import csa2
from nanogpt.csa2_varlen import VarlenCsa2Attention


class LibraryCsa2Attention(VarlenCsa2Attention):
    """VarlenCsa2Attention with HF's dense masked attention."""

    def attend_documents(
        self,
        query: torch.Tensor,
        window_kv: torch.Tensor,
        main_kv: torch.Tensor,
        picks: csa2.Picks,
        starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HF eager attention over cat(raw entries, main KV).

        Returns:
            Output (B, S, H, D) and the attention weights on the picks (B, H, S, k),
            all the indexer loss reads.
        """
        batch, seq_len, _ = window_kv.shape
        n = main_kv.shape[1]
        positions = torch.arange(seq_len, device=query.device)
        # raw entry j for query t: causal, inside the window, same document
        offset = positions[:, None] - positions
        window = (offset >= 0) & (offset < self.sliding_window)
        window &= positions >= starts[:, None]
        mask = torch.zeros(seq_len, seq_len, dtype=query.dtype, device=query.device)
        mask = mask.masked_fill(~window, float("-inf")).expand(batch, 1, -1, -1)
        # main KV: HF's block_bias, 0 at valid picks, -inf elsewhere
        safe = torch.where(picks.valid, picks.entry, n)
        block = mask.new_full((batch, 1, seq_len, n + 1), float("-inf"))
        block = block.scatter(-1, safe.unsqueeze(1), 0.0)[..., :n]
        kv = torch.cat([window_kv, main_kv], dim=1).unsqueeze(1)
        output, weights = eager_attention_forward(
            self, query, kv, kv, torch.cat([mask, block], -1), self.scaling
        )
        entry = picks.entry.unsqueeze(1).expand(-1, weights.shape[1], -1, -1)
        on_picks = weights[..., seq_len:].gather(-1, entry)
        return output, on_picks.masked_fill(~picks.valid.unsqueeze(1), 0.0)
