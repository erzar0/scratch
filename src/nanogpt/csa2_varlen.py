"""CSA2 on record 91's packed batches: documents back to back in one row.

deepseek/csa2.py has no padding or document masks. VarlenCsa2Attention is its
Csa2Attention for record 91's varlen batches (B = 1, cu_seqlens at each document's
first token), with every change marked ``# varlen``:

- positions run over the whole row. RoPE is relative, so a document's offset in the
  row changes nothing;
- a query reads window keys and main KV entries of its own document only. An entry
  straddling a boundary is unfinished for the earlier document and starts before
  the later one, so no query reads it.

Modes, shared main KV and top-k, and the indexer loss are Csa2Attention's.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
)

from deepseek import csa2


def document_starts(cu_seqlens: torch.Tensor, length: int) -> torch.Tensor:
    """(length,) row position where each token's document starts.

    Args:
        cu_seqlens: Document starts, then the row length, possibly repeated
            (record 91 pads it to a fixed size).
        length: Row length.
    """
    cu = cu_seqlens.long()
    positions = torch.arange(length, device=cu.device)
    return cu[torch.searchsorted(cu, positions, right=True) - 1]


class VarlenIndexer(csa2.Csa2Indexer):
    """Csa2Indexer that picks entries of the query's own document only."""

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        index_k: torch.Tensor,
        starts: torch.Tensor,
    ) -> csa2.Picks:
        """Csa2Indexer.forward; starts (S,) from document_starts."""
        batch, seq_len, _ = hidden_states.shape
        hidden_states, q_residual = hidden_states.detach(), q_residual.detach()
        q = self.queries(hidden_states, q_residual, position_ids)
        with torch.no_grad():
            scores = self.scorer(q, index_k, hidden_states)  # (B, S, n)
            # varlen: entries starting before the query's document are ruled out
            entry_start = torch.arange(scores.shape[-1], device=scores.device)
            earlier = entry_start * self.compress_rate < starts[:, None]
            entry, valid, _ = csa2.top_picks(
                scores.masked_fill(earlier, float("-inf")),
                position_ids,
                self.compress_rate,
                self.index_topk,
            )
        keys = index_k[torch.arange(batch, device=entry.device)[:, None, None], entry]
        score = self.scorer(
            q.flatten(0, 1).unsqueeze(1),
            keys.flatten(0, 1),
            hidden_states.flatten(0, 1).unsqueeze(1),
        )
        return csa2.Picks(entry, valid, score.view(batch, seq_len, -1))


def sparse_attention(
    module: csa2.Csa2Attention,
    query: torch.Tensor,
    window_kv: torch.Tensor,
    main_kv: torch.Tensor,
    picks: csa2.Picks,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """csa2.sparse_attention with window keys cut at the query's document start."""
    batch, seq_len, _ = window_kv.shape
    window = module.sliding_window
    padded = F.pad(window_kv, (0, 0, window - 1, 0))
    window_keys = padded.unfold(1, window, 1).transpose(-1, -2)  # (B, S, w, D)
    # varlen: position within the document, not the row
    positions = torch.arange(seq_len, device=query.device) - starts
    window_valid = torch.arange(window, device=query.device) >= (
        window - 1 - positions[:, None]
    )
    rows = torch.arange(batch, device=query.device)[:, None, None]
    keys = torch.cat([main_kv[rows, picks.entry], window_keys], dim=2)
    valid = torch.cat([picks.valid, window_valid.expand(batch, -1, -1)], dim=2)

    logits = torch.einsum("bhsd,bsed->bhse", query, keys) * module.scaling
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    sinks = module.sinks.view(1, -1, 1, 1).expand(batch, -1, seq_len, 1)
    probs = torch.cat([logits, sinks.to(logits.dtype)], dim=-1)
    probs = probs.softmax(-1, dtype=torch.float32)[..., :-1]
    output = torch.einsum("bhse,bsed->bshd", probs.to(keys.dtype), keys)
    return output, probs


class VarlenCsa2Attention(csa2.Csa2Attention):
    """Csa2Attention on one packed row: layer(x, cu_seqlens, shared) -> output.

    Own RoPE and initialization as qwen/model.py's QwenCsa2Attention; no selector.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        if self.indexer is not None:
            self.indexer = VarlenIndexer(config)  # varlen
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
        nn.init.zeros_(self.sinks)

    def attend_documents(
        self,
        query: torch.Tensor,
        window_kv: torch.Tensor,
        main_kv: torch.Tensor,
        picks: csa2.Picks,
        starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """sparse_attention; nanogpt/csa2_library.py swaps in transformers' instead."""
        return sparse_attention(self, query, window_kv, main_kv, picks, starts)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        csa2_shared: csa2.SharedState,
    ) -> torch.Tensor:
        """Csa2Attention.forward on (1, S, hidden), document-masked.

        Args:
            hidden_states: (1, S, hidden) packed row.
            cu_seqlens: Document starts, as document_starts takes them.
            csa2_shared: One SharedState per model forward.

        Returns:
            Output (1, S, hidden).
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        position_ids = torch.arange(input_shape[1], device=hidden_states.device)[None]
        starts = document_starts(cu_seqlens, input_shape[1])  # varlen
        cos, sin = self.rotary_emb(
            hidden_states, position_ids, layer_type=self.rope_layer_type
        )

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = apply_rotary_pos_emb(self.q_b_norm(q), cos, sin)
        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape)
        window_kv = apply_rotary_pos_emb(kv.transpose(1, 2), cos, sin)[:, 0]

        if self.compressor is not None:
            csa2_shared.global_kv[self.layer_idx] = self.compressor(hidden_states)
        main_kv, index_k = csa2_shared.global_kv[self.kv_source]
        if self.indexer is not None:
            csa2_shared.picks[self.layer_idx] = self.indexer(
                hidden_states, q_residual, position_ids, index_k, starts
            )
        picks = csa2_shared.picks[self.picks_source]

        attn_output, attn_weights = self.attend_documents(
            q, window_kv, main_kv, picks, starts
        )
        attn_output = apply_rotary_pos_emb(
            attn_output.transpose(1, 2), cos, -sin
        ).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        output = self.o_b_proj(self.o_a_proj(grouped).flatten(2))
        if self.indexer is not None and torch.is_grad_enabled():
            loss = csa2.indexer_loss(attn_weights, picks)
            output = csa2._AttachLoss.apply(output, loss)
        return output
