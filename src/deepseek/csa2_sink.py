"""CSA2 whose queries also read the first SINKS tokens (not in V4.1).

Qwen3 puts much of its attention weight on the first tokens and reads their
values: with the last 128 tokens and the best 64 blocks visible, zeroing the first
4 tokens' values costs Qwen 1.5 nats (qwen/probe.py mask). CSA2's sink logits take
weight but give nothing, so a CSA2 layer cannot copy that. Here every query also
reads the first SINKS raw entries, like StreamingLLM's attention sinks.
"""

import torch
import torch.nn.functional as F

from deepseek import csa2

SINKS = 4


def sink_attention(
    module: csa2.Csa2Attention,
    query: torch.Tensor,
    window_kv: torch.Tensor,
    main_kv: torch.Tensor,
    picks: csa2.Picks,
) -> tuple[torch.Tensor, torch.Tensor]:
    """csa2.sparse_attention plus the first SINKS raw entries for every query.

    Returns:
        Output (B, S, H, D) and attention weights (B, H, S, k + window + SINKS),
        picks first.
    """
    batch, seq_len, _ = window_kv.shape
    window = module.sliding_window
    padded = F.pad(window_kv, (0, 0, window - 1, 0))
    window_keys = padded.unfold(1, window, 1).transpose(-1, -2)  # (B, S, w, D)
    positions = torch.arange(seq_len, device=query.device)
    window_valid = torch.arange(window, device=query.device) >= (
        window - 1 - positions[:, None]
    )
    # a first token still inside the query's window is read there, not twice
    first = torch.arange(min(SINKS, seq_len), device=query.device)
    sink_valid = positions[:, None] - first >= window  # (S, SINKS)
    sink_keys = window_kv[:, None, : len(first)].expand(-1, seq_len, -1, -1)
    rows = torch.arange(batch, device=query.device)[:, None, None]
    keys = torch.cat([main_kv[rows, picks.entry], window_keys, sink_keys], dim=2)
    valid = torch.cat(
        [
            picks.valid,
            window_valid.expand(batch, -1, -1),
            sink_valid.expand(batch, -1, -1),
        ],
        dim=2,
    )

    logits = torch.einsum("bhsd,bsed->bhse", query, keys) * module.scaling
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    sinks = module.sinks.view(1, -1, 1, 1).expand(batch, -1, seq_len, 1)
    probs = torch.cat([logits, sinks.to(logits.dtype)], dim=-1)
    probs = probs.softmax(-1, dtype=torch.float32)[..., :-1]
    output = torch.einsum("bhse,bsed->bshd", probs.to(keys.dtype), keys)
    return output, probs


class SinkAttend:
    """Mixin for a Csa2Attention subclass: attend with sink_attention."""

    def attend(
        self,
        query: torch.Tensor,
        window_kv: torch.Tensor,
        main_kv: torch.Tensor,
        picks: csa2.Picks,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """sink_attention in place of csa2.sparse_attention."""
        assert isinstance(self, csa2.Csa2Attention)
        return sink_attention(self, query, window_kv, main_kv, picks)
