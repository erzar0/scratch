"""DeepSeek-V4.1 Compressed Sparse Attention 2 (CSA2), as a diff on DeepSeek-V4 CSA.

deepseek/csa.py is the baseline: transformers' DeepSeek-V4, unchanged. Each class
here subclasses or replaces one HF class. Every CSA2 change carries a
``# CSA2 (V4.1 ...)`` comment; code without one is HF's, inherited or copied.

    HF DeepSeek-V4 (CSA)       CSA2 (this file)    what changes, V4.1 §2.3
    DeepseekV4CSACompressor -> Csa2Compressor      one window of m tokens per entry:
                                                   no overlap, no position bias;
                                                   indexer K = projection of main KV
    DeepseekV4Indexer       -> Csa2Indexer         no compressor of its own: scores
                                                   the shared indexer K
    DeepseekV4Attention     -> Csa2Attention       Full / Reindex / Reuse modes share
                                                   main KV, indexer K and top-k
    DeepseekV4DecoderLayer  -> Csa2DecoderLayer    holds a Csa2Attention
    DeepseekV4Model         -> Csa2Model           hands shared tensors to layers
    DeepseekV4ForCausalLM   -> Csa2ForCausalLM     holds a Csa2Model

Three additions the report does not specify, marked ``# not in V4.1``:
    indexer loss    DeepSeek-V3.2's KL from the attention to the indexer. The HF
                    indexer gets no gradient (its scores only feed top-k).
    sparse attend   each query gathers its top-k entries and its window, instead of
                    HF's dense masked softmax over all T + T/m keys: same math,
                    O(T (k + window)) memory. tests/test_csa2.py checks it.
    selector hook   Csa2Attention.selector, None by default: a block selector
                    (deepseek/double_indexing.py, deepseek/reference.py) that picks
                    the top-k instead of the indexer, at evaluation.

Not implemented: the Hierarchical Sparse Indexer (§2.3.2), the causal
encoder-decoder (§2.2), padding masks, and the KV cache (no decoding).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4IndexerScorer,
    DeepseekV4Model,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
)

from deepseek import csa

FULL, REINDEX, REUSE = "full", "reindex", "reuse"
# Every mode; main KV shared in groups of 4 layers like the V4.1 decoder, top-k
# recomputed every second layer (assumed: V4.1 does not specify an 8-layer model).
MODES = (FULL, REUSE, REINDEX, REUSE) * 2


class Picks(NamedTuple):
    """Each query's top-k, all (B, S, k): main KV entry, False where the entry is
    not finished yet, and the indexer score (with gradient, for the indexer loss)."""

    entry: torch.Tensor
    valid: torch.Tensor
    score: torch.Tensor


class SelectorInputs(NamedTuple):
    """What a block selector sees in one Full or Reindex layer (not in V4.1).

    Attributes:
        hidden_states: (B, S, hidden) layer input.
        q_residual: (B, S, q_lora_rank) the attention's query latent.
        position_ids: (B, S) token positions.
        query: (B, H, S, D) rotated main queries.
        main_kv: (B, n, D) shared main KV.
        index_k: (B, n, index_head_dim) shared indexer K.
    """

    hidden_states: torch.Tensor
    q_residual: torch.Tensor
    position_ids: torch.Tensor
    query: torch.Tensor
    main_kv: torch.Tensor
    index_k: torch.Tensor


def finished_entries(position_ids: torch.Tensor, compress_rate: int) -> torch.Tensor:
    """(B, S, 1) main KV entries finished at each query: windows ending by then."""
    return ((position_ids + 1) // compress_rate).unsqueeze(-1)


def top_picks(
    scores: torch.Tensor, position_ids: torch.Tensor, compress_rate: int, top_k: int
) -> Picks:
    """Each query's top_k finished entries by score.

    Args:
        scores: (B, S, n) entry scores, -inf where a selector rules an entry out.
        position_ids: (B, S) token positions.
        compress_rate: Tokens per entry (m).
        top_k: Entries to pick.

    Returns:
        The picks; fewer than top_k allowed entries leave invalid extras.
    """
    n_entries = scores.shape[-1]
    future = torch.arange(n_entries, device=scores.device) >= finished_entries(
        position_ids, compress_rate
    )
    score, entry = scores.masked_fill(future, float("-inf")).topk(
        min(top_k, n_entries), dim=-1
    )
    return Picks(entry, score > float("-inf"), score)


@dataclass
class SharedState:
    """What CSA2 layers hand to later layers within one forward, by source layer.

    Attributes:
        global_kv: Full layer -> (main KV (B, n, D), indexer K (B, n, c)).
        picks: Full or Reindex layer -> its top-k.
    """

    global_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    picks: dict[int, Picks] = field(default_factory=dict)


def small_config(modes: tuple[str, ...] = MODES, **overrides: Any) -> DeepseekV4Config:
    """csa.small_config plus each layer's CSA2 mode (config.csa2_modes).

    Args:
        modes: full, reindex or reuse per layer.
        **overrides: Config fields that replace csa.SMALL's values.

    Returns:
        The config.

    Raises:
        ValueError: If a mode is unknown or layer 0 is not full.
    """
    if modes[0] != FULL or not set(modes) <= {FULL, REINDEX, REUSE}:
        raise ValueError(f"modes must start with full and use {FULL, REINDEX, REUSE}")
    config = csa.small_config(num_hidden_layers=len(modes), **overrides)
    config.csa2_modes = list(modes)
    return config


class Csa2Compressor(nn.Module):
    """Global KV of a Full layer: main KV entries and their indexer keys.

    Replaces DeepseekV4CSACompressor and the compressor inside DeepseekV4Indexer.
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.compress_rate: int = (config.compress_rates or {})[csa.CSA]
        self.head_dim = config.head_dim
        # CSA2 (V4.1 §2.3): head_dim outputs, not 2 x head_dim (no second series
        # for the overlapping window), and no position_bias
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # CSA2 (V4.1 §2.3): indexer K is projected from main KV
        self.index_k_proj = nn.Linear(self.head_dim, config.index_head_dim, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress every finished window of m tokens.

        Args:
            hidden_states: (B, S, hidden) layer input.

        Returns:
            Main KV (B, S // m, head_dim) and indexer K (B, S // m, index_head_dim),
            both rotated to their window's first position.
        """
        batch, seq_len, _ = hidden_states.shape
        n_entries = seq_len // self.compress_rate
        tokens = hidden_states[:, : n_entries * self.compress_rate]
        windows = (batch, n_entries, self.compress_rate, self.head_dim)
        kv = self.kv_proj(tokens).view(windows)
        # CSA2 (V4.1 §2.3): entry i pools window i only (V4 adds window i - 1's
        # tokens), with no absolute position bias on the gate
        gate = self.gate_proj(tokens).view(windows).softmax(2, dtype=torch.float32)
        entries = self.kv_norm((kv * gate.to(kv.dtype)).sum(dim=2))
        positions = torch.arange(n_entries, device=entries.device) * self.compress_rate
        cos, sin = self.rotary_emb(
            entries, positions.expand(batch, -1), layer_type=self.rope_layer_type
        )
        # CSA2 (V4.1 §2.3): indexer K from main KV, not from a second compressor.
        # Before RoPE (assumed); detached as in V3.2, so only the indexer loss moves it
        index_k = self.index_k_proj(entries.detach())
        main_kv, index_k = (
            apply_rotary_pos_emb(x.unsqueeze(1), cos, sin).squeeze(1)
            for x in (entries, index_k)
        )
        return main_kv, index_k


class Csa2Indexer(nn.Module):
    """Indexer of a Full or Reindex layer: own queries, shared indexer K.

    Replaces DeepseekV4Indexer; keeps its query path and DeepseekV4IndexerScorer.
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.compress_rate: int = (config.compress_rates or {})[csa.CSA]
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        # CSA2 (V4.1 §2.3): no kv_proj, gate_proj, position_bias or kv_norm: the
        # keys come from Csa2Compressor
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, config.index_n_heads * self.head_dim, bias=False
        )
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.scorer = DeepseekV4IndexerScorer(config)

    def queries(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """(B, S, index_n_heads, index_head_dim) rotated indexer queries; RoPE in
        hidden_states' dtype."""
        batch, seq_len, _ = q_residual.shape
        cos, sin = self.rotary_emb(
            hidden_states, position_ids, layer_type=self.rope_layer_type
        )
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim)
        return apply_rotary_pos_emb(q.transpose(1, 2), cos, sin).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        index_k: torch.Tensor,
    ) -> Picks:
        """Pick each query's top-k finished entries.

        Args:
            hidden_states: (B, S, hidden) layer input.
            q_residual: (B, S, q_lora_rank) the attention's query latent.
            position_ids: (B, S) token positions.
            index_k: (B, n, index_head_dim) shared indexer K.

        Returns:
            The picks. Queries with fewer than k finished entries get invalid extras.
        """
        batch, seq_len, _ = hidden_states.shape
        # not in V4.1: detached inputs, as in V3.2, so the indexer loss moves only
        # the indexer and the LM loss does not reach it
        hidden_states, q_residual = hidden_states.detach(), q_residual.detach()
        q = self.queries(hidden_states, q_residual, position_ids)
        with torch.no_grad():
            scores = self.scorer(q, index_k, hidden_states)  # (B, S, n)
            entry, valid, _ = top_picks(
                scores, position_ids, self.compress_rate, self.index_topk
            )
        # not in V4.1: the picked scores again, with gradient, for the indexer loss.
        # The HF scorer, one query per row: never builds the (B, S, n) graph.
        keys = index_k[torch.arange(batch, device=entry.device)[:, None, None], entry]
        score = self.scorer(
            q.flatten(0, 1).unsqueeze(1),
            keys.flatten(0, 1),
            hidden_states.flatten(0, 1).unsqueeze(1),
        )
        return Picks(entry, valid, score.view(batch, seq_len, -1))


def sparse_attention(
    module: DeepseekV4Attention,
    query: torch.Tensor,
    window_kv: torch.Tensor,
    main_kv: torch.Tensor,
    picks: Picks,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF eager_attention_forward over each query's own keys (not in V4.1).

    Query t reads its valid picks from main KV and the sliding_window raw entries
    ending at t (shared K = V), plus each head's sink logit, as the HF dense mask
    allows; nothing else is computed.

    Args:
        module: The layer: sinks, scaling and sliding_window.
        query: (B, H, S, D) rotated queries.
        window_kv: (B, S, D) rotated raw entries.
        main_kv: (B, n, D) rotated main KV.
        picks: Each query's top-k.

    Returns:
        Output (B, S, H, D), as eager returns it, and attention weights
        (B, H, S, k + window), picks first.
    """
    batch, seq_len, _ = window_kv.shape
    window = module.sliding_window
    padded = F.pad(window_kv, (0, 0, window - 1, 0))
    window_keys = padded.unfold(1, window, 1).transpose(-1, -2)  # (B, S, w, D)
    positions = torch.arange(seq_len, device=query.device)
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


def indexer_loss(probs: torch.Tensor, picks: Picks) -> torch.Tensor:
    """DeepSeek-V3.2's indexer loss (not in V4.1).

    Cross-entropy (KL up to a constant) from the attention's weight on the picks,
    summed over heads and renormalized, to the softmax of the picked scores.

    Args:
        probs: (B, H, S, k + window) attention weights from sparse_attention.
        picks: The layer's picks.

    Returns:
        Mean loss over queries.
    """
    target = probs[..., : picks.entry.shape[-1]].detach().sum(1)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-9)
    # -1e9, not -inf: a query with no valid pick gets a finite log-softmax, times 0
    log_p = picks.score.masked_fill(~picks.valid, -1e9).log_softmax(-1)
    return -(target * log_p).sum(-1).mean()


# not in V4.1: a block selector (deepseek/double_indexing.py, deepseek/reference.py)
# picks a Full or Reindex layer's top-k instead of its indexer, under no_grad
type Selector = Callable[["Csa2Attention", SelectorInputs], Picks]


class _AttachLoss(torch.autograd.Function):
    """Identity on the output; the backward gives the loss gradient 1, as if the
    loss were added to the training loss."""

    @staticmethod
    def forward(ctx: Any, output: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        ctx.loss_like = (loss.dtype, loss.device)
        return output.view_as(output)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype, device = ctx.loss_like
        return grad, torch.ones((), dtype=dtype, device=device)


class Csa2Attention(DeepseekV4Attention):
    """DeepseekV4Attention with CSA2's global branch.

    Query, sliding-window KV, sinks and the grouped output projection are HF's.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        # CSA2 (V4.1 §2.3.1): a static mode per layer. Full owns the global KV and
        # an indexer, Reindex only an indexer, Reuse neither.
        modes = config.csa2_modes[: layer_idx + 1]
        self.mode = modes[-1]
        self.kv_source = max(i for i, mode in enumerate(modes) if mode == FULL)
        self.picks_source = max(i for i, mode in enumerate(modes) if mode != REUSE)
        self.compressor = Csa2Compressor(config) if self.mode == FULL else None
        self.indexer = Csa2Indexer(config) if self.mode != REUSE else None
        # not in V4.1: picks the top-k instead of the indexer when set
        self.selector: Selector | None = None

    def attend(
        self,
        query: torch.Tensor,
        window_kv: torch.Tensor,
        main_kv: torch.Tensor,
        picks: Picks,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """sparse_attention; variants override it (not in V4.1)."""
        return sparse_attention(self, query, window_kv, main_kv, picks)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: object | None = None,
        csa2_shared: SharedState | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention output (B, S, hidden) and weights (B, H, S, k + window).

        Raises:
            NotImplementedError: If called with a KV cache.
        """
        if past_key_values is not None or csa2_shared is None:
            raise NotImplementedError("CSA2 runs uncached, from Csa2Model")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = apply_rotary_pos_emb(self.q_b_norm(q), cos, sin)
        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape)
        window_kv = apply_rotary_pos_emb(kv.transpose(1, 2), cos, sin)[:, 0]

        # CSA2 (V4.1 §2.3.1): Full computes main KV and indexer K, later layers
        # reuse the latest Full layer's
        if self.compressor is not None:
            csa2_shared.global_kv[self.layer_idx] = self.compressor(hidden_states)
        main_kv, index_k = csa2_shared.global_kv[self.kv_source]
        # CSA2 (V4.1 §2.3.1): Full and Reindex pick their own top-k from the shared
        # indexer K; Reuse takes the latest picks
        if self.indexer is not None and self.selector is not None:
            # not in V4.1: a block selector instead of the indexer
            inputs = SelectorInputs(
                hidden_states, q_residual, position_ids, q, main_kv, index_k
            )
            csa2_shared.picks[self.layer_idx] = self.selector(self, inputs)
        elif self.indexer is not None:
            csa2_shared.picks[self.layer_idx] = self.indexer(
                hidden_states, q_residual, position_ids, index_k
            )
        picks = csa2_shared.picks[self.picks_source]

        attn_output, attn_weights = self.attend(q, window_kv, main_kv, picks)
        attn_output = apply_rotary_pos_emb(
            attn_output.transpose(1, 2), cos, -sin
        ).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        output = self.o_b_proj(self.o_a_proj(grouped).flatten(2))
        if (
            self.indexer is not None
            and self.selector is None
            and torch.is_grad_enabled()
        ):
            output = _AttachLoss.apply(output, indexer_loss(attn_weights, picks))
        return output, attn_weights


class Csa2DecoderLayer(DeepseekV4DecoderLayer):
    """DeepseekV4DecoderLayer (mHC, MoE) holding a Csa2Attention."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.self_attn = Csa2Attention(config, layer_idx)  # CSA2


class Csa2Model(DeepseekV4Model):
    """DeepseekV4Model whose layers share one SharedState per forward."""

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(  # CSA2
            Csa2DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.post_init()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """DeepseekV4Model.forward; layers receive a fresh SharedState."""
        # CSA2 (V4.1 §2.3.1): layers hand main KV, indexer K and top-k on
        return super().forward(*args, csa2_shared=SharedState(), **kwargs)


class Csa2ForCausalLM(DeepseekV4ForCausalLM):
    """DeepseekV4ForCausalLM holding a Csa2Model."""

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__(config)
        self.model = Csa2Model(config)  # CSA2
        self.post_init()


def build(modes: tuple[str, ...] = MODES, **overrides: Any) -> Csa2ForCausalLM:
    """A freshly initialized SMALL CSA2 language model.

    Args:
        modes: full, reindex or reuse per layer.
        **overrides: Config fields that replace csa.SMALL's values.

    Returns:
        The model, on the CPU in float32.
    """
    return Csa2ForCausalLM(small_config(modes, **overrides))
