"""Qwen3-0.6B-Base with CSA2 attention; every other weight frozen.

Each Qwen3DecoderLayer's self_attn becomes a QwenCsa2Attention: deepseek/csa2.py's
Csa2Attention, unchanged, behind Qwen's call signature. Sizes are V4-Flash's
attention scaled to width 1024 (V4-Flash: 64 heads of 512, query latent 1024,
indexer 64 x 128, top-512, window 128, m = 4):

    16 heads of 128, RoPE on 16 channels, query latent 256, output 2 groups x 256,
    indexer 16 heads of 32, top-64 entries, window 128, m = 4,
    modes (Full, Reuse, Reindex, Reuse) x 7.

Qwen's own attentions stay in model.teachers; teacher() puts them back for a
forward. convert(model, None) is the dense control: Qwen's attention trainable.
"""

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4RotaryEmbedding,
)

from deepseek import csa2

QWEN = "Qwen/Qwen3-0.6B-Base"

# V4-Flash attention scaled to width 1024; heads and head_dim come from Qwen
SIZES: dict[str, Any] = {
    "partial_rotary_factor": 0.125,
    "q_lora_rank": 256,
    "o_groups": 2,
    "o_lora_rank": 256,
    "index_n_heads": 16,
    "index_head_dim": 32,
    "index_topk": 64,
    "sliding_window": 128,
}


def csa2_config(qwen: Qwen3Config, **overrides: Any) -> DeepseekV4Config:
    """CSA2 attention config for a Qwen3 model: its width, heads and context.

    Args:
        qwen: The Qwen3 config.
        **overrides: Fields that replace SIZES (tests use tiny ones).

    Returns:
        The config, modes (Full, Reuse, Reindex, Reuse) repeated over the layers.
    """
    pattern = (csa2.FULL, csa2.REUSE, csa2.REINDEX, csa2.REUSE)
    modes = (pattern * qwen.num_hidden_layers)[: qwen.num_hidden_layers]
    return csa2.small_config(
        modes,
        **(
            SIZES
            | {
                "vocab_size": qwen.vocab_size,
                "hidden_size": qwen.hidden_size,
                "num_attention_heads": qwen.num_attention_heads,
                "head_dim": qwen.head_dim,
                "rms_norm_eps": qwen.rms_norm_eps,
                "max_position_embeddings": qwen.max_position_embeddings,
            }
            | overrides
        ),
    )


class Current:
    """The SharedState of the forward in progress, reset before each model forward."""

    def __init__(self) -> None:
        self.state = csa2.SharedState()

    def reset(self, *_: object) -> None:
        """Forward pre-hook: a fresh SharedState."""
        self.state = csa2.SharedState()


class QwenCsa2Attention(csa2.Csa2Attention):
    """Csa2Attention called the way Qwen3DecoderLayer calls Qwen3Attention."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int, current: Current):
        super().__init__(config, layer_idx)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.current = current
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
        nn.init.zeros_(self.sinks)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Csa2Attention.forward on Qwen's arguments.

        Qwen's causal mask and RoPE are unused: CSA2 masks by position and has its
        own RoPE. Padding is unsupported, so only right padding is safe.
        """
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0)
        cos_sin = self.rotary_emb(
            hidden_states, position_ids, layer_type=self.rope_layer_type
        )
        return super().forward(
            hidden_states,
            {self.rope_layer_type: cos_sin},
            position_ids,
            None,
            past_key_values,
            csa2_shared=self.current.state,
        )


def load(name: str = QWEN, device: str = "cuda") -> Qwen3ForCausalLM:
    """Pretrained Qwen3 in bfloat16, uncached, eval mode."""
    model = Qwen3ForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map=device
    )
    model.config.use_cache = False
    # Qwen ships max_new_tokens 2048, which overrides lm-eval's per-task max_length
    model.generation_config.max_new_tokens = None
    return model.eval()


def convert(
    model: Qwen3ForCausalLM, config: DeepseekV4Config | None
) -> Qwen3ForCausalLM:
    """Freeze model, keep its attentions as teachers, add trainable attentions.

    Args:
        model: A Qwen3 model; changed in place.
        config: CSA2 attention config (csa2_config), or None for the dense
            control: trainable copies of Qwen's own attentions.

    Returns:
        model, with float32 trainable attentions on its device.
    """
    model.requires_grad_(False)
    layers = model.model.layers
    model.teachers = nn.ModuleList(layer.self_attn for layer in layers)
    device = model.lm_head.weight.device
    current = Current()
    for i, layer in enumerate(layers):
        student = (
            copy.deepcopy(layer.self_attn)
            if config is None
            else QwenCsa2Attention(config, i, current)
        )
        layer.self_attn = student.to(device, torch.float32).requires_grad_(True)
    model.model.register_forward_pre_hook(current.reset)
    return model


def students(model: Qwen3ForCausalLM) -> list[nn.Module]:
    """The trainable attention of every layer."""
    return [layer.self_attn for layer in model.model.layers]


def csa2_students(model: Qwen3ForCausalLM) -> list[QwenCsa2Attention]:
    """The CSA2 attention of every layer.

    Raises:
        TypeError: If model was converted without a CSA2 config.
    """
    attentions = students(model)
    if not all(isinstance(a, QwenCsa2Attention) for a in attentions):
        raise TypeError("model has no CSA2 attention")
    return [a for a in attentions if isinstance(a, QwenCsa2Attention)]


def teachers(model: Qwen3ForCausalLM) -> nn.ModuleList:
    """Qwen's original attentions, kept by convert."""
    kept = model.teachers
    assert isinstance(kept, nn.ModuleList)
    return kept


@contextmanager
def teacher(model: Qwen3ForCausalLM) -> Iterator[None]:
    """Run model with Qwen's original attentions inside the block."""
    layers = model.model.layers
    trained = students(model)
    for layer, original in zip(layers, teachers(model), strict=True):
        layer.self_attn = original
    try:
        yield
    finally:
        for layer, attention in zip(layers, trained, strict=True):
            layer.self_attn = attention


def trainable_state(model: Qwen3ForCausalLM) -> dict[str, torch.Tensor]:
    """The trainable weights only: what a checkpoint stores."""
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
