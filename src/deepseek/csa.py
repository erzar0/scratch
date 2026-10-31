"""DeepSeek-V4 Compressed Sparse Attention: the Hugging Face model, unchanged.

This is the CSA baseline that deepseek/csa2.py edits. It is transformers'
DeepseekV4ForCausalLM with every layer set to CSA and the sizes shrunk to our
WikiText-103 runs (about 50M parameters, like the earlier JAX models):

    d_model 512, 8 layers, 8 query heads of 64 sharing one KV head, RoPE on 16
    channels, query latent 256, 4 indexer heads of 32, top_k 8 compressed entries,
    window 32 raw tokens, compression ratio m = 4, grouped output 2 x 128.

Two pieces are trimmed because attention is what we study (assumed choices):
    MoE    one routed and one shared expert, so a dense SwiGLU; no hash routing.
    head   input and output embeddings tied.

The HF indexer gets no gradient: its scores only feed top-k. A model trained with
this file keeps its random indexer; csa2.py adds the indexer loss.
"""

from typing import Any

from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
)

VOCAB = 50304  # GPT-2's 50257 padded to a multiple of 64
N_LAYERS = 8
CSA = "compressed_sparse_attention"

SMALL: dict[str, Any] = {
    "vocab_size": VOCAB,
    "hidden_size": 512,
    "num_hidden_layers": N_LAYERS,
    "num_attention_heads": 8,
    "head_dim": 64,
    "partial_rotary_factor": 0.25,
    "q_lora_rank": 256,
    "o_groups": 2,
    "o_lora_rank": 128,
    "index_n_heads": 4,
    "index_head_dim": 32,
    "index_topk": 8,
    "sliding_window": 32,
    "compress_rates": {CSA: 4, "heavily_compressed_attention": 128},
    "n_routed_experts": 1,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 768,
    # HF's per-expert loop: grouped_mm falls back to a host-synced custom op under
    # torch.compile with float32 weights
    "experts_implementation": "eager",
    "tie_word_embeddings": True,
    "max_position_embeddings": 8192,
}


def small_config(**overrides: Any) -> DeepseekV4Config:
    """The SMALL DeepSeek-V4 config with all-CSA layers.

    Args:
        **overrides: Config fields that replace SMALL's values.

    Returns:
        The config.
    """
    fields = SMALL | overrides
    n_layers = fields["num_hidden_layers"]
    return DeepseekV4Config(
        layer_types=[CSA] * n_layers, mlp_layer_types=["moe"] * n_layers, **fields
    )


def build(**overrides: Any) -> DeepseekV4ForCausalLM:
    """A freshly initialized SMALL DeepSeek-V4 CSA language model.

    Args:
        **overrides: Config fields that replace SMALL's values.

    Returns:
        The model, on the CPU in float32.
    """
    return DeepseekV4ForCausalLM(small_config(**overrides))
