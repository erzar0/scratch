"""DeepSeek-V4 Compressed Sparse Attention: the transformers layer, unchanged.

Latency only. CSA computes its keys (one head, K = V), compressed entries and
indexer from hidden states through learned weights, so it cannot run on the suite's
q, k, v; with random weights its speed is real but its output is not, so it has no
emulation and no accuracy rows.

The library's own classes, not subclassed or patched: a one-layer DeepseekV4Model
with a CSA layer, seeded random init, bf16, eager attention (the only kind it
supports). V4-Flash CSA settings: compression 4, window 128 raw keys, indexer heads
of 128, top-k = the budget in compressed entries (512 in V4-Flash). Shapes match the
suite: one sequence per KV head, 2 query heads of 128. Projections are sized down
(hidden 256): their cost is not CSA's. LibraryCsaFlash is the same layer at the
library's default sizes (DeepSeek-V4-Flash attention), one sequence.

The model prefills a cache with random tokens, once per length: the weights are the
same at every budget and the cache does not depend on top-k, so every budget shares
it. A step is one new token through the attention layer, so it includes the layer's
q, KV and output projections. Each step first restores the cache as prefill left
it, so every timed step and CUDA-graph replay is the same token.
"""

from dataclasses import dataclass, replace
from functools import cache
from typing import Any, NamedTuple, cast

import torch
from transformers import DynamicCache
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Model,
)

from attention.fast_pytorch import KV

CSA = "compressed_sparse_attention"
CHUNK = 512  # most prefill tokens per model call
SCRATCH = 2**30  # bytes the indexer may expand per prefill call


class State(NamedTuple):
    """A prefilled library layer and one new token's inputs."""

    attention: DeepseekV4Attention
    cache: DynamicCache
    inputs: dict[str, Any]  # attention.forward's arguments for the token
    saved: dict[str, Any]  # the cache layer's attributes after prefill


def copied(attributes: dict[str, Any]) -> dict[str, Any]:
    """The attributes with their dicts copied: enough to save and restore a cache
    layer, as the library reassigns, never edits, its cache tensors."""
    return {k: dict(v) if isinstance(v, dict) else v for k, v in attributes.items()}


@dataclass(frozen=True)
class LibraryCsa:
    """The transformers DeepSeek-V4 CSA layer's decode step: q, kv give only shapes."""

    keep: int = 512  # index_topk: compressed entries attended
    window: int = 128  # raw recent keys attended
    compression: int = 4  # raw keys per compressed entry (entries overlap: 8 wide)
    heads: int = 2  # query heads; also indexer heads
    dim: int = 128  # head dim; also indexer head dim

    def sizes(self) -> dict[str, Any]:
        """Head and projection sizes: the suite's heads, projections sized down."""
        hidden = self.heads * self.dim
        return {
            "hidden_size": hidden,
            "num_attention_heads": self.heads,
            "head_dim": self.dim,
            "q_lora_rank": hidden,
            "o_groups": 2,
            "o_lora_rank": self.dim,
            "index_n_heads": self.heads,
            "index_head_dim": self.dim,
            "moe_intermediate_size": hidden,
        }

    def sequences(self, kv_heads: int) -> int:
        """Sequences in the batch: one per KV head of the suite's shapes."""
        return kv_heads

    def config(self) -> DeepseekV4Config:
        """One CSA layer of these sizes, one routed expert."""
        return DeepseekV4Config(
            **self.sizes(),
            num_hidden_layers=1,
            index_topk=self.keep,
            sliding_window=self.window,
            compress_rates={CSA: self.compression, "heavily_compressed_attention": 128},
            layer_types=[CSA],
            mlp_layer_types=["moe"],
            n_routed_experts=1,
            num_experts_per_tok=1,
            vocab_size=64,  # inputs are embeddings
        )

    def model(self, device: torch.device) -> DeepseekV4Model:
        """The library model; its weights are the same at every keep."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = DeepseekV4Model(self.config()).eval()
        model.to(device=device, dtype=torch.bfloat16)  # type: ignore[call-arg]  # HF stub
        return model

    def state(self, kv: KV) -> State:
        """The prefilled layer for kv's KV heads and length, built once per kv."""
        groups, n, _ = kv.k.shape
        return kv.once(
            f"csa{self.keep}",
            lambda: self.token(self.sequences(groups), n, kv.k.device),
        )

    @torch.no_grad()
    def token(self, groups: int, n: int, device: torch.device) -> State:
        """This keep's layer on the shared prefilled cache, and a new token's inputs."""
        cache, saved = filled(replace(self, keep=CHUNK), groups, n, device)
        model = self.model(device)
        gen = torch.Generator(device).manual_seed(1)
        token = torch.randn(
            groups, 1, model.config.hidden_size, generator=gen, device=device
        ).bfloat16()
        pos = torch.full((1, 1), n, device=device)
        mask = create_sliding_window_causal_mask(
            config=model.config,
            inputs_embeds=token,
            attention_mask=None,
            past_key_values=cache,
            position_ids=pos,
        )
        # without a tensor mask the layer drops the indexer's top-k mask
        assert isinstance(mask, torch.Tensor), type(mask)
        rope = {
            kind: model.rotary_emb(token, position_ids=pos, layer_type=kind)
            for kind in ("main", "compress")
        }
        inputs = {
            "hidden_states": token,
            "position_embeddings": rope,
            "position_ids": pos,
            "attention_mask": mask,
            "past_key_values": cache,
        }
        attention = cast(DeepseekV4Attention, model.layers[0].self_attn)
        return State(attention, cache, inputs, saved)

    def __call__(self, q: torch.Tensor, kv: KV) -> tuple[torch.Tensor, None]:
        """(sequences, 1, hidden) the layer's output for one new token."""
        state = self.state(kv)
        vars(state.cache.layers[0]).update(copied(state.saved))
        return state.attention(**state.inputs)[0], None


@dataclass(frozen=True)
class LibraryCsaFlash(LibraryCsa):
    """The same layer at the library's default sizes (DeepSeek-V4-Flash attention):
    64 query heads of 512, indexer 64 x 128, hidden 4096, q latent 1024, output 8
    groups x 1024. One sequence: all its query heads share one KV head."""

    heads: int = 64  # query heads; also indexer heads (of 128)
    dim: int = 512  # head dim

    def sizes(self) -> dict[str, Any]:
        """The library defaults (V4-Flash) for everything else."""
        return {
            "num_attention_heads": self.heads,
            "head_dim": self.dim,
            "index_n_heads": self.heads,
        }

    def sequences(self, kv_heads: int) -> int:
        """One sequence, whatever the suite's KV heads."""
        return 1


@cache
@torch.no_grad()
def filled(
    csa: LibraryCsa, groups: int, n: int, device: torch.device
) -> tuple[DynamicCache, dict[str, Any]]:
    """A cache the library model filled with n random tokens per sequence, and its
    layer's attributes after."""
    model = csa.model(device)
    gen = torch.Generator(device).manual_seed(0)
    # the library's indexer matmul copies the float32 compressed keys once per
    # query (broadcast batch dims), so a call's memory is chunk * n/4 * dim * 4
    entries = groups * n // csa.compression * model.config.index_head_dim * 4
    chunk = max(1, min(CHUNK, SCRATCH // entries))
    cache = DynamicCache(config=model.config)
    for start in range(0, n, chunk):
        length = min(chunk, n - start)
        model(
            inputs_embeds=torch.randn(
                groups, length, model.config.hidden_size, generator=gen, device=device
            ).bfloat16(),
            past_key_values=cache,
            use_cache=True,
        )
        if device.type == "cuda":  # each call needs a slightly bigger block:
            torch.cuda.empty_cache()  # cached smaller ones fill the GPU, then OOM
    return cache, copied(vars(cache.layers[0]))
