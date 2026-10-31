"""One attention layer vs sequence length: a CSA2 Full layer (deepseek/csa2.py: own
compressor and indexer, m = 4) against exact causal softmax attention (PyTorch SDPA,
FlashAttention backend), sized like the trained models (d_model 512, 8 heads of 64),
bf16, random weights.

    prefill  forward of one sequence of T tokens. CSA2's indexer scores every
             finished entry for every query: (T, T / m) scores per indexer head, so
             its memory is O(T²).
    decode   one new token from a cache of the T tokens before it, batch 1 and
             BATCH sequences, each with its own copy of the cache. Exact attention
             reads the whole KV cache; CSA2 reads top_k + window entries plus one
             indexer key per entry. csa2.py has no KV cache: decode_step here runs
             the layer's own modules on one token. Both steps timed as a CUDA
             graph replay (bench/timing.py).
    cache    bytes kept per sequence: CSA2 the window, main KV and indexer K of a
             Full layer (Reuse layers keep only the window); exact K and V, with
             one free slot the decode step writes the new token into.

    uv run python -m bench.layer  -> runs/bench/layer.csv
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

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

from bench.timing import graphed, measure
from deepseek import csa, csa2
from experiments.results import Results

logger = logging.getLogger(__name__)

OUT = Path("runs/bench/layer.csv")
D_MODEL, N_HEADS, BATCH = csa.SMALL["hidden_size"], csa.SMALL["num_attention_heads"], 8
LENGTHS = [2**i for i in range(10, 18)]  # 1k ... 128k
DEVICE, DTYPE = torch.device("cuda"), torch.bfloat16


class Cache(NamedTuple):
    """A CSA2 Full layer's decode cache: the last window - 1 raw entries (B, w-1, D),
    main KV (B, n, D) and indexer K (B, n, c)."""

    window: torch.Tensor
    main_kv: torch.Tensor
    index_k: torch.Tensor


def csa2_layer(
    config: DeepseekV4Config | None = None,
) -> tuple[csa2.Csa2Attention, DeepseekV4RotaryEmbedding]:
    """Layer 0 (a Full layer) of a CSA2 model, SMALL by default, and its RoPE."""
    config = config or csa2.small_config()
    attention = csa2.Csa2Attention(config, 0).to(DEVICE, DTYPE).eval()
    return attention, DeepseekV4RotaryEmbedding(config).to(DEVICE)


def rope(
    rotary: DeepseekV4RotaryEmbedding, x: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the CSA layers' "compress" RoPE at positions (B, S)."""
    return rotary(x, positions, layer_type="compress")


def csa2_prefill(
    attention: csa2.Csa2Attention, rotary: DeepseekV4RotaryEmbedding
) -> Callable[[torch.Tensor], torch.Tensor]:
    """The layer's forward on (1, T, d_model)."""

    def run(x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=DEVICE)[None]
        embeddings = {"compress": rope(rotary, x, positions)}
        shared = csa2.SharedState()
        return attention(x, embeddings, positions, None, csa2_shared=shared)[0]

    return run


def csa2_cache(
    attention: csa2.Csa2Attention, rotary: DeepseekV4RotaryEmbedding, x: torch.Tensor
) -> Cache:
    """The decode cache of context x (1, T, d_model)."""
    assert attention.compressor is not None
    positions = torch.arange(x.shape[1], device=DEVICE)[None]
    cos, sin = rope(rotary, x, positions)
    kv = attention.kv_norm(attention.kv_proj(x))
    window = apply_rotary_pos_emb(kv.unsqueeze(1), cos, sin)[:, 0]
    main_kv, index_k = attention.compressor(x)
    return Cache(window[:, -(attention.sliding_window - 1) :], main_kv, index_k)


def csa2_decode_step(
    attention: csa2.Csa2Attention, rotary: DeepseekV4RotaryEmbedding, seq_len: int
) -> Callable[[Cache, torch.Tensor], torch.Tensor]:
    """Output for one new token x_t (B, 1, d_model) at position seq_len.

    Raises:
        ValueError: If x_t finishes an entry (it would have to be compressed).
    """
    indexer = attention.indexer
    assert indexer is not None
    if (seq_len + 1) % indexer.compress_rate == 0:
        raise ValueError(f"token {seq_len} finishes an entry: not modeled")

    def step(cache: Cache, x_t: torch.Tensor) -> torch.Tensor:
        batch = x_t.shape[0]
        position = torch.full((1, 1), seq_len, device=DEVICE)
        cos, sin = rope(rotary, x_t, position)
        q_residual = attention.q_a_norm(attention.q_a_proj(x_t))
        q = attention.q_b_proj(q_residual).view(batch, 1, attention.num_heads, -1)
        q = q.transpose(1, 2)
        q = apply_rotary_pos_emb(attention.q_b_norm(q), cos, sin)  # (B, H, 1, D)
        kv_t = attention.kv_norm(attention.kv_proj(x_t))
        kv_t = apply_rotary_pos_emb(kv_t.unsqueeze(1), cos, sin)[:, 0]
        index_q = indexer.queries(x_t, q_residual, position)
        scores = indexer.scorer(index_q, cache.index_k, x_t)
        m, top_k = indexer.compress_rate, indexer.index_topk
        picks = csa2.top_picks(scores, position, m, top_k)
        rows = torch.arange(batch, device=DEVICE)[:, None, None]
        picked = cache.main_kv[rows, picks.entry][:, 0]  # (B, k, D)
        keys = torch.cat([picked, cache.window, kv_t], 1)
        logits = torch.einsum("bhd,bnd->bhn", q[:, :, 0], keys) * attention.scaling
        sinks = attention.sinks.view(1, -1, 1).expand(batch, -1, 1)
        weight = torch.cat([logits, sinks.to(logits.dtype)], -1)
        weight = weight.softmax(-1, dtype=torch.float32)[..., :-1].to(keys.dtype)
        out = torch.einsum("bhn,bnd->bhd", weight, keys)[:, None]  # (B, 1, H, D)
        out = apply_rotary_pos_emb(out.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = out.reshape(batch, 1, attention.config.o_groups, -1)
        return attention.o_b_proj(attention.o_a_proj(grouped).flatten(2))

    return step


class Exact(nn.Module):
    """Exact causal softmax attention: fused QKV, SDPA, output projection; grouped
    KV heads when n_kv_heads < n_heads (Qwen3: 16 query heads, 8 KV heads)."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_kv_heads: int = N_HEADS,
        head_dim: int = D_MODEL // N_HEADS,
    ) -> None:
        super().__init__()
        self.sizes = (n_heads, n_kv_heads, n_kv_heads)
        self.head_dim = head_dim
        self.qkv = nn.Linear(d_model, sum(self.sizes) * head_dim, bias=False)
        self.proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def heads(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """q (B, H, S, d), k and v (B, H_kv, S, d)."""
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, -1, self.head_dim)
        return tuple(t.transpose(1, 2) for t in qkv.split(self.sizes, dim=2))

    def attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
    ) -> torch.Tensor:
        """SDPA over q's heads; grouped KV heads shared."""
        gqa = k.shape[1] < q.shape[1]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=gqa)
        return self.proj(out.transpose(1, 2).flatten(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prefill: (B, T, d_model) -> (B, T, d_model)."""
        q, k, v = self.heads(x)
        return self.attend(q, k, v, causal=True)

    def cache(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values (B, H, T + 1, d) of context x; the last slot is free."""
        _, k, v = self.heads(F.pad(x, (0, 0, 0, 1)))
        return k, v

    def decode_step(
        self, cache: tuple[torch.Tensor, torch.Tensor], x_t: torch.Tensor
    ) -> torch.Tensor:
        """One new token, written into the cache's free slot in place (as a KV
        cache appends), attending to the whole cache."""
        q, k_t, v_t = self.heads(x_t)
        k, v = cache
        k[:, :, -1:], v[:, :, -1:] = k_t, v_t
        return self.attend(q, k, v, causal=False)


def tiled(cache: tuple[torch.Tensor, ...], batch: int) -> tuple[torch.Tensor, ...]:
    """Each sequence its own copy of a batch-1 cache, of the same tuple type."""
    copies = [t.expand(batch, *t.shape[1:]).contiguous() for t in cache]
    return Cache(*copies) if isinstance(cache, Cache) else tuple(copies)


def mib(tensors: tuple[torch.Tensor, ...]) -> float:
    """Bytes of a batch-1 cache in MiB."""
    return sum(t.numel() * t.element_size() for t in tensors) / 2**20


@torch.no_grad()
def main(
    config: DeepseekV4Config | None = None,
    exact: Exact | None = None,
    out: Path = OUT,
    lengths: list[int] = LENGTHS,
) -> None:
    """Time both layers at every length; a failed prefill (OOM) is NaN.

    Args:
        config: CSA2 config, SMALL by default.
        exact: The exact layer, the SMALL size by default.
        out: Result CSV.
        lengths: Context lengths.
    """
    torch._dynamo.config.cache_size_limit = 64
    attention, rotary = csa2_layer(config)
    exact = (exact or Exact()).to(DEVICE, DTYPE)
    d_model = attention.config.hidden_size
    results = Results(out, {})
    for seq_len in lengths:
        x = torch.randn(1, seq_len, d_model, device=DEVICE, dtype=DTYPE)
        layers: dict[str, tuple[Callable, tuple[torch.Tensor, ...], Callable]] = {
            "csa2": (
                torch.compile(csa2_prefill(attention, rotary)),
                csa2_cache(attention, rotary, x),
                csa2_decode_step(attention, rotary, seq_len),
            ),
            "exact (flash)": (exact, exact.cache(x), exact.decode_step),
        }
        for name, (forward, cache, step) in layers.items():
            prefill = measure(forward, x)
            row = {"layer": name, "T": seq_len, "prefill_ms": prefill.ms}
            row |= {"prefill_peak_gib": prefill.peak_gib, "cache_mib": mib(cache)}
            for batch in (1, BATCH):
                caches = tiled(cache, batch)
                x_t = torch.randn(batch, 1, d_model, device=DEVICE, dtype=DTYPE)
                timing = measure(graphed(torch.compile(step), caches, x_t))
                row[f"decode_us_batch{batch}"] = timing.ms * 1e3
            logger.info("%s", row)
            results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
