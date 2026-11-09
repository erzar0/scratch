"""The transformers DeepSeek-V4 CSA layer, timed as is (src/attention/deepseek_csa.py):
the library's prefill fills its cache, a step repeats exactly, and attention uses
the window plus exactly top-k compressed entries. CPU, 2 sequences of 1024 tokens.
The V4-Flash-size check needs CUDA (about 50 s on CPU):
    uv run python -m tests.attention.test_deepseek_csa
"""

from typing import cast

import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACache

from attention.deepseek_csa import LibraryCsa, LibraryCsaFlash, copied
from attention.fast_pytorch import KV

N, KEEP = 1024, 32
CSA = LibraryCsa(keep=KEEP)


def kv(device: str = "cpu") -> KV:
    """Only its shape matters: 2 KV heads (sequences), N keys."""
    k = torch.zeros(2, N, 128, device=device)
    return KV(k, k)


def test_prefill_fills_the_library_cache() -> None:
    saved = CSA.state(kv()).saved  # DeepseekV4CSACache attributes after prefill
    assert saved["entry_count"] == {"compressor": N // 4, "indexer": N // 4}
    assert saved["keys"].shape == (2, 1, CSA.window - 1, 128)


def test_every_budget_shares_one_prefill() -> None:
    cache = kv()
    assert CSA.state(cache).cache is LibraryCsa(keep=KEEP * 2).state(cache).cache


@torch.no_grad()
def test_step_repeats_and_leaves_the_cache_as_prefill_did() -> None:
    cache = kv()
    first, _ = CSA(torch.empty(0), cache)
    second, _ = CSA(torch.empty(0), cache)
    assert first.shape == (2, 1, 256) and bool(first.isfinite().all())
    assert torch.equal(first, second)
    layer = cast(DeepseekV4CSACache, CSA.state(cache).cache.layers[0])
    assert layer.entry_count == {"compressor": N // 4, "indexer": N // 4}


@torch.no_grad()
def test_attention_reads_the_window_and_top_k_entries() -> None:
    state = CSA.state(kv())
    vars(state.cache.layers[0]).update(copied(state.saved))
    _, weights = state.attention(**state.inputs)  # (2, heads, 1, window + N / 4)
    assert weights.shape[-1] == CSA.window + N // 4
    assert bool(((weights[..., : CSA.window] > 0).sum(-1) == CSA.window).all())
    assert bool(((weights[..., CSA.window :] > 0).sum(-1) == KEEP).all())


@torch.no_grad()
def flash_size_is_one_sequence_of_64_heads() -> None:
    """GPU: one sequence whatever the KV heads, 64 heads of 512, top-k applied."""
    state = LibraryCsaFlash(keep=KEEP).state(kv("cuda"))
    assert state.saved["keys"].shape == (1, 1, CSA.window - 1, 512)
    assert state.saved["entry_count"] == {"compressor": N // 4, "indexer": N // 4}
    vars(state.cache.layers[0]).update(copied(state.saved))
    _, weights = state.attention(**state.inputs)  # (1, 64, 1, window + N / 4)
    assert weights.shape == (1, 64, 1, CSA.window + N // 4)
    assert bool(((weights[..., CSA.window :] > 0).sum(-1) == KEEP).all())


if __name__ == "__main__":
    flash_size_is_one_sequence_of_64_heads()
    print("V4-Flash size: one sequence, 64 heads of 512, top-k entries")
