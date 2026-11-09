"""The harness (src/attention/harness.py): its random chunk for every attention test,
every method exact when it keeps every key, K sector counts, attach to a HF model."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from attention import harness, loki
from attention.block_topk import BlockTopK
from attention.dense import dense
from attention.exact_topk import ExactTopK
from attention.sparq import SparQ
from attention.tournament.general import Tournament


def make_chunk(
    groups: int = 2, queries: int = 3, keys: int = 12, dim: int = 16
) -> harness.Chunk:
    """Random float32 chunk: the last `queries` positions of `keys`, causal."""
    torch.manual_seed(0)
    query = torch.randn(groups, 2, queries, dim)
    key, value = torch.randn(2, groups, keys, dim)
    positions = torch.arange(keys)
    valid = positions[None] <= positions[keys - queries :, None]
    scaling = dim**-0.5
    logits = ((query @ key[:, None].mT) * scaling).masked_fill(~valid, float("-inf"))
    return harness.Chunk(query, key, value, logits, valid, 0, scaling)


@pytest.mark.parametrize(
    "method",
    [
        ExactTopK(16),
        SparQ(16, 16),
        loki.Loki(4, 16),
        BlockTopK(block=4, keep=16, window=4),
        *(
            Tournament(order, rounds, losers, keep=16)
            for order in ("random", "hadamard", "query", "pca")
            for rounds in ("bracket", "halving")
            for losers in ("approx", "drop", "mean")
        ),
    ],
    ids=repr,
)
def test_keeping_every_key_is_exact_attention(method: harness.Method) -> None:
    chunk = make_chunk()
    torch.testing.assert_close(method(chunk).output, dense(chunk).output)


def test_k_sectors_per_layout() -> None:
    rank = torch.arange(16).expand(1, 1, 16)  # 16 dimensions, read in order
    level = torch.zeros(1, 1, 32, dtype=torch.long)
    level[..., :16] = 1  # keys 0-15 read 8 dimensions
    level[..., 20] = 2  # key 20 read in full

    sectors = harness.k_sectors(rank, torch.tensor([0, 8, 16]), level)

    assert {name: int(s) for name, s in sectors.items()} == {
        "sectors_rows": 16 + 1,
        "sectors_chunks": 8 + 2,
        "sectors_transpose": 8 + 16,
        "sectors_copies": 8 + 8 + 1,
    }


def test_attached_dense_matches_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "CHUNK", 4)
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    model = LlamaForCausalLM(config).eval()
    tokens = torch.randint(0, 64, (1, 30))
    with torch.no_grad():
        sdpa = model(input_ids=tokens).logits
        state = harness.attach(model, dense, scored=10)
        emulated = model(input_ids=tokens).logits

    torch.testing.assert_close(emulated, sdpa, atol=1e-5, rtol=1e-4)
    assert set(state.sums) == {0, 1}
    assert state.sums[0]["recall"] == state.sums[0]["count"]
