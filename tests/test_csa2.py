"""deepseek/csa2.py: HF DeepSeek-V4's attention math, causality, modes, indexer loss."""

import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    eager_attention_forward,
)

from deepseek import csa2, csa2_sink
from model import train

TINY = {
    "vocab_size": 97,
    "hidden_size": 64,
    "num_attention_heads": 4,
    "head_dim": 16,
    "q_lora_rank": 32,
    "o_lora_rank": 16,
    "index_n_heads": 2,
    "index_head_dim": 8,
    "index_topk": 3,
    "sliding_window": 4,
    "moe_intermediate_size": 32,
}
MODES = (csa2.FULL, csa2.REUSE, csa2.REINDEX, csa2.REUSE)


def test_sparse_attention_matches_hf_eager() -> None:
    # Arrange: random picks, some invalid, and non-zero sinks
    torch.manual_seed(0)
    attention = csa2.build(MODES, **TINY).model.layers[0].self_attn
    attention.sinks.data.normal_()
    batch, seq_len, n_entries, dim = 2, 10, 6, attention.head_dim
    query = torch.randn(batch, attention.num_heads, seq_len, dim)
    window_kv = torch.randn(batch, seq_len, dim)
    main_kv = torch.randn(batch, n_entries, dim)
    entry = torch.rand(batch, seq_len, n_entries).argsort(-1)[..., :3]
    valid = torch.rand(batch, seq_len, 3) > 0.3
    picks = csa2.Picks(entry, valid, torch.zeros(batch, seq_len, 3))

    # Act
    output, _ = csa2.sparse_attention(attention, query, window_kv, main_kv, picks)

    # Assert: HF eager over [raw tokens ; main KV] with the dense mask it implies
    positions = torch.arange(seq_len)
    behind = positions[:, None] - positions[None, :]
    in_window = (behind >= 0) & (behind < attention.sliding_window)
    picked = torch.zeros(batch, seq_len, n_entries, dtype=torch.bool)
    picked.scatter_(-1, entry, valid)
    allowed = torch.cat([in_window.expand(batch, -1, -1), picked], -1).unsqueeze(1)
    mask = torch.zeros(allowed.shape).masked_fill(~allowed, float("-inf"))
    kv = torch.cat([window_kv, main_kv], 1).unsqueeze(1)
    want, _ = eager_attention_forward(
        attention, query, kv, kv, mask, scaling=attention.scaling
    )
    torch.testing.assert_close(output, want)


def test_sink_attention_matches_hf_eager() -> None:
    # Arrange: as above; the sequence outruns window + SINKS
    torch.manual_seed(0)
    attention = csa2.build(MODES, **TINY).model.layers[0].self_attn
    attention.sinks.data.normal_()
    batch, seq_len, n_entries, dim = 2, 12, 6, attention.head_dim
    query = torch.randn(batch, attention.num_heads, seq_len, dim)
    window_kv = torch.randn(batch, seq_len, dim)
    main_kv = torch.randn(batch, n_entries, dim)
    entry = torch.rand(batch, seq_len, n_entries).argsort(-1)[..., :3]
    valid = torch.rand(batch, seq_len, 3) > 0.3
    picks = csa2.Picks(entry, valid, torch.zeros(batch, seq_len, 3))

    # Act
    output, _ = csa2_sink.sink_attention(attention, query, window_kv, main_kv, picks)

    # Assert: HF eager, raw tokens allowed in the window or among the first SINKS
    positions = torch.arange(seq_len)
    behind = positions[:, None] - positions[None, :]
    first = positions[None, :] < csa2_sink.SINKS
    raw = (behind >= 0) & ((behind < attention.sliding_window) | first)
    picked = torch.zeros(batch, seq_len, n_entries, dtype=torch.bool)
    picked.scatter_(-1, entry, valid)
    allowed = torch.cat([raw.expand(batch, -1, -1), picked], -1).unsqueeze(1)
    mask = torch.zeros(allowed.shape).masked_fill(~allowed, float("-inf"))
    kv = torch.cat([window_kv, main_kv], 1).unsqueeze(1)
    want, _ = eager_attention_forward(
        attention, query, kv, kv, mask, scaling=attention.scaling
    )
    torch.testing.assert_close(output, want)


def test_csa2_is_causal() -> None:
    # Arrange
    torch.manual_seed(0)
    model = csa2.build(MODES, **TINY).eval()
    tokens = torch.randint(0, TINY["vocab_size"], (1, 40))
    changed = tokens.clone()
    changed[:, 25:] = (changed[:, 25:] + 1) % TINY["vocab_size"]

    # Act
    with torch.no_grad():
        logits, changed_logits = (
            model(input_ids=t, use_cache=False).logits for t in (tokens, changed)
        )

    # Assert
    torch.testing.assert_close(logits[:, :25], changed_logits[:, :25])
    assert not torch.allclose(logits[:, 25:], changed_logits[:, 25:])


def test_modes_own_and_share() -> None:
    attentions = [layer.self_attn for layer in csa2.build(MODES, **TINY).model.layers]

    owners = [
        (a.compressor is not None, a.indexer is not None, a.kv_source, a.picks_source)
        for a in attentions
    ]

    assert owners == [
        (True, True, 0, 0),
        (False, False, 0, 0),
        (False, True, 0, 2),
        (False, False, 0, 2),
    ]


def test_indexer_loss_moves_only_the_indexer() -> None:
    # Arrange
    torch.manual_seed(0)
    model = csa2.build(MODES, **TINY)
    tokens = torch.randint(0, TINY["vocab_size"], (2, 24))

    # Act: a zero LM loss, so every gradient comes from the attached indexer loss
    (0 * model(input_ids=tokens, use_cache=False).logits.sum()).backward()

    # Assert
    for name, param in model.named_parameters():
        moved = param.grad is not None and bool(param.grad.abs().sum() > 0)
        assert moved == train.is_indexer(name), name
