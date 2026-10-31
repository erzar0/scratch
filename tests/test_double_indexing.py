"""deepseek/double_indexing.py: picks against a per-query loop, keep-all = flat,
causality, the learned router at init, and read counts."""

from typing import Any

import pytest
import torch

from deepseek import csa2
from deepseek.double_indexing import (
    DoubleIndexing,
    add_router,
    coarse_scores,
    head_weights,
    pick,
    reads,
    summary_keys,
    superblocks,
)
from experiments.selector_forward import selecting
from tests.test_csa2 import MODES, TINY

M = 2  # compress rate: 4 entries per superblock of S = 2 are 8 tokens
CONFIG: dict[str, Any] = TINY | {"compress_rates": {"compressed_sparse_attention": M}}


def positive_inputs(
    attention: csa2.Csa2Attention, batch: int = 2, seq_len: int = 40
) -> csa2.SelectorInputs:
    """Inputs whose indexer scores are all positive, so top-k has no ReLU-zero ties:
    positive queries, head weights and keys, and keys zero on the RoPE channels."""
    indexer = attention.indexer
    assert indexer is not None
    with torch.no_grad():
        indexer.q_b_proj.weight.abs_()
        indexer.scorer.weights_proj.weight.abs_()
    hidden = torch.rand(batch, seq_len, CONFIG["hidden_size"])
    position_ids = torch.arange(seq_len)[None]  # batch 1, as HF passes them
    rope_dim = 2 * indexer.rotary_emb(hidden, position_ids, "compress")[0].shape[-1]
    index_k = torch.rand(batch, seq_len // M, CONFIG["index_head_dim"])
    index_k[..., -rope_dim:] = 0
    q_residual = torch.rand(batch, seq_len, CONFIG["q_lora_rank"])
    query = torch.randn(batch, attention.num_heads, seq_len, attention.head_dim)
    main_kv = torch.randn(batch, seq_len // M, attention.head_dim)
    return csa2.SelectorInputs(
        hidden, q_residual, position_ids, query, main_kv, index_k
    )


@torch.no_grad()
def naive_picks(
    attention: csa2.Csa2Attention,
    inputs: csa2.SelectorInputs,
    route: DoubleIndexing,
) -> list[list[set[int]]]:
    """Each query's picked entries, one query and one superblock at a time."""
    indexer = attention.indexer
    assert indexer is not None
    scorer, hidden = indexer.scorer, inputs.hidden_states
    q = indexer.queries(hidden, inputs.q_residual, inputs.position_ids)
    weights = scorer.weights_proj(hidden) * scorer.weights_scaling
    size, picks = route.superblock_size, []
    for b in range(q.shape[0]):
        rows = []
        for t in range(q.shape[1]):
            q_t, hidden_t = q[b : b + 1, t : t + 1], hidden[b : b + 1, t : t + 1]
            finished = (t + 1) // M
            coarse = []
            for g in range(finished // size):
                keys = inputs.index_k[b, g * size : (g + 1) * size]
                if route.coarse == "bound":  # sum_c of the best q_c k_c over keys
                    best = (q[b, t, :, None] * keys).amax(1).sum(-1)
                    score = (weights[b, t].relu() * best.relu()).sum()
                else:
                    summary = (
                        keys.mean(0, keepdim=True) if route.coarse == "mean" else keys
                    )
                    score = scorer(q_t, summary[None], hidden_t).max()
                coarse.append(float(score))
            kept = sorted(range(len(coarse)), key=lambda g: -coarse[g])
            kept = kept[: route.kept_superblocks]
            allowed = [
                j
                for j in range(finished)
                if j // size in kept or j // size == len(coarse)
            ]
            fine = scorer(q_t, inputs.index_k[b : b + 1], hidden_t).flatten()
            ranked = sorted(allowed, key=lambda j: -float(fine[j]))
            rows.append(set(ranked[: indexer.index_topk]))
        picks.append(rows)
    return picks


@pytest.mark.parametrize(
    "route",
    [
        DoubleIndexing(1, 2),
        DoubleIndexing(2, 1),
        DoubleIndexing(3, 2),
        DoubleIndexing(2, 1, "bound"),
        DoubleIndexing(3, 1, "bound"),
        DoubleIndexing(2, 2, "oracle"),
    ],
)
def test_matches_per_query_loop(route: DoubleIndexing) -> None:
    # Arrange
    torch.manual_seed(0)
    attention = csa2.build(MODES, **CONFIG).model.layers[0].self_attn
    inputs = positive_inputs(attention)

    # Act
    picks = route(attention, inputs)

    # Assert
    want = naive_picks(attention, inputs, route)
    for b, rows in enumerate(want):
        for t, entries in enumerate(rows):
            got = picks.entry[b, t][picks.valid[b, t]].tolist()
            assert set(got) == entries, (b, t)


@pytest.mark.parametrize("coarse", ["mean", "bound", "oracle"])
@pytest.mark.parametrize("superblock_size", [1, 2, 3])
def test_keeping_every_superblock_is_flat(coarse: str, superblock_size: int) -> None:
    # Arrange
    torch.manual_seed(0)
    model = csa2.build(MODES, **CONFIG).eval()
    tokens = torch.randint(0, CONFIG["vocab_size"], (2, 40))
    route = DoubleIndexing(superblock_size, 10**6, coarse)  # type: ignore[arg-type]

    # Act
    with torch.no_grad(), selecting(model, route):
        routed = model(input_ids=tokens, use_cache=False).logits

    # Assert
    with torch.no_grad():
        flat = model(input_ids=tokens, use_cache=False).logits
    torch.testing.assert_close(routed, flat)


def test_routed_model_is_causal() -> None:
    # Arrange
    torch.manual_seed(0)
    model = csa2.build(MODES, **CONFIG).eval()
    tokens = torch.randint(0, CONFIG["vocab_size"], (1, 48))
    changed = tokens.clone()
    changed[:, 30:] = (changed[:, 30:] + 1) % CONFIG["vocab_size"]

    # Act
    with torch.no_grad(), selecting(model, DoubleIndexing(2, 1, "bound")):
        logits, changed_logits = (
            model(input_ids=t, use_cache=False).logits for t in (tokens, changed)
        )

    # Assert
    torch.testing.assert_close(logits[:, :30], changed_logits[:, :30])


def test_learned_router_starts_as_mean() -> None:
    # Arrange
    torch.manual_seed(0)
    attention = csa2.build(MODES, **CONFIG).model.layers[0].self_attn
    inputs = positive_inputs(attention)
    assert attention.indexer is not None
    q = attention.indexer.queries(
        inputs.hidden_states, inputs.q_residual, inputs.position_ids
    )

    # Act
    add_router(attention, 3)
    learned = coarse_scores(attention, inputs, q, DoubleIndexing(3, 1, "learned"))

    # Assert
    mean = coarse_scores(attention, inputs, q, DoubleIndexing(3, 1, "mean"))
    torch.testing.assert_close(learned, mean)


def test_learned_without_router_raises() -> None:
    attention = csa2.build(MODES, **CONFIG).model.layers[0].self_attn
    inputs = positive_inputs(attention)
    with pytest.raises(ValueError, match="no router"):
        DoubleIndexing(2, 1, "learned")(attention, inputs)


def test_reads() -> None:
    flat = reads(8, 2, None)
    mean = reads(8, 2, DoubleIndexing(2, 1))
    bound = reads(8, 2, DoubleIndexing(2, 1, "bound"))

    # 8 queries, m = 2: finished entries 0 1 1 2 2 3 3 4 (sum 16). S = 2, s = 1:
    # complete superblocks 0 0 0 1 1 1 1 2 (sum 6; one summary key each, two for
    # bound), 2 kept entries once one is complete (10), finished % 2 in the own (4)
    assert (flat, mean, bound) == (16, 6 + 10 + 4, 12 + 10 + 4)


@pytest.mark.parametrize(
    "route",
    [
        DoubleIndexing(2, 1),
        DoubleIndexing(3, 2),
        DoubleIndexing(2, 2, "bound"),
        DoubleIndexing(3, 10**6, "oracle"),
    ],
)
def test_fast_pick_matches_mask(route: DoubleIndexing) -> None:
    # Arrange
    torch.manual_seed(0)
    attention = csa2.build(MODES, **CONFIG).model.layers[0].self_attn
    inputs = positive_inputs(attention)
    indexer = attention.indexer
    assert indexer is not None
    q = indexer.queries(inputs.hidden_states, inputs.q_residual, inputs.position_ids)
    weights = head_weights(indexer.scorer, inputs.hidden_states)
    blocks = superblocks(inputs.index_k, route.superblock_size)

    # Act
    picks = pick(
        q,
        weights.expand(2, -1, -1),
        inputs.index_k,
        summary_keys(blocks, route.coarse),
        csa2.finished_entries(inputs.position_ids, M).expand(2, -1, -1),
        indexer.index_topk,
        route,
        indexer.scorer.softmax_scale,
    )

    # Assert
    want = route(attention, inputs)
    for b in range(2):
        for t in range(40):
            got = set(picks.entry[b, t][picks.valid[b, t]].tolist())
            assert got == set(want.entry[b, t][want.valid[b, t]].tolist()), (b, t)
