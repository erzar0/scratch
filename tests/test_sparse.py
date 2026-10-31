"""sparse/: decode attentions against exact attention, the tournament's rounds."""

import copy

import pytest
import torch
from transformers import Qwen3ForCausalLM

from sparse import attend, loki
from sparse.sparq import SparQ
from sparse.tournament import Tournament
from tests.test_qwen import TINY


def make_chunk(
    groups: int = 2, queries: int = 3, keys: int = 12, dim: int = 16
) -> attend.Chunk:
    """Random float32 chunk: the last `queries` positions of `keys`, causal."""
    torch.manual_seed(0)
    query = torch.randn(groups, 2, queries, dim)
    key, value = torch.randn(2, groups, keys, dim)
    positions = torch.arange(keys)
    valid = positions[None] <= positions[keys - queries :, None]
    scaling = dim**-0.5
    logits = ((query @ key[:, None].mT) * scaling).masked_fill(~valid, float("-inf"))
    return attend.Chunk(query, key, value, logits, valid, 0, scaling)


@pytest.fixture(autouse=True)
def identity_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loki's PCA basis without a calibration file: the identity."""
    monkeypatch.setattr(
        loki, "basis", lambda keys, device: torch.eye(16).expand(1, 2, 16, 16)
    )


@pytest.mark.parametrize(
    "method",
    [
        attend.ExactTopK(16),
        SparQ(16, 16),
        loki.Loki("pre", 4, 16),
        *(
            Tournament(order, rounds, losers, keep=16)
            for order in ("random", "hadamard", "query", "pca")
            for rounds in ("bracket", "halving")
            for losers in ("approx", "drop", "mean")
        ),
    ],
    ids=repr,
)
def test_keeping_every_key_is_exact_attention(method: attend.Method) -> None:
    chunk = make_chunk()
    torch.testing.assert_close(method(chunk).output, attend.dense(chunk).output)


def test_sparq_with_every_dimension_picks_the_exact_top_k() -> None:
    chunk = make_chunk()
    sparq = SparQ(r=16, k=4, local=0)(chunk)
    exact = attend.ExactTopK(4)(chunk)
    assert torch.equal(sparq.selected, exact.selected)


def test_bracket_and_halving_survivors() -> None:
    score = torch.tensor([[[3.0, 1, 0, 5, 2, 2.5, 9, 4]]])
    alive = torch.ones_like(score, dtype=torch.bool)

    bracket = Tournament(rounds="bracket").survivors(score, alive, 1, 4)
    halving = Tournament(rounds="halving").survivors(score, alive, 1, 4)

    assert bracket[0, 0].nonzero().flatten().tolist() == [0, 3, 5, 6]
    assert halving[0, 0].nonzero().flatten().tolist() == [0, 3, 6, 7]


def test_local_window_always_wins() -> None:
    chunk = make_chunk(groups=1, queries=2, keys=16)
    n = chunk.valid.sum(-1)

    selected = Tournament("random", "halving", "drop", keep=4, local=3)(chunk).selected

    for q in range(2):
        assert selected[0, q, n[q] - 3 : n[q]].all()


def test_k_sectors_per_layout() -> None:
    rank = torch.arange(16).expand(1, 1, 16)  # 16 dimensions, read in order
    level = torch.zeros(1, 1, 32, dtype=torch.long)
    level[..., :16] = 1  # keys 0-15 read 8 dimensions
    level[..., 20] = 2  # key 20 read in full

    sectors = attend.k_sectors(rank, torch.tensor([0, 8, 16]), level)

    assert {name: int(s) for name, s in sectors.items()} == {
        "sectors_rows": 16 + 1,
        "sectors_chunks": 8 + 2,
        "sectors_transpose": 8 + 16,
        "sectors_copies": 8 + 8 + 1,
    }


def test_bracket_leaves_one_key_per_block_and_counts_reads() -> None:
    chunk = make_chunk(groups=1, queries=1, keys=8)

    outcome = Tournament(keep=2)(chunk)  # 2 rounds of 8 dimensions

    assert outcome.selected.unflatten(-1, (2, 4)).sum(-1).tolist() == [[[1, 1]]]
    # K: 4 keys out after 8 dims, 2 after 16, 2 winners whole; V: every key
    assert outcome.reads.item() == 4 * 8 + 2 * 16 + 2 * 16 + 8 * 16


def test_attached_dense_matches_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attend, "CHUNK", 4)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(copy.deepcopy(TINY)).eval()  # attach sets its config
    tokens = torch.randint(0, 64, (1, 30))
    with torch.no_grad():
        sdpa = model(input_ids=tokens).logits
        state = attend.attach(model, attend.dense, scored=10)
        emulated = model(input_ids=tokens).logits

    torch.testing.assert_close(emulated, sdpa, atol=1e-5, rtol=1e-4)
    assert set(state.sums) == set(range(TINY.num_hidden_layers))
    assert state.sums[0]["recall"] == state.sums[0]["count"]
