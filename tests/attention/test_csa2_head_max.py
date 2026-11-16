"""csa2_head_max: with one head per KV head it picks as csa2_exact_picks does; with
two it can pick a key only one head wants."""

import torch

from attention import harness
from attention.csa2_exact_picks import Csa2ExactPicks
from attention.csa2_head_max import Csa2HeadMax
from tests.attention.test_harness import make_chunk


def test_one_head_is_exact_picks() -> None:
    chunk = make_chunk(keys=203)
    one = harness.Chunk(
        chunk.query[:, :1], chunk.key, chunk.value, chunk.logits[:, :1],
        chunk.valid, 0, chunk.scaling,
    )  # fmt: skip
    got = Csa2HeadMax(keep=8, window=16)(one).selected
    assert torch.equal(got, Csa2ExactPicks(keep=8, window=16)(one).selected)


def test_one_head_outvoted_by_sum_is_kept() -> None:
    logits = torch.zeros(1, 2, 1, 4)
    logits[0, 0, 0] = torch.tensor([5.0, 1.0, 1.0, 0.0])  # head 0 wants key 0
    logits[0, 1, 0] = torch.tensor([-5.0, 1.0, 1.0, 0.0])  # head 1 dislikes it
    chunk = harness.Chunk(
        torch.zeros(1, 2, 1, 2), torch.zeros(1, 4, 2), torch.zeros(1, 4, 2),
        logits, torch.ones(1, 4, dtype=torch.bool), 0, 1.0,
    )  # fmt: skip
    assert Csa2HeadMax(keep=1, window=1)(chunk).selected[0, 0, 0]
    assert not Csa2ExactPicks(keep=1, window=1)(chunk).selected[0, 0, 0]
