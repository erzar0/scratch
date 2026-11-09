"""Bound bandit (src/attention/bandit.py)."""

import torch

from attention import harness
from attention.bandit import BoundBandit
from tests.attention.test_harness import make_chunk


def test_exact_bound_keeps_the_exact_top_keep() -> None:
    # 4 rounds of 8 channels; the score is logsumexp over the group's heads
    chunk = make_chunk(keys=64, dim=32)
    want = harness.top_mask(chunk.logits.logsumexp(1), 8, chunk.valid)

    got = BoundBandit("exact", 1.0, keep=8)(chunk)
    gauss = BoundBandit("gauss", 1.0, keep=8)(chunk)

    assert torch.equal(got.selected, want)
    torch.testing.assert_close(got.output, harness.attend(chunk, want))
    assert bool((gauss.selected.sum(-1) == 8).all())
