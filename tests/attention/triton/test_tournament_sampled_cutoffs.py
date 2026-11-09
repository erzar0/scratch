"""Sampled halving kernel
(src/attention/triton/tournament_sampled_cutoffs.py) against the emulation.

pytest skips this (tests/conftest.py hides the GPU); run it on the GPU with
    uv run python -m tests.attention.triton.test_tournament_sampled_cutoffs
"""

import pytest
import torch

from attention import harness
from attention.tournament.sampled_cutoffs import SampledHalving
from attention.tournament.wide_first_round import FirstRound
from tests.attention.triton import inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def chunk_of(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> harness.Chunk:
    """One decode query per KV head, every key valid."""
    scaling = k.shape[-1] ** -0.5
    logits = (q[:, :, None] @ k[:, None].mT) * scaling
    valid = torch.ones(1, k.shape[1], dtype=torch.bool, device="cuda")
    return harness.Chunk(q[:, :, None], k, v, logits, valid, 0, scaling)


def overlap(won: torch.Tensor, selected: torch.Tensor) -> float:
    """Share of the emulation's winners (G, 1, n) the kernel also picked."""
    want = selected[:, 0]
    got = torch.zeros_like(want).scatter_(1, won.long(), True)
    return float((got & want).sum() / want.sum())


def unique(won: torch.Tensor) -> bool:
    """Every slot holds a key, none twice."""
    return bool((won >= 0).all() and won.sort(-1).values.diff(dim=-1).ne(0).all())


def test_sampled_kernel_matches_the_emulation() -> None:
    from attention.triton import common, tournament_sampled_cutoffs

    q, k, k_t, v = inputs(6000)  # a ragged tail of the last block
    chunk = chunk_of(q, k, v)
    for first in (8, 16):
        # keep 128 of 6000: 6 rounds, 4 sampled; a sample of 8192 holds every
        # key, so the cutoffs are exact and only the fp16 partials differ
        want = SampledHalving(first=first, keep=128, sample=8192)(chunk)
        won = tournament_sampled_cutoffs.select(q, k_t, 128, first, sample=8192)
        assert unique(won) and overlap(won, want.selected) >= 0.99
        got = common.attend(q, k, v, won)
        torch.testing.assert_close(got, want.output[:, :, 0], atol=0.05, rtol=0)

        approx = SampledHalving(first=first, keep=128, sample=8192, losers="approx")
        got = tournament_sampled_cutoffs.tournament_approx(
            q, k, k_t, v, 128, first, sample=8192
        )
        want_approx = approx(chunk).output[:, :, 0]
        torch.testing.assert_close(got, want_approx, atol=0.05, rtol=0)


def test_sampled_kernel_at_128k_keys() -> None:
    from attention.triton import tournament_sampled_cutoffs

    q, k, k_t, v = inputs(131072)
    chunk = chunk_of(q, k, v)
    won = tournament_sampled_cutoffs.select(q, k_t, 512)  # 8 rounds, 6 sampled

    assert unique(won)
    sampled = overlap(won, SampledHalving(first=8, keep=512)(chunk).selected)
    exact = overlap(won, FirstRound(first=8, keep=512)(chunk).selected)
    print(f"128K winners shared with the emulation {sampled:.3f}, exact {exact:.3f}")
    assert sampled >= 0.98 and exact >= 0.95


if __name__ == "__main__":
    test_sampled_kernel_matches_the_emulation()
    test_sampled_kernel_at_128k_keys()
    print("sampled kernel matches")
