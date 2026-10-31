"""Sampled halving kernel (src/sparse/kernel_sampled.py) against the emulation.

pytest skips this (tests/conftest.py hides the GPU); run it on the GPU with
    uv run python -m tests.test_kernel_sampled
"""

import pytest
import torch

from sparse import attend
from sparse.first_round import FirstRound
from sparse.sampled_halving import SampledHalving
from tests.test_sparse_kernel import inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def chunk_of(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> attend.Chunk:
    """One decode query per KV head, every key valid."""
    scaling = k.shape[-1] ** -0.5
    logits = (q[:, :, None] @ k[:, None].mT) * scaling
    valid = torch.ones(1, k.shape[1], dtype=torch.bool, device="cuda")
    return attend.Chunk(q[:, :, None], k, v, logits, valid, 0, scaling)


def overlap(won: torch.Tensor, selected: torch.Tensor) -> float:
    """Share of the emulation's winners (G, 1, n) the kernel also picked."""
    want = selected[:, 0]
    got = torch.zeros_like(want).scatter_(1, won.long(), True)
    return float((got & want).sum() / want.sum())


def unique(won: torch.Tensor) -> bool:
    """Every slot holds a key, none twice."""
    return bool((won >= 0).all() and won.sort(-1).values.diff(dim=-1).ne(0).all())


def test_sampled_kernel_matches_the_emulation() -> None:
    from sparse import kernel, kernel_sampled

    q, k, k_t, v = inputs(6000)  # a ragged tail of the last block
    chunk = chunk_of(q, k, v)
    for first in (8, 16):
        # keep 128 of 6000: 6 rounds, 4 sampled; a sample of 8192 holds every
        # key, so the cutoffs are exact and only the fp16 partials differ
        want = SampledHalving(first=first, keep=128, sample=8192)(chunk)
        won = kernel_sampled.select(q, k_t, 128, first, sample=8192)
        assert unique(won) and overlap(won, want.selected) >= 0.99
        got = kernel.attend(q, k, v, won)
        torch.testing.assert_close(got, want.output[:, :, 0], atol=0.05, rtol=0)

        want = SampledHalving(first=first, keep=128, sample=8192, losers="approx")
        got = kernel_sampled.tournament_approx(q, k, k_t, v, 128, first, sample=8192)
        torch.testing.assert_close(got, want(chunk).output[:, :, 0], atol=0.05, rtol=0)


def test_sampled_kernel_at_128k_keys() -> None:
    from sparse import kernel_sampled

    q, k, k_t, v = inputs(131072)
    chunk = chunk_of(q, k, v)
    won = kernel_sampled.select(q, k_t, 512)  # 8 rounds, 6 sampled

    assert unique(won)
    sampled = overlap(won, SampledHalving(first=8, keep=512)(chunk).selected)
    exact = overlap(won, FirstRound(first=8, keep=512)(chunk).selected)
    print(f"128K winners shared with the emulation {sampled:.3f}, exact {exact:.3f}")
    assert sampled >= 0.98 and exact >= 0.95


if __name__ == "__main__":
    test_sampled_kernel_matches_the_emulation()
    test_sampled_kernel_at_128k_keys()
    print("sampled kernel matches")
