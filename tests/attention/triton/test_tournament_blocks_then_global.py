"""Blocks-then-global tournament kernel
(src/attention/triton/tournament_blocks_then_global.py) against the emulation.

pytest skips this (tests/conftest.py hides the GPU); run it on the GPU with
    uv run python -m tests.attention.triton.test_tournament_blocks_then_global
"""

import pytest
import torch

from attention import harness
from attention.tournament.blocks_then_global import HybridTournament
from tests.attention.triton import inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_tournament_kernel_matches_the_emulation() -> None:
    from attention.triton import common, tournament_blocks_then_global

    q, k, k_t, v = inputs()
    n, scaling = k.shape[1], 128**-0.5
    logits = (q[:, :, None] @ k[:, None].mT) * scaling
    valid = torch.ones(1, n, dtype=torch.bool, device="cuda")
    chunk = harness.Chunk(q[:, :, None], k, v, logits, valid, 0, scaling)
    # keep 128 at 4096 keys: 5 rounds, the kernel runs 2 in blocks and 3 global
    want = HybridTournament(block=1024, block_rounds=2, keep=128)(chunk)

    won = tournament_blocks_then_global.select(q, k_t, 128)

    assert bool((won >= 0).all())
    mask = torch.zeros(8, n, dtype=torch.bool, device="cuda")
    assert torch.equal(mask.scatter_(1, won.long(), True), want.selected[:, 0])
    torch.testing.assert_close(common.attend(q, k, v, won), want.output[:, :, 0])


if __name__ == "__main__":
    test_tournament_kernel_matches_the_emulation()
    print("kernel matches")
