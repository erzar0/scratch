"""Global halving kernel
(src/attention/triton/tournament_global_halving.py) against the emulation.

pytest skips this (tests/conftest.py hides the GPU); run it on the GPU with
    uv run python -m tests.attention.triton.test_tournament_global_halving
"""

import pytest
import torch

from attention import harness
from attention.tournament.general import Tournament
from tests.attention.triton import inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_global_kernel_matches_the_emulation() -> None:
    from attention.triton import common, tournament_global_halving

    q, k, k_t, v = inputs(6000)  # not a multiple of the block: a ragged tail
    n, scaling = k.shape[1], 128**-0.5
    logits = (q[:, :, None] @ k[:, None].mT) * scaling
    valid = torch.ones(1, n, dtype=torch.bool, device="cuda")
    chunk = harness.Chunk(q[:, :, None], k, v, logits, valid, 0, scaling)
    # keep 128 of 6000 keys: 6 rounds, 3 across programs and 3 in common._global_rounds
    want = Tournament("query", "halving", "drop", keep=128)(chunk)

    won = tournament_global_halving.select(q, k_t, 128)

    assert bool((won >= 0).all())
    mask = torch.zeros(8, n, dtype=torch.bool, device="cuda")
    assert torch.equal(mask.scatter_(1, won.long(), True), want.selected[:, 0])
    torch.testing.assert_close(common.attend(q, k, v, won), want.output[:, :, 0])


if __name__ == "__main__":
    test_global_kernel_matches_the_emulation()
    print("global kernel matches")
