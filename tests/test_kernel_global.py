"""Global halving kernel (src/sparse/kernel_global.py) against the emulation.

pytest skips this (tests/conftest.py hides the GPU); run it on the GPU with
    uv run python -m tests.test_kernel_global
"""

import pytest
import torch

from sparse import attend
from sparse.tournament import Tournament
from tests.test_sparse_kernel import inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_global_kernel_matches_the_emulation() -> None:
    from sparse import kernel, kernel_global

    q, k, k_t, v = inputs(6000)  # not a multiple of the block: a ragged tail
    n, scaling = k.shape[1], 128**-0.5
    logits = (q[:, :, None] @ k[:, None].mT) * scaling
    valid = torch.ones(1, n, dtype=torch.bool, device="cuda")
    chunk = attend.Chunk(q[:, :, None], k, v, logits, valid, 0, scaling)
    # keep 128 of 6000 keys: 6 rounds, 3 across programs and 3 in kernel._global_rounds
    want = Tournament("query", "halving", "drop", keep=128)(chunk)

    won = kernel_global.select(q, k_t, 128)

    assert bool((won >= 0).all())
    mask = torch.zeros(8, n, dtype=torch.bool, device="cuda")
    assert torch.equal(mask.scatter_(1, won.long(), True), want.selected[:, 0])
    torch.testing.assert_close(kernel.attend(q, k, v, won), want.output[:, :, 0])


if __name__ == "__main__":
    test_global_kernel_matches_the_emulation()
    print("global kernel matches")
