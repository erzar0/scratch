"""SparQ kernel (src/attention/triton/sparq.py) against the emulation.

pytest skips this (tests/conftest.py hides the GPU); run it on the GPU with
    uv run python -m tests.attention.triton.test_sparq
"""

import pytest
import torch

from attention import harness
from attention.sparq import SparQ
from tests.attention.triton import inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_sparq_kernel_matches_the_emulation() -> None:
    from attention.triton import sparq

    q, k, k_t, v = inputs()
    n, scaling = k.shape[1], 128**-0.5
    logits = (q[:, :, None] @ k[:, None].mT) * scaling
    valid = torch.ones(1, n, dtype=torch.bool, device="cuda")
    chunk = harness.Chunk(q[:, :, None], k, v, logits, valid, 0, scaling)
    want = SparQ(r=32, k=128, local=16)(chunk)

    won = sparq.select(q, k_t, 32, 128)

    mask = torch.zeros(8, n, dtype=torch.bool, device="cuda")
    assert torch.equal(mask.scatter_(1, won.long(), True), want.selected[:, 0])
    got = sparq.sparq(q, k, k_t, v, 32, 128)
    torch.testing.assert_close(got, want.output[:, :, 0])


if __name__ == "__main__":
    test_sparq_kernel_matches_the_emulation()
    print("kernel matches")
