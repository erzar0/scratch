"""Triton decode kernels (src/sparse/kernel.py) against torch; needs CUDA.

pytest skips these (tests/conftest.py hides the GPU); run them on the GPU with
    uv run python -m tests.test_sparse_kernel
"""

import pytest
import torch

from sparse import attend
from sparse.hybrid_tournament import HybridTournament

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def inputs(
    n: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random float32 q (8, 2, 128), K, K transposed, V (8, n, 128)."""
    torch.manual_seed(0)
    q = torch.randn(8, 2, 128, device="cuda")
    k = torch.randn(8, n, 128, device="cuda")
    return q, k, k.transpose(1, 2).contiguous(), torch.randn_like(k)


def test_tournament_kernel_matches_the_emulation() -> None:
    from sparse import kernel

    q, k, k_t, v = inputs()
    n, scaling = k.shape[1], 128**-0.5
    logits = (q[:, :, None] @ k[:, None].mT) * scaling
    valid = torch.ones(1, n, dtype=torch.bool, device="cuda")
    chunk = attend.Chunk(q[:, :, None], k, v, logits, valid, 0, scaling)
    # keep 128 at 4096 keys: 5 rounds, the kernel runs 2 in blocks and 3 global
    want = HybridTournament(block=1024, block_rounds=2, keep=128)(chunk)

    won = kernel.select(q, k_t, 128)

    assert bool((won >= 0).all())
    mask = torch.zeros(8, n, dtype=torch.bool, device="cuda")
    assert torch.equal(mask.scatter_(1, won.long(), True), want.selected[:, 0])
    torch.testing.assert_close(kernel.attend(q, k, v, won), want.output[:, :, 0])


def test_sparq_kernel_matches_torch() -> None:
    from sparse import kernel

    q, k, k_t, v = inputs()
    scaling, q_abs = 128**-0.5, q.abs()
    dims = q_abs.sum(1).topk(32, -1).indices
    mask = torch.zeros_like(q_abs[:, 0]).scatter_(-1, dims, 1.0)[:, None]
    frac = (q_abs * mask).sum(-1) / q_abs.sum(-1)
    approx = ((q * mask) @ k.mT) * (scaling / frac.sqrt())[..., None]
    idx = approx.logsumexp(1).topk(128, -1).indices[..., None].expand(-1, -1, 128)
    picked_k, picked_v = k.gather(1, idx), v.gather(1, idx)
    want = ((q @ picked_k.mT) * scaling).softmax(-1) @ picked_v

    torch.testing.assert_close(kernel.sparq(q, k, k_t, v, 32, 128), want)


if __name__ == "__main__":
    test_tournament_kernel_matches_the_emulation()
    test_sparq_kernel_matches_torch()
    print("kernels match")
