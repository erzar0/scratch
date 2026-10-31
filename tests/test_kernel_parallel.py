"""Parallel tournament kernel (src/sparse/kernel_parallel.py); needs CUDA.

pytest skips these (tests/conftest.py hides the GPU); run them on the GPU with
    uv run python -m tests.test_kernel_parallel
"""

import pytest
import torch

from sparse import attend
from sparse.block_tournament import BlockTournament
from tests.test_sparse_kernel import inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_parallel_kernel_matches_block_halving() -> None:
    from sparse import kernel, kernel_parallel

    q, k, k_t, v = inputs()
    n, scaling = k.shape[1], 128**-0.5
    logits = (q[:, :, None] @ k[:, None].mT) * scaling
    valid = torch.ones(1, n, dtype=torch.bool, device="cuda")
    chunk = attend.Chunk(q[:, :, None], k, v, logits, valid, 0, scaling)
    # keep 128 at 4096 keys: 5 rounds in 4 blocks of 1024, 32 winners each
    want = BlockTournament(block=1024, keep=128)(chunk)

    won = kernel_parallel.select(q, k_t, 128, blocks=4)

    assert bool((won >= 0).all())
    mask = torch.zeros(8, n, dtype=torch.bool, device="cuda")
    assert torch.equal(mask.scatter_(1, won.long(), True), want.selected[:, 0])
    torch.testing.assert_close(kernel.attend(q, k, v, won), want.output[:, :, 0])


def test_parallel_kernel_at_131072_keys() -> None:
    # 8 rounds in 32 blocks of 4096: 512 distinct winners per KV head
    from sparse import kernel_parallel

    q, _, k_t, _ = (t.bfloat16() for t in inputs(131072))
    won = kernel_parallel.select(q, k_t, 512)
    assert bool((won >= 0).all())
    assert all(len(row.unique()) == 512 for row in won)


if __name__ == "__main__":
    test_parallel_kernel_matches_block_halving()
    test_parallel_kernel_at_131072_keys()
    print("parallel kernel matches")
