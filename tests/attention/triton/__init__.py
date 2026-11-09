"""Kernel tests (src/attention/triton/); they need CUDA. pytest skips them
(tests/conftest.py hides the GPU); each file runs on the GPU with
    uv run python -m tests.attention.triton.<file>
"""

import torch


def inputs(
    n: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random float32 q (8, 2, 128), K, K transposed, V (8, n, 128)."""
    torch.manual_seed(0)
    q = torch.randn(8, 2, 128, device="cuda")
    k = torch.randn(8, n, 128, device="cuda")
    return q, k, k.transpose(1, 2).contiguous(), torch.randn_like(k)
