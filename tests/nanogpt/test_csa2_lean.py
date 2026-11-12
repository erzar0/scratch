"""csa2_lean.q_norm_rope_pad = the q output of record 91's fused QKNormRoPEPad
(experiments/csa2/b/triton_kernels.py), forward and the gradient it sends to the q half
of the projection, bf16 at training shapes (6 heads, 96 qk channels, padded to 128).
Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_lean
"""

import sys
from pathlib import Path

import torch

from nanogpt import csa2_lean
from tests.nanogpt.test_csa2_flex import tables

sys.path.insert(0, str(Path(__file__).parents[2] / "experiments/csa2/b"))
from triton_kernels import QKNormRoPEPad  # noqa: E402

T, H, QK, PAD = 1024, 6, 96, 128

if __name__ == "__main__":
    torch.manual_seed(0)
    f1, f2 = (f.bfloat16() for f in tables(T, QK))
    qk = (3 * torch.randn(T, 2 * H, QK, device="cuda")).bfloat16().requires_grad_()
    want, _ = QKNormRoPEPad(qk, f1, f2, H, PAD, False, False)
    got = csa2_lean.q_norm_rope_pad(qk[:, :H], f1, f2, PAD)
    assert got.shape == want.shape == (T, H, PAD)
    diff = (got.float() - want.float()).abs()
    assert diff.max() <= 2**-7 * want.float().abs().max(), diff.max()  # 1 bf16 ulp
    assert (diff == 0).float().mean() > 0.99, (diff == 0).float().mean()
    grad = torch.randn_like(want)
    (expect,) = torch.autograd.grad(want, qk, grad)
    (back,) = torch.autograd.grad(got, qk, grad)
    assert torch.equal(back[:, H:], torch.zeros_like(back[:, H:]))
    assert torch.allclose(
        back[:, :H].float(), expect[:, :H].float(), atol=3e-2, rtol=2e-2
    )
    print("q_norm_rope_pad: = QKNormRoPEPad's q, forward and gradient")
