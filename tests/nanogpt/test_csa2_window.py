"""csa2_window.shared_window: G = 1, 2, 6 shared K=V vectors per token; keys =
csa2_flex.rotate of the vector at m = 1, values = the vector, head h reads group
h // (H / G); gradients reach every weight. Window + entries through csa2_flex.attention
= SDPA with the same masks. Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_window
"""

import torch
import torch.nn.functional as F

from nanogpt import csa2_flex, csa2_window

T, C, H, D, RR = 256, 64, 6, 128, 96

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(T, C, device="cuda")
    angle = F.pad(torch.rand(T, RR // 4, device="cuda") * 6, (0, RR // 4))
    angle = angle.repeat_interleave(2, 1)
    sign = torch.tensor([1.0, -1.0], device="cuda").repeat(RR // 2)
    f1, f2 = angle.cos(), angle.sin() * sign
    for groups in (1, 2, 6):
        w = [
            torch.randn(D, C, device="cuda", requires_grad=True) for _ in range(groups)
        ]
        k, v = csa2_window.shared_window(x, tuple(w), H, f1, f2)
        assert k.shape == v.shape == (T, H, D)
        for h in range(H):
            e = F.rms_norm(x @ w[h // (H // groups)].T, (D,))
            assert torch.allclose(v[:, h], e, atol=1e-5)
            want = csa2_flex.rotate(e, 1, f1, f2)
            assert torch.allclose(k[:, h], want, atol=1e-5), (
                (k[:, h] - want).abs().max()
            )
        (k.square().sum() + v.sum()).backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in w)
    print("shared_window: = rotate(rms_norm(x w_g)) / rms_norm(x w_g), G = 1, 2, 6")
