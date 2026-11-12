"""q-only form of record 91's fused QKNormRoPEPad (triton_kernels.py) for models whose
heads read a shared K=V window (csa2_window) instead of their own k: the k half of the
QK projection is then never read, so neither it nor its norm and RoPE are computed.
Same q as the kernel: rms_norm over the qk channels (eps 2^-23), RoPE pairs
(2i, 2i + 1) y = f1 x + f2 x_flip on every channel (the tables are 1 / 0 past the
rotary half), zero padded to `padded` channels; fp32 inside, stored in q's dtype.
Checked against the kernel, forward and gradients: tests/nanogpt/test_csa2_lean.py.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1.1920928955078125e-7  # the kernel's rms_norm eps


def q_norm_rope_pad(q: Tensor, factor1: Tensor, factor2: Tensor, padded: int) -> Tensor:
    """(T, H, padded) normed, rotated, padded queries of q (T, H, D); factor1,
    factor2 (T, D) record 91's yarn tables."""
    x = q.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    flip = x.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
    y = factor1[:, None].float() * x + factor2[:, None].float() * flip
    return F.pad(y.to(q.dtype), (0, padded - q.shape[-1]))
