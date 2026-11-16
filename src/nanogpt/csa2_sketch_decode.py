"""Decode-time scores of cheap per-row summaries (the first step of block picks at
V4.1-Flash-1017 size), on CUDA cores: one program per R rows, every row read once.

fp4: rows stored as V4.1's FP4 main-cache row (csa2_ref.fake_quant "main"): E2M1
     values, channel 2i in the low nibble of byte i, one E4M3 scale per 16 channels:
     D / 2 + D / 16 bytes per row (288 at 512 channels; FP8: 512). Decoded in-kernel
     arithmetically, scored against q in fp32.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from nanogpt.csa2_ref import e2m1

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # E2M1 magnitudes by code


def pack_fp4(x: Tensor) -> tuple[Tensor, Tensor]:
    """x (N, D) -> codes (N, D / 2) uint8, scales (N, D / 16) float8_e4m3fn;
    unpack_fp4 of them = csa2_ref.fake_quant(x, "main")."""
    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(-1, keepdim=True)
    scale = (amax.clamp(min=6 * 2**-9) / 6).to(torch.float8_e4m3fn)
    y = e2m1(blocks / scale.float()).flatten(-2)
    code = torch.searchsorted(y.new_tensor(LEVELS), y.abs())
    code = (code | torch.signbit(y).long() << 3).to(torch.uint8)
    return code[:, 0::2] | code[:, 1::2] << 4, scale[..., 0]


def unpack_fp4(codes: Tensor, scales: Tensor) -> Tensor:
    """(N, D) fp32 values of pack_fp4's rows."""
    levels = torch.tensor(LEVELS, device=codes.device)
    code = torch.stack([codes & 15, codes >> 4], -1).flatten(-2).long()
    value = levels[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    return (value.unflatten(-1, (-1, 16)) * scales.float()[..., None]).flatten(-2)


@triton.jit
def _e2m1(code):
    """fp32 value of 4-bit E2M1 codes."""
    mag = code & 7
    exp = mag >> 1
    mant = (mag & 1).to(tl.float32)
    size = tl.where(exp == 0, mant * 0.5, (1 + mant * 0.5) * tl.exp2(exp - 1.0))
    return tl.where((code & 8) != 0, -size, size)


@triton.jit
def _fp4_gemv(code_ptr, scale_ptr, q_ptr, out_ptr, n, D: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    ok = rows < n
    b = tl.arange(0, D // 2)
    code = tl.load(code_ptr + rows[:, None] * (D // 2) + b[None, :], ok[:, None], 0)
    scale = tl.load(scale_ptr + rows[:, None] * (D // 16) + (b // 8)[None, :],
                    ok[:, None], 0).to(tl.float32)  # fmt: skip
    q_even = tl.load(q_ptr + 2 * b).to(tl.float32)
    q_odd = tl.load(q_ptr + 2 * b + 1).to(tl.float32)
    lo, hi = _e2m1(code & 15), _e2m1(code >> 4)
    score = tl.sum(scale * (lo * q_even[None, :] + hi * q_odd[None, :]), 1)
    tl.store(out_ptr + rows, score, ok)


def fp4_scores(codes: Tensor, scales: Tensor, q: Tensor, rows: int = 32) -> Tensor:
    """(N,) fp32 scores of pack_fp4's rows against q (D,)."""
    n, dim = codes.shape[0], codes.shape[1] * 2
    out = torch.empty(n, dtype=torch.float32, device=codes.device)
    _fp4_gemv[(triton.cdiv(n, rows),)](codes, scales, q, out, n, D=dim, R=rows,
                                       num_warps=4)  # fmt: skip
    return out
