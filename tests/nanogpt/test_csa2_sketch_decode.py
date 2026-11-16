"""csa2_sketch_decode: packed FP4 rows unpack to csa2_ref.fake_quant's rows (CPU);
fp4_scores = their fp32 scores (CUDA)."""

import pytest
import torch

from nanogpt.csa2_ref import fake_quant
from nanogpt.csa2_sketch_decode import fp4_scores, pack_fp4, unpack_fp4


def test_fp4_round_trip_is_fake_quant() -> None:
    torch.manual_seed(0)
    x = torch.randn(1000, 512) * torch.rand(1000, 1) * 3
    x[0] = 0  # all-zero row: smallest scale
    codes, scales = pack_fp4(x)
    assert codes.shape == (1000, 256) and scales.shape == (1000, 32)
    torch.testing.assert_close(unpack_fp4(codes, scales), fake_quant(x, "main"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp4_scores() -> None:
    torch.manual_seed(0)
    x = torch.randn(1003, 512, device="cuda").bfloat16()
    q = torch.randn(512, device="cuda").bfloat16()
    codes, scales = pack_fp4(x)
    want = unpack_fp4(codes, scales) @ q.float()
    torch.testing.assert_close(fp4_scores(codes, scales, q), want, rtol=1e-4, atol=1e-3)
