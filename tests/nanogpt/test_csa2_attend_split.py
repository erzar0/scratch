"""Split decode attention (src/nanogpt/csa2_attend_split.py) = csa2_decode's one
program per head: CSA2's (sink, inverse RoPE) and ours (6-head window 96 wide, row-
and block-major keys, q_pick), 16, 32 and 64 rows per program, twice in a row (the
counters reset). Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_attend_split
"""

import torch

from nanogpt import csa2_attend_split as split
from nanogpt import csa2_decode as cd

H, D, W, ROPE, RR, N = 6, 128, 128, 16, 96, 4096


def rand(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


def close(got: torch.Tensor, want: torch.Tensor) -> None:
    assert torch.allclose(got.float(), want.float(), atol=2e-2), (
        (got.float() - want.float()).abs().max()
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    q, picks = rand(H, D), torch.randperm(N, device="cuda")[:64]
    window, main = rand(W, D), rand(N, D)
    sinks, cos, sin = rand(H, dtype=torch.float32), rand(ROPE), rand(ROPE)
    want = cd.csa2_attend(q, window, main, picks, sinks, cos, sin)
    for rows in (32, 64, 32):
        close(split.csa2_attend(q, window, main, picks, sinks, cos, sin, rows), want)
    print("csa2_attend split = one program per head (32, 64 rows)")
    keys, k_win, v_win, qp = rand(N, D), rand(W, H, RR), rand(W, H, D), rand(H, D)
    angle = torch.rand(N, 32, device="cuda") * 6
    table = torch.cat([angle.cos(), angle.sin()], 1).to(torch.bfloat16)
    major = keys.view(N, -1, 16).transpose(0, 1).contiguous()
    for q_pick in (None, qp):
        want = cd.b_attend(q, k_win, v_win, keys, picks, table, 0.1, q_pick)
        for rows in (16, 64, 16):
            for k in (keys, major):
                got = split.b_attend(
                    q, k_win, v_win, k, picks, table, 0.1, q_pick, rows
                )
                close(got, want)
    assert (split._counts[q.device] == 0).all()
    print("b_attend split = one program per head (row-, block-major, q_pick)")
