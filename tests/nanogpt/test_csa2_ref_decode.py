"""csa2_ref_decode: block_max = PyTorch; the decode picks of the candidate source
(its top 64 inside its candidate blocks) and of a later index layer (inside the
same blocks) = csa2_ref's training picks (mode 1, mode 2) for the last query of one
document; bf16 and FP8 keys. Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_ref_decode
"""

import torch

from nanogpt import csa2_decode, csa2_ref, csa2_ref_decode

N, H, D, BLOCKS, SIZE = 4096, 6, 32, 256, 8


def same_picks(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    keys = torch.randn(N, D, device="cuda").to(dtype)
    q = torch.randn(2, H, D, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(2, H, device="cuda") + 0.1  # positive: relu ties cannot decide

    score = (keys.float() @ q[0].float().T).relu() @ w[0] * D**-0.5
    got = csa2_ref_decode.block_max(q[0], w[0], keys, SIZE)
    assert torch.allclose(got, score.view(-1, SIZE).amax(1), atol=1e-4)

    cand = csa2_ref_decode.candidates(q[0], w[0], keys, BLOCKS, SIZE)
    picks = [cand[csa2_decode.top64(q[i], keys[cand], w[i])] for i in range(2)]

    begin = torch.zeros(N, dtype=torch.long, device="cuda")
    kept = begin[:0]
    for i, mode in enumerate((1, 2)):  # the query (last row) is q[i] at every row
        entry, valid, out = csa2_ref._picks(
            q[i].expand(N, -1, -1), w[i].expand(N, -1), keys, begin, 1, 64, kept,
            mode, BLOCKS if mode == 1 else 0, SIZE,
        )  # fmt: skip
        assert valid[-1].all()
        assert set(entry[-1].tolist()) == set(picks[i].tolist()), (dtype, mode)
        if mode == 1:
            blocks = set((cand[::SIZE] // SIZE).tolist())
            assert set(out[-1].tolist()) == blocks, dtype
            kept = out


if __name__ == "__main__":
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        same_picks(dtype)
    print("ref decode: block_max = PyTorch; candidate picks = csa2_ref (bf16, FP8)")
