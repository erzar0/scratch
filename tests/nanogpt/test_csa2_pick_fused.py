"""csa2_pick_fused.picks = the block-mean rule in PyTorch (top 64 blocks by q . mean,
then the top 64 of their entries and the incomplete newest block's by q . k), from
one merge group to many, bf16 and FP8 E4M3 caches; the merge counters are left
zero. Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_pick_fused
"""

import torch

from nanogpt import csa2_decode, csa2_pick_fused

D, SIZE = 128, csa2_pick_fused.SIZE
BF, FP8 = torch.bfloat16, torch.float8_e4m3fn


def test_picks_match() -> None:
    torch.manual_seed(0)
    for n, dtype in ((100, BF), (517, BF), (4096, BF), (32773, FP8), (1 << 20, BF)):
        keys = torch.randn(n, D, device="cuda").to(dtype)
        q = torch.randn(D, device="cuda")
        nb = n // SIZE
        means = keys[: nb * SIZE].view(nb, SIZE, D).float().mean(1).to(dtype)
        blocks = (means.float() @ q).topk(min(64, nb)).indices
        cand = (blocks[:, None] * SIZE + torch.arange(SIZE, device="cuda")).flatten()
        cand = torch.cat([cand, torch.arange(nb * SIZE, n, device="cuda")])
        want = cand[(keys[cand].float() @ q).topk(64).indices]
        got = csa2_pick_fused.picks(q, keys, means)
        assert set(got.tolist()) == set(want.tolist()), n
    for buffers in csa2_decode._buffers.values():
        assert not any(b.any() for b in buffers), buffers
    print("fused block-mean picks = PyTorch (1 to 8 merge groups; bf16, FP8)")


if __name__ == "__main__":
    test_picks_match()
