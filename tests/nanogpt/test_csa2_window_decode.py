"""csa2_window_decode.kv_attend = the attention written out: G = 1, 2, 6 shared K=V
window vectors (head h reads h // (H / G)) + picked entries, values turned back by
their pair_table rows; row- and block-major keys, q_pick, 16 and 32 rows per
program, twice (counters reset). Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_window_decode
"""

import torch

from nanogpt import csa2_attend_split, csa2_window_decode

H, D, W, N = 6, 128, 128, 4096
BF = torch.bfloat16


def rand(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=BF)


def turned(key: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Values of rotated keys (n, 128): first 64 channels turned back in pairs."""
    cos, sin = table[:, :32].float(), table[:, 32:].float()
    x0, x1 = key[:, 0:64:2].float(), key[:, 1:64:2].float()
    v0, v1 = (cos * x0 - sin * x1).to(BF), (cos * x1 + sin * x0).to(BF)
    return torch.cat([torch.stack([v0, v1], -1).flatten(1), key[:, 64:]], 1)


def want(q, qp, win, wtab, keys, picks, table, scale):
    groups, out = win.shape[1], []
    kp, vp = keys[picks].float(), turned(keys[picks], table[picks]).float()
    for h in range(H):
        kw = win[:, h // (H // groups)]
        vw = turned(kw, wtab).float()
        logits = torch.cat([kw.float() @ q[h].float(), kp @ qp[h].float()]) * scale
        out.append(logits.softmax(0) @ torch.cat([vw, vp]))
    return torch.stack(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    q, qp, keys = rand(H, D), rand(H, D), rand(N, D)
    picks = torch.randperm(N, device="cuda")[:64]
    angle = torch.rand(N, 32, device="cuda") * 6
    table = torch.cat([angle.cos(), angle.sin()], 1).to(BF)
    wtab = table[:W].contiguous()
    major = keys.view(N, -1, 16).transpose(0, 1).contiguous()
    for groups in (1, 2, 6):
        win = rand(W, groups, D)
        for q_pick in (None, qp):
            ref = want(q, q if q_pick is None else q_pick, win, wtab, keys, picks,
                       table, 0.1)  # fmt: skip
            for rows in (16, 32, 16):
                for k in (keys, major):
                    got = csa2_window_decode.kv_attend(
                        q, win, wtab, k, picks, table, 0.1, q_pick, rows
                    )
                    err = (got.float() - ref).abs().max()
                    assert err < 2e-2, (groups, rows, err)
    assert (csa2_attend_split._counts[q.device] == 0).all()
    print("kv_attend: = written out (G = 1, 2, 6; row-, block-major; q_pick)")
