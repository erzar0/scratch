"""Fused CSA2-layout decode (src/nanogpt/csa2_decode.py) = PyTorch: top64 = torch.topk
of the exact and indexer scores (one block and many); prefilter64 = the decode rule
csa2_select_decode.prefilter; top 256 and in-place block entries (within) = torch.topk;
csa2_attend and b_attend (6
heads and MQA) = the attention written out; routed channels (csa2_route): top64 on
q's top 2 blocks and b_attend with the picks' own query. Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_decode
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from nanogpt import csa2_decode, csa2_select_decode

H, D, W, ROPE, RR = 6, 128, 128, 16, 96
BF = torch.bfloat16


def rand(*shape: int, dtype: torch.dtype = BF) -> Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


def major(keys: Tensor) -> Tensor:
    """Entries block-major: (D // 16, n, 16)."""
    return keys.view(len(keys), -1, 16).transpose(0, 1).contiguous()


def cleared() -> None:
    """Every kernel leaves the shared merge counters and histogram zero."""
    for buffers in csa2_decode._buffers.values():
        assert not any(b.any() for b in buffers), buffers


def picks_match() -> None:
    for n in (200, 2048, 131072):
        keys, q = rand(n, D), rand(H, D)
        want = (keys.float() @ q.float().sum(0)).topk(64).indices
        got = csa2_decode.top64(q.float().sum(0), keys)
        assert set(got.tolist()) == set(want.tolist()), n
        index_k, qi, w = rand(n, 32), rand(H, 32), rand(H, dtype=torch.float32)
        score = (index_k.float() @ qi.float().T).relu() * 32**-0.5 @ w
        got = csa2_decode.top64(qi, index_k, w)
        assert set(got.tolist()) == set(score.topk(64).indices.tolist()), n
    cleared()
    print("top64: = torch.topk (exact, indexer; 1 to 8 merge groups)")


def block_picks_match() -> None:
    """top64 top=256 = torch.topk(256); within = torch.topk over the gathered entries
    of the given blocks of 8 (real entry ids)."""
    off = torch.arange(8, device="cuda")
    for n in (2048, 131072, 1048576):
        keys, q = rand(n, D), rand(D, dtype=torch.float32)
        means = keys.view(-1, 8, D).float().mean(1).to(BF)
        score = means.float() @ q
        if len(means) >= 256:
            got = csa2_decode.top64(q, means, top=256)
            assert set(got.tolist()) == set(score.topk(256).indices.tolist()), n
        for pool in (64, 256):
            blocks = score.topk(min(pool, len(means))).indices
            cand = (blocks[:, None] * 8 + off).flatten()
            want = cand[(keys[cand].float() @ q).topk(64).indices]
            got = csa2_decode.top64(q, keys, within=blocks)
            assert set(got.tolist()) == set(want.tolist()), (n, pool)
    cleared()
    print("top64 top=256 and within blocks of 8: = torch.topk")


def routed_picks_match() -> None:
    """top64 blocks=2: = torch.topk of q . k when q is zero outside 2 of its 16-channel
    blocks (exact, as csa2_route), and of the top-2-block score for a full q."""
    for n in (2048, 131072):
        keys = rand(n, D)
        q = rand(D, dtype=torch.float32)
        q[96:] = 0
        top = q.abs().view(-1, 16).sum(1).topk(2).indices
        c = (top[:, None] * 16 + torch.arange(16, device="cuda")).flatten()
        want = (keys[:, c].float() @ q[c]).topk(64).indices
        got = csa2_decode.top64(q, keys, blocks=2)
        assert set(got.tolist()) == set(want.tolist()), n
        routed = torch.zeros_like(q)
        routed[c] = q[c]
        want = (keys.float() @ routed).topk(64).indices
        got = csa2_decode.top64(routed, keys, blocks=2)
        assert set(got.tolist()) == set(want.tolist()), n
        got = csa2_decode.top64(routed, major(keys), blocks=2)
        assert set(got.tolist()) == set(want.tolist()), n
    cleared()
    print("top64 blocks=2: = torch.topk on the 2 blocks (routed, full q, block-major)")


def prefilter_matches() -> None:
    """prefilter64 = csa2_select_decode.prefilter's last query (one document); q
    zero beyond 96 channels as record 91 pads it, scored on 96."""
    for n, keep in ((8192, 512), (131072, 1024), (131072, 8192), (1024, 1024)):
        q = rand(n, H, D)
        q[..., 96:] = 0
        keys = rand(n // 4, D)
        starts = torch.zeros(n, dtype=torch.long, device="cuda")
        entry, _ = csa2_select_decode.prefilter(q, keys, starts, 4, 64, keep)
        got = csa2_decode.prefilter64(q[-1].float().sum(0), keys, keep, width=96)
        assert set(got.tolist()) == set(entry[-1].tolist()), (n, keep)
    cleared()
    print("prefilter64: = csa2_select_decode.prefilter")


def csa2_matches() -> None:
    n = 4096
    q, window, main = rand(H, D), rand(W, D), rand(n, D)
    picks = torch.randperm(n, device="cuda")[:64]
    sinks, cos, sin = rand(H, dtype=torch.float32), rand(ROPE), rand(ROPE)
    keys = torch.cat([main[picks], window])
    logits = (q @ keys.T).float() * D**-0.5
    probs = torch.cat([logits, sinks[:, None]], 1).softmax(-1)[:, :-1]
    out = probs.to(BF) @ keys
    x = out[:, -ROPE:]
    half = torch.cat([-x[:, ROPE // 2 :], x[:, : ROPE // 2]], 1)
    want = torch.cat([out[:, :-ROPE], x * cos - half * sin], 1)
    got = csa2_decode.csa2_attend(q, window, main, picks, sinks, cos, sin)
    assert torch.allclose(got.float(), want.float(), atol=2e-2), (
        (got - want).abs().max()
    )
    print("csa2_attend: = PyTorch")


def b_matches() -> None:
    n = 4096
    q, keys = rand(H, D), rand(n, D)
    picks = torch.randperm(n, device="cuda")[:64]
    # record 91's tables: RR wide, the first RR / 2 channels turn (pairs share angles)
    angle = rand(n, RR // 4, dtype=torch.float32)
    angle = F.pad(angle, (0, RR // 4)).repeat_interleave(2, 1)
    f1 = angle.cos().to(BF)
    f2 = (angle.sin() * torch.tensor([1.0, -1.0], device="cuda").repeat(RR // 2)).to(BF)
    kp = keys[picks]
    y = kp[:, :RR].float()
    flip = y.view(64, -1, 2).flip(-1).view(64, RR)
    vp = torch.cat([(f1[picks] * y - f2[picks] * flip).to(BF), kp[:, RR:]], 1)
    table = csa2_decode.pair_table(f1, f2)
    for k_win, v_win in (
        (rand(W, H, RR), rand(W, H, D)),
        (rand(W, 1, RR).expand(W, H, RR), rand(W, 1, D).expand(W, H, D)),
        (rand(32, H, RR), rand(32, H, D)),
    ):
        k = F.pad(k_win, (0, D - RR))
        k = torch.cat([k, kp[:, None].expand(-1, H, -1)]).transpose(0, 1)
        v = torch.cat([v_win, vp[:, None].expand(-1, H, -1)]).transpose(0, 1)
        want = F.scaled_dot_product_attention(q[:, None], k, v, scale=0.1)[:, 0]
        got = csa2_decode.b_attend(q, k_win, v_win, keys, picks, table, 0.1)
        assert torch.allclose(got.float(), want.float(), atol=2e-2), (
            (got - want).abs().max()
        )
    print("b_attend: = SDPA (6 heads, MQA, window 32; keys 96 wide, bf16 tables)")
    qp = rand(H, D)
    k_win, v_win = rand(W, H, RR), rand(W, H, D)
    window = (q.float() * 0.1)[:, None, :RR] @ k_win.float().permute(1, 2, 0)
    entries = (qp.float() * 0.1)[:, None] @ kp.float().T
    probs = torch.cat([window, entries], -1).softmax(-1)
    v = torch.cat([v_win, vp[:, None].expand(-1, H, -1)]).transpose(0, 1).float()
    want = (probs @ v)[:, 0]
    got = csa2_decode.b_attend(q, k_win, v_win, keys, picks, table, 0.1, q_pick=qp)
    assert torch.allclose(got.float(), want, atol=2e-2), (got - want).abs().max()
    got = csa2_decode.b_attend(q, k_win, v_win, major(keys), picks, table, 0.1, qp)
    assert torch.allclose(got.float(), want, atol=2e-2), (got - want).abs().max()
    print("b_attend q_pick: picks' logits use their own query (row- and block-major)")


if __name__ == "__main__":
    torch.manual_seed(0)
    picks_match()
    block_picks_match()
    routed_picks_match()
    prefilter_matches()
    csa2_matches()
    b_matches()
