"""CSA2 KV layout in record 91 attention (src/nanogpt/csa2_flex.py): entry RoPE keeps
logits relative (record 91's tables) and rotates entry e to token e m; compress is
the gated pool; attention() equals a dense masked softmax over window keys and
picked entries, forward and backward, eager and compiled (fullgraph). Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_flex
"""

import torch
from torch import Tensor

from nanogpt import csa2_flex, csa2_select

M = 4
DOCS = (100, 700, 224)  # 1024 tokens: 8 row tiles, 2 entry tiles


def tables(length: int, width: int = 96) -> tuple[Tensor, Tensor]:
    """record 91's Yarn factor1, factor2 (half-truncated RoPE) for `length` rows."""
    freq = (1 / 1024) ** torch.linspace(0, 1, width // 4)
    freq = torch.cat([freq.repeat_interleave(2), torch.zeros(width // 2)])
    theta = torch.outer(torch.arange(length).float(), freq)
    f2 = theta.sin()
    f2[..., 1::2] *= -1
    return theta.cos().cuda(), f2.cuda()


def rope_is_relative() -> None:
    f1, f2 = tables(64)
    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(2, 128, generator=g).cuda()
    rq = csa2_flex.rotate(q.expand(64, -1), 1, f1, f2)
    rk = csa2_flex.rotate(k.expand(64, -1), 1, f1, f2)
    assert torch.allclose(rq[10] @ rk[3], rq[40] @ rk[33], atol=1e-4)
    assert torch.equal(rq[:, 96:], q[96:].expand(64, -1))
    entries = torch.randn(16, 128, generator=g).cuda()
    tokens = entries.repeat_interleave(M, 0)
    assert torch.equal(
        csa2_flex.rotate(entries, M, f1, f2), csa2_flex.rotate(tokens, 1, f1, f2)[::M]
    )


def compress_is_gated_pool() -> None:
    g = torch.Generator().manual_seed(1)
    x, w_kv, w_gate = (
        torch.randn(s, generator=g).cuda() for s in ((10, 8), (6, 8), (6, 8))
    )
    got = csa2_flex.compress(x, w_kv, w_gate, M)
    for e in range(2):
        rows = x[e * M : (e + 1) * M]
        pool = ((rows @ w_gate.T).softmax(0) * (rows @ w_kv.T)).sum(0)
        assert torch.allclose(got[e], pool / pool.square().mean().sqrt(), atol=1e-4)


def dense(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    keys: Tensor,
    values: Tensor,
    entry: Tensor,
    valid: Tensor,
    begin: Tensor,
    scale: float,
) -> Tensor:
    """The masked softmax over [k ; entries] in plain torch."""
    length, heads, _ = q.shape
    t = torch.arange(length, device=q.device)
    near = (t[None] <= t[:, None]) & (t[:, None] - t[None] < csa2_flex.WINDOW)
    near &= t[None] >= begin[:, None]
    picked = torch.zeros(length, keys.shape[0], dtype=torch.bool, device=q.device)
    picked.scatter_(1, entry, valid)
    allow = torch.cat([near, picked], 1)
    k_all = torch.cat([k, keys[:, None].expand(-1, heads, -1)])
    v_all = torch.cat([v, values[:, None].expand(-1, heads, -1)])
    logits = torch.einsum("thd,shd->hts", q, k_all) * scale
    weights = logits.masked_fill(~allow, float("-inf")).softmax(-1)
    return torch.einsum("hts,shd->thd", weights, v_all)


def attention_is_dense(compiled: bool) -> None:
    g = torch.Generator().manual_seed(2)
    length, heads, dim = sum(DOCS), 2, 64
    q, k, v = (torch.randn(length, heads, dim, generator=g).cuda() for _ in range(3))
    keys, values = (torch.randn(length // M, dim, generator=g).cuda() for _ in range(2))
    cu = torch.tensor((0, *torch.tensor(DOCS).cumsum(0).tolist(), length)).cuda()
    begin = csa2_flex.starts(cu, length)
    entry, valid = csa2_select.exact(q, keys, begin, M, 8)
    inputs = q, k, v, keys, values
    for x in inputs:
        x.requires_grad_()
    want = dense(q, k, v, keys, values, entry, valid, begin, 0.2)
    run = csa2_flex.attention
    if compiled:  # as in training: fullgraph, mask tensors from the custom op
        run = torch.compile(run, fullgraph=True, dynamic=False)
    got = run(q, k, v, keys, values, entry, valid, cu, 0.2, M)
    assert torch.allclose(got, want, atol=2e-5), (got - want).abs().max()
    grad = torch.randn_like(want)
    expect = torch.autograd.grad(want, inputs, grad)
    for a, b in zip(torch.autograd.grad(got, inputs, grad), expect, strict=True):
        assert torch.allclose(a, b, atol=2e-4), (a - b).abs().max()


if __name__ == "__main__":
    rope_is_relative()
    print("rotate: relative logits, entry e at token e m")
    compress_is_gated_pool()
    print("compress: gated pool, rms-normed")
    for compiled in (False, True):
        attention_is_dense(compiled)
    print("attention: dense masked softmax, forward and backward, eager and compiled")
