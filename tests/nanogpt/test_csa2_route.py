"""Routed channels for CSA2's entries (src/nanogpt/csa2_route.py): route() keeps each
token's top blocks with gates summing to keep * sqrt(blocks / keep) and passes
gradient to the router; attention() equals a dense masked softmax with window
logits q . k and entry logits q_routed . keys, forward and backward, eager and
compiled (fullgraph). Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_route
"""

import math

import torch
from torch import Tensor

from nanogpt import csa2_flex, csa2_route, csa2_select

M = 4
DOCS = (100, 700, 224)  # 1024 tokens: 8 row tiles, 2 entry tiles


def route_keeps_top_blocks() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 8, generator=g).cuda()
    q = torch.randn(5, 2, 128, generator=g).cuda()
    w = torch.zeros(6, 8, device="cuda")  # equal gates: ties keep blocks 0, 1
    got = csa2_route.route(x, w, q, 96)
    assert torch.equal(got[..., 32:], torch.zeros_like(got[..., 32:]))
    assert torch.allclose(got[..., :32], q[..., :32] * math.sqrt(3))
    w = torch.randn(6, 8, generator=g).cuda().requires_grad_()
    got = csa2_route.route(x, w, q, 96)
    kept = (got[0, 0, :96] != 0).view(6, 16).all(1)
    assert kept.sum() == 2 and (got[0, 0, :96] == 0).view(6, 16)[~kept].all()
    (grad,) = torch.autograd.grad(got.square().sum(), w)
    assert grad.abs().sum() > 0


def dense(
    q: Tensor,
    q_routed: Tensor,
    k: Tensor,
    v: Tensor,
    keys: Tensor,
    values: Tensor,
    entry: Tensor,
    valid: Tensor,
    begin: Tensor,
    scale: float,
) -> Tensor:
    """The masked softmax over [k ; entries] in plain torch, entries on q_routed."""
    length, heads, _ = q.shape
    t = torch.arange(length, device=q.device)
    near = (t[None] <= t[:, None]) & (t[:, None] - t[None] < csa2_flex.WINDOW)
    near &= t[None] >= begin[:, None]
    picked = torch.zeros(length, keys.shape[0], dtype=torch.bool, device=q.device)
    picked.scatter_(1, entry, valid)
    logits = torch.cat(
        [
            torch.einsum("thd,shd->hts", q, k),
            torch.einsum("thd,sd->hts", q_routed, keys),
        ],
        -1,
    )
    allow = torch.cat([near, picked], 1)
    weights = (logits * scale).masked_fill(~allow, float("-inf")).softmax(-1)
    v_all = torch.cat([v, values[:, None].expand(-1, heads, -1)])
    return torch.einsum("hts,shd->thd", weights, v_all)


def attention_is_dense(compiled: bool) -> None:
    g = torch.Generator().manual_seed(2)
    length, heads, dim = sum(DOCS), 2, 128
    x = torch.randn(length, 16, generator=g).cuda()
    w = torch.randn(6, 16, generator=g).cuda()
    q, k, v = (torch.randn(length, heads, dim, generator=g).cuda() for _ in range(3))
    q_routed = csa2_route.route(x, w, q, 96)
    keys, values = (torch.randn(length // M, dim, generator=g).cuda() for _ in range(2))
    cu = torch.tensor((0, *torch.tensor(DOCS).cumsum(0).tolist(), length)).cuda()
    begin = csa2_flex.starts(cu, length)
    entry, valid = csa2_select.exact(q_routed, keys, begin, M, 8)
    inputs = q, q_routed, k, v, keys, values
    for t in inputs:
        t.requires_grad_()
    want = dense(*inputs, entry, valid, begin, 0.2)
    run = csa2_route.attention
    if compiled:  # as in training: fullgraph, mask tensors from the custom op
        run = torch.compile(run, fullgraph=True, dynamic=False)
    got = run(*inputs, entry, valid, cu, 0.2, M)
    assert torch.allclose(got, want, atol=2e-5), (got - want).abs().max()
    grad = torch.randn_like(want)
    expect = torch.autograd.grad(want, inputs, grad)
    for a, b in zip(torch.autograd.grad(got, inputs, grad), expect, strict=True):
        assert torch.allclose(a, b, atol=2e-4), (a - b).abs().max()


if __name__ == "__main__":
    route_keeps_top_blocks()
    print("route: top blocks kept, gates scaled, router gets gradient")
    for compiled in (False, True):
        attention_is_dense(compiled)
    print("attention: dense masked softmax, forward and backward, eager and compiled")
