"""csa2_flex_gqa.attention with one window head = csa2_flex.attention with that head
copied to every query head, forward and gradients (window k, v, entries, q), eager and
compiled (fullgraph, as in training). Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_flex_gqa
"""

import torch

from nanogpt import csa2_flex, csa2_flex_gqa, csa2_select
from tests.nanogpt.test_csa2_flex import DOCS, M


def gqa_is_expanded(compiled: bool) -> None:
    g = torch.Generator().manual_seed(3)
    length, heads, dim = sum(DOCS), 6, 64
    q = torch.randn(length, heads, dim, generator=g).cuda()
    k, v = (torch.randn(length, 1, dim, generator=g).cuda() for _ in range(2))
    keys, values = (torch.randn(length // M, dim, generator=g).cuda() for _ in range(2))
    cu = torch.tensor((0, *torch.tensor(DOCS).cumsum(0).tolist(), length)).cuda()
    entry, valid = csa2_select.exact(q, keys, csa2_flex.starts(cu, length), M, 8)
    inputs = q, k, v, keys, values
    for x in inputs:
        x.requires_grad_()
    want = csa2_flex.attention(
        q, k.expand(-1, heads, -1), v.expand(-1, heads, -1), keys, values,
        entry, valid, cu, 0.2, M,
    )  # fmt: skip
    run = csa2_flex_gqa.attention
    if compiled:
        run = torch.compile(run, fullgraph=True, dynamic=False)
    got = run(q, k, v, keys, values, entry, valid, cu, 0.2, M)
    assert torch.allclose(got, want, atol=2e-5), (got - want).abs().max()
    grad = torch.randn_like(want)
    expect = torch.autograd.grad(want, inputs, grad)
    for a, b in zip(torch.autograd.grad(got, inputs, grad), expect, strict=True):
        assert torch.allclose(a, b, atol=2e-4), (a - b).abs().max()


if __name__ == "__main__":
    for compiled in (False, True):
        gqa_is_expanded(compiled)
    print("gqa: = csa2_flex.attention with the window expanded, fwd + grads, compiled")
