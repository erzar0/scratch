"""csa2_pf = csa2_ours with no parameters beyond q, wkv + kv_norm and o:
- only the host projections remain (no compressor, indexer or sink);
- picks avoid the window: every pick ends at least WINDOW tokens before its query;
- fast picks = csa2_select's (PyTorch) under the window shift;
- every layer's output = csa2_ours's when its compressor is tied to wkv / kv_norm
  with a zero gate (mean pooling), its sink is the class's SINK (-inf) and its picks
  are ours (tests/nanogpt/test_csa2_pf_sink.py: the same with a constant sink);
- a compiled bf16 training step reaches every parameter;
- drop_in (any host's q, k, v) with m = 1 and every entry picked = exact causal
  attention within the host's span, per document, GQA, also with sink tokens kept
  apart; m = 4 keeps the window exact.
Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_pf
"""

import torch
from torch import Tensor

from nanogpt import csa2_flex, csa2_ours, csa2_pf, csa2_select, csa2_select_fast
from tests.nanogpt.test_csa2_ref import CFG

LENGTH = 1024
HOST = {"wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "wo_a", "wo_b"}


def run(stack: torch.nn.ModuleList, xs: Tensor, cu: Tensor) -> tuple[list, list]:
    shared: dict = {}
    ys, picks = [], []
    for layer, x in zip(stack, xs, strict=True):
        ys.append(layer(x, cu, shared))
        picks.append(shared.get("picks") if getattr(layer, "picks", False) else None)
    return ys, picks


@torch.no_grad()
def test_same_as_ours(
    cls: type[csa2_pf.Csa2PfAttention] = csa2_pf.Csa2PfAttention,
) -> None:
    """cls's layers (csa2_pf or a subclass) = csa2_ours's, tied, sink = cls.SINK."""
    torch.manual_seed(0)
    pf_layers = [cls(i, CFG) for i in range(10)]
    ours_layers = [csa2_ours.Csa2OursAttention(i, CFG) for i in range(10)]
    pf, ours = torch.nn.ModuleList(pf_layers), torch.nn.ModuleList(ours_layers)
    assert {n.split(".")[1] for n, _ in pf.named_parameters()} == HOST
    ours.load_state_dict(pf.state_dict(), strict=False)
    for a, b in zip(pf_layers, ours_layers, strict=True):
        b.attn_sink.fill_(cls.SINK)
        if b.compressor is not None:  # mean pooling of the layer's own kv
            b.compressor.wkv.weight.copy_(a.wkv.weight)
            b.compressor.norm.weight.copy_(a.kv_norm.weight)
            if b.ratio > 1:
                b.compressor.wgate.weight.zero_()
    pf, ours = pf.cuda(), ours.cuda()
    xs = torch.randn(10, 1, LENGTH, CFG.dim, device="cuda")
    cu = torch.tensor([0, 300, LENGTH], device="cuda")
    ys, picks = run(pf, xs, cu)

    begin, t = csa2_flex.starts(cu, LENGTH), torch.arange(LENGTH, device="cuda")
    for layer, got in zip(pf_layers, picks, strict=True):
        if got is not None:
            entry, valid = got
            last = (entry + 1) * layer.ratio - 1  # each pick's newest token
            assert valid.any(), layer.layer
            assert (last <= t[:, None] - csa2_flex.WINDOW)[valid].all(), layer.layer
            assert (entry * layer.ratio >= begin[:, None])[valid].all(), layer.layer

    csa2_ours.RULE = csa2_select.exact
    try:
        ys_torch, _ = run(pf, xs, cu)
    finally:
        csa2_ours.RULE = csa2_select_fast.exact
    for i, (a, b) in enumerate(zip(ys, ys_torch, strict=True)):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (i, (a - b).abs().max())

    given = iter(p for p in picks if p is not None)
    csa2_ours.RULE = lambda *a: next(given)  # type: ignore[assignment]
    try:
        ys_ours, _ = run(ours, xs, cu)
    finally:
        csa2_ours.RULE = csa2_select_fast.exact
    for i, (a, b) in enumerate(zip(ys, ys_ours, strict=True)):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (i, (a - b).abs().max())
    print(f"{cls.__name__}: host parameters only; picks avoid the window; = csa2_ours"
          f" tied, sink {cls.SINK:.3f}")  # fmt: skip


def test_training_step() -> None:
    """Compiled bf16 forward + backward of the SMALL stack on 2 documents."""
    torch.manual_seed(5)
    stack = torch.nn.ModuleList(csa2_pf.Csa2PfAttention(i) for i in range(10))
    stack = stack.cuda()
    cu = torch.tensor([0, 1500, 3000, 4096], device="cuda")
    x = torch.randn(1, 4096, csa2_pf.csa2_ref.SMALL.dim, device="cuda")
    x.requires_grad_()
    csa2_ours.RULE = csa2_select_fast.tournament

    def step(x: Tensor) -> Tensor:
        shared: dict = {}
        total = x.new_zeros(())
        with torch.autocast("cuda", torch.bfloat16):
            for layer in stack:
                total = total + layer(x, cu, shared).float().square().mean()
        return total

    try:
        loss = torch.compile(step, dynamic=False)(x)
        loss.backward()
    finally:
        csa2_ours.RULE = csa2_select_fast.exact
    assert torch.isfinite(loss)
    for name, p in stack.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    print("training step: compiled bf16 fwd + bwd (tournament), every parameter")


def test_drop_in() -> None:
    """drop_in against masked softmax attention, on CPU."""
    torch.manual_seed(1)
    length, heads, kv_heads, dim = 700, 4, 2, 16
    q, k, v = torch.randn(3, length, heads, dim).unbind(0)
    k, v = k[:, :kv_heads], v[:, :kv_heads]
    cu = torch.tensor([0, 250, length])
    t = torch.arange(length)
    same_doc = csa2_flex.starts(cu, length)[:, None] <= t
    for span in (length, 300):
        mask = (t <= t[:, None]) & same_doc & (t[:, None] - t <= span)
        logits = torch.einsum("thd,shd->hts", q, k.repeat_interleave(2, 1)) * 0.25
        want = logits.masked_fill(~mask, float("-inf")).softmax(-1)
        want = (want @ v.repeat_interleave(2, 1).transpose(0, 1)).transpose(0, 1)
        got = csa2_pf.drop_in(q, k, v, cu, 0.25, span, 1, length, chunk=128)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    got = csa2_pf.drop_in(q, k, v, cu, 0.25, 300, 1, length, chunk=128, sinks=4)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)  # sinks: same keys
    got = csa2_pf.drop_in(q, k, v, cu, 0.25, length, 4, 16, chunk=128)
    for i in (5, 127, 350):  # whole document within WINDOW keys: exact
        torch.testing.assert_close(got[i], want[i], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(got[length - 1], want[length - 1])
    print("drop_in: m = 1 with every pick = exact attention in span; GQA, documents")


if __name__ == "__main__":
    test_same_as_ours()
    test_training_step()
    test_drop_in()
