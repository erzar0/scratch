"""csa2_ours = csa2_ref with our summed-query picks in place of its indexer:
- csa2_select_fast's picks = csa2_select's (PyTorch) inside the 10-layer stack;
- every layer's output = csa2_ref's when csa2_ref.picks returns our picks;
- no indexer parameters; a compiled bf16 training step reaches every parameter.
Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_ours
"""

import torch
from torch import Tensor

from nanogpt import csa2_ours, csa2_ref, csa2_select, csa2_select_fast
from tests.nanogpt.test_csa2_ref import CFG

LENGTH = 512


def stacks() -> tuple[torch.nn.ModuleList, torch.nn.ModuleList, Tensor, Tensor]:
    """Ours and csa2_ref's 10 layers with the same weights, inputs xs and cu."""
    torch.manual_seed(0)
    ours = torch.nn.ModuleList(csa2_ours.Csa2OursAttention(i, CFG) for i in range(10))
    ref = torch.nn.ModuleList(csa2_ref.Csa2RefAttention(i, CFG) for i in range(10))
    for module in (*ours, *ref):
        torch.nn.init.normal_(module.attn_sink)
    ref.load_state_dict(ours.state_dict(), strict=False)
    xs = torch.randn(10, 1, LENGTH, CFG.dim, device="cuda")
    cu = torch.tensor([0, 200, LENGTH], device="cuda")
    return ours.cuda(), ref.cuda(), xs, cu


def run(stack: torch.nn.ModuleList, xs: Tensor, cu: Tensor) -> tuple[list, list]:
    shared: dict = {}
    ys, picks = [], []
    for layer, x in zip(stack, xs, strict=True):
        ys.append(layer(x, cu, shared))
        picks.append(shared.get("picks"))
    return ys, picks


@torch.no_grad()
def test_same_as_reference() -> None:
    ours, ref, xs, cu = stacks()
    assert not any("indexer" in name for name, _ in ours.named_parameters())
    ys, picks = run(ours, xs, cu)
    csa2_ours.RULE = csa2_select.exact  # PyTorch picks: same outputs
    try:
        ys_torch, _ = run(ours, xs, cu)
    finally:
        csa2_ours.RULE = csa2_select_fast.exact
    for i, (a, b) in enumerate(zip(ys, ys_torch, strict=True)):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (i, (a - b).abs().max())
    given = iter(p for i, p in enumerate(picks) if ours[i].picks)
    forced = csa2_ref.picks
    csa2_ref.picks = lambda *a: (  # type: ignore[assignment]
        *next(given),
        a[2].new_zeros(LENGTH, a[8], dtype=int),
    )
    try:
        ys_ref, _ = run(ref, xs, cu)
    finally:
        csa2_ref.picks = forced
    for i, (a, b) in enumerate(zip(ys, ys_ref, strict=True)):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (i, (a - b).abs().max())
    print("same as csa2_ref given our picks; fast picks = PyTorch picks")


def test_training_step() -> None:
    """Compiled bf16 forward + backward of the SMALL stack on 2 documents."""
    torch.manual_seed(5)
    stack = torch.nn.ModuleList(csa2_ours.Csa2OursAttention(i) for i in range(10))
    stack = stack.cuda()
    cu = torch.tensor([0, 1500, 3000, 4096], device="cuda")
    x = torch.randn(1, 4096, csa2_ref.SMALL.dim, device="cuda", requires_grad=True)

    def step(x: Tensor) -> Tensor:
        shared: dict = {}
        total = x.new_zeros(())
        with torch.autocast("cuda", torch.bfloat16):
            for layer in stack:
                total = total + layer(x, cu, shared).float().square().mean()
        return total

    loss = torch.compile(step, dynamic=False)(x)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in stack.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    print("training step: compiled bf16 fwd + bwd, every parameter has a gradient")


if __name__ == "__main__":
    test_same_as_reference()
    test_training_step()
