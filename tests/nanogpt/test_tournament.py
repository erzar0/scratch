"""Training tournament (src/nanogpt/tournament.py): each query of a packed row picks
what Tournament: shuffled blocks, growing budget (emulation, budget 512) picks with
its document's prefix as the cache, and attention() is the softmax over those
winners, forward and backward. The kernel cases need CUDA:
    uv run python -m tests.nanogpt.test_tournament
"""

from itertools import accumulate

import torch

from attention import harness
from attention.tournament.shuffled_blocks import shuffle
from attention.tournament.shuffled_blocks_growing import GrowingShuffledBlocks
from nanogpt.tournament import BLOCK, QUOTA, attention, layouts, picked

EMULATION = GrowingShuffledBlocks(keep=512)
DOCS = (5, 700, 1500, 2600)  # 1, 1, 2 and 3 blocks at their ends


def test_settings_are_the_growing_version_at_budget_512() -> None:
    assert (QUOTA, BLOCK) == (EMULATION.winners(), EMULATION.block)
    tables = layouts(3, "cpu")
    assert torch.equal(tables[:2048], shuffle(2048, "cpu").int())
    assert torch.equal(tables[2048:], shuffle(3072, "cpu").int())


def packed(
    docs: tuple[int, ...], dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, k, v (T, 2 heads, 128) and cu_seqlens with record 91's repeated end."""
    torch.manual_seed(0)
    length = sum(docs)
    q, k, v = torch.randn(3, length, 2, 128, device="cuda", dtype=dtype)
    ends = [0, *accumulate(docs), length]
    return q, k, v, torch.tensor(ends, dtype=torch.int32, device="cuda")


def winners(q: torch.Tensor, k: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    """(H, T, T) picked()'s bit words as a mask."""
    keys = torch.arange(q.shape[0], device=q.device)
    return ((picked(q, k, cu)[..., keys // 32] >> (keys % 32)) & 1) == 1


def kernel_picks_the_emulation_keys() -> None:
    """Queries at the block edges (n near 1, 1024, 2048) and every 97th. Float32
    sums in another order may swap keys tied to 1e-8 (one of 40K winners here)."""
    q, k, v, cu = packed(DOCS)
    got = winners(q, k, cu)
    edges = {*range(1, 40), *range(1000, 1050), *range(2030, 2060)}
    shared = total = 0
    for start, end in zip(cu[:-2].tolist(), cu[1:-1].tolist(), strict=True):
        for t in range(start, end):
            n = t - start + 1
            if n not in edges and t % 97:
                continue
            query = q[t, :, None, None]  # (H, 1 head, 1 query, D)
            key, value = (x[start : t + 1].transpose(0, 1) for x in (k, v))
            logits = query @ key[:, None].mT * 128**-0.5
            valid = torch.ones(1, n, dtype=torch.bool, device="cuda")
            chunk = harness.Chunk(query, key, value, logits, valid, 0, 128**-0.5)
            want = EMULATION(chunk).selected[:, 0]
            mine = got[:, t, start : t + 1]
            assert int(got[:, t].sum()) == int(mine.sum()), t  # only its prefix
            assert torch.equal(mine.sum(-1), want.sum(-1)), (t, n)
            shared += int((mine & want).sum())
            total += int(want.sum())
    assert shared >= 0.999 * total, (shared, total)


def attention_is_the_softmax_over_the_winners() -> None:
    """bf16 forward and gradients within 1% of float32 math on the same winners."""
    q, k, v, cu = packed((300, 1200, 548), torch.bfloat16)
    for x in (q, k, v):
        x.requires_grad_()
    scale = 0.12
    out = torch.compile(attention)(q, k, v, cu, scale)
    grad = torch.randn_like(out)
    out.backward(grad)
    mask = winners(q.detach(), k.detach(), cu)
    exact = [x.detach().float().requires_grad_() for x in (q, k, v)]
    logits = torch.einsum("thd,shd->hts", exact[0], exact[1]) * scale
    weights = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    want = torch.einsum("hts,shd->thd", weights, exact[2])
    want.backward(grad.float())
    for name, a, b in zip(
        ("output", "q grad", "k grad", "v grad"),
        (out.detach(), q.grad, k.grad, v.grad),
        (want.detach(), *(x.grad for x in exact)),
        strict=True,
    ):
        assert a is not None and b is not None
        error = float((a.float() - b).norm() / b.norm())
        assert error < 0.01, (name, error)


if __name__ == "__main__":
    kernel_picks_the_emulation_keys()
    print("each query picks the emulation's keys on its document prefix")
    attention_is_the_softmax_over_the_winners()
    print("attention = softmax over the winners, forward and backward")
