"""Union of top dims (src/nanogpt/union_dims.py): union_logits() is the float64 sum
over each pair's union of top channels (all channels: q.k); picks() keeps each
window's best keys up to its quota in one pass or halving rounds, as a float64
loop over sampled queries does, on windows small enough that queries span several;
attention() over every key on every channel is dense document-causal softmax.
Needs CUDA:
    uv run python -m tests.nanogpt.test_union_dims
"""

import math

import torch
from torch import Tensor

from nanogpt.tournament_fast import layouts
from nanogpt.union_dims import Settings, attention, dims_of, picks, union_logits
from tests.nanogpt.test_tournament import packed
from tests.nanogpt.test_tournament_local import local_mask


def logits_are_the_union_sum(dims: int) -> None:
    torch.manual_seed(dims)
    q, k = torch.randn(2, 64, 2, 128, device="cuda", dtype=torch.float64)
    mq = torch.zeros_like(q, dtype=torch.bool).scatter_(
        -1, q.abs().topk(min(dims, 128), -1).indices, True
    )
    mk = torch.zeros_like(k, dtype=torch.bool).scatter_(
        -1, k.abs().topk(min(dims, 128), -1).indices, True
    )
    union = mq[:, None] | mk[None]  # (Q, S, H, D)
    want = (q[:, None] * k[None] * union).sum(-1).permute(2, 0, 1)
    got = union_logits(q, k, dims).double()
    assert torch.allclose(got, want, atol=1e-4), (dims, (got - want).abs().max())
    if dims >= 128:
        assert torch.allclose(got, torch.einsum("qhd,shd->hqs", q, k), atol=1e-4)


def reference(q: Tensor, k: Tensor, cu: Tensor, s: Settings, query: int) -> Tensor:
    """(H, T) the query's picks, float64, one window at a time."""
    length, heads, _ = q.shape
    start = max(c for c in cu.tolist() if c <= query)
    n = query - start + 1
    b = math.ceil(n / s.block)
    if b == 1:
        offset = torch.arange(s.block, device=q.device)
    else:
        table = layouts(b, s.block, str(q.device), True)
        offset = table[s.block * ((b - 1) * b // 2 - 1) :][: b * s.block]
    out = torch.zeros(heads, length, dtype=torch.bool, device=q.device)
    quota = s.keys // b
    for h in range(heads):
        for w in range(b):
            slots = offset[w * s.block : (w + 1) * s.block]
            alive = (start + slots[slots < n]).tolist()
            for d in dims_of(s, b):
                score = union_logits(
                    q[query : query + 1].double(), k[alive].double(), d
                )
                target = max(quota, (len(alive) + 1) // 2)
                kept = min(len(alive), quota if s.mode == "pass" else target)
                best = score[h, 0].topk(kept).indices.tolist()
                alive = [alive[i] for i in best]
            out[h, alive] = True
    return out


def picks_match_the_reference(s: Settings) -> None:
    """Same counts per query; picks differ at most on float32 near-ties."""
    q, k, _, cu = packed((5, 1500, 2591))
    needed = {
        d for b in range(1, math.ceil(q.shape[0] / s.block) + 1) for d in dims_of(s, b)
    }
    got = picks({d: union_logits(q, k, d) for d in needed}, cu, 0, s)
    for query in range(0, q.shape[0], 37):
        want = reference(q, k, cu, s, query)
        mine = got[:, query]
        assert torch.equal(mine.sum(-1), want.sum(-1)), (s, query)
        assert int((mine & want).sum()) >= 0.99 * int(want.sum()), (s, query)


def all_keys_all_dims_is_dense() -> None:
    q, k, v, cu = packed((300, 1200, 548), torch.bfloat16)
    s = Settings(512, 4 * 2048, 128, "pass")
    got = attention(q, k, v, cu, 0.12, s).float()
    mask = local_mask(q, cu, q.shape[0])
    logits = torch.einsum("thd,shd->hts", q.float(), k.float()) * 0.12
    weights = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    want = torch.einsum("hts,shd->thd", weights, v.float())
    error = float((got - want).norm() / want.norm())
    assert error < 0.01, error


if __name__ == "__main__":
    for dims in (8, 16, 128):
        logits_are_the_union_sum(dims)
    print("logits: the sum over each pair's union of top channels")
    for s in (
        Settings(512, 256, 16, "pass"),
        Settings(512, 256, 16, "rounds"),
        Settings(512, 384, 8, "rounds"),
    ):
        picks_match_the_reference(s)
    print("picks: each window's best keys up to its quota, one pass and rounds")
    all_keys_all_dims_is_dense()
    print("attention: every key on every channel is dense")
