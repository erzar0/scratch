"""SparQ-rule picks, exact attention (src/nanogpt/sparq_pick.py): sparq_logits() is
the float64 sum over the query's top channels; attention() is the float32 softmax
of exact q.k over union_dims.picks() of those scores made on the whole row at once
(several query chunks, windows small enough that queries span several). Needs CUDA:
    uv run python -m tests.nanogpt.test_sparq_pick
"""

import torch

from nanogpt.sparq_pick import attention, sparq_logits
from nanogpt.union_dims import Settings, picks
from tests.nanogpt.test_tournament import packed


def logits_are_the_query_top_sum(dims: int) -> None:
    torch.manual_seed(dims)
    q, k = torch.randn(2, 64, 2, 128, device="cuda", dtype=torch.float64)
    top = q.abs().topk(dims, -1).indices
    mask = torch.zeros_like(q, dtype=torch.bool).scatter_(-1, top, True)
    want = torch.einsum("qhd,shd->hqs", q * mask, k)
    got = sparq_logits(q, k, dims).double()
    assert torch.allclose(got, want, atol=1e-4), (dims, (got - want).abs().max())


def is_exact_softmax_over_sparq_picks(s: Settings) -> None:
    q, k, v, cu = packed((5, 1500, 2591))
    keep = picks({s.dims: sparq_logits(q, k, s.dims)}, cu, 0, s)
    logits = torch.einsum("thd,shd->hts", q, k) * 0.12
    weights = logits.masked_fill(~keep, float("-inf")).softmax(-1)
    want = torch.einsum("hts,shd->thd", weights, v)
    got = attention(q, k, v, cu, 0.12, s)
    error = float((got - want).norm() / want.norm())
    assert error < 1e-4, (s, error)


if __name__ == "__main__":
    for dims in (8, 16, 128):
        logits_are_the_query_top_sum(dims)
    print("logits: the sum over the query's top channels")
    for s in (Settings(512, 256, 16), Settings(512, 384, 32)):
        is_exact_softmax_over_sparq_picks(s)
    print("attention: exact softmax over the SparQ-rule picks")
