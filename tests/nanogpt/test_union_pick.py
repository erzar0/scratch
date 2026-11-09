"""Union picks, exact attention (src/nanogpt/union_pick.py): attention() is the
float32 softmax of exact q.k over union_dims.picks() made on the whole row at once
(several query chunks, windows small enough that queries span several). Needs CUDA:
    uv run python -m tests.nanogpt.test_union_pick
"""

import torch

from nanogpt.union_dims import Settings, picks, union_logits
from nanogpt.union_pick import attention
from tests.nanogpt.test_tournament import packed


def is_exact_softmax_over_union_picks(s: Settings) -> None:
    q, k, v, cu = packed((5, 1500, 2591))
    keep = picks({s.dims: union_logits(q, k, s.dims)}, cu, 0, s)
    logits = torch.einsum("thd,shd->hts", q, k) * 0.12
    weights = logits.masked_fill(~keep, float("-inf")).softmax(-1)
    want = torch.einsum("hts,shd->thd", weights, v)
    got = attention(q, k, v, cu, 0.12, s)
    error = float((got - want).norm() / want.norm())
    assert error < 1e-4, (s, error)


if __name__ == "__main__":
    for s in (Settings(512, 256, 16), Settings(512, 384, 32)):
        is_exact_softmax_over_union_picks(s)
    print("attention: exact softmax over the union picks")
