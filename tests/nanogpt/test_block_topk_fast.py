"""Exact top-k per block in two kernels (src/nanogpt/block_topk_fast.py): winners
are the float64 per-query reference's and block_topk.select's (at least 99.9% of
them, same count; float32 near-ties differ), on documents of one block and of
several; equal scores go to the lowest slots, exactly min(winners, alive) per
block; the bit words hold the winners and attention() is the softmax over them,
forward and backward, in chunks too, and compiles in a no-grad model. Needs CUDA:
    uv run python -m tests.nanogpt.test_block_topk_fast
"""

from unittest import mock

import torch

from nanogpt import block_topk
from nanogpt import block_topk_fast as fast
from tests.nanogpt import test_tournament_fast as fast_tests
from tests.nanogpt.test_block_topk import RUN2, reference
from tests.nanogpt.test_tournament import packed
from tests.nanogpt.test_tournament_fast import winner_mask


def picks_the_top_per_block(docs: tuple[int, ...]) -> None:
    q, k, _, cu = packed(docs)
    got = winner_mask(fast.select(q, k, cu, max(docs), RUN2), len(q))
    torch_way = winner_mask(block_topk.select(q, k, cu, max(docs), RUN2), len(q))
    for name, want in (("float64", reference(q, k, cu, RUN2)), ("torch", torch_way)):
        shared, total = int((got & want).sum()), int(want.sum())
        assert int(got.sum()) == total, (name, int(got.sum()), total)
        assert shared >= 0.999 * total, (name, shared, total)


def ties_go_to_the_lowest_slots() -> None:
    """Every key equal: each query keeps its document's first 64 positions."""
    q, k, _, cu = packed((300,))
    k = k[:1].expand_as(k).contiguous()
    got = winner_mask(fast.select(q, k, cu, 300, RUN2), 300)
    n = torch.arange(1, 301, device="cuda")
    want = torch.arange(300, device="cuda")[None] < n.clamp(max=64)[:, None]
    assert torch.equal(got, want.expand_as(got))


if __name__ == "__main__":
    picks_the_top_per_block((300, 1200, 548))
    picks_the_top_per_block((5, 1500, 4500))
    print("winners = the top 64 per block (one block and several)")
    ties_go_to_the_lowest_slots()
    print("ties: the lowest slots, 64 per block")
    names = ("attention", "picked", "select")
    with mock.patch.multiple(fast_tests, **{n: getattr(fast, n) for n in names}):
        fast_tests.bits_are_the_winners(RUN2)
        fast_tests.attention_is_the_softmax_over_the_winners(RUN2)
        fast_tests.chunks_are_the_whole_row(RUN2)
        fast_tests.compiles_in_a_no_grad_model()
    print("bit words = winners; attention = softmax over them; chunks; compiles")
