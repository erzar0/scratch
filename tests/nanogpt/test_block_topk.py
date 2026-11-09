"""Exact top-k per block (src/nanogpt/block_topk.py): each query's winners are a
per-query float64 reference's top `winners` keys in each of tournament_fast's
blocks (at least 99.9% of them; float32 near-ties differ), on documents of one
block and of several; the bit words hold the winners and attention() is the
softmax over them, forward and backward, in chunks too, and compiles in a
no-grad model. Needs CUDA:
    uv run python -m tests.nanogpt.test_block_topk
"""

from unittest import mock

import torch
from torch import Tensor

from nanogpt import block_topk
from nanogpt.tournament_fast import Settings, layouts
from tests.nanogpt import test_tournament_fast as fast_tests
from tests.nanogpt.test_tournament import packed
from tests.nanogpt.test_tournament_fast import winner_mask

RUN2 = Settings(2048, 64, 16)


def reference(q: Tensor, k: Tensor, cu: Tensor, settings: Settings) -> Tensor:
    """(H, T, T) top `winners` of q.k in float64 per block, query by query."""
    length, heads, _ = q.shape
    block, quota = settings.block, settings.winners
    table = layouts(-(-length // block), block, "cuda", True).long()
    mask = torch.zeros(heads, length, length, dtype=torch.bool, device=q.device)
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True):
        for t in range(s, e):
            n = t - s + 1
            logits = torch.einsum("hd,nhd->hn", q[t].double(), k[s : t + 1].double())
            b = -(-n // block)
            seg = table[block * ((b - 1) * b // 2 - 1) :][: b * block]
            slots = (
                seg.view(b, block) if b > 1 else torch.arange(n, device="cuda")[None]
            )
            for part in slots:
                part = part[part < n]
                top = logits[:, part].topk(min(quota, len(part)), dim=1).indices
                mask[torch.arange(heads)[:, None], t, s + part[top]] = True
    return mask


def picks_the_top_per_block(docs: tuple[int, ...]) -> None:
    q, k, _, cu = packed(docs)
    got = winner_mask(block_topk.select(q, k, cu, max(docs), RUN2), len(q))
    want = reference(q, k, cu, RUN2)
    shared, total = int((got & want).sum()), int(want.sum())
    assert int(got.sum()) == total and shared >= 0.999 * total, (shared, total)


if __name__ == "__main__":
    picks_the_top_per_block((300, 1200, 548))
    picks_the_top_per_block((5, 1500, 4500))
    print("winners = the top 64 per block (one block and several)")
    names = ("attention", "picked", "select")
    with mock.patch.multiple(fast_tests, **{n: getattr(block_topk, n) for n in names}):
        fast_tests.bits_are_the_winners(RUN2)
        fast_tests.attention_is_the_softmax_over_the_winners(RUN2)
        fast_tests.chunks_are_the_whole_row(RUN2)
        fast_tests.compiles_in_a_no_grad_model()
    print("bit words = winners; attention = softmax over them; chunks; compiles")
