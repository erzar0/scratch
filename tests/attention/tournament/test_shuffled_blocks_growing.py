"""Shuffled blocks with a growing budget: the same winners in every 1024-slot block.

src/attention/tournament/shuffled_blocks_growing.py. The kernel case needs CUDA:
    uv run python -m tests.attention.tournament.test_shuffled_blocks_growing
"""

import torch

from attention.registry import versions
from attention.tournament.shuffled_blocks import shuffle
from attention.tournament.shuffled_blocks_growing import GrowingShuffledBlocks
from tests.attention.test_correctness import chunk_of

KEEP = 128  # keys at 32K: 4 winners per block of 1024


def test_every_block_keeps_the_same_winners_so_the_total_grows() -> None:
    method = GrowingShuffledBlocks(keep=KEEP)
    assert method.winners() == 4
    for n in (2048, 4096, 8192):
        # outlier-channels: no duplicate keys, so no ties at a cutoff
        selected = method(chunk_of(n, "outlier-channels")).selected  # position order
        blocks = selected[..., shuffle(n, "cpu")].unflatten(-1, (-1, 1024))
        assert bool((blocks.sum(-1) == 4).all()), n
        assert bool((selected.sum(-1) == 4 * n // 1024).all()), n


def test_equals_shuffled_blocks_at_32k() -> None:
    built = versions(KEEP)
    chunk = chunk_of(32768)
    growing = built["tournament/shuffled-blocks-growing"].emulation
    fixed = built["tournament/shuffled-blocks"].emulation
    assert growing is not None and fixed is not None
    assert torch.equal(growing(chunk).selected, fixed(chunk).selected)


def kernel_keeps_the_same_winners_per_block() -> None:
    """GPU: 16 winners in each of 8 blocks of 1024 at 8192 keys (budget 512)."""
    from attention.triton import tournament_shuffled_blocks
    from tests.attention.triton import inputs

    q, _, k_t, _ = (t.bfloat16() for t in inputs(8192))
    keep, blocks = GrowingShuffledBlocks(keep=512).split(8192)
    won = tournament_shuffled_blocks.select(q, k_t, keep, blocks)  # slots in k_t
    assert (keep, blocks) == (128, 8) and bool((won >= 0).all())
    for row in won.long().cpu():
        assert torch.equal(torch.bincount(row // 1024), torch.full((8,), 16))


if __name__ == "__main__":
    kernel_keeps_the_same_winners_per_block()
    print("kernel keeps 16 winners per block")
