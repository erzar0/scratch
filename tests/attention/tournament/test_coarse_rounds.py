"""Coarse rounds: src/attention/tournament/coarse_rounds.py.

uv run pytest tests/attention/tournament/test_coarse_rounds.py
"""

import torch

from attention.tournament.coarse_rounds import CoarseRounds
from attention.tournament.shuffled_blocks import ShuffledBlocks, shuffle
from tests.attention.test_correctness import chunk_of

N, BLOCK, KEEP = 4096, 1024, 64  # 16 winners per block: 6 halvings


def test_cut_2_is_shuffled_blocks() -> None:
    chunk = chunk_of(N, "outlier-channels")
    half = CoarseRounds(block=BLOCK, keep=KEEP, cut=2)(chunk)
    plain = ShuffledBlocks(block=BLOCK, keep=KEEP)(chunk)
    assert torch.equal(half.selected, plain.selected)


def test_coarser_cuts_keep_the_quota_on_fewer_channels() -> None:
    # outlier-channels: no duplicate keys, so no ties at a cutoff
    chunk = chunk_of(N, "outlier-channels")
    reads = []
    for cut in (2, 4, 8):
        method = CoarseRounds(block=BLOCK, keep=KEEP, cut=cut)
        assert method.read_by(6) == method.dims * 6 // (cut.bit_length() - 1)
        outcome = method(chunk)
        blocks = outcome.selected[..., shuffle(N, "cpu")].unflatten(-1, (-1, BLOCK))
        assert bool((blocks.sum(-1) == KEEP * BLOCK // N).all()), cut
        reads.append(float(outcome.reads.float().mean()))
    assert reads[0] > reads[1] > reads[2], reads
