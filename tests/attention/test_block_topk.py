"""Block top-k (src/attention/block_topk.py)."""

import pytest
import torch

from attention import harness
from attention.block_topk import BlockTopK, summary_keys
from attention.dense import dense
from tests.attention.test_harness import make_chunk

FLAT = BlockTopK(block=4, keep=12, window=4)


def double(coarse: str) -> BlockTopK:
    return BlockTopK(4, 12, 4, superblock=4, kept_superblocks=1, coarse=coarse)


def planted() -> harness.Chunk:
    """One query over 256 keys; key 37 is 4 x the query's first head."""
    c = make_chunk(groups=1, queries=1, keys=256)
    key = c.key.clone()
    key[0, 37] = 4 * c.query[0, 0, 0]
    logits = (c.query @ key[:, None].mT) * c.scaling
    return harness.Chunk(c.query, key, c.value, logits, c.valid, 0, c.scaling)


@pytest.mark.parametrize(
    "method",
    [
        BlockTopK(block=4, keep=16, window=4),
        *(BlockTopK(4, 16, 4, 2, 2, coarse) for coarse in ("mean", "bound")),
    ],
    ids=repr,
)
def test_keeping_every_block_is_exact_attention(method: BlockTopK) -> None:
    chunk = make_chunk()
    torch.testing.assert_close(method(chunk).output, dense(chunk).output)


def test_bound_summary_bounds_every_key() -> None:
    keys, q = torch.randn(3, 16, 8), torch.randn(8)
    mid, half = summary_keys(keys, "bound").unbind(-2)
    assert bool(((keys @ q).amax(-1) <= mid @ q + half @ q.abs() + 1e-5).all())


@pytest.mark.parametrize("method", [FLAT, double("mean"), double("bound")], ids=repr)
def test_planted_key_is_kept(method: BlockTopK) -> None:
    assert bool(method(planted()).selected[0, 0, 37])


def test_reads() -> None:
    chunk = planted()
    flat, coarse = FLAT(chunk), double("mean")(chunk)
    # flat: 63 block means (the window is the last block); double: 16 superblock
    # means + 4 blocks of the kept one
    assert flat.reads.item() == 63 * 16 + 32 * flat.selected.sum()
    assert coarse.reads.item() == (16 + 4) * 16 + 32 * coarse.selected.sum()
