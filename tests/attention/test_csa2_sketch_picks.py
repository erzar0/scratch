"""csa2_sketch_picks: a pool of every older key picks as csa2_exact_picks does; the
fast step picks the emulation's keys at 8 heads per KV head."""

import pytest
import torch

from attention import harness
from attention.csa2_exact_picks import Csa2ExactPicks
from attention.csa2_sketch_picks import Csa2SketchPicks
from attention.fast_pytorch import KV, csa2_sketch_picks
from tests.attention.test_harness import make_chunk

SKETCHES = ["sign", "pca", "random", "channels"]


@pytest.mark.parametrize("sketch", SKETCHES)
def test_full_pool_is_exact_picks(sketch: str) -> None:
    chunk = make_chunk(keys=203, dim=32)  # at most 187 keys before each window
    pooled = Csa2SketchPicks(8, 16, sketch=sketch, fraction=2, pool=24)(chunk)
    exact = Csa2ExactPicks(keep=8, window=16)(chunk)
    assert torch.equal(pooled.selected, exact.selected)


@pytest.mark.parametrize("sketch", SKETCHES)
def test_fast_step_picks_the_emulation_keys(sketch: str) -> None:
    torch.manual_seed(0)
    n, dim = 4096, 64
    m = Csa2SketchPicks(64, sketch=sketch, fraction=4)
    q = torch.randn(1, 8, dim)
    k, v = torch.randn(2, 1, n, dim)
    logits = (q[:, :, None] @ k[:, None].mT) * dim**-0.5
    valid = torch.ones(1, n, dtype=torch.bool)
    chunk = harness.Chunk(q[:, :, None], k, v, logits, valid, 0, dim**-0.5)
    want = m(chunk).selected[:, 0]

    _, won = csa2_sketch_picks(m)(q, KV(k, v))

    assert won is not None
    got = torch.zeros(1, n, dtype=torch.bool).scatter_(1, won.long(), True)
    assert torch.equal(got, want)
