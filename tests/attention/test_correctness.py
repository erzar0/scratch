"""Properties every registered version must keep, on its float32 emulation (CPU).

LLM-like synthetic data, budget 128, 2 KV heads × 2 query heads, head dim 128,
decode queries at the last positions of a causal context. Known exceptions
(docs/methods.md): block top-k and double indexing attend up to block - 1 keys
fewer than the budget, since one chosen block can straddle the recent window; halving
tournaments keep every key tied at a cutoff (LLM-like sinks and needles are exact
copies), so the budget test runs on outlier-channels data, which has no copies;
block and bracket tournaments end with n / 2^rounds keys, fewer than the budget
(but more than half) unless n is the budget times a power of two; shuffled blocks
with a growing budget keep the same winners per 1024-key block at every length, so
budget * n / 32K keys, and are not dense on short contexts. The DeepSeek-V4 CSA
layer is latency only (no emulation), so it is not here.
"""

import pytest
import torch

from attention import harness
from attention.block_topk import BlockTopK
from attention.dense import dense
from attention.registry import Version, versions
from attention.tournament.shuffled_blocks_growing import GrowingShuffledBlocks
from benchmarks import synthetic

KEEP, N, QUERIES = 128, 2048, 4
QUOTAS = {"tournament/blocks", "tournament/original"}  # fixed quotas per block
GROWING = "tournament/shuffled-blocks-growing"  # winners per block, not in all
VERSIONS = versions(KEEP)
EMULATED = [v for v in VERSIONS.values() if v.emulation is not None]
every = pytest.mark.parametrize("version", EMULATED, ids=lambda v: v.id)


def chunk_of(n: int, distribution: str = "llm-like") -> harness.Chunk:
    """Synthetic chunk: the last QUERIES positions of n, causal."""
    q, k, v = synthetic.qkv(distribution, n, QUERIES, 0, groups=2, device="cpu")
    positions = torch.arange(n)
    valid = positions[None] <= positions[n - QUERIES :, None]
    scaling = q.shape[-1] ** -0.5
    logits = ((q @ k[:, None].mT) * scaling).masked_fill(~valid, float("-inf"))
    return harness.Chunk(q, k, v, logits, valid, 0, scaling)


def emulate(version: Version) -> harness.Method:
    """The version's emulation (every parametrized version has one)."""
    assert version.emulation is not None
    return version.emulation


@every
def test_shapes(version: Version) -> None:
    chunk = chunk_of(N)
    out = emulate(version)(chunk)
    assert out.output.shape == chunk.query.shape  # (G, g, Q, D)
    assert out.selected.shape == (2, QUERIES, N)  # one pick per KV head


@every
@pytest.mark.parametrize("n", [N, N + 37])
def test_budget(version: Version, n: int) -> None:
    chunk = chunk_of(n, "outlier-channels")
    method = emulate(version)
    selected = method(chunk).selected.sum(-1)
    if isinstance(method, GrowingShuffledBlocks):  # ragged n: padded slots can win
        most, _ = method.split(n)
        assert bool((selected == most).all() if n == N else (selected <= most).all())
    elif version.id == "dense":
        assert torch.equal(selected, chunk.valid.sum(-1).expand(2, -1))
    elif isinstance(method, BlockTopK):  # a block may straddle the recent window
        assert bool((selected <= KEEP).all() & (selected > KEEP - method.block).all())
    elif version.id in QUOTAS and n != N:
        assert bool((selected <= KEEP).all() & (selected > KEEP // 2).all())
    else:
        assert bool((selected == KEEP).all()), selected.tolist()


@every
@pytest.mark.parametrize("n", [N, N + 37])  # 37: no block size divides it
def test_causal_and_valid(version: Version, n: int) -> None:
    chunk = chunk_of(n)
    selected = emulate(version)(chunk).selected
    assert not bool((selected & ~chunk.valid).any())


@every
def test_short_context_is_dense(version: Version) -> None:
    if version.id == GROWING:
        pytest.skip("keeps its winners per block at any length")
    chunk = chunk_of(KEEP - 28)
    torch.testing.assert_close(
        emulate(version)(chunk).output, dense(chunk).output, atol=1e-5, rtol=0
    )


@every
def test_deterministic(version: Version) -> None:
    a, b = emulate(version)(chunk_of(N)), emulate(version)(chunk_of(N))
    assert torch.equal(a.output, b.output) and torch.equal(a.selected, b.selected)


def test_dense_and_exact_topk_by_hand() -> None:
    chunk = chunk_of(N)
    weights = torch.softmax(chunk.logits, -1)
    torch.testing.assert_close(dense(chunk).output, weights @ chunk.value[:, None])
    top = weights.sum(1).topk(KEEP, -1).indices  # group score: summed weights
    want = torch.zeros(2, QUERIES, N, dtype=torch.bool).scatter_(-1, top, True)
    assert torch.equal(emulate(VERSIONS["exact-topk"])(chunk).selected, want)
