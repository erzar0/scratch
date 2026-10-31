"""attention/key_bandit.py and attention/chunk_bandit.py against float64 references."""

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from attention.dense.chunk_bandit import (
    ChunkBanditConfig,
    chunk_attn,
    chunk_reads,
    chunk_sparse,
)
from attention.dense.key_bandit import (
    KeyBanditConfig,
    exact_causal,
    key_bandit,
    key_bandit_cost,
)
from model.heads import WEIGHTS, get_heads
from tests.reference import KEY, max_diff, naive_exact, naive_key_bandit

SEQ_LEN, DIM = 48, 32
SELECTORS = ("bandit", "oracle", "random")
type Selector = Literal["bandit", "oracle", "random"]
type SelectionQuery = Literal["all", "mean"]


def qkv(seed: int, seq_len: int = SEQ_LEN, scale: float = 1.0) -> tuple[jax.Array, ...]:
    q, k, v = jax.random.normal(jax.random.fold_in(KEY, seed), (3, seq_len, DIM))
    return q * scale, k * scale, v


def replace_after(arrays: tuple[jax.Array, ...], start: int) -> tuple[jax.Array, ...]:
    """Each array with rows from start on replaced by large noise."""
    return tuple(
        a.at[start:].set(
            5 * jax.random.normal(jax.random.fold_in(KEY, 99 + j), a[start:].shape)
        )
        for j, a in enumerate(arrays)
    )


def test_exact_causal_matches_reference() -> None:
    q, k, v = qkv(1)
    assert (
        max_diff(exact_causal(q, k, v), naive_exact(*map(np.asarray, (q, k, v))))
        < 1e-12
    )


@pytest.mark.parametrize(
    "settings", [(0.5, 3.0, 8), (2.0, 3.0, 8), (1.0, 0.0, 4), (2.0, 1.0, 16)]
)
def test_key_bandit_matches_reference(settings: tuple[float, float, int]) -> None:
    # Arrange
    q, k, v = qkv(5)
    order = jax.random.permutation(KEY, DIM)

    # Act
    out, cost = key_bandit_cost(q, k, v, order, KeyBanditConfig(*settings))

    # Assert
    want, want_cost = naive_key_bandit(
        (np.asarray(q), np.asarray(k), np.asarray(v)), np.asarray(order), settings
    )
    assert max_diff(out, want) < 1e-10
    assert abs(float(cost) - want_cost) < 1e-9


def test_key_bandit_huge_width_is_exact() -> None:
    q, k, v = qkv(13)
    out = key_bandit(q, k, v, KEY, KeyBanditConfig(ucb_width=1e6))
    assert max_diff(out, exact_causal(q, k, v)) < 1e-12


def test_key_bandit_cost_below_exact_and_monotone_in_width() -> None:
    # Arrange
    q, k, v = qkv(61, seq_len=64, scale=1.5)
    order = jax.random.permutation(KEY, DIM)

    # Act
    costs = [
        float(key_bandit_cost(q, k, v, order, KeyBanditConfig(w))[1])
        for w in (0.5, 1, 2, 4)
    ]

    # Assert
    assert costs == sorted(costs)
    assert costs[0] < (64 + 1) / 2


def test_key_bandit_is_causal() -> None:
    # Arrange
    arrays, half = qkv(20), SEQ_LEN // 2
    config = KeyBanditConfig(ucb_width=1.0)

    # Act
    q, k, v = arrays
    q2, k2, v2 = replace_after(arrays, half + 1)
    before = key_bandit(q, k, v, KEY, config)
    after = key_bandit(q2, k2, v2, KEY, config)

    # Assert
    assert max_diff(before[: half + 1], after[: half + 1]) < 1e-9


def test_key_bandit_gradients_finite() -> None:
    q, k, v = qkv(31, seq_len=32)
    loss = lambda q, k, v: jnp.sum(key_bandit(q, k, v, KEY, KeyBanditConfig(1.0)) ** 2)  # noqa: E731
    grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    assert all(bool(jnp.isfinite(g).all()) for g in grads)
    assert float(jnp.abs(grads[2]).max()) > 0


@pytest.mark.skipif(not WEIGHTS.exists(), reason="needs the trained softmax T=1024 LM")
def test_key_bandit_on_real_heads() -> None:
    # Arrange
    heads = [h for h in get_heads() if (h.layer, h.head) in ((0, 2), (3, 2))]

    # Act
    errors = []
    for head in heads:
        q, k, v = (jnp.asarray(a[:256]) for a in (head.q, head.k, head.v))
        exact = exact_causal(q, k, v)
        out = key_bandit(q, k, v, KEY, KeyBanditConfig(ucb_width=2.0))
        errors.append(float(jnp.linalg.norm(out - exact) / jnp.linalg.norm(exact)))

    # Assert
    assert max(errors) < 0.05  # old NumPy sweep: 0.009 mean at ucb_width 2


@pytest.mark.parametrize("selection_query", ["all", "mean"])
@pytest.mark.parametrize("summaries", [True, False])
@pytest.mark.parametrize("selector", SELECTORS)
def test_chunk_exact_when_every_chunk_refined(
    selector: Selector, summaries: bool, selection_query: SelectionQuery
) -> None:
    # Arrange: 6 chunks of 8, the last one padded
    q, k, v = qkv(70, seq_len=45)
    config = ChunkBanditConfig(
        selector, 8, 6, summaries=summaries, selection_query=selection_query
    )

    # Act
    out = chunk_attn(q, k, v, KEY, config)

    # Assert
    assert max_diff(out, exact_causal(q, k, v)) < 1e-12


def test_chunk_reads() -> None:
    # Arrange: T 16, chunks of 4, window 8; by hand, query chunks 0-3 read
    # 10, 26, 26 + 16 refined + 12 pulls, 26 + 16 + 4 summaries + 24 pulls
    bandit = ChunkBanditConfig("bandit", 4, 1, first_pulls=2, second_pulls=1)
    every_chunk = ChunkBanditConfig("random", 8, 6)

    # Act
    reads = chunk_reads(16, bandit), chunk_reads(45, every_chunk)

    # Assert: refining every chunk reads exactly what exact causal attention reads
    assert reads == (160 / 16, (45 + 1) / 2)


@pytest.mark.parametrize("selector", ["bandit", "random"])
def test_chunk_sparse_exact_when_every_chunk_refined(selector: str) -> None:
    q, k, v = qkv(70, seq_len=45)
    out = chunk_sparse(q, k, v, KEY, ChunkBanditConfig(selector, 8, 6, impl="xla"))  # type: ignore[arg-type]
    assert max_diff(out, exact_causal(q, k, v)) < 1e-12


@pytest.mark.parametrize("selector", ["bandit", "random"])
def test_chunk_sparse_equals_chunk_attn_mean_query(selector: str) -> None:
    # Arrange
    q, k, v = qkv(71, seq_len=45)
    config = ChunkBanditConfig(selector, 8, 2, impl="xla")  # type: ignore[arg-type]
    dense = dataclasses.replace(config, summaries=False, selection_query="mean")

    # Act
    out = chunk_sparse(q, k, v, KEY, config)

    # Assert
    assert max_diff(out, chunk_attn(q, k, v, KEY, dense)) < 1e-12


def test_chunk_sparse_rejects_oracle() -> None:
    q, k, v = qkv(71, seq_len=45)
    with pytest.raises(ValueError):
        chunk_sparse(q, k, v, KEY, ChunkBanditConfig("oracle", 8, 2, impl="xla"))


@pytest.mark.parametrize("selector", ["bandit", "random"])
def test_chunk_causal_across_chunks(selector: str) -> None:
    # Rows of chunks < c never see chunks >= c; within a chunk, picks use the chunk's
    # mean query (a documented leak). Chunk 5 picks 1 old chunk, so a leak shows.
    # Arrange
    arrays, start = qkv(73, seq_len=64), 6 * 8
    config = ChunkBanditConfig(selector, 8, 1, impl="xla")  # type: ignore[arg-type]

    # Act
    before = chunk_sparse(*arrays, KEY, config)
    after = chunk_sparse(*replace_after(arrays, start), KEY, config)

    # Assert
    assert max_diff(before[:start], after[:start]) < 1e-12


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="cuDNN needs a GPU")
def test_chunk_split_path_bf16_matches_f32() -> None:
    # Arrange
    q, k, v = (a.astype(jnp.float32) for a in qkv(72, seq_len=1000, scale=1.5))
    config = ChunkBanditConfig(chunk=64, refined_chunks=4, impl="xla")

    # Act
    want = chunk_sparse(q, k, v, KEY, config)
    bf16 = (a.astype(jnp.bfloat16) for a in (q, k, v))
    out = chunk_sparse(*bf16, KEY, dataclasses.replace(config, impl="split"))

    # Assert
    assert (
        float(jnp.linalg.norm(out.astype(jnp.float32) - want) / jnp.linalg.norm(want))
        < 0.02
    )
