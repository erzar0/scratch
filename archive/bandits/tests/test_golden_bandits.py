"""Cut from tests/test_golden.py when the bandits were archived (frozen, not run).

Used GOLDEN, KEY, SEQ_LEN, SMALL, assert_golden and the csa_params / x fixtures
from tests/test_golden.py, plus the forward_bandit golden via test_forward's
"bandit" mixer. Keys stay in tests/data/golden.npz.
"""

from attention.dense.chunk_bandit import ChunkBanditConfig, chunk_attn, chunk_sparse
from attention.dense.key_bandit import KeyBanditConfig, exact_causal, key_bandit_cost
from attention.selectors import superblock_bandit
from attention.selectors.double_indexing import superblock_candidates
from attention.selectors.superblock_bandit import SuperblockBandit, bandit_superblocks

BANDIT_ROUTES = {
    "b1": SuperblockBandit(2, 1, first_pulls=2, survivors=3, second_pulls=0),
    "b2": SuperblockBandit(3, 1, first_pulls=1, survivors=2, second_pulls=1),
    "b3": SuperblockBandit(4, 2, first_pulls=1, survivors=3, second_pulls=2),
}

@pytest.mark.parametrize("name", BANDIT_ROUTES)
def test_csa_bandit(csa_params: dict, x: jax.Array, name: str) -> None:
    # Arrange
    route = BANDIT_ROUTES[name]
    indexer = indexer_inputs(csa_params, x)
    n_blocks = indexer.block_keys.shape[0]
    finished = jnp.arange(SEQ_LEN) // 4

    # Act
    kept = bandit_superblocks(indexer, finished, route)
    candidates = superblock_candidates(kept, finished, route.superblock_size, n_blocks)
    picks = superblock_bandit.pick(indexer, finished, 2, route)
    y = csa_attention(csa_params, x, SMALL, route.select(csa_params, x).candidates)

    # Assert
    assert_golden(f"bandit_cand_{name}", candidates)
    for field, actual in zip(("score", "block", "valid"), picks, strict=True):
        assert_golden(f"bandit_pick_{field}_{name}", actual)
    assert_golden(f"bandit_csa_{name}", y)


@pytest.fixture(scope="module")
def head() -> tuple[jax.Array, jax.Array, jax.Array]:
    q, k, v = (
        jax.random.normal(jax.random.fold_in(KEY, 20 + i), (70, 16)) for i in range(3)
    )
    return q, k, v


@pytest.mark.parametrize("selector", ["oracle", "bandit"])
@pytest.mark.parametrize("summaries", [True, False])
@pytest.mark.parametrize("selection_query", ["all", "mean"])
def test_chunk_attn(
    head: tuple[jax.Array, ...], selector: str, summaries: bool, selection_query: str
) -> None:
    # Arrange
    config = ChunkBanditConfig(
        selector=selector,  # type: ignore[arg-type]
        chunk=8,
        refined_chunks=2,
        summaries=summaries,
        selection_query=selection_query,  # type: ignore[arg-type]
    )

    # Act
    out = chunk_attn(*head, KEY, config)

    # Assert
    assert_golden(f"chunk_attn_{selector}_{summaries}_{selection_query}", out)


@pytest.mark.parametrize("selector", ["bandit", "random"])
def test_chunk_sparse(head: tuple[jax.Array, ...], selector: str) -> None:
    # Arrange
    config = ChunkBanditConfig(
        selector=selector,  # type: ignore[arg-type]
        chunk=8,
        refined_chunks=2,
        impl="xla",
    )

    # Act
    out = chunk_sparse(*head, KEY, config)

    # Assert
    assert_golden(f"chunk_sparse_{selector}", out)


@pytest.mark.parametrize("ucb_width", [0.5, 2.0])
def test_key_bandit(head: tuple[jax.Array, ...], ucb_width: float) -> None:
    # Arrange
    order = jax.random.permutation(KEY, 16)

    # Act
    q, k, v = head
    out, cost = key_bandit_cost(q, k, v, order, KeyBanditConfig(ucb_width=ucb_width))

    # Assert
    assert_golden(f"bandit_cost_{ucb_width}", out)
    assert_golden(f"bandit_cost_cost_{ucb_width}", cost)
    assert_golden("exact_causal", exact_causal(*head))


