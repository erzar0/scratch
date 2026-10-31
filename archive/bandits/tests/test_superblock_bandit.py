"""attention/selectors/superblock_bandit.py against a loop-style float64 reference
of its candidates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from attention.csa import CsaConfig, csa_attention, csa_attention_picks, indexer_inputs
from attention.selectors import superblock_bandit
from attention.selectors.double_indexing import superblock_candidates
from attention.selectors.superblock_bandit import (
    SuperblockBandit,
    bandit_superblocks,
    pull_order,
)
from tests.reference import (
    CSA_CONFIGS,
    csa_arrays,
    csa_setup,
    max_diff,
    naive_bandit_candidates,
)

ROUTES = [  # (S, s, first_pulls, survivors, second_pulls)
    (2, 1, 2, 3, 0),
    (3, 1, 1, 2, 1),
    (4, 2, 1, 3, 2),
    (4, 1, 2, 2, 2),
    (2, 10**6, 1, 10**6, 1),
]
CONFIG_IDS = range(len(CSA_CONFIGS))


def padded(index: int) -> tuple[dict, jax.Array, int, CsaConfig]:
    params, x = csa_setup(index)
    _, _, _, block_size, _, top_k, window = CSA_CONFIGS[index]
    x = jnp.pad(x, ((0, -len(x) % block_size), (0, 0)))
    return params, x, block_size, CsaConfig(top_k, window)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("index", CONFIG_IDS)
def test_candidates_match_reference(index: int, route: tuple) -> None:
    # Arrange
    params, x, block_size, _ = padded(index)
    bandit = SuperblockBandit(*route)
    indexer = indexer_inputs(params, x)
    n_superblocks = indexer.block_keys.shape[0] // route[0]
    order = np.asarray(pull_order(n_superblocks, route[0], 0))

    # Act
    finished = jnp.arange(len(x)) // block_size
    kept = bandit_superblocks(indexer, finished, bandit)
    got = superblock_candidates(kept, finished, route[0], indexer.block_keys.shape[0])

    # Assert
    want = naive_bandit_candidates(csa_arrays(params, x), order, route, 1.0)
    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("index", CONFIG_IDS)
def test_pick_matches_masked_csa(index: int, route: tuple) -> None:
    # Arrange
    params, x, block_size, config = padded(index)
    bandit = SuperblockBandit(*route)
    mask = bandit.select(params, x).candidates
    finished = jnp.arange(len(x)) // block_size

    # Act
    picks = superblock_bandit.pick(
        indexer_inputs(params, x), finished, config.top_k, bandit
    )
    _, picked, _ = csa_attention_picks(params, x, config, mask)

    # Assert
    block, valid = np.asarray(picks.block), np.asarray(picks.valid)
    picked_np, mask_np = np.asarray(picked), np.asarray(mask)
    for t in range(len(x)):
        want = {b for b in picked_np[t] if b < t // block_size and mask_np[t, b]}
        assert set(block[t][valid[t]].tolist()) == want, f"t={t}"


@pytest.mark.parametrize("index", CONFIG_IDS)
def test_keeping_everything_is_flat(index: int) -> None:
    # Arrange
    params, x, _, config = padded(index)
    keep_all = SuperblockBandit(2, 10**6, 1, 10**6, 1)

    # Act
    masked = csa_attention(params, x, config, keep_all.select(params, x)[0])

    # Assert
    assert max_diff(masked, csa_attention(params, x, config)) == 0


def test_causal_every_prefix() -> None:
    # Arrange
    params, x, _, config = padded(2)
    route = SuperblockBandit(2, 1, 1, 2, 1)

    def run(x: jax.Array) -> jax.Array:
        return csa_attention(params, x, config, route.select(params, x)[0])

    # Act
    full = run(x)

    # Assert
    assert max(max_diff(run(x[:n]), full[:n]) for n in range(1, len(x))) < 1e-12


def test_selection_reads() -> None:
    # Arrange
    params, x, block_size, _ = padded(2)
    route = SuperblockBandit(2, 1, 1, 2, 1)
    finished = np.arange(len(x)) // block_size
    complete = finished // 2

    # Act
    reads = np.asarray(route.select(params, x).reads)

    # Assert
    want = (
        complete + np.minimum(2, complete) + np.minimum(1, complete) * 2 + finished % 2
    )
    np.testing.assert_array_equal(reads, want)


def test_invalid_pulls_raise() -> None:
    with pytest.raises(ValueError):
        SuperblockBandit(4, 1, first_pulls=3, survivors=2, second_pulls=2)
