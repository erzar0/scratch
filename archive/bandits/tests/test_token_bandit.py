"""attention/selectors/token_bandit.py against a loop-style float64 reference."""

import jax
import numpy as np
import pytest

from attention.csa import CsaConfig, csa_attention, pad_to_blocks
from attention.selectors.token_bandit import TokenBandit
from tests.reference import (
    CSA_CONFIGS,
    csa_arrays,
    csa_setup,
    max_diff,
    naive_token_bandit,
)

SETTINGS = [  # (first_pulls, second_pulls, survivors, ucb_width)
    (1, 1, 4, 1.0),
    (2, 0, 3, 0.0),
    (1, 2, 8, 2.0),
    (2, 2, 4, 0.5),
]
CONFIG_IDS = [1, 2, 4, 5]


def setup(index: int) -> tuple[dict, jax.Array, int, CsaConfig]:
    params, x = csa_setup(index)
    _, _, _, block_size, _, top_k, window = CSA_CONFIGS[index]
    return params, x, block_size, CsaConfig(top_k, window)


def pull_order(n_blocks: int, block_size: int, seed: int) -> np.ndarray:
    base = jax.random.PRNGKey(seed)
    return np.array(
        [
            jax.random.permutation(jax.random.fold_in(base, b), block_size)
            for b in range(n_blocks)
        ]
    )


@pytest.mark.parametrize("settings", SETTINGS)
@pytest.mark.parametrize("index", CONFIG_IDS)
def test_matches_reference(index: int, settings: tuple) -> None:
    # Arrange
    params, x, block_size, config = setup(index)
    first, second, survivors, width = settings
    survivors = max(survivors, config.top_k)
    bandit = TokenBandit(config.top_k, first, second, survivors, width, seed=5)
    n_blocks = len(pad_to_blocks(x, block_size)) // block_size

    # Act
    selection = bandit.select(params, x)

    # Assert
    order = pull_order(n_blocks, block_size, 5)
    settings = (first, second, survivors, width)
    want = naive_token_bandit(csa_arrays(params, x), order, config.top_k, settings)
    np.testing.assert_array_equal(np.asarray(selection.candidates), want)


@pytest.mark.parametrize("index", CONFIG_IDS)
def test_top_k_covering_everything_is_flat(index: int) -> None:
    # Arrange
    params, x, _, config = setup(index)
    bandit = TokenBandit(top_k=10**6, survivors=10**6)

    # Act
    selection = bandit.select(params, x)

    # Assert
    masked = csa_attention(params, x, config, selection.candidates)
    assert max_diff(masked, csa_attention(params, x, config)) == 0


def test_causal_every_prefix() -> None:
    # Arrange
    params, x, _, config = setup(2)
    bandit = TokenBandit(config.top_k, survivors=4)

    def run(x: jax.Array) -> jax.Array:
        selection = bandit.select(params, x)
        return csa_attention(params, x, config, selection.candidates)

    # Act
    full = run(x)

    # Assert
    assert max(max_diff(run(x[:n]), full[:n]) for n in range(1, len(x))) < 1e-12


@pytest.mark.parametrize(
    "bad", [dict(first_pulls=0), dict(first_pulls=3, second_pulls=2), dict(survivors=1)]
)
def test_invalid_settings_raise(bad: dict) -> None:
    params, x, _, _ = setup(2)
    with pytest.raises(ValueError):
        TokenBandit(**{"top_k": 2, **bad}).select(params, x)
