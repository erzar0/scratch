"""attention/selectors/channel_bandit.py against a loop-style float64 reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from attention.csa import CsaConfig, csa_attention, indexer_inputs, pad_to_blocks
from attention.selectors.channel_bandit import (
    ChannelBandit,
    channel_bandit_candidates,
)
from tests.reference import (
    CSA_CONFIGS,
    csa_arrays,
    csa_setup,
    max_diff,
    naive_channel_bandit,
)

SETTINGS = [  # (ucb_width, tolerance, dims_per_round)
    (1.0, 0.0, 2),
    (0.5, 0.0, 4),
    (2.0, 0.1, 2),
    (0.0, 0.0, 4),
]
CONFIG_IDS = [1, 2, 3, 4, 5]


def setup(index: int) -> tuple[dict, jax.Array, int, CsaConfig]:
    params, x = csa_setup(index)
    _, _, _, block_size, _, top_k, window = CSA_CONFIGS[index]
    return params, pad_to_blocks(x, block_size), block_size, CsaConfig(top_k, window)


@pytest.mark.parametrize("settings", SETTINGS)
@pytest.mark.parametrize("index", CONFIG_IDS)
def test_matches_reference(index: int, settings: tuple) -> None:
    # Arrange
    params, x, block_size, config = setup(index)
    ucb_width, tolerance, step = settings
    bandit = ChannelBandit(config.top_k, ucb_width, tolerance, step, seed=3)
    dim = params["indexer"]["key_compressor"]["bias_own"].shape[1]
    order = np.asarray(jax.random.permutation(jax.random.PRNGKey(3), dim))

    # Act
    survivors, reads = channel_bandit_candidates(
        indexer_inputs(params, x), jnp.arange(len(x)) // block_size, bandit
    )

    # Assert
    want, want_reads = naive_channel_bandit(
        csa_arrays(params, x), order, config.top_k, settings
    )
    np.testing.assert_array_equal(np.asarray(survivors), want)
    np.testing.assert_allclose(np.asarray(reads), want_reads, atol=1e-12)


@pytest.mark.parametrize("index", CONFIG_IDS)
def test_infinite_width_is_flat(index: int) -> None:
    # Arrange
    params, x, _, config = setup(index)
    bandit = ChannelBandit(config.top_k, ucb_width=1e9, dims_per_round=2)

    # Act
    selection = bandit.select(params, x)

    # Assert
    masked = csa_attention(params, x, config, selection.candidates)
    assert max_diff(masked, csa_attention(params, x, config)) == 0


def test_eliminates_and_saves_reads() -> None:
    # Arrange
    params, x, block_size, config = setup(4)
    bandit = ChannelBandit(config.top_k, ucb_width=0.5, dims_per_round=4)

    # Act
    selection = bandit.select(params, x)

    # Assert
    finished = np.arange(len(x)) // block_size
    assert (np.asarray(selection.reads) <= finished + 1e-12).all()
    assert np.asarray(selection.reads).sum() < 0.9 * finished.sum()


def test_causal_every_prefix() -> None:
    # Arrange
    params, x, _, config = setup(2)
    bandit = ChannelBandit(config.top_k, ucb_width=0.5, dims_per_round=2)

    def run(x: jax.Array) -> jax.Array:
        selection = bandit.select(params, x)
        return csa_attention(params, x, config, selection.candidates)

    # Act
    full = run(x)

    # Assert
    assert max(max_diff(run(x[:n]), full[:n]) for n in range(1, len(x))) < 1e-12
