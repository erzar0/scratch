"""model/train.py: data windows, schedule and optimizer steps."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from deepseek import csa2
from model import train
from tests.test_csa2 import MODES, TINY

CONFIG = train.TrainConfig(seq_len=32, batch=2, steps=100, warmup=10)


def test_load_memmaps_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    tokens = np.arange(1000, dtype=np.uint16)
    tokens.tofile(tmp_path / "wt103_validation.bin")
    monkeypatch.setattr(train, "DATA", tmp_path)

    # Act
    data = train.load("validation")

    # Assert
    assert isinstance(data, np.memmap)
    np.testing.assert_array_equal(data, tokens)


def test_windows_full_batches_only() -> None:
    # Arrange
    data = np.arange(100)

    # Act
    batches = list(train.windows(data, seq_len=10, batch=3))

    # Assert: 9 windows of 10 inputs fit, so 3 full batches
    assert [tuple(b.shape) for b in batches] == [(3, 11)] * 3
    np.testing.assert_array_equal(batches[1][0].numpy(), np.arange(30, 41))
    assert len(list(train.windows(data, 10, 3, limit=2))) == 2


def test_learning_rate_schedule() -> None:
    assert train.learning_rate(0, CONFIG) == 0
    assert train.learning_rate(5, CONFIG) == pytest.approx(CONFIG.peak_lr / 2)
    assert train.learning_rate(10, CONFIG) == pytest.approx(
        CONFIG.peak_lr * (0.1 + 0.45 * (1 + np.cos(np.pi / 10)))
    )
    assert train.learning_rate(100, CONFIG) == pytest.approx(0.1 * CONFIG.peak_lr)


def test_train_steps_reduce_loss() -> None:
    # Arrange
    torch.manual_seed(0)
    model = csa2.build(MODES, **TINY).train()
    config = dataclasses.replace(CONFIG, peak_lr=1e-2)
    optimizer = train.make_optimizer(model, config)
    batch = torch.randint(0, TINY["vocab_size"], (2, config.seq_len + 1))

    # Act
    losses = []
    for _ in range(5):
        loss, norm = train.train_step(model, optimizer, batch, config.peak_lr)
        losses.append(float(loss))

    # Assert
    assert losses[-1] < losses[0]
    assert np.isfinite(float(norm))


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEQ", "2048")
    monkeypatch.setenv("PEAK_LR", "1e-4")
    config = train.TrainConfig.from_env()
    assert (config.seq_len, config.peak_lr, config.batch) == (2048, 1e-4, 8)
