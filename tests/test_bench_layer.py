"""bench/layer.py: the hand-written decode steps equal the layers' forward."""

from collections.abc import Iterator

import pytest
import torch

from bench import layer


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    """A loaded CPU can run matmuls on fewer threads, which reorders float sums and
    can flip a near-tie in CSA2's top-k; one thread keeps the sums fixed."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("seq_len", [64, 129])  # T % m == 0 and != 0
def test_decode_step_matches_forward(
    monkeypatch: pytest.MonkeyPatch, seq_len: int
) -> None:
    # Arrange
    monkeypatch.setattr(layer, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(layer, "DTYPE", torch.float32)
    torch.manual_seed(0)
    attention, rotary = layer.csa2_layer()
    x = torch.randn(2, seq_len + 1, layer.D_MODEL)
    prefill = layer.csa2_prefill(attention, rotary)
    caches = [
        layer.csa2_cache(attention, rotary, x[i : i + 1, :seq_len]) for i in (0, 1)
    ]
    cache = layer.Cache(*(torch.cat(parts) for parts in zip(*caches, strict=True)))

    # Act
    with torch.no_grad():
        step = layer.csa2_decode_step(attention, rotary, seq_len)(cache, x[:, -1:])
        full = torch.cat([prefill(x[i : i + 1])[:, -1:] for i in (0, 1)])

    # Assert
    torch.testing.assert_close(step, full)


def test_exact_decode_step_matches_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(layer, "DEVICE", torch.device("cpu"))
    torch.manual_seed(0)
    exact = layer.Exact()
    x = torch.randn(2, 65, layer.D_MODEL)

    # Act
    with torch.no_grad():
        step = exact.decode_step(exact.cache(x[:, :-1]), x[:, -1:])
        full = exact(x)[:, -1:]

    # Assert
    torch.testing.assert_close(step, full)


def test_decode_step_refuses_a_finishing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(layer, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(layer, "DTYPE", torch.float32)
    attention, rotary = layer.csa2_layer()

    # Act / Assert: token 131 closes entry 32 (m = 4)
    with pytest.raises(ValueError, match="finishes an entry"):
        layer.csa2_decode_step(attention, rotary, 131)
