"""experiments/train_router.py: targets from the frozen model, and router steps."""

import torch

from deepseek import csa2
from deepseek.double_indexing import add_router
from experiments import train_router
from experiments.selector_forward import selecting_layers
from tests.test_csa2 import MODES
from tests.test_double_indexing import CONFIG


def test_router_steps_reduce_loss() -> None:
    # Arrange: a frozen model, its targets on one batch, fresh routers
    torch.manual_seed(0)
    model = csa2.build(MODES, **CONFIG).eval().requires_grad_(False)
    tokens = torch.randint(0, CONFIG["vocab_size"], (2, 64))
    data = train_router.layer_data(model, tokens, 2)
    routers = [add_router(a, 2) for a in selecting_layers(model)]
    optimizer = torch.optim.Adam([p for r in routers for p in r.parameters()], 1e-2)

    # Act
    losses = []
    for _ in range(20):
        loss = train_router.router_loss(model, data, 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach()))

    # Assert: targets are attention shares; the loss falls
    for _, target in data.values():
        assert bool((target >= 0).all()) and bool((target.sum(-1) <= 1 + 1e-5).all())
        assert float(target.sum()) > 0
    assert losses[-1] < losses[0]
