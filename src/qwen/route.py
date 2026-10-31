"""Routers of double indexing's learned coarse step on CSA2-Qwen (tier D).

experiments/train_router.py on the frozen CSA2-Qwen: training windows of 4096
tokens from the web + books mix, recall on WikiText val; superblocks of SIZE.

    uv run python -m qwen.route    -> runs/qwen/router_S16_r1.pt
    (STEPS, LR from the environment)
"""

import logging

import torch
from transformers import Qwen3ForCausalLM

from deepseek.double_indexing import add_router
from experiments import train_router
from experiments.selector_forward import selecting_layers
from qwen import data
from qwen import train as qt

logger = logging.getLogger(__name__)

SIZE = 16
SEQ = 4096
PATH = qt.RUNS / f"router_S{SIZE}_r1.pt"


def load(model: Qwen3ForCausalLM) -> None:
    """Give every Full and Reindex layer its trained router, in the indexer's dtype."""
    state = torch.load(PATH, map_location=model.device, weights_only=True)
    for attention in selecting_layers(model):
        assert attention.indexer is not None
        router = add_router(attention, SIZE)
        router.load_state_dict(state[str(attention.layer_idx)])
        router.to(attention.indexer.q_b_proj.weight.dtype)


def main() -> None:
    """Train and save the routers."""
    model = qt.build("csa2", qt.final_stage("csa2")).eval().requires_grad_(False)
    tokens = data.mix(), data.load("validation")
    config = train_router.RouterTraining.from_env()
    torch.save(train_router.train_routers(model, SIZE, SEQ, config, tokens), PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
