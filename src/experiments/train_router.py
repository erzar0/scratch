"""R: train the routers of double indexing's coarse="learned" mode
(deepseek/double_indexing.py Router) on the frozen CSA2 LM: per Full or Reindex layer
a coarse indexer whose top-s superblocks should hold the entries the model attends to.

Target per query and complete superblock g: the attention weight (mean over heads
and over the layers sharing the picks: the layer and its Reuse followers) on the
layer's picks inside g. Picks in the query's incomplete superblock are always
candidates and carry no target. Loss: cross-entropy from the target to the coarse
softmax (with a learned temperature), summed over layers.

    SEQ=2048 uv run python -m experiments.train_router 8 16
    -> runs/route/router_seq2048_S{S}_r1.pt
    (STEPS, LR, KEYS from the environment; logs attention recall on 4 val windows)
"""

import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from transformers import PreTrainedModel

from deepseek import csa2
from deepseek.double_indexing import (
    CoarseMode,
    DoubleIndexing,
    Router,
    add_router,
    coarse_scores,
)
from experiments.selector_forward import selecting, selecting_layers
from model import train

logger = logging.getLogger(__name__)

OUT = Path("runs/route")
VAL_WINDOWS = 4

# Per Full or Reindex layer: its selector inputs and (B, S, G) target.
type LayerData = dict[int, tuple[csa2.SelectorInputs, torch.Tensor]]


@dataclass(frozen=True)
class RouterTraining:
    """Router optimizer settings (STEPS, LR, KEYS).

    Attributes:
        steps: Adam steps, one sequence each.
        lr: Peak learning rate (cosine to 0).
        n_keys: Pooled router keys per superblock.
    """

    steps: int = 600
    lr: float = 1e-3
    n_keys: int = 1

    @classmethod
    def from_env(cls) -> "RouterTraining":
        """Read STEPS, LR and KEYS; unset ones keep the defaults."""
        env = os.environ
        return cls(
            int(env.get("STEPS", cls.steps)),
            float(env.get("LR", cls.lr)),
            int(env.get("KEYS", cls.n_keys)),
        )


def router_path(seq_len: int, superblock_size: int, n_keys: int, seed: int = 0) -> Path:
    """Where the routers of every layer are saved; seed is the base model's."""
    name = f"router_seq{seq_len}_S{superblock_size}_r{n_keys}"
    return OUT / f"{name}{train.seed_suffix(seed)}.pt"


def load_routers(
    model: PreTrainedModel, seq_len: int, superblock_size: int, n_keys: int, seed: int
) -> None:
    """Give every Full and Reindex layer its trained router from router_path."""
    path = router_path(seq_len, superblock_size, n_keys, seed)
    state = torch.load(path, map_location=train.DEVICE, weights_only=True)
    for attention in selecting_layers(model):
        router = add_router(attention, superblock_size, n_keys)
        router.load_state_dict(state[str(attention.layer_idx)])


class Recorded(NamedTuple):
    """One selecting layer in a frozen forward: its selector inputs (float32), its
    indexer's picks, and the attention weight on them (B, S, k), the mean over
    heads and over the layers sharing the picks (the layer and its Reuse followers).
    """

    inputs: csa2.SelectorInputs
    picks: csa2.Picks
    weight: torch.Tensor


class _Recorder:
    """A selector that runs the layer's own indexer and keeps its inputs and picks."""

    def __init__(self) -> None:
        self.inputs: dict[int, csa2.SelectorInputs] = {}
        self.picks: dict[int, csa2.Picks] = {}

    def __call__(
        self, attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
    ) -> csa2.Picks:
        assert attention.indexer is not None
        picks = attention.indexer(
            inputs.hidden_states,
            inputs.q_residual,
            inputs.position_ids,
            inputs.index_k,
        )
        self.inputs[attention.layer_idx] = inputs
        self.picks[attention.layer_idx] = picks
        return picks


@torch.no_grad()
def record(model: PreTrainedModel, tokens: torch.Tensor) -> dict[int, Recorded]:
    """Frozen forward over tokens (B, S), bf16 on CUDA, without the LM head."""
    recorder, weights = _Recorder(), {}
    attentions = [layer.self_attn for layer in model.model.layers]

    def keep_weights(module: torch.nn.Module, _: object, output: tuple) -> None:
        weights[module.layer_idx] = output[1]

    hooks = [a.register_forward_hook(keep_weights) for a in attentions]
    cuda = train.DEVICE.type == "cuda"
    try:
        with (
            selecting(model, recorder),
            torch.autocast(train.DEVICE.type, torch.bfloat16, enabled=cuda),
        ):
            model.model(input_ids=tokens, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    found = {}
    for source, picks in recorder.picks.items():
        top_k = picks.entry.shape[-1]
        sharing = [a.layer_idx for a in attentions if a.picks_source == source]
        weight = torch.stack([weights[i][..., :top_k].float() for i in sharing])
        inputs = csa2.SelectorInputs(
            *(
                x.float() if x.is_floating_point() else x
                for x in recorder.inputs[source]
            )
        )
        found[source] = Recorded(inputs, picks, weight.mean((0, 2)))
    return found


def layer_data(
    model: PreTrainedModel, tokens: torch.Tensor, superblock_size: int
) -> LayerData:
    """Frozen forward over tokens (B, S): every selecting layer's inputs and target."""
    data: LayerData = {}
    for source, (inputs, picks, weight) in record(model, tokens).items():
        indexer = model.model.layers[source].self_attn.indexer
        n_superblocks = inputs.index_k.shape[1] // superblock_size
        complete = (
            csa2.finished_entries(inputs.position_ids, indexer.compress_rate)
            // superblock_size
        )
        superblock = picks.entry // superblock_size
        weight = weight * (picks.valid & (superblock < complete))
        target = torch.zeros(*weight.shape[:2], n_superblocks, device=weight.device)
        target.scatter_add_(-1, superblock.clamp(max=n_superblocks - 1), weight)
        data[source] = inputs, target
    return data


def masked_scores(
    attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs, route: DoubleIndexing
) -> torch.Tensor:
    """(B, S, G) coarse scores, -1e9 where not complete before the query."""
    assert attention.indexer is not None
    q = attention.indexer.queries(
        inputs.hidden_states, inputs.q_residual, inputs.position_ids
    )
    score = coarse_scores(attention, inputs, q, route)
    m = attention.indexer.compress_rate
    complete = csa2.finished_entries(inputs.position_ids, m) // route.superblock_size
    visible = torch.arange(score.shape[-1], device=score.device) < complete
    return score.masked_fill(~visible, -1e9)


def router_loss(model: PreTrainedModel, data: LayerData, size: int) -> torch.Tensor:
    """Cross-entropy of every layer's coarse softmax against its target, summed."""
    route = DoubleIndexing(size, 1, "learned")
    total = torch.zeros((), device=train.DEVICE)
    for attention in selecting_layers(model):
        inputs, target = data[attention.layer_idx]
        router = attention.router
        assert isinstance(router, Router)
        logits = masked_scores(attention, inputs, route)
        log_prob = (logits * router.temperature.exp()).log_softmax(-1)
        total = total + -(target * log_prob).sum(-1).mean()
    return total


@torch.no_grad()
def attention_recall(
    model: PreTrainedModel, data: list[LayerData], route: DoubleIndexing
) -> float:
    """Share of the target inside the coarse top-s, mean over layers and windows;
    queries with more than s complete superblocks only."""
    keep, recalls = route.kept_superblocks, []
    for layer_data_ in data:
        for attention in selecting_layers(model):
            inputs, target = layer_data_[attention.layer_idx]
            score = masked_scores(attention, inputs, route)
            kept = score.topk(min(keep, score.shape[-1]), dim=-1).indices
            rows = (score > -1e9).sum(-1, keepdim=True) > keep
            target = target * rows
            recalls.append(float(target.gather(-1, kept).sum() / target.sum()))
    return float(np.mean(recalls))


def log_recall(
    model: PreTrainedModel, data: list[LayerData], size: int, mode: CoarseMode
) -> None:
    """Log attention recall at s = 2 and 4."""
    for keep in (2, 4):
        recall = attention_recall(model, data, DoubleIndexing(size, keep, mode))
        logger.info("S=%d %s s=%d: attention recall %.3f", size, mode, keep, recall)


def train_routers(
    model: PreTrainedModel,
    size: int,
    seq_len: int,
    config: RouterTraining,
    splits: tuple[np.ndarray, np.ndarray],
) -> dict[str, dict[str, torch.Tensor]]:
    """Adam on the routers of every layer at once, on token splits (train,
    validation); returns layer -> router state."""
    train_data, val = splits
    windows = [val[i * seq_len : (i + 1) * seq_len][None] for i in range(VAL_WINDOWS)]
    val_data = [layer_data(model, train.to_device(w), size) for w in windows]
    for mode in ("mean", "bound", "oracle"):
        log_recall(model, val_data, size, mode)  # type: ignore[arg-type]
    routers = [add_router(a, size, config.n_keys) for a in selecting_layers(model)]
    params = [p for router in routers for p in router.parameters()]
    optimizer = torch.optim.Adam(params, lr=config.lr)
    rng = np.random.default_rng(0)
    for step in range(1, config.steps + 1):
        start = rng.integers(0, len(train_data) - seq_len)
        tokens = train.to_device(train_data[start : start + seq_len][None])
        loss = router_loss(model, layer_data(model, tokens, size), size)
        loss.backward()
        for group in optimizer.param_groups:
            group["lr"] = (
                config.lr * 0.5 * (1 + math.cos(math.pi * step / config.steps))
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == 1:
            logger.info("S=%d step %d loss %.4f", size, step, float(loss.detach()))
    log_recall(model, val_data, size, "learned")
    layers = selecting_layers(model)
    return {
        str(a.layer_idx): r.state_dict() for a, r in zip(layers, routers, strict=True)
    }


def main(run: train.TrainConfig, sizes: list[int], config: RouterTraining) -> None:
    """Train and save the routers for every superblock size."""
    model = train.load_model("csa2", train.run_name("csa2", run).with_suffix(".pt"))
    model.requires_grad_(False)
    OUT.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        splits = train.load("train"), train.load("validation")
        state = train_routers(model, size, run.seq_len, config, splits)
        torch.save(state, router_path(run.seq_len, size, config.n_keys, run.seed))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(
        train.TrainConfig.from_env(),
        [int(a) for a in sys.argv[1:]],
        RouterTraining.from_env(),
    )
