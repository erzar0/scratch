"""Run a trained CSA2 model with a block selector in its Full and Reindex layers.

The one path every selection method runs through: set_selector puts a csa2.Selector
(deepseek/double_indexing.py, deepseek/reference.py) in place of each layer's
indexer, or None for the indexer itself (pure CSA2). Reads are indexer-key
equivalents per sequence and Full or Reindex layer; Reuse layers read none.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import torch
from transformers import PreTrainedModel

from deepseek import csa2, double_indexing
from model import train

# One selector for every layer, or one per layer (entries of Reuse layers unused).
type LayerSelectors = csa2.Selector | Sequence[csa2.Selector | None] | None


def selecting_layers(model: PreTrainedModel) -> list[csa2.Csa2Attention]:
    """Attention of every Full and Reindex layer."""
    layers = (layer.self_attn for layer in model.model.layers)
    return [a for a in layers if a.indexer is not None]


def layer_selector(selector: LayerSelectors, layer: int) -> csa2.Selector | None:
    """The selector of one layer."""
    if isinstance(selector, Sequence):
        return selector[layer]
    return selector


@contextmanager
def selecting(model: PreTrainedModel, selector: LayerSelectors) -> Iterator[None]:
    """Set each Full and Reindex layer's selector, and clear them afterwards."""
    for attention in selecting_layers(model):
        attention.selector = layer_selector(selector, attention.layer_idx)
    try:
        yield
    finally:
        for attention in selecting_layers(model):
            attention.selector = None


def reads_per_seq(
    model: PreTrainedModel, seq_len: int, selector: LayerSelectors, n_keys: int = 1
) -> float:
    """Indexer reads per sequence, the mean over Full and Reindex layers.

    Selectors other than DoubleIndexing count as the flat indexer's reads.
    """
    counts = []
    for attention in selecting_layers(model):
        chosen = layer_selector(selector, attention.layer_idx)
        route = chosen if isinstance(chosen, double_indexing.DoubleIndexing) else None
        assert attention.indexer is not None
        m = attention.indexer.compress_rate
        counts.append(double_indexing.reads(seq_len, m, route, n_keys))
    return float(np.mean(counts))


def batch_losses(
    model: PreTrainedModel,
    data: np.ndarray,
    run: train.TrainConfig,
    selector: LayerSelectors,
) -> np.ndarray:
    """Mean loss of every train.windows batch of data under selector."""
    with torch.no_grad(), selecting(model, selector):
        batches = train.windows(data, run.seq_len, run.batch)
        return np.array([float(train.loss_fn(model, batch)) for batch in batches])
