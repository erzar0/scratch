"""Copy stress test: each val window's first half, repeated, so the second half can be
predicted by copying from T/2 tokens back.

Loss counts only the second half, whose useful entries lie T/2 back: a selector that
misses them loses the copy. Δ vs pure CSA2 with paired SE over the batches, for
paired_se's headline methods, all through experiments/selector_forward.

    SEQ=2048 BATCH=2 uv run python -m experiments.copy_test
    -> runs/paper/copy_test.csv (rows for SEQ and SEED replaced)
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from experiments import paired_se
from experiments.results import Results, paired_delta
from experiments.selector_forward import LayerSelectors, selecting
from model import train

logger = logging.getLogger(__name__)

OUT = paired_se.OUT.with_name("copy_test.csv")


def copy_batch(batch: torch.Tensor) -> torch.Tensor:
    """(B, T + 1) windows -> (B, T + 1): the first T/2 tokens twice, then token 0."""
    half = (batch.shape[1] - 1) // 2
    first = batch[:, :half]
    return torch.cat([first, first, first[:, :1]], 1)


@torch.no_grad()
def copy_losses(
    model: PreTrainedModel,
    data: np.ndarray,
    run: train.TrainConfig,
    selector: LayerSelectors,
) -> np.ndarray:
    """Mean loss on the repeated half of copy_batch, per train.windows batch."""
    cuda = train.DEVICE.type == "cuda"
    losses = []
    with selecting(model, selector):
        for batch in train.windows(data, run.seq_len, run.batch):
            seq = copy_batch(batch)
            with torch.autocast(train.DEVICE.type, torch.bfloat16, enabled=cuda):
                logits = model(input_ids=seq[:, :-1], use_cache=False).logits
            half = run.seq_len // 2
            nll = F.cross_entropy(
                logits[:, half:].flatten(0, 1).float(), seq[:, half + 1 :].flatten()
            )
            losses.append(float(nll))
    return np.array(losses)


def main(run: train.TrainConfig) -> None:
    """Copy-half Δ vs pure CSA2 for every headline; see the module docstring."""
    model = train.load_model("csa2", train.run_name("csa2", run).with_suffix(".pt"))
    val = train.load("validation")
    reference = copy_losses(model, val, run, None)
    results = Results(OUT, {"T": run.seq_len, "seed": run.seed})
    runs: list[tuple[str, str, float | None, np.ndarray]] = [
        ("pure CSA2", "flat indexer", 1.0, reference)
    ]
    for found in paired_se.headlines(model, run):
        selector = paired_se.prepared(model, run, found)
        losses = copy_losses(model, val, run, selector)
        runs.append((found.method, found.config, found.reads_ratio, losses))
    for name, config, ratio, losses in runs:
        delta, se = paired_delta(losses, reference)
        row = {"T": run.seq_len, "seed": run.seed, "method": name, "config": config}
        row |= {"reads_ratio": ratio, "copy_loss": float(losses.mean())}
        row |= {"delta": delta, "se": se, "n_batches": len(reference)}
        logger.info("%s", row)
        results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(train.TrainConfig.from_env())
