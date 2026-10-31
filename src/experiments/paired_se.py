"""E8: paired standard error of Δ val loss vs pure CSA2 (top_k 8, window 32) for each
method's headline config, every method on the same val batches.

Headline: the config reading least at Δ ≤ TOLERANCE; none matches → the lowest Δ.
Random entries and the attention oracle at top_k 8. Every method and pure CSA2 run
through experiments/selector_forward, so the pairs differ only in the selector.
Headlines always come from the seed-0 model's sweeps, so SEED=1 (a second trained
model) tests them on a model they were not chosen on.

    SEQ=2048 BATCH=2 uv run python -m experiments.paired_se [smoke]
    -> runs/paper/paired_se.csv (rows for SEQ replaced; SEED=n: paired_se_s<n>.csv)
"""

import logging
import sys
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl
from transformers import PreTrainedModel

from deepseek.reference import Oracle, RandomEntries
from experiments import eval_route
from experiments.eval_csa import TOP_K
from experiments.results import Results, paired_delta, read
from experiments.selector_forward import LayerSelectors, batch_losses
from experiments.train_router import RouterTraining
from model import train

logger = logging.getLogger(__name__)

OUT = Path("runs/paper/paired_se.csv")
TOLERANCE = 0.01


class Headline(NamedTuple):
    """One method's headline: its label, settings text, reads ratio (None for the
    references) and its selector of every layer."""

    method: str
    config: str
    reads_ratio: float | None
    selector: LayerSelectors


def headline(frame: pl.DataFrame) -> dict[str, Any]:
    """Row with the highest reads_ratio at delta_vs_flat ≤ TOLERANCE, else lowest Δ."""
    matched = frame.filter(pl.col("delta_vs_flat") <= TOLERANCE)
    if matched.is_empty():
        return frame.sort("delta_vs_flat").row(0, named=True)
    return matched.sort("reads_ratio", descending=True).row(0, named=True)


def headlines(model: PreTrainedModel, run: train.TrainConfig) -> list[Headline]:
    """References, then each double-indexing mode's headline from the seed-0 sweep."""
    found = [
        Headline("random", f"top_k={TOP_K}", None, RandomEntries(TOP_K)),
        Headline("oracle", f"top_k={TOP_K}", None, Oracle(TOP_K)),
    ]
    n_keys = RouterTraining.from_env().n_keys
    sweep = read(eval_route.OUT).filter(
        (pl.col("T") == run.seq_len) & (pl.col("seed") == 0)
    )
    for mode in ("mean", "bound", "learned", "mixed"):
        label = f"{mode}_r{n_keys}" if mode in ("learned", "mixed") else mode
        row = headline(sweep.filter(pl.col("coarse") == label))
        wanted = str(row["S"]), str(row["s"])
        grid = eval_route.route_grid(mode, model, run, n_keys)
        routes = next(r for r in grid if eval_route.describe(r) == wanted)
        text = f"S={wanted[0]}, s={wanted[1]}"
        found.append(
            Headline(f"double indexing {mode}", text, row["reads_ratio"], routes)
        )
    return found


def prepared(
    model: PreTrainedModel, run: train.TrainConfig, found: Headline
) -> LayerSelectors:
    """The headline's selector, after loading its routers into the model if any."""
    if not isinstance(found.selector, (RandomEntries, Oracle)):
        n_keys = RouterTraining.from_env().n_keys
        eval_route.with_routers_for(model, run, found.selector, n_keys)  # type: ignore[arg-type]
    return found.selector


def main(run: train.TrainConfig, smoke: bool) -> None:
    """Evaluate pure CSA2 and every headline; see the module docstring."""
    model = train.load_model("csa2", train.run_name("csa2", run).with_suffix(".pt"))
    val = train.load("validation")
    if smoke:
        val = val[: 2 * run.batch * run.seq_len + 1]
    out = OUT.with_stem(OUT.stem + train.seed_suffix(run.seed) + "_smoke" * smoke)
    results = Results(out, {"T": run.seq_len})
    reference = batch_losses(model, val, run, None)
    logger.info("pure CSA2: %.5f", reference.mean())
    for found in headlines(model, run):
        losses = batch_losses(model, val, run, prepared(model, run, found))
        delta, se = paired_delta(losses, reference)
        row = {"T": run.seq_len, "method": found.method, "config": found.config}
        row |= {
            "reads_ratio": found.reads_ratio,
            "delta": delta,
            "se": se,
            "n_batches": len(reference),
        }
        logger.info("%s", row)
        results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(train.TrainConfig.from_env(), "smoke" in sys.argv)
