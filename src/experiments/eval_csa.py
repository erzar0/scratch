"""E0, pure CSA2: the trained CSA2 LM zero-shot under other top_k, window and block
selectors (its own indexer, random entries, attention oracle).

    SEQ=2048 BATCH=2 uv run python -m experiments.eval_csa [smoke]
    -> runs/csa/csa_loss.csv
    SEQ=8192 BATCH=1 uv run python -m experiments.eval_csa

Rows for SEQ and SEED are replaced. reads_per_seq: indexer-key equivalents per
sequence and Full or Reindex layer (experiments/selector_forward.py). smoke: two
windows, one config per selector, rows to csa_loss_smoke.csv.
"""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl
from transformers import PreTrainedModel

from deepseek import csa, csa2
from deepseek.reference import Oracle, RandomEntries
from experiments.results import Results, read
from experiments.selector_forward import (
    batch_losses,
    reads_per_seq,
    selecting_layers,
)
from model import train

logger = logging.getLogger(__name__)

OUT = Path("runs/csa/csa_loss.csv")
TOP_K, WINDOW = csa.SMALL["index_topk"], csa.SMALL["sliding_window"]
TOP_KS, WINDOWS = (2, 4, 8, 16, 32), (1, 16, 32, 64)
CORNERS = [(2, 1), (2, 64), (32, 1), (32, 64)]
SELECTOR_TOP_KS = (2, 8)


def config_grid() -> list[tuple[int, int]]:
    """(top_k, window): one axis at a time around the trained (8, 32), plus corners."""
    pairs = [(k, WINDOW) for k in TOP_KS] + [(TOP_K, w) for w in WINDOWS]
    return list(dict.fromkeys(pairs + CORNERS))


@contextmanager
def configured(model: PreTrainedModel, top_k: int, window: int) -> Iterator[None]:
    """Run every indexer at top_k and every layer at window, then restore them."""
    attentions = [layer.self_attn for layer in model.model.layers]
    indexers = [a.indexer for a in selecting_layers(model) if a.indexer is not None]
    saved = [a.sliding_window for a in attentions], [i.index_topk for i in indexers]
    for attention in attentions:
        attention.sliding_window = window
    for indexer in indexers:
        indexer.index_topk = top_k
    try:
        yield
    finally:
        for attention, old_window in zip(attentions, saved[0], strict=True):
            attention.sliding_window = old_window
        for indexer, old_top_k in zip(indexers, saved[1], strict=True):
            indexer.index_topk = old_top_k


def main_read_cost(model: PreTrainedModel) -> float:
    """Multiply-adds of one main-attention key read (all heads) in indexer-key reads."""
    config = model.config
    main = config.num_attention_heads * config.head_dim
    return main / (config.index_n_heads * config.index_head_dim)


def flat_reference(run: train.TrainConfig) -> tuple[float, float]:
    """Pure CSA2's (val loss, reads per sequence) at its trained top_k and window.

    Raises:
        SystemExit: If OUT has no such row for this run (run this module first).
    """
    frame = read(OUT).filter(
        (pl.col("T") == run.seq_len)
        & (pl.col("seed") == run.seed)
        & (pl.col("selector") == "flat")
        & (pl.col("top_k") == TOP_K)
        & (pl.col("window") == WINDOW)
    )
    if frame.is_empty():
        logger.error("no flat row for %s in %s: run experiments.eval_csa", run, OUT)
        raise SystemExit(1)
    return float(frame["val_loss"][0]), float(frame["reads_per_seq"][0])


def main(run: train.TrainConfig, smoke: bool) -> None:
    """Evaluate every config and selector; see the module docstring."""
    model = train.load_model("csa2", train.run_name("csa2", run).with_suffix(".pt"))
    val = train.load("validation")
    grid = config_grid()
    if smoke:
        val, grid = val[: 2 * run.batch * run.seq_len + 1], [(TOP_K, WINDOW)]
    flat_reads = reads_per_seq(model, run.seq_len, None)
    oracle_reads = flat_reads * main_read_cost(model)
    runs: list[tuple[str, csa2.Selector | None, int, int, float]] = [
        ("flat", None, top_k, window, flat_reads) for top_k, window in grid
    ]
    for top_k in SELECTOR_TOP_KS[: 1 if smoke else None]:
        runs += [
            ("random", RandomEntries(top_k), top_k, WINDOW, 0.0),
            ("oracle", Oracle(top_k), top_k, WINDOW, oracle_reads),
        ]
    out = OUT.with_stem(OUT.stem + "_smoke") if smoke else OUT
    results = Results(out, {"T": run.seq_len, "seed": run.seed})
    for name, selector, top_k, window, n_reads in runs:
        with configured(model, top_k, window):
            loss = float(batch_losses(model, val, run, selector).mean())
        row = {"T": run.seq_len, "seed": run.seed, "selector": name}
        row |= {"top_k": top_k, "window": window, "reads_per_seq": n_reads}
        row |= {"val_loss": loss}
        logger.info("%s", row)
        results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(train.TrainConfig.from_env(), "smoke" in sys.argv)
