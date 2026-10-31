"""E5: the per-key bandit (attention/dense/key_bandit.py) on real heads of the trained
softmax T = 1024 LM: output relative error vs reads (exact-logit equivalents per
query; exact causal attention averages (T + 1) / 2), and val loss with every head
swapped for it (delta_se: paired standard error over batches). "loss" skips the
error sweep.

    uv run python -m experiments.sweep_key_bandit [loss] [smoke]
    -> runs/key_bandit/key_error.csv, runs/key_bandit/key_loss.csv
"""

import dataclasses
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from attention.dense.key_bandit import (
    KeyBanditConfig,
    exact_causal,
    key_bandit,
    key_bandit_cost,
)
from experiments.results import Results, paired_delta
from model import train
from model.heads import SEQ_LEN, WEIGHTS, get_heads
from model.transformer import ForwardOptions

logger = logging.getLogger(__name__)

OUT = Path("runs/key_bandit")
SEEDS = 3
GRID = [
    KeyBanditConfig(width, tolerance, step)
    for width in (0.5, 1.0, 2.0, 4.0)
    for tolerance in (1.0, 3.0, 10.0)
    for step in (4, 8, 16)
]
LOSS_GRID = [  # error-front configs, cheapest first
    KeyBanditConfig(width, tolerance, step)
    for width, tolerance, step in (
        (0.5, 3.0, 4),
        (0.5, 3.0, 8),
        (1.0, 1.0, 8),
        (1.0, 3.0, 8),
        (2.0, 1.0, 4),
        (1.0, 3.0, 16),
        (2.0, 1.0, 8),
        (2.0, 3.0, 8),
        (4.0, 3.0, 8),
    )
]
EVAL_WINDOWS = 64


@dataclasses.dataclass(frozen=True)
class KeyHead:
    """key_bandit as a head attention (fixed channel order), hashable for jit."""

    config: KeyBanditConfig

    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        return key_bandit(q, k, v, jax.random.PRNGKey(0), self.config)


_cost = jax.jit(key_bandit_cost, static_argnames="config")


def error_sweep(grid: list[KeyBanditConfig]) -> None:
    """Relative error and reads of every config on every head and seed."""
    results = Results(OUT / "key_error.csv", {})
    for head in get_heads():
        q, k, v = (jnp.asarray(a, jnp.float32) for a in (head.q, head.k, head.v))
        exact = exact_causal(q, k, v)
        for config in grid:
            for seed in range(SEEDS):
                order = jax.random.permutation(jax.random.PRNGKey(seed), q.shape[1])
                out, reads = _cost(q, k, v, order, config)
                error = float(jnp.linalg.norm(out - exact) / jnp.linalg.norm(exact))
                row = {"layer": head.layer, "head": head.head, "seed": seed}
                row |= dataclasses.asdict(config) | {
                    "reads": float(reads),
                    "rel_err": error,
                }
                results.add(row)
        logger.info("layer %d head %d done", head.layer, head.head)


def loss_sweep(grid: list[KeyBanditConfig], limit: int) -> None:
    """Val loss of the T = 1024 model with every head swapped for the bandit."""
    run = train.TrainConfig(seq_len=SEQ_LEN, batch=1)  # batch 4 OOMs at 4 dims/round
    params = train.load_params(WEIGHTS)
    val = train.load("validation")
    results = Results(OUT / "key_loss.csv", {})
    heads = [None, *(KeyHead(c) for c in grid)]
    exact_losses = None  # per batch; exact is the first head
    for head in heads:
        options = ForwardOptions(n_heads=run.n_heads, head_attention=head)
        losses = np.asarray(
            [
                float(train.eval_loss(params, batch, options))
                for batch in train.windows(val, run.seq_len, run.batch, limit)
            ]
        )
        if exact_losses is None:
            exact_losses = losses
        delta, delta_se = paired_delta(losses, exact_losses)
        row: dict[str, object] = {"method": "key bandit" if head else "exact"}
        row |= dataclasses.asdict(head.config) if head else {}
        row |= {"val_loss": float(losses.mean()), "delta_vs_exact": delta}
        row |= {"delta_se": delta_se}
        logger.info("%s", row)
        results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    smoke = "smoke" in sys.argv
    if "loss" not in sys.argv:
        error_sweep(GRID[:1] if smoke else GRID)
    loss_sweep(LOSS_GRID[:1] if smoke else LOSS_GRID, 1 if smoke else EVAL_WINDOWS)
