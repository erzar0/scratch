"""E2 / E6 / E7: bandit block selectors on the trained pure-CSA LM, zero-shot.

    superblock  attention/selectors/superblock_bandit.py: superblocks are arms, pulls
                read block keys
    channel     attention/selectors/channel_bandit.py: blocks are arms, pulls read
                channels
    token       attention/selectors/token_bandit.py: blocks are arms, pulls read raw
                tokens with the main query (replaces the learned indexer)

Stages: 1 the grid; 2 (superblock only) ucb_width and second_pulls around 5
stage-1 configs spread evenly along the reads/loss Pareto front; 3 two more seeds
of every front config. reads_ratio = flat reads / reads, flat from
runs/csa/csa_loss.csv.

    SEQ=2048 BATCH=2 uv run python -m experiments.eval_csa_bandit superblock 1 [smoke]
    -> runs/csa_bandit/{study}.csv (rows for SEQ and stage replaced)
"""

import dataclasses
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import polars as pl

from attention.csa import CsaConfig
from attention.selectors import superblock_bandit
from attention.selectors.channel_bandit import ChannelBandit
from attention.selectors.superblock_bandit import SuperblockBandit
from attention.selectors.token_bandit import TokenBandit
from experiments.results import Results, pareto_front, read
from experiments.selector_forward import evaluate, load_csa_model
from model import train

logger = logging.getLogger(__name__)

OUT = Path("runs/csa_bandit")
FLAT_RESULTS = Path("runs/csa/csa_loss.csv")
BLOCK_SIZE, TOP_K = 4, 8
MIN_READS_RATIO = 2.5  # stage-1 superblock configs must read this much less than flat
FRONT_SIZE, SEEDS = 5, (1, 2)


def superblock_grid(seq_len: int) -> list[SuperblockBandit]:
    """S, s, first_pulls, survivors, second_pulls; MIN_READS_RATIO fewer reads."""
    finished = jnp.arange(seq_len) // BLOCK_SIZE
    routes = [
        SuperblockBandit(size, keep, first, keep * factor, second)
        for size in (8, 16, 32, 64)
        for keep in (1, 2, 4, 8)
        for first in (1, 2)
        for factor in (2, 4)
        for second in (size // 8, size // 4)
    ]
    ratio = [
        float(finished.sum() / superblock_bandit.reads(finished, r).sum())
        for r in routes
    ]
    return [r for r, x in zip(routes, ratio, strict=True) if x >= MIN_READS_RATIO]


def superblock_variants(route: SuperblockBandit) -> list[SuperblockBandit]:
    """Stage 2: ucb_width (0 = no UCB) and second_pulls (0 = no round 2)."""
    size = route.superblock_size
    widths = [dataclasses.replace(route, ucb_width=w) for w in (0.0, 0.5, 2.0, 4.0)]
    seconds = {0, size // 8, size // 4, size // 2} - {route.second_pulls}
    pulls = [
        dataclasses.replace(route, second_pulls=s)
        for s in sorted(seconds)
        if route.first_pulls + s <= size
    ]
    return widths + pulls


def channel_grid(_: int) -> list[ChannelBandit]:
    return [
        ChannelBandit(TOP_K, width, tolerance, step)
        for width in (0.25, 0.5, 1.0, 2.0)  # 4.0 read 1.00x of flat at T=2048
        for tolerance in (0.0, 0.5, 1.0, 2.0)
        for step in (4, 8, 16)
    ]


def token_grid(_: int) -> list[TokenBandit]:
    return [
        TokenBandit(TOP_K, first, second, TOP_K * factor, width)
        for first in (1, 2)
        for second in (0, 1, 2)
        for factor in (2, 4)
        for width in (0.0, 1.0, 2.0)
    ]


@dataclasses.dataclass(frozen=True)
class Study:
    """One bandit: its selector class, stage-1 grid and stage-2 variants."""

    settings: type
    grid: Callable[[int], list[Any]]
    variants: Callable[[Any], list[Any]] | None = None


STUDIES = {
    "superblock": Study(SuperblockBandit, superblock_grid, superblock_variants),
    "channel": Study(ChannelBandit, channel_grid),
    "token": Study(TokenBandit, token_grid),
}


def from_row(settings: type, row: dict[str, Any]) -> Any:
    """Rebuild a settings dataclass from a results row."""
    fields = dataclasses.fields(settings)
    return settings(**{f.name: f.type(row[f.name]) for f in fields})  # type: ignore[operator]


def front_configs(
    path: Path, seq_len: int, study: Study, stages: tuple[int, ...]
) -> list[Any]:
    """Seed-0 configs on the reads/loss front of earlier stages."""
    frame = read(path).filter(
        (pl.col("T") == seq_len) & pl.col("stage").is_in(stages) & (pl.col("seed") == 0)
    )
    front = pareto_front(frame, "reads_per_seq", "val_loss")
    return [from_row(study.settings, row) for row in front.to_dicts()]


def flat_reference(seq_len: int) -> tuple[float, float]:
    """Pure CSA's (val loss, reads) at top_k 8, window 32, from E0."""
    frame = read(FLAT_RESULTS).filter(
        (pl.col("T") == seq_len)
        & (pl.col("selector") == "flat")
        & (pl.col("top_k") == TOP_K)
        & (pl.col("window") == 32)
    )
    if frame.is_empty():
        logger.error(
            "no flat row for T=%d in %s: run experiments.eval_csa",
            seq_len,
            FLAT_RESULTS,
        )
        raise SystemExit(1)
    return float(frame["val_loss"][0]), float(frame["reads_per_seq"][0])


def stage_configs(name: str, stage: int, seq_len: int) -> list[Any]:
    """The configs a stage evaluates."""
    study, path = STUDIES[name], OUT / f"{name}.csv"
    if stage == 1:
        return study.grid(seq_len)
    if stage == 2:
        if study.variants is None:
            return []
        front = front_configs(path, seq_len, study, (1,))
        picks = sorted(
            {round(i * (len(front) - 1) / (FRONT_SIZE - 1)) for i in range(FRONT_SIZE)}
        )
        variants = (v for i in picks for v in study.variants(front[i]))
        return list(dict.fromkeys(variants))  # dedupe, keep order
    front = front_configs(path, seq_len, study, (1, 2))
    return [dataclasses.replace(config, seed=s) for config in front for s in SEEDS]


def main(name: str, stage: int, run: train.TrainConfig, smoke: bool) -> None:
    """Evaluate one stage of one study; see the module docstring."""
    params = load_csa_model(run)
    val = train.load("validation")
    flat_loss, flat_reads = flat_reference(run.seq_len)
    configs = stage_configs(name, stage, run.seq_len)
    if smoke:
        val, configs = val[: 2 * run.batch * run.seq_len + 1], configs[:1]
    results = Results(OUT / f"{name}.csv", {"T": run.seq_len, "stage": stage})
    logger.info("%s stage %d: %d configs", name, stage, len(configs))
    for config in configs:
        loss, reads = evaluate(params, val, run, config, CsaConfig(TOP_K))
        row = {"T": run.seq_len, "stage": stage, **dataclasses.asdict(config)}
        row |= {"reads_per_seq": reads, "reads_ratio": flat_reads / max(reads, 1e-9)}
        row |= {"val_loss": loss, "delta_vs_flat": loss - flat_loss}
        logger.info("%s", row)
        results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = [a for a in sys.argv[1:] if a != "smoke"]
    main(args[0], int(args[1]), train.TrainConfig.from_env(), "smoke" in sys.argv)
