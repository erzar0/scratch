"""E1: double indexing (deepseek/double_indexing.py) on the trained CSA2 LM,
zero-shot: val loss vs indexer reads.

mode is the coarse score: mean (default), bound, oracle (reads every entry: a
diagnostic), learned (routers from experiments/train_router.py; S without one are
skipped), mixed (learned in most layers, a bigger bound route in "needle" layers,
those whose output moves most under the learned route: relative squared error
above the mean over selecting layers, on 2 train windows; reads are the mean over
layers).

    SEQ=2048 BATCH=2 uv run python -m experiments.eval_route [mode] [smoke]
    -> runs/route/route_loss.csv (rows for SEQ, SEED and mode replaced)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel

from deepseek.double_indexing import DoubleIndexing
from experiments.eval_csa import flat_reference
from experiments.results import Results
from experiments.selector_forward import (
    batch_losses,
    reads_per_seq,
    selecting,
    selecting_layers,
)
from experiments.train_router import RouterTraining, load_routers, router_path
from model import train

logger = logging.getLogger(__name__)

OUT = Path("runs/route/route_loss.csv")
GRIDS = {
    2048: [(size, keep) for size in (8, 16, 32) for keep in (1, 2, 4, 8)],
    8192: [(size, keep) for size in (16, 32, 64) for keep in (2, 4, 8)],
}
# ((S, s) learned in most layers, (S, s) bound in needle layers)
MIXED = {
    2048: [((8, keep), (8, wide)) for keep in (1, 2) for wide in (8, 16)],
    8192: [
        ((16, keep), needle)
        for keep in (2, 4)
        for needle in ((16, 8), (16, 16), (16, 32), (32, 16))
    ],
}

type LayerRoutes = DoubleIndexing | tuple[DoubleIndexing | None, ...]


@torch.no_grad()
def needle_layers(
    model: PreTrainedModel, tokens: torch.Tensor, route: DoubleIndexing
) -> list[int]:
    """Selecting layers whose attention output moves more than the mean over
    selecting layers (relative squared error) under route alone (every other layer
    flat, so its input is the flat one), on tokens (B, S)."""
    outputs: dict[object, torch.Tensor] = {}  # by layer_idx

    def keep(module: torch.nn.Module, _: object, output: tuple) -> None:
        outputs[module.layer_idx] = output[0].float()

    layers = selecting_layers(model)
    hooks = [a.register_forward_hook(keep) for a in layers]
    try:
        model.model(input_ids=tokens, use_cache=False)
        flat = dict(outputs)
        errors = {}
        for attention in layers:
            index = attention.layer_idx
            n_layers = len(model.model.layers)
            routes = tuple(route if i == index else None for i in range(n_layers))
            with selecting(model, routes):
                model.model(input_ids=tokens, use_cache=False)
            error = (outputs[index] - flat[index]).square().sum() / flat[
                index
            ].square().sum()
            errors[index] = float(error)
    finally:
        for hook in hooks:
            hook.remove()
    logger.info("calibration: relative output error per layer %s", errors)
    mean = sum(errors.values()) / len(errors)
    return [index for index, error in errors.items() if error > mean]


def route_grid(
    mode: str, model: PreTrainedModel, run: train.TrainConfig, n_keys: int
) -> list[LayerRoutes]:
    """Routes this mode evaluates; learned and mixed need trained routers (loaded
    into the model by with_routers_for)."""
    seq_len = run.seq_len
    if mode != "mixed":
        return [
            DoubleIndexing(size, keep, mode)  # type: ignore[arg-type]
            for size, keep in GRIDS[seq_len]
            if mode != "learned"
            or router_path(seq_len, size, n_keys, run.seed).exists()
        ]
    size = MIXED[seq_len][0][0][0]
    learned = DoubleIndexing(size, 4, "learned")
    load_routers(model, seq_len, size, n_keys, run.seed)
    tokens = train.load("train")[: 2 * seq_len].reshape(2, seq_len)
    needle = needle_layers(model, train.to_device(tokens), learned)
    n_layers = len(model.model.layers)
    return [
        tuple(
            DoubleIndexing(*wide, "bound")
            if index in needle
            else DoubleIndexing(*narrow, "learned")
            for index in range(n_layers)
        )
        for narrow, wide in MIXED[seq_len]
    ]


def describe(routes: LayerRoutes) -> tuple[str, str]:
    """(S, s) columns; mixed: 'most layers/needle layers'."""
    if isinstance(routes, DoubleIndexing):
        return str(routes.superblock_size), str(routes.kept_superblocks)
    chosen = [r for r in routes if r is not None]
    learned = next(r for r in chosen if r.coarse == "learned")
    wide = next((r for r in chosen if r.coarse == "bound"), learned)
    sizes = f"{learned.superblock_size}/{wide.superblock_size}"
    return sizes, f"{learned.kept_superblocks}/{wide.kept_superblocks}"


def with_routers_for(
    model: PreTrainedModel, run: train.TrainConfig, routes: LayerRoutes, n_keys: int
) -> None:
    """Load the trained routers when a layer is learned."""
    layers = routes if isinstance(routes, tuple) else (routes,)
    learned = next((r for r in layers if r is not None and r.coarse == "learned"), None)
    if learned is not None:
        load_routers(model, run.seq_len, learned.superblock_size, n_keys, run.seed)


def main(mode: str, run: train.TrainConfig, smoke: bool) -> None:
    """Evaluate every route of one mode; see the module docstring."""
    n_keys = RouterTraining.from_env().n_keys
    model = train.load_model("csa2", train.run_name("csa2", run).with_suffix(".pt"))
    val = train.load("validation")
    flat_loss, flat_reads = flat_reference(run)
    grid = route_grid(mode, model, run, n_keys)
    if smoke:
        val, grid = val[: 2 * run.batch * run.seq_len + 1], grid[:1]
    label = f"{mode}_r{n_keys}" if mode in ("learned", "mixed") else mode
    out = OUT.with_stem(OUT.stem + "_smoke") if smoke else OUT
    results = Results(out, {"T": run.seq_len, "seed": run.seed, "coarse": label})
    for routes in grid:
        with_routers_for(model, run, routes, n_keys)
        loss = float(np.mean(batch_losses(model, val, run, routes)))
        n_reads = reads_per_seq(model, run.seq_len, routes, n_keys)
        size, keep = describe(routes)
        row = {"T": run.seq_len, "seed": run.seed, "coarse": label}
        row |= {"S": size, "s": keep}
        row |= {"reads_per_seq": n_reads, "reads_ratio": flat_reads / n_reads}
        row |= {"val_loss": loss, "delta_vs_flat": loss - flat_loss}
        logger.info("%s", row)
        results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = [a for a in sys.argv[1:] if a != "smoke"]
    main(args[0] if args else "mean", train.TrainConfig.from_env(), "smoke" in sys.argv)
