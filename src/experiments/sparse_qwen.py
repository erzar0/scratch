"""Sparse decode attentions in the original Qwen3-0.6B-Base (src/sparse/).

Rows name a method and its settings, joined by "-":
    dense                                    plain SDPA (the reference)
    exact-<k>                                exact top-k
    sparq-<r>-<k>                            SparQ
    loki-<pre|post>-<d>-<k>                  Loki
    tournament-<order>-<rounds>-<losers>-<keep>
    blocks-<block>-<keep>                    tournament, halving inside blocks
    hybrid-<block>-<rounds>-<keep>           first rounds inside blocks, then global
    first-<dims>-<losers>-<keep>             global halving, first round on <dims> dims
                                             (losers drop, approx or meanv)
    sampled-<dims>-<losers>-<keep>           the same, sampled-median cutoffs
    groups-<grouping>-<dims>-<losers>-<keep> the same, exact cutoffs, dims read in
                                             fixed groups of 8 (pairs or contiguous)
    shuffled-<tournament or blocks row>      that row on keys in a random order
    bandit-<exact|gauss>-<width>-<keep>      bound bandit, |q| order, radius × width

    uv run python -m experiments.sparse_qwen calibrate
        Loki's PCA bases from 2 windows of LONG WikiText-103 train tokens
        -> runs/sparse/loki_pca.pt
    uv run python -m experiments.sparse_qwen pg19 <row>...
        the first LONG tokens of the PG-19 test books longer than LONG: loss on the
        last SCORED positions, whose attention runs the row's method
        -> runs/sparse/pg19.csv (per book), runs/sparse/layers.csv (per layer)
LONG (32768), SCORED (1024) and LIMIT (16 books) come from the environment.
"""

import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.results import Results
from model import train
from qwen import data
from qwen import model as qm
from sparse import attend
from sparse.block_tournament import BlockTournament
from sparse.bound_bandit import BoundBandit
from sparse.first_round import FirstRound
from sparse.group_order import GroupOrder
from sparse.hybrid_tournament import HybridTournament
from sparse.loki import Loki, calibrate
from sparse.mean_losers import MeanLosers
from sparse.sampled_halving import SampledHalving
from sparse.shuffled import ShuffledBlocks, ShuffledTournament
from sparse.sparq import SparQ
from sparse.tournament import Tournament

logger = logging.getLogger(__name__)

OUT = Path(os.environ.get("SPARSE_OUT", "runs/sparse"))
LONG = int(os.environ.get("LONG", 32768))
SCORED = int(os.environ.get("SCORED", 1024))
BOOKS = int(os.environ.get("LIMIT", 16))
METHODS = {"exact": attend.ExactTopK, "sparq": SparQ, "loki": Loki}
SHUFFLED = {Tournament: ShuffledTournament, BlockTournament: ShuffledBlocks}


def method(row: str) -> attend.Method | None:
    """The row's method; None for dense."""
    name, *settings = row.split("-")
    if name == "dense":
        return None
    args = [int(s) if s.isdigit() else s for s in settings]
    if name == "shuffled":
        inner = method("-".join(settings))
        return SHUFFLED[type(inner)](**asdict(inner))  # type: ignore[call-overload,index]
    if name == "blocks":
        return BlockTournament(block=int(args[0]), keep=int(args[1]))
    if name == "first":
        width, losers, kept = settings
        kind = MeanLosers if losers == "meanv" else FirstRound
        return kind(first=int(width), losers=losers, keep=int(kept))
    if name == "sampled":
        width, losers, kept = settings
        return SampledHalving(first=int(width), losers=losers, keep=int(kept))
    if name == "groups":
        grouping, width, losers, kept = settings
        return GroupOrder(
            grouping=grouping, first=int(width), losers=losers, keep=int(kept)
        )
    if name == "bandit":
        bound, width, kept = settings
        return BoundBandit(bound=bound, width=float(width), keep=int(kept))
    if name == "hybrid":
        block, rounds, keep = map(int, args)
        return HybridTournament(block=block, block_rounds=rounds, keep=keep)
    return Tournament(*args) if name == "tournament" else METHODS[name](*args)  # type: ignore[arg-type]


@torch.no_grad()
def pg19(row: str) -> None:
    """One row on the PG-19 books; see the module docstring."""
    model = qm.load()
    chosen = method(row)
    state = None if chosen is None else attend.attach(model, chosen, SCORED)
    books = Results(OUT / "pg19.csv", {"row": row})
    for i, book in enumerate([b for b in data.books() if len(b) > LONG][:BOOKS]):
        start = time.perf_counter()
        tokens = train.to_device(book[None, : LONG + 1])
        hidden = model.model(input_ids=tokens[:, :-1]).last_hidden_state[0, -SCORED:]
        logits = model.lm_head(hidden).float()
        loss = float(F.cross_entropy(logits, tokens[0, -SCORED:]))
        books.add({"row": row, "book": i, "ce": loss})
        logger.info(
            "[%s] book %d: %.4f (%.0f s)", row, i, loss, time.perf_counter() - start
        )
    if state is None:
        return
    layers = Results(OUT / "layers.csv", {"row": row})
    for layer, sums in sorted(state.sums.items()):
        count = sums.pop("count")
        means = {name: sums[name] / count for name in ("recall", "mass", "error")}
        ratios = {
            name: sums[name]
            / sums["dense_sectors" if name.startswith("sectors") else "dense_reads"]
            for name in sums
            if name == "reads" or name.startswith("sectors_")
        }
        layers.add({"row": row, "layer": layer} | means | ratios)


def calibrate_loki() -> None:
    """Loki's PCA bases; see the module docstring."""
    tokens = data.load("train")[: 2 * LONG].reshape(2, LONG)
    calibrate(qm.load(), train.to_device(tokens))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    study, rows = sys.argv[1], sys.argv[2:]
    if study == "calibrate":
        calibrate_loki()
    for name in rows:
        {"pg19": pg19}[study](name)
