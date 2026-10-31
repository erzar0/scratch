"""Pick recall: how many of pure CSA2's entries each headline method also picks.

Every selecting layer gets the frozen pure-CSA2 model's own inputs
(train_router.record), so only the selection differs between methods. Per method
and Full or Reindex layer, over the queries of WINDOWS val windows that have more
than top_k finished entries (with fewer, every method picks them all):

    pick_recall  share of pure CSA2's top_k picks the method also picks
    mass_recall  share of pure CSA2's attention weight on its picks that those carry

The weight is the head mean over the layer and its Reuse followers. Methods and
configs are paired_se's headlines. The last query of the first window is also saved
for the pick strip: per layer, the indexer score of every finished entry
("score|<layer>") and the entries each method picks ("<method>|<layer>").

    SEQ=2048 uv run python -m experiments.pick_recall
    -> runs/paper/pick_recall.csv (rows for SEQ and SEED replaced),
       runs/paper/pick_strip_T<SEQ>.npz
"""

import logging
from typing import Any

import numpy as np
import torch

from deepseek import csa2
from experiments import paired_se
from experiments.eval_csa import TOP_K
from experiments.results import Results
from experiments.selector_forward import layer_selector
from experiments.train_router import record
from model import train

logger = logging.getLogger(__name__)

OUT = paired_se.OUT.with_name("pick_recall.csv")
WINDOWS = 8


def overlap(
    picks: csa2.Picks, flat: csa2.Picks, flat_weight: torch.Tensor
) -> np.ndarray:
    """Shared picks, flat picks, shared flat weight and flat weight, summed.

    Args:
        picks: A method's picks, (N, k) rows.
        flat: Pure CSA2's picks, (N, k) rows.
        flat_weight: (N, k) attention weight on pure CSA2's picks.

    Returns:
        The four sums (4,).
    """
    match = picks.entry[:, :, None] == flat.entry[:, None, :]
    shared = (match & picks.valid[:, :, None]).any(1) & flat.valid
    flat_weight = flat_weight * flat.valid
    return np.array(
        [
            float(shared.sum()),
            float(flat.valid.sum()),
            float((flat_weight * shared).sum()),
            float(flat_weight.sum()),
        ]
    )


def last_scores(
    attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs, finished: torch.Tensor
) -> np.ndarray:
    """Batch 0's last query's indexer score of every finished entry."""
    indexer = attention.indexer
    assert indexer is not None
    q = indexer.queries(inputs.hidden_states, inputs.q_residual, inputs.position_ids)
    keys = inputs.index_k[:, : int(finished[-1])]
    score = indexer.scorer(q[:1, -1:], keys[:1], inputs.hidden_states[:1, -1:])
    return score[0, 0].cpu().numpy()


def rows_of(picks: csa2.Picks, rows: torch.Tensor) -> csa2.Picks:
    """Batch 0's picks at the given query rows."""
    return csa2.Picks(*(x[0, rows] for x in picks))


@torch.no_grad()
def main(run: train.TrainConfig) -> None:
    """Pick and mass recall per method and layer; see the module docstring."""
    model = train.load_model("csa2", train.run_name("csa2", run).with_suffix(".pt"))
    methods: list[tuple[str, paired_se.Headline | None]] = [("pure CSA2", None)]
    methods += [(h.method, h) for h in paired_se.headlines(model, run)]
    val = train.load("validation")
    layers = [layer.self_attn for layer in model.model.layers]
    totals: dict[tuple[str, int], np.ndarray] = {}
    strip: dict[str, Any] = {}  # Any: np.savez's stub reads ** as allow_pickle
    # Method by method: loading one method's routers replaces another's.
    for name, found in methods:
        selector = None if found is None else paired_se.prepared(model, run, found)
        for window in range(WINDOWS):
            start = window * run.seq_len
            tokens = train.to_device(val[start : start + run.seq_len][None])
            for index, (inputs, flat, weight) in record(model, tokens).items():
                attention = layers[index]
                assert attention.indexer is not None
                m = attention.indexer.compress_rate
                finished = csa2.finished_entries(inputs.position_ids, m)[0, :, 0]
                rows = finished > TOP_K
                chosen = layer_selector(selector, index)
                picks = flat if chosen is None else chosen(attention, inputs)
                totals[name, index] = totals.get((name, index), 0) + overlap(
                    rows_of(picks, rows), rows_of(flat, rows), weight[0, rows]
                )
                if window > 0:
                    continue
                last = rows_of(picks, torch.tensor([-1]))
                strip[f"{name}|{index}"] = last.entry[last.valid].cpu().numpy()
                if found is None:
                    strip[f"score|{index}"] = last_scores(attention, inputs, finished)
        logger.info("%s done", name)
    results = Results(OUT, {"T": run.seq_len, "seed": run.seed})
    indices = sorted({index for _, index in totals})
    for name, _ in methods:
        per_layer = [(str(i), totals[name, i]) for i in indices]
        everything = np.sum([totals[name, i] for i in indices], 0)
        for label, (shared, picks, shared_mass, mass) in [
            *per_layer,
            ("all", everything),
        ]:
            row = {"T": run.seq_len, "seed": run.seed, "method": name, "layer": label}
            row |= {"pick_recall": shared / picks, "mass_recall": shared_mass / mass}
            logger.info("%s", row)
            results.add(row)
    name = f"pick_strip_T{run.seq_len}{train.seed_suffix(run.seed)}.npz"
    np.savez(OUT.with_name(name), **strip)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(train.TrainConfig.from_env())
