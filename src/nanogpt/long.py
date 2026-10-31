"""Zero-shot loss by position on long FineWeb val documents, for record 91 checkpoints
(dense, stripped dense, CSA2): does any of them use context past the 2048-token
documents record 91 trains on? Dense attends within its sliding windows; CSA2 picks
blocks from the whole document. The history study scores the same last SPAN tokens
of each document after more and more of the document before them: more context should
help, never hurt.

PYTHONPATH=src ~/repos/modded-nanogpt-r91/.venv/bin/python \
    -m nanogpt.long SCRIPT RUN_ID [TOPK]
    -> runs/nanogpt/long.csv, history.csv
TOPK: blocks each CSA2 query picks (trained: 64)
"""

import sys
from itertools import pairwise
from pathlib import Path

import torch

from experiments.results import Results
from nanogpt import model as nm

RUNS = Path("runs/nanogpt")
# a run_r91.sh checkpoint's YaRN tables and flash-attn max_seqlen stop at 8192 positions
LENGTH = 8192
DOCS = 64
EDGES = (0, 128, 512, 1024, 2048, 4096, LENGTH)
SPAN = 512
# tokens of the document before the scored span; a cut document restarts with BOS
HISTORIES = (0, 256, 1024, 2048, 4096, LENGTH - SPAN)


@torch.no_grad()
def main(script: str, run: str, topk: str = "") -> None:
    """Mean loss per position range over the first LENGTH tokens of DOCS val documents,
    for the run's final checkpoint; topk, if given, overrides every CSA2 indexer's."""
    nm.SCRIPT = script
    (checkpoint,) = (nm.R91 / "logs" / run).glob("state_step*.pt")
    model, config = nm.load(checkpoint)
    if topk:
        indexers = [m for m in model.modules() if hasattr(m, "index_topk")]
        assert indexers, "topk needs a CSA2 checkpoint"
        for indexer in indexers:
            setattr(indexer, "index_topk", int(topk))  # noqa: B010 (Module typing)
    tg = nm.module()
    shards = sorted(nm.FINEWEB.glob("fineweb_val_*.bin"))
    tokens = torch.cat([tg._load_data_shard(f).int() for f in shards])
    windows = nm.first_tokens(tokens, tg.BOS_ID, LENGTH + 1)[:DOCS]
    assert len(windows) == DOCS, len(windows)
    losses = torch.stack(
        [nm.token_losses(model, config, w).float().cpu() for w in windows]
    ).mean(0)
    results = Results(RUNS / "long.csv", {"run": run, "topk": topk})
    for start, end in pairwise(EDGES):
        loss = float(losses[start:end].mean())
        results.add(
            {"run": run, "script": script, "topk": topk, "start": start, "end": end}
            | {"loss": loss}
        )
    print(run, topk, [f"{float(losses[a:b].mean()):.3f}" for a, b in pairwise(EDGES)])

    results = Results(RUNS / "history.csv", {"run": run, "topk": topk})
    for history in HISTORIES:
        span = []
        for w in windows:
            # BOS (w[0]) + the last SPAN + history tokens: a multiple of 16 inputs, as
            # dense attention needs; the whole window at the longest history
            cut = torch.cat([w[:1], w[LENGTH + 1 - SPAN - history :]])
            span.append(nm.token_losses(model, config, cut).float().cpu()[-SPAN:])
        loss = float(torch.stack(span).mean())
        results.add(
            {"run": run, "script": script, "topk": topk, "history": history}
            | {"loss": loss}
        )
        print(run, topk, "history", history, f"{loss:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
