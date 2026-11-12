"""Per-token loss on the validation rows exactly as record 91 validates (8192-token
rows, unaligned, a segment per row start and per BOS), binned by the token's position
in its segment. No canonical-token mask (only the final validation uses it).

python packed_val.py SCRIPT RUN_ID ROWS OUT.csv
"""

import csv
import os
import sys
from itertools import pairwise

import torch
from nanogpt_old import model as nm

ROW = 8192
EDGES = (0, 128, 512, 1024, 2048, 2560, 4096, ROW)


@torch.no_grad()
def main(script: str, run: str, rows: str, out: str) -> None:
    nm.SCRIPT = script
    (checkpoint,) = (nm.R91 / "logs" / run).glob("state_step*.pt")
    model, config = nm.load(checkpoint)
    if int(os.environ.get("EVAL_COMPILE", 0)):  # ~8x faster with flex attention
        model = torch.compile(model, dynamic=False)
    tg = nm.module()
    shard = sorted(nm.FINEWEB.glob("fineweb_val_*.bin"))[0]
    tokens = tg._load_data_shard(shard)
    idx = torch.arange(ROW)
    loss_sum = torch.zeros(len(EDGES) - 1, dtype=torch.float64)
    count = torch.zeros(len(EDGES) - 1, dtype=torch.float64)
    kept = []
    for r in range(int(rows)):
        buf = tokens[r * ROW : (r + 1) * ROW + 1]
        inputs, targets = buf[:-1], buf[1:]
        bos = torch.nonzero(inputs == tg.BOS_ID)[:, 0]
        cu = torch.full((128,), ROW)
        cu[0] = 0
        cu[1 : len(bos) + 1] = bos
        losses = (
            model(
                inputs.int().cuda(),
                targets.long().cuda(),
                cu.int().cuda(),
                tg.get_bigram_hash(inputs.int()).cuda(),
                config,
            )
            .double()
            .cpu()
        )
        start = torch.where(inputs == tg.BOS_ID, idx, 0).cummax(0).values
        pos = idx - start
        first = (
            start == 0 if inputs[0] != tg.BOS_ID else torch.zeros(ROW, dtype=torch.bool)
        )
        kept.append(torch.stack([losses, pos.double(), idx.double(), first.double()]))
        for i, (a, b) in enumerate(pairwise(EDGES)):
            m = (pos >= a) & (pos < b)
            loss_sum[i] += losses[m].sum()
            count[i] += m.sum()
    torch.save(torch.stack(kept), out.removesuffix(".csv") + "_tokens.pt")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "start", "end", "tokens", "loss"])
        for i, (a, b) in enumerate(pairwise(EDGES)):
            w.writerow(
                [run, a, b, int(count[i]), f"{float(loss_sum[i] / count[i]):.5f}"]
            )
        w.writerow(
            [
                run,
                0,
                ROW,
                int(count.sum()),
                f"{float(loss_sum.sum() / count.sum()):.5f}",
            ]
        )
    print(open(out).read(), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
