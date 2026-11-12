"""Appends one eval's token-group losses (packed_val.py's OUT_tokens.pt) to
runs/nanogpt/val_groups.csv, as rows MODEL, RUN, group, tokens, share, loss.

python groups.py MODEL RUN TOKENS.pt
"""

import csv
import sys
from pathlib import Path

import torch

OUT = Path.home() / "repos/scratch/runs/nanogpt/checkpoints/val_groups.csv"


def main(model: str, run: str, path: str) -> None:
    t = torch.load(path)  # (rows, 4, 8192): loss, position, index, first segment
    loss, pos, first = t[:, 0], t[:, 1], t[:, 3].bool()
    short = pos < 2048
    groups = {
        "every token": torch.ones_like(first),
        "document from its start, position < 2048": ~first & short,
        "row starts mid-document (no BOS), position < 2048": first & short,
        "position 2048 or later": ~short,
    }
    with OUT.open("a", newline="") as f:
        w = csv.writer(f)
        for group, m in groups.items():
            n = int(m.sum())
            w.writerow(
                [
                    model,
                    run,
                    group,
                    n,
                    f"{n / m.numel():.4f}",
                    f"{float(loss[m].mean()):.4f}",
                ]
            )
    print(f"{model}: {float(loss.mean()):.4f} on every token", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
