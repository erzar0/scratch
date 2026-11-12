"""Loss gate for an eval rule: per-token loss minus exact picks on the same tokens
(rule_eval.py's logs/rule_<id8>_<tag>_tokens.pt), by position bin, with one standard
error over documents. Gate: all positions <= +0.005, every bin past 2048 <= +0.01.
BASE=<tag> compares against that rule instead of exact. From experiments/csa2:
    python eval/paired.py 1b0702e3 share04 block_mean_8x64
"""

import os
import sys
from pathlib import Path

import torch

LOGS = Path(__file__).resolve().parents[1] / "logs"
BINS = ((0, 8192), (2048, 2560), (2560, 4096), (4096, 8192))


def main(run: str, tags: list[str]) -> None:
    vs = os.environ.get("BASE", "exact")
    base = torch.load(LOGS / f"rule_{run}_{vs}_tokens.pt")  # (rows, 4, 8192)
    pos, idx = base[:, 1].flatten(), base[:, 2].flatten()
    row = torch.arange(base.shape[0]).repeat_interleave(base.shape[2]).double()
    doc = row * 1e5 + (idx - pos)  # one id per document
    for tag in tags:
        d = (
            torch.load(LOGS / f"rule_{run}_{tag}_tokens.pt")[:, 0] - base[:, 0]
        ).flatten()
        out = []
        for a, b in BINS:
            m = (pos >= a) & (pos < b)
            ids, inv = torch.unique(doc[m], return_inverse=True)
            sums = torch.zeros(len(ids), dtype=torch.float64).index_add_(0, inv, d[m])
            n = torch.zeros_like(sums).index_add_(0, inv, torch.ones_like(d[m]))
            mean = sums.sum() / n.sum()  # cluster-robust SE of a ratio estimator
            se = ((sums - mean * n).square().sum() * len(ids) / (len(ids) - 1)).sqrt()
            out.append(f"{a}-{b}: {mean:+.4f} ± {se / n.sum():.4f} ({len(ids)} docs)")
        print(f"{tag:26s}", " | ".join(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
