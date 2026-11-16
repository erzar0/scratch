"""Share of attention the learned sink takes, by layer and by the query's position in
its segment: sigmoid(sink - lse) per head and token (csa2_ref.attend scales each
head's output by 1 minus it), mean and largest head mean per position range.
Model eager, flex_attention compiled alone. Tests whether the sinks matter most at
early positions (few keys), where the parameter-free block (no sink) loses.
-> logs/sink_<id8>.csv (layer, start, end, mean, max head). From experiments/csa2:
    python eval/sink_share.py SCRIPT RUN ROWS
"""

import csv
import sys
from itertools import pairwise

import packed_val
import torch
from torch.nn.attention.flex_attention import flex_attention

from nanogpt import csa2_ref
from nanogpt.csa2_flex import starts

script, run, rows = sys.argv[1:]
compiled = torch.compile(flex_attention, dynamic=False)
csa2_ref.flex_attention = compiled  # type: ignore[assignment]
attend, calls = csa2_ref.attend, [0]
sums: dict[tuple[int, int], torch.Tensor] = {}  # (layer, bin): per-head share sum
counts: dict[tuple[int, int], int] = {}


def recorded(q, kv, entries, entry, valid, cu_seqlens, scale, m, sink):  # type: ignore[no-untyped-def]  # noqa: E501
    """csa2_ref.attend, adding each head's sink share by layer and position bin."""
    out, lse = attend(q, kv, entries, entry, valid, cu_seqlens, scale, m, sink)
    layer, calls[0] = calls[0] % 10, calls[0] + 1
    share = torch.sigmoid(sink - lse.float())  # (T, H)
    pos = torch.arange(len(q), device=q.device) - starts(cu_seqlens, len(q))
    for b, (lo, hi) in enumerate(pairwise(packed_val.EDGES)):
        at = (pos >= lo) & (pos < hi)
        sums[layer, b] = sums.get((layer, b), 0) + share[at].sum(0)
        counts[layer, b] = counts.get((layer, b), 0) + int(at.sum())
    return out, lse


csa2_ref.attend = recorded
id8 = run[:8]
packed_val.main(script, run, rows, f"logs/sink_{id8}_loss.csv")
with open(f"logs/sink_{id8}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(("layer", "start", "end", "mean", "max_head"))
    for (layer, b), total in sorted(sums.items()):
        head = total / counts[layer, b]
        lo, hi = packed_val.EDGES[b], packed_val.EDGES[b + 1]
        w.writerow((layer, lo, hi, f"{head.mean():.4f}", f"{head.max():.4f}"))
print(f"wrote logs/sink_{id8}.csv", flush=True)
