"""Decode time of run 7's setup (1 shared K=V window head, picks every 4 layers) with
picks shared across groups (IndexCache: layer 0 only, or layers 0 and 4) and with the
block prefilter (csa2_select_block): stage 1 csa2_decode.top64 over block summaries
(64 blocks of 8 or 16 entries; mean / max: 128 channels, minmax / meanstd: [max, min]
/ [mean, std] packed on q's 96 nonzero channels, 192 read), stage 2 top64 over the
kept entries (gathered). Summaries come from the random keys. Same timing as
decode_heat.py (CUDA graph, L2 flushed); stage picks checked against PyTorch first.
-> runs/nanogpt/long_context/decode_block.csv (version, part, n, us_per_token).
Fork venv, from the repo root:
    PYTHONPATH=src:experiments/csa2/bench python experiments/csa2/bench/decode_block.py
"""

import csv
from pathlib import Path

import torch
from decode_csa2 import GROUP, RR, W, graphed_ms, rand, state
from decode_heat import (
    LENGTHS,
    WINDOWS,
    Pick,
    csa2_attend,
    csa2_pick,
    exact,
    parts,
    prefilter,
    window_attend,
)
from torch import Tensor

from nanogpt import csa2_decode as cd

OUT = Path("runs/nanogpt/long_context/decode_block.csv")
SUMMARIES = ("mean", "max", "minmax", "meanstd")
SIZES = (8, 16)


def summaries(keys: Tensor, size: int) -> dict[str, Tensor]:
    """Block summaries of keys (N, 128) bf16, aligned blocks of size."""
    k = keys.float().view(-1, size, keys.shape[1])[..., :RR]
    pad = k.new_zeros(k.shape[0], 256 - 2 * RR)
    mean, top, low, std = k.mean(1), k.amax(1), k.amin(1), k.std(1, correction=0)
    return {
        "mean": torch.cat([mean, pad[:, : 128 - RR]], 1).to(keys.dtype),
        "max": torch.cat([top, pad[:, : 128 - RR]], 1).to(keys.dtype),
        "minmax": torch.cat([top, low, pad], 1).to(keys.dtype),
        "meanstd": torch.cat([mean, std, pad], 1).to(keys.dtype),
    }


def q_side(qs: Tensor, summary: str) -> Tensor:
    """The query (128 or 256 channels) that scores a summary row by one dot product."""
    if summary in ("mean", "max"):
        return qs
    q = qs[:RR]
    a, b = (q.clamp(min=0), q.clamp(max=0)) if summary == "minmax" else (q, q.abs())
    return torch.cat([a, b, q.new_zeros(256 - 2 * RR)])


def block_pick(summary: str, size: int) -> Pick:
    """Pick rule: top 64 blocks by summary, then the top 64 of their entries."""
    width = 2 * RR if summary in ("minmax", "meanstd") else RR
    offsets = torch.arange(size, device="cuda")

    def pick(s: dict, i: int) -> Tensor:
        g = i // GROUP
        blocks = cd.top64(s[f"q_{summary}"][i], s[f"{summary}_{size}"][g], width=width)
        cand = (blocks[:, None] * size + offsets).flatten()
        return cand[cd.top64(s["qs"][i], s["keys"][g][cand], width=RR)]

    return pick


def reference(s: dict, summary: str, size: int, i: int) -> set[int]:
    """PyTorch: the same two stages."""
    g = i // GROUP
    score = s[f"{summary}_{size}"][g].float() @ s[f"q_{summary}"][i].float()
    blocks = score.topk(64).indices
    cand = (blocks[:, None] * size + torch.arange(size, device="cuda")).flatten()
    keys = s["keys"][g][cand].float()
    return set(cand[(keys @ s["qs"][i]).topk(64).indices].tolist())


WIN = window_attend(1, False)
VERSIONS = {
    "CSA2, learned indexer": (csa2_pick, 2, csa2_attend),
    f"Ours, exact, picks every 4 layers, {WINDOWS[1]} (run 7)": (exact, 4, WIN),
    f"Ours, prefilter keeps 512, picks every 4 layers, {WINDOWS[1]}": (
        prefilter, 4, WIN),
    f"Ours, exact, picks at layers 0, 4, {WINDOWS[1]}": (exact, (0, 4), WIN),
    f"Ours, exact, picks at layer 0 only, {WINDOWS[1]}": (exact, 10, WIN),
    f"Ours, prefilter keeps 512, picks at layer 0 only, {WINDOWS[1]}": (
        prefilter, 10, WIN),
}  # fmt: skip
for summary in SUMMARIES:
    for size in SIZES:
        name = f"Ours, block {summary}, 64 blocks of {size}, picks every 4 layers"
        VERSIONS[f"{name}, {WINDOWS[1]}"] = (block_pick(summary, size), 4, WIN)
for summary in ("mean", "meanstd"):
    name = f"Ours, block {summary}, 64 blocks of 8, picks at layers 0, 4"
    VERSIONS[f"{name}, {WINDOWS[1]}"] = (block_pick(summary, 8), (0, 4), WIN)

if __name__ == "__main__":
    rows = []
    for n in LENGTHS:
        torch.manual_seed(0)
        s = state(n)
        s["picks"] = torch.randperm(n // 4, device="cuda")[:64]
        s["kv_win1"] = [rand(W, 1, 128) for _ in range(10)]
        for size in SIZES:
            built = [summaries(keys, size) for keys in s["keys"]]
            for summary in SUMMARIES:
                s[f"{summary}_{size}"] = [b[summary] for b in built]
        for summary in SUMMARIES:
            s[f"q_{summary}"] = [q_side(qs, summary) for qs in s["qs"]]
        if n == LENGTHS[0]:
            for summary in SUMMARIES:
                for size in SIZES:
                    got = set(block_pick(summary, size)(s, 4).tolist())
                    assert got == reference(s, summary, size, 4), (summary, size)
            print("block picks = PyTorch (every summary, size)", flush=True)
        for name, version in VERSIONS.items():
            for part, fn in parts(s, *version).items():
                us = 1000 * graphed_ms(fn)
                rows.append((name, part, n, round(us, 2)))
                print(*rows[-1], sep=" | ", flush=True)
        del s
        torch.cuda.empty_cache()
    with OUT.open("w", newline="") as f:
        csv.writer(f).writerows([("version", "part", "n", "us_per_token"), *rows])
    print("wrote", OUT)
