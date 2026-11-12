"""Decode time of run 7's setup (1 shared K=V window head, picks every 4 layers) with
picks shared across groups (IndexCache: layer 0 only, or layers 0 and 4) and with the
block prefilter (csa2_select_block): stage 1 csa2_decode.top64 over block summaries
(64 blocks of 8 or 16 entries; mean / max: 128 channels, minmax / meanstd: [max, min]
/ [mean, std] packed on q's 96 nonzero channels, 192 read), stage 2 top64 over the
kept entries (gathered). Summaries come from the random keys. FP8 rows: keys and
summaries stored float8_e4m3fn (the kernels load any dtype as fp32). Same timing as
decode_heat.py (CUDA graph, L2 flushed); stage picks checked against PyTorch first.
CSA2 as DeepSeek's V4.1-Flash-1017 code (src/nanogpt/csa2_ref.py's layout): layer 0
window only; layers 1-4 read entries of 2 tokens, picks by layer 1; layers 5-9 one
entry per token, picks by layers 5 and 9; TensorRT-LLM's decode algorithm with our
kernels: layer 5 keeps 256 candidate blocks of 8 (csa2_ref_decode, its block maxima)
and takes its top 64 inside them, layer 9 its own top 64 inside the same blocks;
attention csa2_attend (128 window rows + 64 picks, sinks). FP8: entries, index keys
and windows float8_e4m3fn (TensorRT-LLM stores FP4 entries and index keys, which
our SM89 kernels do not read).
-> runs/nanogpt/long_context/decode_block.csv (version, part, n, us_per_token).
Fork venv, from the repo root:
    PYTHONPATH=src:experiments/csa2/bench python experiments/csa2/bench/decode_block.py
"""

import csv
from collections.abc import Callable
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

from nanogpt import csa2_attend_split as sp
from nanogpt import csa2_decode as cd
from nanogpt import csa2_ref_decode as rd

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
FP8 = {  # same rules on the FP8 state
    "Ours, exact, picks at layers 0, 4": exact,
    "Ours, block mean, 64 blocks of 8, picks at layers 0, 4": block_pick("mean", 8),
}
FP8 = {f"{k}, FP8 entries, {WINDOWS[1]}": (v, (0, 4), WIN) for k, v in FP8.items()}
REF = "CSA2, V4.1 reference layout (TensorRT-LLM decode)"


def ref_parts(s: dict) -> dict[str, Callable]:
    """REF's three timed loops (decode_heat.parts): picks at layers 1, 5, 9."""

    def attend(i: int, picks: Tensor) -> None:
        main = s["ref_main"][i >= 5]
        picks = picks[:0] if i == 0 else picks  # window only
        sp.csa2_attend(s["q"][i], s["window_kv"][i], main, picks, s["sinks"][i],
                       s["cos"], s["sin"])  # fmt: skip

    def pick(i: int) -> Tensor:
        if i == 1:
            return cd.top64(s["qi"][1], s["ref_index"][0], s["w"][1])
        if i == 5:
            s["cand"] = rd.candidates(s["qi"][5], s["w"][5], s["ref_index"][1])
            s["cand_k"] = s["ref_index"][1][s["cand"]]
        return s["cand"][cd.top64(s["qi"][i], s["cand_k"], s["w"][i])]

    def picks() -> None:
        for i in (1, 5, 9):
            pick(i)

    def attention() -> None:
        for i in range(10):
            attend(i, s["picks"])

    def both() -> None:
        chosen = s["picks"]
        for i in range(10):
            if i in (1, 5, 9):
                chosen = pick(i)
            attend(i, chosen)

    return {"Picks": picks, "Attention": attention, "Picks + attention": both}


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
        s["ref_main"] = [rand(n // 2, 128), rand(n, 128)]
        s["ref_index"] = [rand(n // 2, 32), rand(n, 32)]
        s8 = dict(s)
        for key in ("keys", "mean_8", "ref_main", "ref_index", "window_kv"):
            s8[key] = [x.to(torch.float8_e4m3fn) for x in s[key]]
        if n == LENGTHS[0]:
            assert set(block_pick("mean", 8)(s8, 4).tolist()) == reference(
                s8, "mean", 8, 4
            )
            print("FP8 block picks = PyTorch", flush=True)
        for name, version, st in [
            *((k, v, s) for k, v in VERSIONS.items()),
            *((k, v, s8) for k, v in FP8.items()),
        ]:
            for part, fn in parts(st, *version).items():
                us = 1000 * graphed_ms(fn)
                rows.append((name, part, n, round(us, 2)))
                print(*rows[-1], sep=" | ", flush=True)
        for name, st in ((REF, s), (f"{REF}, FP8 caches", s8)):
            for part, fn in ref_parts(st).items():
                us = 1000 * graphed_ms(fn)
                rows.append((name, part, n, round(us, 2)))
                print(*rows[-1], sep=" | ", flush=True)
        del s, s8
        torch.cuda.empty_cache()
    with OUT.open("w", newline="") as f:
        csv.writer(f).writerows([("version", "part", "n", "us_per_token"), *rows])
    print("wrote", OUT)
