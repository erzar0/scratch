"""Decode time of the V4.1-Flash-1017 reference layout (decode_block.ref_parts: picks
at layers 1, 5, 9 over entries of 2 tokens, then of 1 token; csa2_attend with 128
window rows, 64 picks, sinks) with our summed-query picks in place of its indexer,
the rules eval/ref_rule_eval.py scored on its checkpoint 65614cd9:
    exact          csa2_decode.top64 of (q1 + ... + q6) . entry over every entry.
    block mean P   blocks of 8 entries, their mean entry kept as entries are written;
                   top 64 * P blocks by summed query . mean (P = 1: csa2_decode.top64;
                   P = 4: one GEMV + torch.topk, as csa2_ref_decode.candidates), then
                   top64 over their 64 * 8 * P entries. Layer 9 takes its own blocks.
    one launch     block mean 1 as csa2_pick_fused.picks: the same picks, one kernel;
                   also with every cache FP8 (E4M3), as the reference's FP8 row.
R4 rows: the same timing with both entry caches at 1 entry per 4 tokens (the layout
of queue_round4.sh's C3, ratio 4 at layers 1-9). PF rows: the parameter-free block
(src/nanogpt/csa2_pf.py) in the same layout: picks among the entries that end before
the 128-token window, no sinks (-inf); cache writes (a mean in place of the
compressor's projections) are not timed.
The reference (its indexer, TensorRT-LLM's candidate blocks) is timed again in the
same process. Same timing as decode_heat.py; picks checked against PyTorch first;
kernels per token (torch.profiler, one eager call) printed at the first length.
-> runs/nanogpt/long_context/decode_ref_picks.csv (version, part, n, us_per_token).
Fork venv, from the repo root:
    PYTHONPATH=src:experiments/csa2/bench python \
        experiments/csa2/bench/decode_ref_picks.py
"""

import csv
from collections.abc import Callable
from pathlib import Path

import torch
from decode_block import REF, ref_parts
from decode_csa2 import graphed_ms, rand, state
from decode_heat import LENGTHS
from torch import Tensor

from nanogpt import csa2_decode as cd
from nanogpt import csa2_pick_fused

OUT = Path("runs/nanogpt/long_context/decode_ref_picks.csv")
OFFSETS = torch.arange(8, device="cuda")
Pick = Callable[[dict, int], Tensor]


def exact(s: dict, i: int) -> Tensor:
    return cd.top64(s["qa"][i], s["ref_main"][i >= 5])


def block_mean(pool: int) -> Pick:
    def pick(s: dict, i: int) -> Tensor:
        q, means, keys = s["qa"][i], s["ref_mean"][i >= 5], s["ref_main"][i >= 5]
        if len(means) <= 64 * pool:  # every block a candidate
            return cd.top64(q, keys[: len(means) * 8])
        # best blocks, then their entries scored in place (no gather copy)
        return cd.top64(q, keys, within=cd.top64(q, means, top=64 * pool))

    return pick


def fused(s: dict, i: int) -> Tensor:
    g = i >= 5
    return csa2_pick_fused.picks(s["qa"][i], s["ref_main"][g], s["ref_mean"][g])


def kernels(parts: dict) -> int:
    """GPU kernels one eager call of picks + attention launches (cache writes not
    included)."""
    fn = parts["Picks + attention"]
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(e.device_type == torch.autograd.DeviceType.CUDA for e in p.events())


def reference(s: dict, i: int, pool: int) -> set[int]:
    """PyTorch: the same picks (pool 0: exact)."""
    q, keys = s["qa"][i], s["ref_main"][i >= 5]
    score = keys.float() @ q
    if pool:
        means = s["ref_mean"][i >= 5]
        blocks = (means.float() @ q).topk(min(64 * pool, len(means))).indices
        cand = (blocks[:, None] * 8 + OFFSETS).flatten()
        return set(cand[score[cand].topk(64).indices].tolist())
    return set(score.topk(64).indices.tolist())


BLOCK64 = "summed query, block-mean picks, best 64 blocks of 8"
BLOCK256 = "summed query, block-mean picks, best 256 blocks of 8"
FUSED = f"V4.1 reference layout, {BLOCK64}, one launch"
R4 = "V4.1 reference, entries of 4 tokens at layers 1-9"  # round 4's C3
PF = "parameter-free CSA2, V4.1 layout"


def parameter_free(st: dict, n: int) -> dict:
    """st with each entry cache cut to the entries that end before the window and
    sinks of -inf (src/nanogpt/csa2_pf.py)."""
    out = dict(st)
    out["ref_main"] = [
        k[: (n - 128) // m] for k, m in zip(st["ref_main"], (2, 1), strict=True)
    ]
    out["ref_mean"] = [
        x[: k.shape[0] // 8]
        for x, k in zip(st["ref_mean"], out["ref_main"], strict=True)
    ]
    out["sinks"] = [torch.full_like(x, float("-inf")) for x in st["sinks"]]
    return out


OURS = {
    "V4.1 reference layout, summed query, exact picks": (exact, 0),
    "V4.1 reference layout, summed query, block-mean picks, best 64 blocks of 8": (
        block_mean(1), 1),
    "V4.1 reference layout, summed query, block-mean picks, best 256 blocks of 8": (
        block_mean(4), 4),
    FUSED: (fused, 1),
}  # fmt: skip

if __name__ == "__main__":
    rows = []
    for n in LENGTHS:
        torch.manual_seed(0)
        s = state(n)
        s["picks"] = torch.randperm(n // 4, device="cuda")[:64]
        s["ref_main"] = [rand(n // 2, 128), rand(n, 128)]
        s["ref_index"] = [rand(n // 2, 32), rand(n, 32)]
        # block means kept at cache write: not timed
        s["ref_mean"] = [k.view(-1, 8, 128).float().mean(1).to(k.dtype)
                         for k in s["ref_main"]]  # fmt: skip
        s8 = dict(s)  # FP8 rows: every cache float8_e4m3fn (as decode_block.py)
        for key in ("ref_main", "ref_index", "ref_mean", "window_kv"):
            s8[key] = [x.to(torch.float8_e4m3fn) for x in s[key]]
        s4 = dict(s)  # R4: both entry caches 1 per 4 tokens
        s4["ref_main"] = [rand(n // 4, 128), rand(n // 4, 128)]
        s4["ref_mean"] = [k.view(-1, 8, 128).float().mean(1).to(k.dtype)
                          for k in s4["ref_main"]]  # fmt: skip
        s48 = dict(s4)
        for key in ("ref_main", "ref_mean", "window_kv"):
            s48[key] = [x.to(torch.float8_e4m3fn) for x in s4[key]]
        pf, pf8 = parameter_free(s, n), parameter_free(s8, n)
        if n == LENGTHS[0]:  # pf: 248 means at layer 1, every block
            for name, (pick, pool) in OURS.items():
                for i in (1, 5, 9):
                    for st in (s, pf, s8, pf8, s4, s48):
                        got = set(pick(st, i).tolist())
                        assert got == reference(st, i, pool), (name, i)
            print("picks = PyTorch (bf16 and FP8 caches)", flush=True)
        versions: list[tuple[str, Pick | None, dict]] = [
            (REF, None, s),
            *((k, v[0], s) for k, v in OURS.items()),
            (f"{REF}, FP8 caches", None, s8),
            *((f"{k}, FP8 caches", v[0], s8) for k, v in OURS.items()),
            (f"{R4}, summed query, exact picks", exact, s4),
            (f"{R4}, {BLOCK64}", block_mean(1), s4),
            (f"{R4}, {BLOCK64}, one launch", fused, s4),
            (f"{R4}, summed query, exact picks, FP8 caches", exact, s48),
            (f"{R4}, {BLOCK64}, FP8 caches", block_mean(1), s48),
            (f"{R4}, {BLOCK64}, one launch, FP8 caches", fused, s48),
            (f"{PF}, summed query, exact picks", exact, pf),
            (f"{PF}, {BLOCK64}", block_mean(1), pf),
            (f"{PF}, {BLOCK64}, one launch", fused, pf),
            (f"{PF}, {BLOCK256}", block_mean(4), pf),
            (f"{PF}, summed query, exact picks, FP8 caches", exact, pf8),
            (f"{PF}, {BLOCK64}, FP8 caches", block_mean(1), pf8),
            (f"{PF}, {BLOCK64}, one launch, FP8 caches", fused, pf8),
            (f"{PF}, {BLOCK256}, FP8 caches", block_mean(4), pf8),
        ]
        for name, rule, st in versions:
            if n == LENGTHS[0]:
                print(name, "| kernels per token:", kernels(ref_parts(st, rule)),
                      flush=True)  # fmt: skip
            for part, fn in ref_parts(st, rule).items():
                us = 1000 * graphed_ms(fn)
                rows.append((name, part, n, round(us, 2)))
                print(*rows[-1], sep=" | ", flush=True)
        del s, s8, s4, s48, pf, pf8
        torch.cuda.empty_cache()
    with OUT.open("w", newline="") as f:
        csv.writer(f).writerows([("version", "part", "n", "us_per_token"), *rows])
    print("wrote", OUT)
