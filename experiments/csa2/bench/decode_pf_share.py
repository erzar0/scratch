"""Decode time of the parameter-free block (decode_ref_picks.py's PF rows: picks among
the entries that end before the 128-token window, no sinks) when layer 9 reuses
layer 5's picks (eval/variant_eval.py SHARE9): picks at layers 1 and 5 only, both
layers 5 and 9 reading the same 1-token entries. Exact picks and best 256 blocks of
8, bf16 and FP8 caches, each with and without sharing; the reference (its indexer)
timed again in the same process. Same timing as decode_ref_picks.py; picks checked
against PyTorch at the first length.
-> runs/nanogpt/long_context/decode_pf_share.csv (version, part, n, us_per_token).
Fork venv, from the repo root:
    PYTHONPATH=src:experiments/csa2/bench python \
        experiments/csa2/bench/decode_pf_share.py
"""

import csv
from pathlib import Path

import torch
from decode_block import REF, ref_parts
from decode_csa2 import graphed_ms, rand, state
from decode_heat import LENGTHS
from decode_ref_picks import BLOCK256, PF, Pick, block_mean, exact, parameter_free
from decode_ref_picks import reference as torch_picks

OUT = Path("runs/nanogpt/long_context/decode_pf_share.csv")
SHARE = ", layer 9 reuses layer 5's picks"


def share9(rule: Pick) -> Pick:
    """rule at layers 1 and 5; layer 9 returns layer 5's picks (ref_parts picks at
    1, 5, 9 in that order)."""
    held: dict[int, torch.Tensor] = {}

    def pick(s: dict, i: int) -> torch.Tensor:
        if i == 9:
            return held[5]
        held[i] = rule(s, i)
        return held[i]

    return pick


RULES: dict[str, tuple[Pick, int]] = {
    "summed query, exact picks": (exact, 0),
    BLOCK256: (block_mean(4), 4),
}

if __name__ == "__main__":
    rows = []
    for n in LENGTHS:
        torch.manual_seed(0)
        s = state(n)
        s["picks"] = torch.randperm(n // 4, device="cuda")[:64]
        s["ref_main"] = [rand(n // 2, 128), rand(n, 128)]
        s["ref_index"] = [rand(n // 2, 32), rand(n, 32)]
        s["ref_mean"] = [k.view(-1, 8, 128).float().mean(1).to(k.dtype)
                         for k in s["ref_main"]]  # fmt: skip
        s8 = dict(s)
        for key in ("ref_main", "ref_index", "ref_mean", "window_kv"):
            s8[key] = [x.to(torch.float8_e4m3fn) for x in s[key]]
        pf, pf8 = parameter_free(s, n), parameter_free(s8, n)
        if n == LENGTHS[0]:
            for name, (rule, pool) in RULES.items():
                for st in (pf, pf8):
                    shared = share9(rule)
                    got = [set(shared(st, i).tolist()) for i in (1, 5, 9)]
                    want = [torch_picks(st, i, pool) for i in (1, 5, 5)]
                    assert got == want, name
            print("picks = PyTorch, layer 9 = layer 5 (bf16 and FP8)", flush=True)
        versions: list[tuple[str, Pick | None, dict]] = [(REF, None, s)]
        for fmt, st in (("", pf), (", FP8 caches", pf8)):
            if fmt:
                versions.append((REF + fmt, None, s8))
            for name, (rule, _) in RULES.items():
                versions.append((f"{PF}, {name}{fmt}", rule, st))
                versions.append((f"{PF}, {name}{SHARE}{fmt}", share9(rule), st))
        for name, chosen, st in versions:
            for part, fn in ref_parts(st, chosen).items():
                us = 1000 * graphed_ms(fn)
                rows.append((name, part, n, round(us, 2)))
                print(*rows[-1], sep=" | ", flush=True)
        del s, s8, pf, pf8
        torch.cuda.empty_cache()
    with OUT.open("w", newline="") as f:
        csv.writer(f).writerows([("version", "part", "n", "us_per_token"), *rows])
    print("wrote", OUT)
