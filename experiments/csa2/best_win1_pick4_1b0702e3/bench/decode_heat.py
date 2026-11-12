"""Decode time by part and context: one token through the 10 attention layers
(batch 1, bf16, CUDA graph, L2 flushed, least of 5 medians), picks alone, attention
alone and both, contexts 4K-1M, CSA2 (learned indexer) and our pick rules in CSA2's
KV layout in stripped record 91: picks with csa2_decode's Triton kernels, attention
with csa2_attend_split's (each version at its fastest rows per program). Block-major:
entries stored as (8, entries, 16), so routed picks read 2 contiguous slabs. K=V
windows (csa2_window): G shared rotated K=V vectors per window token instead of the
heads' own K and V, attention with csa2_window_decode.kv_attend.
-> runs/nanogpt/long_context/decode_heat.csv (version, part, n, us_per_token).
Fork venv, from the repo root:
    PYTHONPATH=src python experiments/csa2/bench/decode_heat.py
"""

import csv
from collections.abc import Callable
from pathlib import Path

import torch
from decode_csa2 import GROUP, LAYERS, RR, SCALE, W, graphed_ms, rand, state
from torch import Tensor

from nanogpt import csa2_attend_split as sp
from nanogpt import csa2_decode as cd
from nanogpt import csa2_window_decode as wd

OUT = Path("runs/nanogpt/long_context/decode_heat.csv")
LENGTHS = [4096 << i for i in range(9)]  # 4K .. 1M
Pick = Callable[[dict, int], Tensor]


def csa2_pick(s: dict, i: int) -> Tensor:
    return cd.top64(s["qi"][i], s["index_k"][i // GROUP], s["w"][i])


def exact(s: dict, i: int) -> Tensor:
    return cd.top64(s["qs"][i], s["keys"][i // GROUP], width=RR)


def prefilter(s: dict, i: int) -> Tensor:
    return cd.prefilter64(s["qs"][i], s["keys"][i // GROUP], 512, width=RR)


def routed(s: dict, i: int, keys: str = "keys") -> Tensor:
    return cd.top64(s["qrs"][i], s[keys][i // GROUP], blocks=2)


def routed_major(s: dict, i: int) -> Tensor:
    return routed(s, i, "keys_major")


def csa2_attend(s: dict, i: int, picks: Tensor) -> None:
    sp.csa2_attend(
        s["q"][i], s["window_kv"][i], s["main_kv"][i // GROUP], picks,
        s["sinks"][i], s["cos"], s["sin"],
    )  # fmt: skip


def b_attend(
    s: dict, i: int, picks: Tensor, q_pick: str | None = None, keys: str = "keys"
) -> None:
    k, v = s["win"][i]
    pick_q = s[q_pick][i] if q_pick else None
    sp.b_attend(s["q"][i], k, v, s[keys][i // GROUP], picks, s["table"], SCALE, pick_q)


def routed_attend(s: dict, i: int, picks: Tensor) -> None:
    b_attend(s, i, picks, "qr")


def major_attend(s: dict, i: int, picks: Tensor) -> None:
    b_attend(s, i, picks, "qr", "keys_major")


def window_attend(groups: int, routed: bool) -> Callable:
    """kv_attend over G shared K=V window vectors; routed: q_pick and block-major."""

    def attend(s: dict, i: int, picks: Tensor) -> None:
        keys, q_pick = ("keys_major", s["qr"][i]) if routed else ("keys", None)
        win = s[f"kv_win{groups}"][i]
        wd.kv_attend(s["q"][i], win, s["table"][:W], s[keys][i // GROUP], picks,
                     s["table"], SCALE, q_pick)  # fmt: skip

    return attend


WINDOWS = {1: "1 shared K=V window head", 2: "2 shared K=V window heads",
           6: "per-head K=V window"}  # fmt: skip
# name -> (pick rule, pick every n layers, attention)
VERSIONS: dict[str, tuple[Pick, int, Callable]] = {
    "CSA2, learned indexer": (csa2_pick, 2, csa2_attend),
    "CSA2, picks every 4 layers": (csa2_pick, 4, csa2_attend),
    "Ours, exact": (exact, 2, b_attend),
    "Ours, prefilter keeps 512": (prefilter, 2, b_attend),
    "Ours, exact, picks every 4 layers": (exact, 4, b_attend),
    "Ours, routed 2 of 6 blocks": (routed, 2, routed_attend),
    "Ours, routed, picks every 4 layers": (routed, 4, routed_attend),
    "Ours, routed, block-major": (routed_major, 2, major_attend),
    "Ours, routed, block-major, picks every 4 layers": (routed_major, 4, major_attend),
}  # fmt: skip
for g, label in WINDOWS.items():
    VERSIONS[f"Ours, exact, {label}"] = (exact, 2, window_attend(g, False))
# exact picks with the 1-head window (trained: d3c62030, 1b0702e3); prefilter keeps
# 512 = exact at eval (rule_eval) and is faster at long context
for rule, label in ((prefilter, "prefilter keeps 512"), (exact, "exact")):
    for every in (2, 4):
        name = f"Ours, {label}{', picks every 4 layers' if every == 4 else ''}, "
        VERSIONS[name + WINDOWS[1]] = (rule, every, window_attend(1, False))
for g, label in WINDOWS.items():
    VERSIONS[f"Ours, routed, block-major, picks every 4 layers, {label}"] = (
        routed_major, 4, window_attend(g, True)
    )  # fmt: skip


def parts(
    s: dict, pick: Pick, every: int | tuple[int, ...], attend: Callable
) -> dict[str, Callable]:
    """The three timed loops of one version: picks every `every` layers, or at the
    listed layers (the layers after each reuse its picks)."""
    layers = range(0, LAYERS, every) if isinstance(every, int) else every

    def picks() -> None:
        for i in layers:
            pick(s, i)

    def attention() -> None:
        for i in range(LAYERS):
            attend(s, i, s["picks"])

    def both() -> None:
        for i in range(LAYERS):
            if i in layers:
                chosen = pick(s, i)
            attend(s, i, chosen)

    return {"Picks": picks, "Attention": attention, "Picks + attention": both}


if __name__ == "__main__":
    rows = []
    for n in LENGTHS:
        torch.manual_seed(0)
        s = state(n)
        s["picks"] = torch.randperm(n // 4, device="cuda")[:64]
        s["keys_major"] = [  # (8, entries, 16): each 16-channel block contiguous
            k.view(len(k), -1, 16).transpose(0, 1).contiguous() for k in s["keys"]
        ]
        for g in WINDOWS:
            s[f"kv_win{g}"] = [rand(W, g, 128) for _ in range(LAYERS)]
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
