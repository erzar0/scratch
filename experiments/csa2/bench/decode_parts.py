"""Decode time by part (decode_csa2.py's Triton rows): the 5 pick calls alone at
csa2_decode.PROGRAMS 16 / 32 / 64 / 128, and the 10 attention calls alone. ms per
token, CUDA graph, L2 flushed, median. Fork venv, from the repo root:
    PYTHONPATH=src python experiments/csa2/bench/decode_parts.py
"""

import torch
from decode_csa2 import GROUP, LAYERS, RR, SCALE, graphed_ms, state

from nanogpt import csa2_decode as cd

PICKS = {
    "CSA2 indexer": lambda s, i: cd.top64(
        s["qi"][i], s["index_k"][i // GROUP], s["w"][i]
    ),
    "ours exact": lambda s, i: cd.top64(s["qs"][i], s["keys"][i // GROUP], width=RR),
    "ours prefilter 512": lambda s, i: cd.prefilter64(
        s["qs"][i], s["keys"][i // GROUP], 512, width=RR
    ),
}


def csa2_attend(s: dict, i: int) -> None:
    picks = s["picks"]
    cd.csa2_attend(
        s["q"][i], s["window_kv"][i], s["main_kv"][i // GROUP], picks,
        s["sinks"][i], s["cos"], s["sin"],
    )  # fmt: skip


def b_attend(s: dict, i: int, window: str) -> None:
    k, v = s[window][i]
    keys = s["keys"][i // GROUP]
    cd.b_attend(s["q"][i], k, v, keys, s["picks"], s["table"], SCALE)


ATTENDS = {
    "CSA2 attention": csa2_attend,
    "ours attention, 6-head window": lambda s, i: b_attend(s, i, "win"),
    "ours attention, MQA window": lambda s, i: b_attend(s, i, "mqa"),
}

if __name__ == "__main__":
    for n in (8192, 32768, 131072, 524288):
        torch.manual_seed(0)
        s = state(n)
        s["picks"] = torch.randperm(n // 4, device="cuda")[:64]
        for name, pick in PICKS.items():
            for programs in (16, 32, 64, 128):
                cd.PROGRAMS = programs
                ms = graphed_ms(
                    lambda: [pick(s, i) for i in range(0, LAYERS, 2)]  # noqa: B023
                )
                print(f"{n:>7} {name:<32} programs {programs:>3}: {ms:.4f}", flush=True)
        cd.PROGRAMS = 128
        for name, attend in ATTENDS.items():
            ms = graphed_ms(lambda: [attend(s, i) for i in range(LAYERS)])  # noqa: B023
            print(f"{n:>7} {name:<32}: {ms:.4f}", flush=True)
        del s
        torch.cuda.empty_cache()
