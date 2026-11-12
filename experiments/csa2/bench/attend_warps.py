"""Decode attention (10 layers, one token) by warps per program: CSA2's csa2_attend
and ours b_attend (6-head window), csa2_decode.WARPS 2 / 4 / 8 / 16. ms per token,
CUDA graph, L2 flushed. Fork venv, from the repo root:
    PYTHONPATH=src python experiments/csa2/bench/attend_warps.py
"""

import torch
from decode_csa2 import graphed_ms, state
from decode_parts import ATTENDS

from nanogpt import csa2_decode as cd

if __name__ == "__main__":
    torch.manual_seed(0)
    s = state(8192)
    s["picks"] = torch.randperm(8192 // 4, device="cuda")[:64]
    for name in ("CSA2 attention", "ours attention, 6-head window"):
        for warps in (2, 4, 8, 16):
            cd.WARPS = warps
            attend = ATTENDS[name]
            ms = graphed_ms(lambda: [attend(s, i) for i in range(10)])  # noqa: B023
            print(f"{name:<32} warps {warps:>2}: {ms:.4f}", flush=True)
