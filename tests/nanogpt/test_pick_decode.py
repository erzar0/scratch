"""Decode with SparQ-rule and union-rule picks (src/nanogpt/pick_decode.py): the
last query of an n-token document gets the output union_pick / sparq_pick give it in
prefill, from a cache longer than n (positions >= n must be ignored). Windows small
enough that it spans several. Needs CUDA:
    uv run python -m tests.nanogpt.test_pick_decode
"""

import torch

from nanogpt import pick_decode, sparq_pick, union_pick
from nanogpt.union_dims import Settings
from tests.nanogpt.test_tournament import packed


def matches_prefill(union: bool, s: Settings) -> None:
    q, k, v, _ = packed((2591,))
    cache = pick_decode.build(k, v, s.dims, union)
    prefill = union_pick.attention if union else sparq_pick.attention
    for n in (200, 450, 1500, 2591):  # all keys; 1 window; 3; 6 windows
        cu = torch.tensor([0, n, n], dtype=torch.int32, device="cuda")
        want = prefill(q[:n], k[:n], v[:n], cu, 0.12, s)[-1]
        got = pick_decode.step(q[n - 1], cache, n, 0.12, s)
        error = float((got - want).norm() / want.norm())
        assert error < 1e-4, (union, s, n, error)


if __name__ == "__main__":
    for union in (False, True):
        for s in (Settings(512, 256, 16), Settings(512, 384, 32)):
            matches_prefill(union, s)
    print("decode: the prefill output of the last query, both rules")
