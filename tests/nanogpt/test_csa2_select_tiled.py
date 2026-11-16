"""csa2_select_tiled.exact = csa2_select_fast.exact on packed rows (document masks,
short prefixes, training rows of 2048-token documents) at 1 and 4 tokens per
entry: same valid counts, the same picked scores; the op passes opcheck. Prints
both kernels' times on the training row. Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_select_tiled
"""

import torch
import triton

from nanogpt import csa2_select_fast, csa2_select_tiled
from tests.nanogpt.test_csa2_select_fast import ROWS, TOP, packed, picked


def main() -> None:
    for m in (1, 4):
        for docs, heads in zip(ROWS, (3, 6), strict=True):
            q, kv, starts = packed(docs, heads)
            kv = torch.randn(len(q) // m, kv.shape[1], device="cuda").bfloat16()
            want = csa2_select_fast.exact(q, kv, starts, m, TOP)
            got = csa2_select_tiled.exact(q, kv, starts, m, TOP)
            assert torch.equal(got[1].sum(1), want[1].sum(1)), (docs, m)
            gap = (picked(q, kv, got) - picked(q, kv, want)).nan_to_num().abs()
            assert gap.max() < 1e-3, (docs, m, gap.max())
            print(f"{docs}, {heads} heads, {m} per entry: = csa2_select_fast")
        torch.library.opcheck(csa2_select_tiled.picks, (q, kv, starts, m, TOP))
        for rule in (csa2_select_fast.exact, csa2_select_tiled.exact):
            args = q, kv, starts, m, TOP
            ms = triton.testing.do_bench(lambda f=rule, a=args: f(*a))
            print(f"  {rule.__module__}, {m} per entry: {ms:.2f} ms")
    print("opcheck: passes")


if __name__ == "__main__":
    main()
