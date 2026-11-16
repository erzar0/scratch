"""csa2_attend_mqa = csa2_attend_split.csa2_attend at V4.1-Flash-1017's attention
size (64 heads x 512, window 128, 512 or no picks, a real or -1e30 sink, bf16 and
FP8 rows), called twice (the arrival counters reset). Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_attend_mqa
"""

import itertools

import pytest
import torch

from nanogpt import csa2_attend_mqa, csa2_attend_split

HEADS, DIM, ROPE, WINDOW, N = 64, 512, 64, 128, 20000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_matches_split() -> None:
    torch.manual_seed(0)
    rand = lambda *s: torch.randn(*s, device="cuda").bfloat16()  # noqa: E731
    q, window, main = rand(HEADS, DIM) * 2, rand(WINDOW, DIM), rand(N, DIM)
    cos, sin = rand(ROPE), rand(ROPE)
    for top, sink, fp8 in itertools.product((512, 0), (0.5, -1e30), (False, True)):
        picks = torch.randperm(N, device="cuda")[:top]
        sinks = torch.full((HEADS,), sink, device="cuda")
        win, kv = (x.to(torch.float8_e4m3fn) if fp8 else x for x in (window, main))
        want = csa2_attend_split.csa2_attend(q, win, kv, picks, sinks, cos, sin)
        for _ in range(2):
            got = csa2_attend_mqa.csa2_attend(q, win, kv, picks, sinks, cos, sin)
            torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    test_matches_split()
    print("csa2_attend_mqa = csa2_attend_split (picks 512 / 0, sinks, bf16 / FP8)")
