"""Fast CSA2 picks (src/nanogpt/csa2_select_fast.py) = csa2_select's on packed rows
(document and finished-entry masks, short prefixes, training's 2048-token documents
with 6 heads): same valid counts; exact picks the same scores; the tournament the
same sets except near ties (fp32 sums in another order); the op passes opcheck.
Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_select_fast
"""

import torch
from torch import Tensor

from nanogpt import csa2_select, csa2_select_fast

M, TOP = 4, 64
ROWS = ((300, 2048, 900, 7), (2048,) * 8)  # the second: a training row


def packed(docs: tuple[int, ...], heads: int) -> tuple[Tensor, Tensor, Tensor]:
    """q (S, heads, 128), kv (S // M, 128) bf16, starts (S,) for docs in one row."""
    g = torch.Generator().manual_seed(0)
    length = sum(docs)
    q = torch.randn(length, heads, 128, generator=g).cuda().bfloat16()
    kv = torch.randn(length // M, 128, generator=g).cuda().bfloat16()
    first = torch.tensor((0, *docs[:-1])).cumsum(0)
    starts = first.repeat_interleave(torch.tensor(docs)).cuda()
    return q, kv, starts


def picked(q: Tensor, kv: Tensor, picks: tuple[Tensor, Tensor]) -> Tensor:
    """(S, TOP) fp64 exact scores of the valid picks, sorted; -inf elsewhere."""
    entry, valid = picks
    score = (kv.double()[entry] * q.double().sum(1)[:, None]).sum(-1)
    return score.masked_fill(~valid, float("-inf")).sort(-1).values


def main() -> None:
    for docs, heads in zip(ROWS, (3, 6), strict=True):
        q, kv, starts = packed(docs, heads)
        for rule in ("exact", "tournament"):
            want = getattr(csa2_select, rule)(q, kv, starts, M, TOP)
            got = getattr(csa2_select_fast, rule)(q, kv, starts, M, TOP)
            assert torch.equal(got[1].sum(1), want[1].sum(1)), rule
            if rule == "exact":
                gap = (picked(q, kv, got) - picked(q, kv, want)).nan_to_num().abs()
                assert gap.max() < 1e-3, (rule, gap.max())
            else:
                same = [
                    set(got[0][t][got[1][t]].tolist())
                    == set(want[0][t][want[1][t]].tolist())
                    for t in range(len(q))
                ]
                assert sum(same) >= 0.995 * len(q), (rule, len(q) - sum(same))
            print(f"{docs}, {heads} heads, {rule}: = csa2_select", flush=True)
        for dims in (0, 16):
            torch.library.opcheck(csa2_select_fast.picks, (q, kv, starts, M, TOP, dims))
    print("opcheck: passes")


if __name__ == "__main__":
    main()
