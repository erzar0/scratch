"""Decode rule prefilter (src/nanogpt/csa2_select_decode.py): with keep at or above
every query's allowed entries it is csa2_select.exact; otherwise each query's picks
are the top 64 by the full score among the entries at or above its keep-th best
bin on its aligned 16-channel block (a per-query loop); opcheck passes. Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_select_decode
"""

import torch

from nanogpt import csa2_select, csa2_select_decode
from tests.nanogpt.test_csa2_select import TOP, M, allowed, best, got, packed


def main() -> None:
    q, kv, starts = packed()
    want = csa2_select.exact(q, kv, starts, M, TOP)
    full = csa2_select_decode.prefilter(q, kv, starts, M, TOP, keep=len(kv))
    for t in range(len(q)):
        assert got(*full, t) == got(*want, t), t
    print("keep >= entries: = exact")
    entry, valid = csa2_select_decode.prefilter(q, kv, starts, M, TOP, keep=128)
    qs = q.sum(1)
    blocks = qs.abs().unflatten(-1, (-1, 16)).sum(-1).argmax(-1)
    for t in range(0, len(q), 7):
        ok = allowed(t, int(starts[t]), len(kv))
        c = slice(16 * int(blocks[t]), 16 * int(blocks[t]) + 16)
        first = kv[ok][:, c] @ qs[t, c]
        bins = csa2_select_decode.bins(first).tolist()
        keep = ok
        if len(ok) > 128:
            cut = sorted(bins, reverse=True)[127]
            keep = [e for e, b in zip(ok, bins, strict=True) if b >= cut]
        score = dict(zip(keep, (kv[keep] @ qs[t]).tolist(), strict=True))
        assert got(entry, valid, t) == best(score, keep, TOP), t
    print("keep 128: = the per-query loop")
    torch.library.opcheck(csa2_select_decode.picks, (q, kv, starts, M, TOP, 128, 16))
    print("opcheck: passes")


if __name__ == "__main__":
    main()
