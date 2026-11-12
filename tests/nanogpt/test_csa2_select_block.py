"""Block prefilter (src/nanogpt/csa2_select_block.py): with keep at or above every
block it is csa2_select.exact; otherwise each query's picks are the top 64 by the full
score among its `keep` best fully readable blocks plus its partly readable ones (a
per-query loop), for every summary; minmax bounds every entry score of its block;
opcheck passes. Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_select_block
"""

import torch

from nanogpt import csa2_select, csa2_select_block
from nanogpt.csa2_select_block import SUMMARIES, block_scores
from tests.nanogpt.test_csa2_select import TOP, M, allowed, best, got, packed

SIZE, KEEP = 8, 16


def main() -> None:
    q, kv, starts = packed()
    qs = q.sum(1)
    want = csa2_select.exact(q, kv, starts, M, TOP)
    for summary in SUMMARIES:
        full = csa2_select_block.block(q, kv, starts, M, TOP, SIZE, len(kv), summary)
        for t in range(len(q)):
            assert got(*full, t) == got(*want, t), (summary, t)
    print("keep >= blocks: = exact (every summary)")
    for summary in SUMMARIES:
        entry, valid = csa2_select_block.block(
            q, kv, starts, M, TOP, SIZE, KEEP, summary
        )
        for t in range(0, len(q), 7):
            ok = set(allowed(t, int(starts[t]), len(kv)))
            ids = sorted({e // SIZE for e in ok})
            members = [range(i * SIZE, (i + 1) * SIZE) for i in ids]
            whole = [i for i, r in zip(ids, members, strict=True) if set(r) <= ok]
            part = [i for i in ids if i not in whole]
            if whole:
                blocks = torch.stack([kv[i * SIZE : (i + 1) * SIZE] for i in whole])
                s = block_scores(qs[t][None], blocks, summary)[0]
                whole = [whole[i] for i in s.argsort(descending=True)[:KEEP].tolist()]
            keep = [e for e in ok if e // SIZE in set(whole) | set(part)]
            score = dict(zip(keep, (kv[keep] @ qs[t]).tolist(), strict=True))
            assert got(entry, valid, t) == best(score, keep, TOP), (summary, t)
    print(f"keep {KEEP} blocks of {SIZE}: = the per-query loop (every summary)")
    blocks = kv[: 64 * SIZE].view(64, SIZE, -1)
    bound = block_scores(qs, blocks, "minmax")
    assert (bound[:, :, None] >= torch.einsum("qd,bsd->qbs", qs, blocks) - 1e-3).all()
    print("minmax: >= every entry score of its block")
    args = (q, kv, starts, M, TOP, SIZE, KEEP, "minmax")
    torch.library.opcheck(csa2_select_block.picks, args)
    print("opcheck: passes")


if __name__ == "__main__":
    main()
