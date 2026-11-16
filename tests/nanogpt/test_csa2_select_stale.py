"""csa2_select_stale: every 1 is csa2_select_block.block; every 8 matches a
per-query loop (blocks ranked by query t - t % 8, entries finished since kept). CPU."""

import torch

from nanogpt import csa2_select_block, csa2_select_stale
from nanogpt.csa2_select_block import block_scores
from tests.nanogpt.test_csa2_select import TOP, M, allowed, best, got, packed

SIZE, KEEP, EVERY = 8, 16, 8


def test_every_query_is_block() -> None:
    q, kv, starts = packed("cpu")
    entry, valid = csa2_select_stale.stale(q, kv, starts, M, TOP, SIZE, KEEP, 1)
    want = csa2_select_block.block(q, kv, starts, M, TOP, SIZE, KEEP)
    assert all(got(entry, valid, t) == got(*want, t) for t in range(len(q)))


def test_stale_blocks_match_loop() -> None:
    q, kv, starts = packed("cpu")
    qs = q.sum(1)
    entry, valid = csa2_select_stale.stale(q, kv, starts, M, TOP, SIZE, KEEP, EVERY)
    for t in range(0, len(q), 7):
        r, start = t - t % EVERY, int(starts[t])
        then, ok = allowed(r, start, len(kv)), allowed(t, start, len(kv))
        ids = sorted({e // SIZE for e in then})
        whole = [i for i in ids if set(range(i * SIZE, (i + 1) * SIZE)) <= set(then)]
        part = [i for i in ids if i not in whole]
        if whole:
            blocks = torch.stack([kv[i * SIZE : (i + 1) * SIZE] for i in whole])
            s = block_scores(qs[r][None], blocks, "mean")[0]
            whole = [whole[i] for i in s.argsort(descending=True)[:KEEP].tolist()]
        keep = sorted(
            {e for e in then if e // SIZE in {*whole, *part}} | {*ok} - {*then}
        )
        score = dict(zip(keep, (kv[keep] @ qs[t]).tolist(), strict=True))
        assert got(entry, valid, t) == best(score, keep, TOP), t
