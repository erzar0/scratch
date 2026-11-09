"""CSA2 picks without an indexer (src/nanogpt/csa2_select.py): exact and tournament
picks equal per-query loop references on a packed row (document and finished-entry
masks, halving rounds, two query chunks); a prefix of top_k or fewer entries picks
every allowed entry; the op passes torch.library.opcheck. Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_select
"""

import torch
from torch import Tensor

from nanogpt import csa2_select
from nanogpt.tournament_channel_blocks import windows
from nanogpt.tournament_fast import Settings

M, TOP, DIMS = 4, 64, 16
DOCS = (300, 2048, 900)  # 812 entries; the 2048 document reaches 3 rounds


def packed() -> tuple[Tensor, Tensor, Tensor]:
    """q (S, 3, 128), kv (S // M, 128), starts (S,) for DOCS in one row."""
    g = torch.Generator().manual_seed(0)
    length = sum(DOCS)
    q = torch.randn(length, 3, 128, generator=g).cuda()
    kv = torch.randn(length // M, 128, generator=g).cuda()
    first = torch.tensor((0, *DOCS[:-1])).cumsum(0)
    starts = first.repeat_interleave(torch.tensor(DOCS)).cuda()
    return q, kv, starts


def allowed(t: int, start: int, entries: int) -> list[int]:
    """Entries query t may pick: finished, inside its document."""
    return [e for e in range(entries) if e * M >= start and e < (t + 1) // M]


def got(entry: Tensor, valid: Tensor, t: int) -> set[int]:
    return set(entry[t][valid[t]].tolist())


def best(score: dict[int, float], keys: set[int] | list[int], n: int) -> set[int]:
    return set(sorted(keys, key=score.__getitem__, reverse=True)[:n])


def exact_matches_loop() -> None:
    q, kv, starts = packed()
    entry, valid = csa2_select.exact(q, kv, starts, M, TOP)
    qs = q.sum(1)
    for t in range(0, len(q), 7):
        ok = allowed(t, int(starts[t]), len(kv))
        score = dict(zip(ok, (kv[ok] @ qs[t]).tolist(), strict=True))
        assert got(entry, valid, t) == best(score, ok, TOP), t


def tournament_matches_loop() -> None:
    q, kv, starts = packed()
    entry, valid = csa2_select.tournament(q, kv, starts, M, TOP, DIMS)
    exact = csa2_select.exact(q, kv, starts, M, TOP)
    rounds = 128 // DIMS
    values, order = windows(q.sum(1)[:, None], Settings(TOP << rounds, TOP, DIMS))
    differs = 0
    for t in range(0, len(q), 13):
        ok = allowed(t, int(starts[t]), len(kv))
        doubles = sum(len(ok) > TOP << d for d in range(rounds))
        alive, score = set(ok), dict.fromkeys(ok, 0.0)
        for r in range(doubles):
            w = slice(r * DIMS, (r + 1) * DIMS)
            keys = sorted(alive)
            part = kv[keys][:, order[0, t, w].long()] @ values[0, t, w]
            for e, p in zip(keys, part.tolist(), strict=True):
                score[e] += p
            floor = sorted((score[e] for e in alive), reverse=True)
            alive = {
                e for e in alive if score[e] >= floor[(TOP << doubles - r - 1) - 1]
            }
        want = best(score, alive, TOP)
        assert got(entry, valid, t) == want, (t, doubles)
        if len(ok) <= TOP:  # short prefix: every allowed entry
            assert want == set(ok), t
        differs += want != got(*exact, t)
    assert differs, "the rounds never changed a pick"


def passes_opcheck() -> None:
    q, kv, starts = packed()
    for dims in (0, DIMS):
        torch.library.opcheck(csa2_select.picks, (q, kv, starts, M, TOP, dims))


if __name__ == "__main__":
    exact_matches_loop()
    print("exact: the loop's top 64 allowed entries")
    tournament_matches_loop()
    print("tournament: the loop's halving rounds")
    passes_opcheck()
    print("opcheck: passes")
