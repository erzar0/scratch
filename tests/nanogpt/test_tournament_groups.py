"""Group-halving training tournament (src/nanogpt/tournament_groups.py): with one
group per round the winners are tournament_fast's (no ties in float32 data), on
documents of one block and of several (key copies); groups of 2, 64 and 256 pick
a float64 torch replay's winners (at least 99.9% of them; near-ties differ); the
bit words hold the winners and attention() is the softmax over them, forward and
backward, in chunks too, and compiles in a no-grad model. Needs CUDA:
    uv run python -m tests.nanogpt.test_tournament_groups
"""

import functools
from unittest import mock

import torch
from torch import Tensor

from nanogpt import tournament_fast
from nanogpt import tournament_groups as groups
from nanogpt.tournament_fast import Settings
from tests.nanogpt import test_tournament_fast as fast_tests
from tests.nanogpt.test_tournament import packed
from tests.nanogpt.test_tournament_fast import first_config, winner_mask

RUN2 = Settings(2048, 64, 16)
DOCS = (300, 1200, 548)  # one block each at RUN2


def replay(q: Tensor, k: Tensor, cu: Tensor, settings: Settings, group: int) -> Tensor:
    """(H, T, T) winners of the group rule in float64, documents of one block."""
    length, heads, dim = q.shape
    query = torch.empty(heads, length, settings.channels, device=q.device)
    order = torch.empty_like(query, dtype=torch.int32)
    tournament_fast._order[(heads * length,)](
        q.contiguous(), query, order, length, heads, D=dim, CHANNELS=settings.channels
    )
    mask = torch.zeros(heads, length, length, dtype=torch.bool, device=q.device)
    dims, quota = settings.dims, settings.winners
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True):
        if e <= s:
            continue
        assert e - s <= settings.block, "one block per document"
        t = torch.arange(s, e, device=q.device)
        n = t - s + 1
        rounds = sum(((quota << d) < n).long() for d in range(settings.rounds))
        for width in sorted(set((quota * 2**rounds).tolist())):
            mine = (quota * 2**rounds) == width
            tq, nq = t[mine], n[mine]
            for h in range(heads):
                cur = torch.arange(width, device=q.device).repeat(len(tq), 1)
                alive = cur < nq[:, None]
                dot = torch.zeros(cur.shape, dtype=torch.float64, device=q.device)
                keys = k[s:e, h].double()
                for r in range(width.bit_length() - quota.bit_length()):
                    ch = order[h, tq, r * dims : (r + 1) * dims].long()
                    qv = query[h, tq, r * dims : (r + 1) * dims].double()
                    kk = keys[cur.clamp(max=e - s - 1)]  # (Q, slots, D)
                    part = kk.gather(2, ch[:, None, :].expand(-1, cur.shape[1], -1))
                    dot = dot + torch.where(alive, (part * qv[:, None]).sum(-1), 0.0)
                    score = torch.where(alive, dot, float("-inf"))
                    w = cur.shape[1]
                    g = w if w < 512 else min(group, w)
                    idx = torch.arange(w, device=q.device).view(g, w // g).T  # strided
                    by = score[:, idx]  # (Q, groups, g), slots in index order
                    best = (-by).argsort(dim=-1, stable=True)[..., : g // 2]
                    keep = idx[None].expand(len(tq), -1, -1).gather(2, best).flatten(1)
                    cur, alive = cur.gather(1, keep), alive.gather(1, keep)
                    dot = dot.gather(1, keep)
                rows = tq[:, None].expand_as(cur)
                mask[h, rows[alive], s + cur[alive]] = True
    return mask


def one_group_is_tournament_fast(docs: tuple[int, ...]) -> None:
    q, k, _, cu = packed(docs)
    length, longest = q.shape[0], max(docs)
    got = groups.select(q, k, cu, longest, RUN2, group=RUN2.block)
    want = tournament_fast.select(q, k, cu, longest, RUN2)
    assert torch.equal(winner_mask(got, length), winner_mask(want, length)), docs


def groups_follow_the_rule(group: int) -> None:
    q, k, _, cu = packed(DOCS)
    got = winner_mask(groups.select(q, k, cu, max(DOCS), RUN2, group=group), len(q))
    want = replay(q, k, cu, RUN2, group)
    shared, total = int((got & want).sum()), int(want.sum())
    assert abs(int(got.sum()) - total) <= 0.001 * total, (group, int(got.sum()), total)
    assert shared >= 0.999 * total, (group, shared, total)


if __name__ == "__main__":
    tournament_fast._rounds.early_config_prune = first_config
    one_group_is_tournament_fast(DOCS)
    one_group_is_tournament_fast((5, 1500, 4500))
    print("one group per round: tournament_fast's winners (one block and several)")
    for g in (2, 64, 256):
        groups_follow_the_rule(g)
    print("groups of 2, 64, 256: the float64 replay's winners")
    for g in (2, 64):
        names = {
            n: functools.partial(getattr(groups, n), group=g)
            for n in ("attention", "picked", "select")
        }
        with mock.patch.multiple(fast_tests, **names):
            fast_tests.bits_are_the_winners(RUN2)
            fast_tests.attention_is_the_softmax_over_the_winners(RUN2)
            fast_tests.chunks_are_the_whole_row(RUN2)
            if g == 64:
                fast_tests.compiles_in_a_no_grad_model()
    print("bit words = winners; attention = softmax over them; chunks; compiles")
