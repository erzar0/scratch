"""csa2_select_head_max: with one head it is csa2_select.exact; with three each
query's picks are the top 64 allowed entries by the largest head logit. CPU."""

import torch

from nanogpt import csa2_select, csa2_select_head_max
from tests.nanogpt.test_csa2_select import TOP, M, allowed, best, got, packed


def test_one_head_is_exact() -> None:
    q, kv, starts = packed("cpu")
    q = q[:, :1]
    entry, valid = csa2_select_head_max.head_max(q, kv, starts, M, TOP)
    want = csa2_select.exact(q, kv, starts, M, TOP)
    assert all(got(entry, valid, t) == got(*want, t) for t in range(len(q)))


def test_picks_top_by_largest_head_logit() -> None:
    q, kv, starts = packed("cpu")
    entry, valid = csa2_select_head_max.head_max(q, kv, starts, M, TOP)
    for t in range(0, len(q), 37):
        ok = allowed(t, int(starts[t]), len(kv))
        score = dict(zip(ok, (kv[ok] @ q[t].T).amax(-1).tolist(), strict=True))
        assert got(entry, valid, t) == best(score, ok, TOP), t
    assert not torch.equal(entry, csa2_select.exact(q, kv, starts, M, TOP)[0])
