"""csa2_select_reuse: layers 1 and 5 pick as their own rules; layer 9 picks the top
64 of its own score inside layer 5's kept entries; out-of-order calls fail. CPU."""

import pytest
import torch
from torch import Tensor

from nanogpt import csa2_select, csa2_select_block, csa2_select_reuse
from tests.nanogpt.test_csa2_select import TOP, M, best, got, packed

SIZE, KEEP = 8, 16
Picks = tuple[Tensor, Tensor]


def same(a: Picks, b: Picks) -> bool:
    return all(got(*a, t) == got(*b, t) for t in range(len(a[0])))


@pytest.mark.parametrize("first", ["block", "exact"])
def test_layers_pick_as_their_rules(first: str) -> None:
    q, kv, starts = packed("cpu")
    q9, kv1 = torch.randn_like(q), torch.randn(len(q), 128)
    csa2_select_reuse.STATE["layer"] = 9
    one = csa2_select_reuse.reuse(q, kv, starts, M, TOP, SIZE, KEEP, first)
    if first == "block":
        want = csa2_select_block.block(q, kv, starts, M, TOP, SIZE, KEEP, "mean")
    else:
        want = csa2_select.exact(q, kv, starts, M, TOP)
    assert same(one, want)
    five = csa2_select_reuse.reuse(q, kv1, starts, 1, TOP, SIZE, KEEP, first)
    assert same(five, csa2_select_block.block(q, kv1, starts, 1, TOP, SIZE, KEEP))
    kept = csa2_select_reuse.STATE["kept"]
    nine = csa2_select_reuse.reuse(q9, kv1, starts, 1, TOP, SIZE, KEEP, first)
    qs = q9.sum(1)
    for t in range(0, len(q), 37):
        inside = [e for e in range(int(starts[t]), t + 1) if kept[t, e]]
        score = dict(zip(inside, (kv1[inside] @ qs[t]).tolist(), strict=True))
        assert got(*nine, t) == best(score, inside, TOP), t


def test_out_of_order_fails() -> None:
    q, _, starts = packed("cpu")
    csa2_select_reuse.STATE["layer"] = 9
    with pytest.raises(AssertionError):
        csa2_select_reuse.reuse(q, torch.randn(len(q), 128), starts, 1)
