"""Picks at layer 9 inside layer 5's kept blocks, as V4.1-Flash-1017's later layers
pick inside the candidate source's blocks (model.py uses_candidates), parameter-free
(an inference rule: eval only, never trained).

The PF model's pick layers run in order 1 (m = 2), 5 and 9 (m = 1, layer 5's
entries), so the op tells them apart by m and call order (asserted):
    layer 1  first = "block": csa2_select_block's rule; "exact": every allowed entry
             (Flash-1017's layers 2/8/14 score every index key)
    layer 5  csa2_select_block's rule; its kept entries are stored
    layer 9  the top_k by its own summed query among layer 5's kept entries
Arguments and outputs as csa2_select_block.block.
"""

import torch
from torch import Tensor

from nanogpt.csa2_select import CHUNK, _allowed
from nanogpt.csa2_select_block import kept_entries

STATE: dict = {"layer": 9, "kept": None}  # last pick layer run; layer 5's kept


@torch.library.custom_op("csa2_select_reuse::reuse", mutates_args=())
def picks(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int,
    size: int,
    keep: int,
    first: str,
) -> tuple[Tensor, Tensor]:
    """reuse's picks; opaque to torch.compile."""
    with torch.no_grad():
        return _reuse(q, kv, starts, m, top_k, size, keep, first)


@picks.register_fake
def _(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int,
    size: int,
    keep: int,
    first: str,
) -> tuple[Tensor, Tensor]:
    shape = (q.shape[0], top_k)
    return (
        q.new_empty(shape, dtype=torch.long),
        q.new_empty(shape, dtype=torch.bool),
    )


def reuse(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int = 64,
    size: int = 8,
    keep: int = 256,
    first: str = "block",
) -> tuple[Tensor, Tensor]:
    """top_k picks; layer 9's inside layer 5's kept blocks (module docstring)."""
    return picks(q.detach(), kv.detach(), starts, m, top_k, size, keep, first)


def _next_layer(m: int) -> int:
    """1, 5 or 9: the pick layer this call runs, from m and the previous call."""
    last = STATE["layer"]
    layer = 1 if m != 1 else {1: 5, 5: 9}.get(last)
    assert layer is not None and (layer == 1) == (last == 9), (last, m)
    STATE["layer"] = layer
    return layer


def _reuse(
    q: Tensor,
    kv: Tensor,
    starts: Tensor,
    m: int,
    top_k: int,
    size: int,
    keep: int,
    first: str,
) -> tuple[Tensor, Tensor]:
    layer = _next_layer(m)
    length = q.shape[0]
    entry = torch.zeros(length, top_k, dtype=torch.long, device=q.device)
    valid = torch.zeros(length, top_k, dtype=torch.bool, device=q.device)
    if layer == 5:
        width = -(-kv.shape[0] // size) * size
        STATE["kept"] = torch.zeros(length, width, dtype=torch.bool, device=q.device)
    qs = q.float().sum(1)
    for a in range(0, length, CHUNK):
        b = min(a + CHUNK, length)
        lo, hi = -(-int(starts[a]) // m), min(b // m, kv.shape[0])
        if hi <= lo:
            continue
        lo, hi = lo // size * size, -(-hi // size) * size  # whole blocks
        keys = kv[lo:hi].float()
        keys = torch.cat([keys, keys.new_zeros(hi - lo - len(keys), keys.shape[1])])
        allowed = _allowed(starts, m, a, b, lo, hi)
        if layer == 9:
            kept = STATE["kept"][a:b, lo:hi] & allowed
        elif layer == 1 and first == "exact":
            kept = allowed
        else:
            kept = kept_entries(qs[a:b], keys, allowed, size, keep, "mean")
        if layer == 5:
            STATE["kept"][a:b, lo:hi] = kept
        exact = (qs[a:b] @ keys.T).masked_fill(~kept, float("-inf"))
        top, index = exact.topk(min(top_k, hi - lo), -1)
        entry[a:b, : top.shape[1]] = index + lo
        valid[a:b, : top.shape[1]] = top > float("-inf")
    return entry, valid
