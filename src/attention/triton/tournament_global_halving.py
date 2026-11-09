"""Triton decode kernel: the global halving tournament, every round over all keys.

The user's sort-based variant with losers dropped and an 8-dim first round:
every key is scored on the top 8 dims of |q|, the better half goes on and adds
8 more per round, the last keep get exact attention. The better half is found by
an exact radix select over all survivors of a KV head, so winners match
Tournament("query", "halving", "drop", keep). Shapes as in attention/triton/common.py.

Launches, while more than keep * 2^GLOBAL_ROUNDS keys survive:
    round     one program per (KV head, BLOCK candidates): the last round's
              survivors (score at or above its cutoff) take slots from an
              atomic counter and add 8 dims; (pos, a0, a1, score) is written
              compacted, and a histogram of each score's top byte added up
    digit x3  the histogram of the next byte over the entries that match the
              bytes so far; every program re-derives them from the histograms
Then one compaction, common._global_rounds (the last rounds, one program per KV
head) and common.attend.
"""

import torch
import triton
import triton.language as tl

from attention.tournament.general import halving_rounds
from attention.triton import common
from attention.triton.common import (
    BLOCK,
    DIMS,
    GLOBAL_ROUNDS,
    _add_round,
    _byte,
    _claim_slots,
    _pick_bin,
    _round_score,
    _sortable,
    _store_entries,
)

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)


@triton.jit
def _cutoff(hist, need, PASSES: tl.constexpr):
    """The first PASSES bytes of the need-th largest key, read off the byte
    histograms hist (4 x 256 bins); True if need or fewer keys are counted."""
    bins = tl.arange(0, 256)
    few = tl.sum(tl.load(hist + bins), 0) <= need
    prefix = tl.full((), 0, tl.uint32)
    for p in tl.static_range(PASSES):
        b, need = _pick_bin(tl.load(hist + p * 256 + bins), need)
        prefix = prefix | (b.to(tl.uint32) << (24 - 8 * p))
    return prefix, few


@triton.jit
def _round(
    q_ptr,
    order_ptr,
    scale_ptr,
    kt_ptr,
    s_kt_g,
    s_kt_d,
    src_pos,
    src_val,
    hist_ptr,
    count_ptr,
    dst_pos,
    dst_val,
    n,
    w_src,
    w_dst,
    r,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    NR: tl.constexpr,
    B: tl.constexpr,
    FIRST: tl.constexpr,
    ADD: tl.constexpr,
):
    """One program per (KV head, B entries): round r - 1's survivors (every key in
    the first round) take slots; with ADD they read round r's dims, and their
    scores' top bytes go into round r's histogram."""
    g = tl.program_id(0)
    keys = kt_ptr + g * s_kt_g  # KV head g's K_T
    groups = tl.num_programs(0)
    i = tl.program_id(1) * B + tl.arange(0, B)
    if FIRST:  # every key, written in place
        pos = i
        alive = i < n
        a0 = tl.zeros((B,), tl.float32)
        a1 = tl.zeros((B,), tl.float32)
        slot = i
    else:  # the last round's survivors, compacted
        pos = tl.load(src_pos + g * w_src + i, mask=i < w_src, other=-1)
        alive = pos >= 0
        a0 = tl.load(src_val + g * w_src + i, mask=alive, other=0.0)
        a1 = tl.load(src_val + (groups + g) * w_src + i, mask=alive, other=0.0)
        score = tl.load(src_val + (2 * groups + g) * w_src + i, mask=alive, other=0.0)
        floor, few = _cutoff(hist_ptr + ((r - 1) * groups + g) * 1024, w_dst, 4)
        alive = alive & ((_sortable(score) >= floor) | few)
        slot = _claim_slots(count_ptr + r * groups + g, alive)
        # ponytail: keys tied at the cutoff fill the last slots in arbitrary order
        alive = alive & (slot < w_dst)
    if ADD:
        a0, a1 = _add_round(
            q_ptr, order_ptr, keys, s_kt_d, g, r, pos, alive, a0, a1, D, DIMS
        )
        score = _round_score(scale_ptr, g, r, a0, a1, alive, NR)
        here = hist_ptr + (r * groups + g) * 1024 + tl.arange(0, 256)
        tl.atomic_add(here, tl.histogram(_byte(_sortable(score), 0), 256, mask=alive))
        tl.store(dst_val + (2 * groups + g) * w_dst + slot, score, mask=alive)
    dst_a0 = dst_val + g * w_dst
    dst_a1 = dst_val + (groups + g) * w_dst
    _store_entries(dst_pos + g * w_dst, dst_a0, dst_a1, slot, pos, a0, a1, alive)


@triton.jit
def _digit(pos_ptr, val_ptr, hist_ptr, w, need, P: tl.constexpr, B: tl.constexpr):
    """One program per (KV head, B entries): the histogram of byte P of the scores
    whose first P bytes match the cutoff's."""
    g = tl.program_id(0)
    groups = tl.num_programs(0)
    i = tl.program_id(1) * B + tl.arange(0, B)
    pos = tl.load(pos_ptr + g * w + i, mask=i < w, other=-1)
    alive = pos >= 0
    score = tl.load(val_ptr + (2 * groups + g) * w + i, mask=alive, other=0.0)
    key = _sortable(score)
    hist = hist_ptr + g * 1024
    prefix, _ = _cutoff(hist, need, P)
    match = alive & ((key >> (32 - 8 * P)) == (prefix >> (32 - 8 * P)))
    tl.atomic_add(
        hist + P * 256 + tl.arange(0, 256),
        tl.histogram(_byte(key, P), 256, mask=match),
    )


def round_buffers(
    groups: int, widths: list[int], dev: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Zeroed histograms (4 x 256 bins per sorting round and KV head) and slot
    counters, and each round's survivor lists: positions (G, w), -1 for empty
    (round 0: every key), and values (3, G, w): a0, a1 and the score."""
    n, spread = widths[0], len(widths) - 1
    size = spread * groups * 1024 + (spread + 1) * groups
    counts = torch.zeros(size, dtype=torch.int32, device=dev)
    hist, count = counts.split([spread * groups * 1024, (spread + 1) * groups])
    empty = torch.full((groups * sum(widths[1:]),), -1, dtype=torch.int32, device=dev)
    pos = [torch.empty(groups, n, dtype=torch.int32, device=dev)]
    pos += [p.view(groups, -1) for p in empty.split([groups * w for w in widths[1:]])]
    val = [torch.empty(3, groups, w, device=dev) for w in widths]
    return hist, count, pos, val


def select(q: torch.Tensor, k_t: torch.Tensor, keep: int) -> torch.Tensor:
    """(G, keep) winner positions of global halving, -1 for empty slots."""
    groups, dim, n = k_t.shape
    total = halving_rounds(n, keep)
    spread = total - GLOBAL_ROUNDS  # rounds over all programs of a KV head
    assert spread >= 1 and total * DIMS <= dim
    read = common.reading(q, k_t)
    budget = [keep << (total - 1 - t) for t in range(total)]
    widths = [n, *budget[:spread]]  # round t's output; the last is the compaction
    hist, count, pos, val = round_buffers(groups, widths, q.device)
    for t in range(spread + 1):
        src = max(t - 1, 0)
        _round[(groups, triton.cdiv(widths[src], BLOCK))](
            *read,
            src_pos=pos[src],
            src_val=val[src],
            hist_ptr=hist,
            count_ptr=count,
            dst_pos=pos[t],
            dst_val=val[t],
            n=n,
            w_src=widths[src],
            w_dst=widths[t],
            r=t,
            **common.sizes(dim),
            B=BLOCK,
            FIRST=t == 0,
            ADD=t < spread,
        )
        if t == spread:
            break
        for p in (1, 2, 3):  # the cutoff's next bytes
            _digit[(groups, triton.cdiv(widths[t], BLOCK))](
                pos[t],
                val[t],
                hist[t * groups * 1024 :],
                widths[t],
                budget[t],
                P=p,
                B=BLOCK,
            )
    won = torch.full((groups, keep), -1, dtype=torch.int32, device=q.device)
    common._global_rounds[(groups,)](
        *read,
        pos_ptr=pos[-1],
        a0_ptr=val[-1][0],
        a1_ptr=val[-1][1],
        win_ptr=won,
        first=spread,
        **common.sizes(dim),
        C=widths[-1],
        KEEP=keep,
        ROUNDS=GLOBAL_ROUNDS,
        num_warps=8,
    )
    return won


def tournament(
    q: torch.Tensor, k: torch.Tensor, k_t: torch.Tensor, v: torch.Tensor, keep: int
) -> torch.Tensor:
    """(G, 2, D) global halving tournament attention for one decode token."""
    return common.attend(q, k, v, select(q, k_t, keep))
