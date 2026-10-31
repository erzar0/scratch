"""Triton decode kernel: global halving with sampled-median cutoffs.

The kernel of sparse/sampled_halving.py (SampledHalving); shapes as in
sparse/kernel.py. Each early round's cutoff comes from a sample of its keys,
picked by the same position hash and scored one round ahead, so a round is one
pass that writes only its survivors. An entry is the key's position (int32)
and dot0, dot1: the partial dot products of the two query heads with the key
over the dims read so far (summed in fp32, stored as fp16), 8 B in all.

Launches, with S = rounds - TAIL sampled rounds:
    first_sample  every position; the sampled ones get their first-round score
    cut           one program per KV head: the cutoff from the round's sample
    round         candidates add one round of dims; those at or above the cut
                  take slots from an atomic counter, losers' scaled partial
                  logits go to approx (approximate losers only), and sampled
                  survivors add the next round's dims into the next sample
                  (cut and round alternate S times)
Then the last TAIL rounds exact: kernel._global_rounds and kernel.attend, or
for approximate losers _tail_approx, the floor (torch), kernel._attend_part on
the winners and _loser_part on every value row, merged as kernel.attend does.
"""

import math

import torch
import triton
import triton.language as tl

from sparse import kernel
from sparse.kernel import (
    BLOCK,
    CHUNK,
    DIMS,
    _add_round,
    _pick_bin,
    _round_score,
    _sortable,
    _top,
)
from sparse.sampled_halving import HEADROOM, MASK, SAMPLE, TAIL

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

SALT = 0x61C88647  # sampled_halving.position_hash's round salt
SPLITS = 64  # programs per KV head in _loser_part


@triton.jit
def _hash(pos, salt):
    """sampled_halving.position_hash, in int64."""
    h = (pos.to(tl.int64) + salt) & 0x7FFFFFFF
    h = ((h ^ (h >> 15)) * 0x5BD1E995) & 0x7FFFFFFF
    h = ((h ^ (h >> 15)) * 0x5BD1E995) & 0x7FFFFFFF
    return h ^ (h >> 15)


@triton.jit
def _append(buf, counter, val, mask, CAP: tl.constexpr):
    """Store the masked values at slots taken from an atomic counter."""
    base = tl.atomic_add(counter, tl.sum(mask.to(tl.int32), 0))
    slot = base + tl.cumsum(mask.to(tl.int32), 0) - 1
    tl.store(buf + slot, val, mask=mask & (slot < CAP))


@triton.jit
def _store_logits(approx_ptr, scale_ptr, g, r, n, pos, a0, a1, mask, NR):
    """Each head's scaled partial logits after group r, at the keys' positions."""
    s0 = a0 * tl.load(scale_ptr + (g * 2) * NR + r)
    s1 = a1 * tl.load(scale_ptr + (g * 2 + 1) * NR + r)
    tl.store(approx_ptr + (g * 2) * n + pos, s0, mask=mask)
    tl.store(approx_ptr + (g * 2 + 1) * n + pos, s1, mask=mask)


@triton.jit
def _first_sample(
    q_ptr,
    order_ptr,
    scale_ptr,
    kt_ptr,
    smp_ptr,
    smp_count,
    n,
    salt,
    lim,
    s_kt_g,
    s_kt_d,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    NR: tl.constexpr,
    B: tl.constexpr,
    LEAD: tl.constexpr,
    SCAP: tl.constexpr,
):
    g = tl.program_id(0)
    pos = tl.program_id(1) * B + tl.arange(0, B)
    pick = (pos < n) & (_hash(pos, salt) <= lim)
    a0 = tl.zeros((B,), tl.float32)
    a1 = tl.zeros((B,), tl.float32)
    for j in tl.static_range(LEAD + 1):
        a0, a1 = _add_round(
            q_ptr, order_ptr, kt_ptr, g, j, pos, pick, a0, a1, s_kt_g, s_kt_d, D, DIMS
        )
    score = _round_score(scale_ptr, g, LEAD, a0, a1, pick, NR)
    _append(smp_ptr + g * SCAP, smp_count + g, score, pick, SCAP)


@triton.jit
def _cut(
    smp_ptr,
    smp_count,
    count_ptr,
    cut_ptr,
    n,
    target,
    FIRST: tl.constexpr,
    SCAP: tl.constexpr,
):
    """Sortable key of the need-th largest sample; 0 (keep all) if few keys."""
    g = tl.program_id(0)
    m = tl.minimum(tl.load(smp_count + g), SCAP)
    if FIRST:
        count = n
    else:
        count = tl.maximum(tl.load(count_ptr + g), 1)
    need = (m * target + count - 1) // count  # ceil
    i = tl.arange(0, SCAP)
    match = i < m
    key = _sortable(tl.load(smp_ptr + g * SCAP + i, mask=match, other=0.0))
    floor = tl.full((), 0, tl.uint32)
    left = need
    for p in tl.static_range(4):
        shift = 24 - 8 * p
        digit = ((key >> shift) & 0xFF).to(tl.int32)
        b, left = _pick_bin(tl.histogram(digit, 256, mask=match), left)
        match = match & (digit == b)
        floor = floor | (b.to(tl.uint32) << shift)
    floor = tl.where((count <= target) | (need <= 0), tl.full((), 0, tl.uint32), floor)
    tl.store(cut_ptr + g, floor.to(tl.int32, bitcast=True))


@triton.jit
def _round(
    q_ptr,
    order_ptr,
    scale_ptr,
    kt_ptr,
    src_pos,
    src_d0,
    src_d1,
    src_count,
    cut_ptr,
    dst_count,
    dst_pos,
    dst_d0,
    dst_d1,
    smp_ptr,
    smp_count,
    approx_ptr,
    n,
    w_src,
    w_dst,
    r,
    salt,
    lim,
    s_kt_g,
    s_kt_d,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    NR: tl.constexpr,
    B: tl.constexpr,
    FIRST: tl.constexpr,
    LEAD: tl.constexpr,
    SAMPLE: tl.constexpr,
    APPROX: tl.constexpr,
    SCAP: tl.constexpr,
):
    g = tl.program_id(0)
    i = tl.program_id(1) * B + tl.arange(0, B)
    if FIRST:  # every key; the first round reads LEAD + 1 groups of DIMS dims
        pos = i
        alive = i < n
        a0 = tl.zeros((B,), tl.float32)
        a1 = tl.zeros((B,), tl.float32)
        for j in tl.static_range(LEAD + 1):
            a0, a1 = _add_round(
                q_ptr,
                order_ptr,
                kt_ptr,
                g,
                j,
                pos,
                alive,
                a0,
                a1,
                s_kt_g,
                s_kt_d,
                D,
                DIMS,
            )
    else:  # the last round's survivors, compacted
        alive = i < tl.minimum(tl.load(src_count + g), w_src)
        pos = tl.load(src_pos + g * w_src + i, mask=alive, other=0)
        a0 = tl.load(src_d0 + g * w_src + i, mask=alive, other=0.0).to(tl.float32)
        a1 = tl.load(src_d1 + g * w_src + i, mask=alive, other=0.0).to(tl.float32)
        a0, a1 = _add_round(
            q_ptr,
            order_ptr,
            kt_ptr,
            g,
            r + LEAD,
            pos,
            alive,
            a0,
            a1,
            s_kt_g,
            s_kt_d,
            D,
            DIMS,
        )
    score = _round_score(scale_ptr, g, r + LEAD, a0, a1, alive, NR)
    cut = tl.load(cut_ptr + g).to(tl.uint32, bitcast=True)
    win = alive & (_sortable(score) >= cut)
    base = tl.atomic_add(dst_count + g, tl.sum(win.to(tl.int32), 0))
    slot = base + tl.cumsum(win.to(tl.int32), 0) - 1
    # ponytail: survivors past the slots lose (an eighth over the budget is
    # about 6 standard deviations away); a second pass could place them
    win = win & (slot < w_dst)
    tl.store(dst_pos + g * w_dst + slot, pos, mask=win)
    tl.store(dst_d0 + g * w_dst + slot, a0, mask=win)
    tl.store(dst_d1 + g * w_dst + slot, a1, mask=win)
    if APPROX:
        _store_logits(
            approx_ptr, scale_ptr, g, r + LEAD, n, pos, a0, a1, alive & ~win, NR
        )
    if SAMPLE:  # the next round's sample, scored on its dims
        pick = win & (_hash(pos, salt) <= lim)
        b0, b1 = _add_round(
            q_ptr,
            order_ptr,
            kt_ptr,
            g,
            r + 1 + LEAD,
            pos,
            pick,
            a0,
            a1,
            s_kt_g,
            s_kt_d,
            D,
            DIMS,
        )
        nxt = _round_score(scale_ptr, g, r + 1 + LEAD, b0, b1, pick, NR)
        _append(smp_ptr + g * SCAP, smp_count + g, nxt, pick, SCAP)


@triton.jit
def _tail_approx(
    q_ptr,
    order_ptr,
    scale_ptr,
    kt_ptr,
    pos_ptr,
    a0_ptr,
    a1_ptr,
    win_ptr,
    approx_ptr,
    n,
    first,
    s_kt_g,
    s_kt_d,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    NR: tl.constexpr,
    C: tl.constexpr,
    KEEP: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    """kernel._global_rounds, also writing losers' logits and -inf for winners."""
    g = tl.program_id(0)
    idx = g * C + tl.arange(0, C)
    pos = tl.load(pos_ptr + idx)
    alive = pos >= 0
    a0 = tl.load(a0_ptr + idx, mask=alive, other=0.0)
    a1 = tl.load(a1_ptr + idx, mask=alive, other=0.0)
    for t in tl.static_range(ROUNDS):
        r = first + t
        a0, a1 = _add_round(
            q_ptr, order_ptr, kt_ptr, g, r, pos, alive, a0, a1, s_kt_g, s_kt_d, D, DIMS
        )
        score = _round_score(scale_ptr, g, r, a0, a1, alive, NR)
        won = _top(score, alive, KEEP << (ROUNDS - 1 - t))
        _store_logits(approx_ptr, scale_ptr, g, r, n, pos, a0, a1, alive & ~won, NR)
        alive = won
    slot = tl.cumsum(alive.to(tl.int32), 0) - 1
    won = alive & (slot < KEEP)
    tl.store(win_ptr + g * KEEP + slot, pos, mask=won)
    last = first + ROUNDS - 1  # keys tied at the last cutoff past KEEP lose
    _store_logits(approx_ptr, scale_ptr, g, last, n, pos, a0, a1, alive & ~won, NR)
    tl.store(approx_ptr + (g * 2) * n + pos, float("-inf"), mask=won)
    tl.store(approx_ptr + (g * 2 + 1) * n + pos, float("-inf"), mask=won)


@triton.jit
def _online(m, total, out, s, v):
    """Running softmax (max, weight sum, weighted values) plus one chunk."""
    top = tl.maximum(m, tl.max(s, 0))
    scale = tl.exp(m - top)
    p = tl.exp(s - top)
    return top, total * scale + tl.sum(p, 0), out * scale + tl.sum(p[:, None] * v, 0)


@triton.jit
def _loser_part(
    approx_ptr,
    floor_ptr,
    v_ptr,
    m_out,
    l_out,
    o_out,
    n,
    span,
    s_g,
    s_n,
    D: tl.constexpr,
    CH: tl.constexpr,
):
    """Partials of the losers: logits min(approx, floor), span keys per program."""
    g = tl.program_id(0)
    c = tl.program_id(1)
    parts = tl.num_programs(1)
    dd = tl.arange(0, D)
    f0 = tl.load(floor_ptr + g * 2)
    f1 = tl.load(floor_ptr + g * 2 + 1)
    m0 = tl.full((), -1e30, tl.float32)  # finite: exp(m - top) stays defined
    m1 = tl.full((), -1e30, tl.float32)
    l0 = tl.full((), 0.0, tl.float32)
    l1 = tl.full((), 0.0, tl.float32)
    o0 = tl.zeros((D,), tl.float32)
    o1 = tl.zeros((D,), tl.float32)
    for j in range(0, span, CH):
        pos = c * span + j + tl.arange(0, CH)
        ok = pos < n
        rows = g * s_g + pos[:, None] * s_n + dd[None, :]
        v = tl.load(v_ptr + rows, mask=ok[:, None], other=0.0).to(tl.float32)
        a0 = tl.load(approx_ptr + (g * 2) * n + pos, mask=ok, other=float("-inf"))
        a1 = tl.load(approx_ptr + (g * 2 + 1) * n + pos, mask=ok, other=float("-inf"))
        m0, l0, o0 = _online(m0, l0, o0, tl.minimum(a0, f0), v)
        m1, l1, o1 = _online(m1, l1, o1, tl.minimum(a1, f1), v)
    at = (g * 2) * parts + c
    tl.store(m_out + at, m0)
    tl.store(l_out + at, l0)
    tl.store(o_out + at * D + dd, o0)
    tl.store(m_out + at + parts, m1)
    tl.store(l_out + at + parts, l1)
    tl.store(o_out + (at + parts) * D + dd, o1)


def select(
    q: torch.Tensor,
    k_t: torch.Tensor,
    keep: int,
    first: int = 8,
    sample: int = SAMPLE,
    approx: torch.Tensor | None = None,
) -> torch.Tensor:
    """(G, keep) winners of sampled halving, -1 for empty slots.

    With approx (G, 2, n), each loser's scaled partial logits go there (-inf
    at the winners).
    """
    groups, dim, n = k_t.shape
    total = math.ceil(math.log2(n / keep))
    spread = total - TAIL  # sampled rounds
    lead = first // DIMS - 1  # extra groups of dims in the first round
    assert spread >= 1 and total + lead <= dim // DIMS
    dims, q_ord, scale = kernel.order(q)
    budget = [keep << (total - 1 - t) for t in range(total)]
    scap = triton.next_power_of_2(min(2 * sample, n))  # sample slots
    dev = q.device
    found, picked = torch.zeros(2, spread, groups, dtype=torch.int32, device=dev)
    samples = torch.empty(spread, groups, scap, device=dev)
    cut = torch.empty(spread, groups, dtype=torch.int32, device=dev)
    # survivor slots per round: an eighth over the budget, and for the tail a
    # power of 2 (8 keep, above the last sampled round's 4.5 keep)
    tail = keep << (TAIL + 1)
    slots = [b + b // HEADROOM for b in budget[: spread - 1]] + [tail]
    early = slots[:-1]  # sampled rounds before the last
    pos = [torch.empty(groups, w, dtype=torch.int32, device=dev) for w in early]
    pos.append(torch.full((groups, tail), -1, dtype=torch.int32, device=dev))
    dots = [torch.empty(2, groups, w, dtype=torch.float16, device=dev) for w in early]
    dots.append(torch.empty(2, groups, tail, device=dev))  # fp32 for the tail

    def lim(m: int, count: int) -> int:  # a hash at most lim: chance m / count
        return min((m << 31) // count, MASK + 1) - 1

    common = (k_t.stride(0), k_t.stride(1), dim, DIMS, dim // DIMS, BLOCK)
    _first_sample[(groups, triton.cdiv(n, BLOCK))](
        q_ord,
        dims,
        scale,
        k_t,
        samples[0],
        picked[0],
        n,
        SALT & MASK,
        lim(sample, n),
        *common,
        lead,
        scap,
    )
    for t in range(spread):
        src = max(t - 1, 0)
        _cut[(groups,)](
            samples[t],
            picked[t],
            found[src],
            cut[t],
            n,
            budget[t],
            t == 0,
            scap,
        )
        nxt = min(t + 1, spread - 1)
        _round[(groups, triton.cdiv(slots[src] if t else n, BLOCK))](
            q_ord,
            dims,
            scale,
            k_t,
            pos[src],
            dots[src][0],
            dots[src][1],
            found[src],
            cut[t],
            found[t],
            pos[t],
            dots[t][0],
            dots[t][1],
            samples[nxt],
            picked[nxt],
            scale if approx is None else approx,  # unused without approx
            n,
            slots[src],
            slots[t],
            t,
            (t + 2) * SALT & MASK,  # the emulation's round t + 2 (counted from 1)
            lim(sample // 2, budget[t]),
            *common,
            t == 0,
            lead,
            t + 1 < spread,
            approx is not None,
            scap,
        )
    won = torch.full((groups, keep), -1, dtype=torch.int32, device=dev)
    head = (q_ord, dims, scale, k_t, pos[spread - 1], *dots[spread - 1], won)
    rest = (k_t.stride(0), k_t.stride(1), dim, DIMS, dim // DIMS, tail, keep, TAIL)
    if approx is None:
        kernel._global_rounds[(groups,)](*head, spread + lead, *rest, num_warps=8)
    else:
        _tail_approx[(groups,)](*head, approx, n, spread + lead, *rest, num_warps=8)
    return won


def tournament(
    q: torch.Tensor,
    k: torch.Tensor,
    k_t: torch.Tensor,
    v: torch.Tensor,
    keep: int,
    first: int = 8,
) -> torch.Tensor:
    """(G, 2, D) sampled halving attention, losers dropped, one decode token."""
    return kernel.attend(q, k, v, select(q, k_t, keep, first))


def tournament_approx(
    q: torch.Tensor,
    k: torch.Tensor,
    k_t: torch.Tensor,
    v: torch.Tensor,
    keep: int,
    first: int = 8,
    sample: int = SAMPLE,
) -> torch.Tensor:
    """(G, 2, D) sampled halving attention, losers approximate, one decode token.

    Winners get exact logits; each loser its last scaled partial logit, capped
    at the lowest winner logit (the floor); one softmax over every key.
    """
    groups, n, dim = k.shape
    assert keep % CHUNK == 0
    approx = torch.empty(groups, 2, n, device=q.device)
    won = select(q, k_t, keep, first, sample, approx)
    # ponytail: the floor takes a few small torch launches; fuse it if it shows
    rows = k.gather(1, won.long().clamp(min=0)[..., None].expand(-1, -1, dim))
    logits = q.float() @ rows.float().mT * dim**-0.5  # (G, 2, keep)
    floor = logits.masked_fill((won < 0)[:, None], float("inf")).amin(-1)
    dev = q.device

    def partials(width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = torch.empty(groups, 2, width, device=dev)
        return m, torch.empty_like(m), torch.empty(groups, 2, width, dim, device=dev)

    parts = keep // CHUNK
    winners = partials(parts)
    kernel._attend_part[(groups, parts)](
        q, k, v, won, *winners, dim**-0.5, k.stride(0), k.stride(1), keep, CHUNK, dim
    )
    span = CHUNK * triton.cdiv(n, CHUNK * SPLITS)
    losers = partials(triton.cdiv(n, span))
    _loser_part[(groups, losers[0].shape[-1])](
        approx, floor, v, *losers, n, span, v.stride(0), v.stride(1), dim, CHUNK
    )
    m, total, out = (torch.cat(x, 2) for x in zip(winners, losers, strict=True))
    weight = (m - m.amax(-1, keepdim=True)).exp()
    return (weight[..., None] * out).sum(2) / (weight * total).sum(2)[..., None]
