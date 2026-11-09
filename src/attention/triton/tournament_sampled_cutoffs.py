"""Triton decode kernel: global halving with sampled-median cutoffs.

The kernel of attention/tournament/sampled_cutoffs.py (SampledHalving); shapes as in
attention/triton/common.py. Each early round's cutoff comes from a sample of its keys,
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
Then the last TAIL rounds exact: common._global_rounds and common.attend, or
for approximate losers _tail_approx, the floor (torch), common._attend_part on
the winners and _loser_part on every value row, merged as common.attend does.
"""

import torch
import triton
import triton.language as tl

from attention.tournament.general import halving_rounds
from attention.tournament.sampled_cutoffs import HEADROOM, MASK, SAMPLE, TAIL
from attention.triton import common
from attention.triton.common import (
    BLOCK,
    CHUNK,
    DIMS,
    _add_round,
    _claim_slots,
    _load_entries,
    _radix_floor,
    _round_score,
    _slots,
    _sortable,
    _store_entries,
    _top,
)

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
    slot = _claim_slots(counter, mask)
    tl.store(buf + slot, val, mask=mask & (slot < CAP))


@triton.jit
def _store_logits(approx_ptr, scale_ptr, g, r, n, pos, a0, a1, mask, NR):
    """Each head's scaled partial logits after group r, at the keys' positions."""
    s0 = a0 * tl.load(scale_ptr + (g * 2) * NR + r)
    s1 = a1 * tl.load(scale_ptr + (g * 2 + 1) * NR + r)
    tl.store(approx_ptr + (g * 2) * n + pos, s0, mask=mask)
    tl.store(approx_ptr + (g * 2 + 1) * n + pos, s1, mask=mask)


@triton.jit
def _first_round_dots(
    q_ptr,
    order_ptr,
    keys,
    s_kt_d,
    g,
    pos,
    mask,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    B: tl.constexpr,
    LEAD: tl.constexpr,
):
    """Both heads' dot products over the first round's LEAD + 1 groups of DIMS."""
    a0 = tl.zeros((B,), tl.float32)
    a1 = tl.zeros((B,), tl.float32)
    for j in tl.static_range(LEAD + 1):
        a0, a1 = _add_round(
            q_ptr, order_ptr, keys, s_kt_d, g, j, pos, mask, a0, a1, D, DIMS
        )
    return a0, a1


@triton.jit
def _first_sample(
    q_ptr,
    order_ptr,
    scale_ptr,
    kt_ptr,
    s_kt_g,
    s_kt_d,
    smp_ptr,
    smp_count,
    n,
    salt,
    lim,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    NR: tl.constexpr,
    B: tl.constexpr,
    LEAD: tl.constexpr,
    SCAP: tl.constexpr,
):
    """One program per (KV head, B positions): the first round's scores of the
    positions whose hash is at most lim, appended to the first sample."""
    g = tl.program_id(0)
    keys = kt_ptr + g * s_kt_g  # KV head g's K_T
    pos = tl.program_id(1) * B + tl.arange(0, B)
    pick = (pos < n) & (_hash(pos, salt) <= lim)
    a0, a1 = _first_round_dots(
        q_ptr, order_ptr, keys, s_kt_d, g, pos, pick, D, DIMS, B, LEAD
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
    """One program per KV head: the sortable key of the sample's need-th largest
    score, need = ceil(sample size * target / count); 0 (keep all) if few keys."""
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
    floor = _radix_floor(key, match, need)
    floor = tl.where((count <= target) | (need <= 0), tl.full((), 0, tl.uint32), floor)
    tl.store(cut_ptr + g, floor.to(tl.int32, bitcast=True))


@triton.jit
def _round(
    q_ptr,
    order_ptr,
    scale_ptr,
    kt_ptr,
    s_kt_g,
    s_kt_d,
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
    t,
    salt,
    lim,
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
    """One program per (KV head, B candidates): sampled round t. Candidates add
    the round's dims; those at or above the cut take slots, losers' logits go to
    approx (APPROX), and sampled survivors score the next round into its sample
    (SAMPLE)."""
    g = tl.program_id(0)
    i = tl.program_id(1) * B + tl.arange(0, B)
    keys = kt_ptr + g * s_kt_g  # KV head g's K_T
    r = t + LEAD  # the round's group of DIMS dims: the first round reads LEAD + 1
    if FIRST:  # every key
        pos = i
        alive = i < n
        a0, a1 = _first_round_dots(
            q_ptr, order_ptr, keys, s_kt_d, g, pos, alive, D, DIMS, B, LEAD
        )
    else:  # the last round's survivors, compacted
        alive = i < tl.minimum(tl.load(src_count + g), w_src)
        pos = tl.load(src_pos + g * w_src + i, mask=alive, other=0)
        a0 = tl.load(src_d0 + g * w_src + i, mask=alive, other=0.0).to(tl.float32)
        a1 = tl.load(src_d1 + g * w_src + i, mask=alive, other=0.0).to(tl.float32)
        a0, a1 = _add_round(
            q_ptr, order_ptr, keys, s_kt_d, g, r, pos, alive, a0, a1, D, DIMS
        )
    score = _round_score(scale_ptr, g, r, a0, a1, alive, NR)
    cut = tl.load(cut_ptr + g).to(tl.uint32, bitcast=True)
    win = alive & (_sortable(score) >= cut)
    slot = _claim_slots(dst_count + g, win)
    # ponytail: survivors past the slots lose (an eighth over the budget is
    # about 6 standard deviations away); a second pass could place them
    win = win & (slot < w_dst)
    at = g * w_dst
    _store_entries(dst_pos + at, dst_d0 + at, dst_d1 + at, slot, pos, a0, a1, win)
    if APPROX:
        _store_logits(approx_ptr, scale_ptr, g, r, n, pos, a0, a1, alive & ~win, NR)
    if SAMPLE:  # the next round's sample, scored on its dims
        pick = win & (_hash(pos, salt) <= lim)
        b0, b1 = _add_round(
            q_ptr, order_ptr, keys, s_kt_d, g, r + 1, pos, pick, a0, a1, D, DIMS
        )
        nxt = _round_score(scale_ptr, g, r + 1, b0, b1, pick, NR)
        _append(smp_ptr + g * SCAP, smp_count + g, nxt, pick, SCAP)


@triton.jit
def _tail_approx(
    q_ptr,
    order_ptr,
    scale_ptr,
    kt_ptr,
    s_kt_g,
    s_kt_d,
    pos_ptr,
    a0_ptr,
    a1_ptr,
    win_ptr,
    approx_ptr,
    n,
    first,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    NR: tl.constexpr,
    C: tl.constexpr,
    KEEP: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    """common._global_rounds, also writing losers' logits and -inf for winners."""
    g = tl.program_id(0)
    keys = kt_ptr + g * s_kt_g  # KV head g's K_T
    pos, alive, a0, a1 = _load_entries(pos_ptr, a0_ptr, a1_ptr, g * C + tl.arange(0, C))
    for t in tl.static_range(ROUNDS):
        r = first + t
        a0, a1 = _add_round(
            q_ptr, order_ptr, keys, s_kt_d, g, r, pos, alive, a0, a1, D, DIMS
        )
        score = _round_score(scale_ptr, g, r, a0, a1, alive, NR)
        won = _top(score, alive, KEEP << (ROUNDS - 1 - t))
        _store_logits(approx_ptr, scale_ptr, g, r, n, pos, a0, a1, alive & ~won, NR)
        alive = won
    slot = _slots(alive)
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


def hash_limit(m: int, count: int) -> int:
    """The hash limit that samples each of count keys with chance m / count."""
    return min((m << 31) // count, MASK + 1) - 1


def survivor_buffers(
    groups: int, slots: list[int], dev: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Each sampled round's survivor list: positions (G, w) and dot products
    (2, G, w) in fp16; the last round's (the tail's input) in fp32, empty slots
    -1."""
    *early, tail = slots
    pos = [torch.empty(groups, w, dtype=torch.int32, device=dev) for w in early]
    pos.append(torch.full((groups, tail), -1, dtype=torch.int32, device=dev))
    dots = [torch.empty(2, groups, w, dtype=torch.float16, device=dev) for w in early]
    dots.append(torch.empty(2, groups, tail, device=dev))
    return pos, dots


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
    total = halving_rounds(n, keep)
    spread = total - TAIL  # sampled rounds
    lead = first // DIMS - 1  # extra groups of dims in the first round
    assert spread >= 1 and total + lead <= dim // DIMS
    read = common.reading(q, k_t)
    sizes = common.sizes(dim)
    budget = [keep << (total - 1 - t) for t in range(total)]
    scap = triton.next_power_of_2(min(2 * sample, n))  # sample slots
    dev = q.device
    found, picked = torch.zeros(2, spread, groups, dtype=torch.int32, device=dev)
    samples = torch.empty(spread, groups, scap, device=dev)
    cut = torch.empty(spread, groups, dtype=torch.int32, device=dev)
    # survivor slots per round: an eighth over the budget, and for the tail a
    # power of 2 (8 keep, above the last sampled round's 4.5 keep)
    slots = [b + b // HEADROOM for b in budget[: spread - 1]] + [keep << (TAIL + 1)]
    pos, dots = survivor_buffers(groups, slots, dev)
    _first_sample[(groups, triton.cdiv(n, BLOCK))](
        *read,
        smp_ptr=samples[0],
        smp_count=picked[0],
        n=n,
        salt=SALT & MASK,
        lim=hash_limit(sample, n),
        **sizes,
        B=BLOCK,
        LEAD=lead,
        SCAP=scap,
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
            FIRST=t == 0,
            SCAP=scap,
        )
        nxt = min(t + 1, spread - 1)
        _round[(groups, triton.cdiv(slots[src] if t else n, BLOCK))](
            *read,
            src_pos=pos[src],
            src_d0=dots[src][0],
            src_d1=dots[src][1],
            src_count=found[src],
            cut_ptr=cut[t],
            dst_count=found[t],
            dst_pos=pos[t],
            dst_d0=dots[t][0],
            dst_d1=dots[t][1],
            smp_ptr=samples[nxt],
            smp_count=picked[nxt],
            approx_ptr=samples if approx is None else approx,  # unused without
            n=n,
            w_src=slots[src],
            w_dst=slots[t],
            t=t,
            salt=(t + 2) * SALT & MASK,  # the emulation's round t + 2 (from 1)
            lim=hash_limit(sample // 2, budget[t]),
            **sizes,
            B=BLOCK,
            FIRST=t == 0,
            LEAD=lead,
            SAMPLE=t + 1 < spread,
            APPROX=approx is not None,
            SCAP=scap,
        )
    won = torch.full((groups, keep), -1, dtype=torch.int32, device=dev)
    tail = {
        "pos_ptr": pos[-1],
        "a0_ptr": dots[-1][0],
        "a1_ptr": dots[-1][1],
        "win_ptr": won,
        "first": spread + lead,
        **sizes,
        "C": slots[-1],
        "KEEP": keep,
        "ROUNDS": TAIL,
        "num_warps": 8,
    }
    if approx is None:
        common._global_rounds[(groups,)](*read, **tail)
    else:
        _tail_approx[(groups,)](*read, approx_ptr=approx, n=n, **tail)
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
    return common.attend(q, k, v, select(q, k_t, keep, first))


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
    common._attend_part[(groups, parts)](
        q,
        k,
        v,
        won,
        *winners,
        scaling=dim**-0.5,
        s_g=k.stride(0),
        s_n=k.stride(1),
        K=keep,
        CH=CHUNK,
        D=dim,
    )
    span = CHUNK * triton.cdiv(n, CHUNK * SPLITS)
    losers = partials(triton.cdiv(n, span))
    _loser_part[(groups, losers[0].shape[-1])](
        approx,
        floor,
        v,
        *losers,
        n=n,
        span=span,
        s_g=v.stride(0),
        s_n=v.stride(1),
        D=dim,
        CH=CHUNK,
    )
    m, total, out = (torch.cat(x, 2) for x in zip(winners, losers, strict=True))
    weight = (m - m.amax(-1, keepdim=True)).exp()
    return (weight[..., None] * out).sum(2) / (weight * total).sum(2)[..., None]
