"""Fused decode for the CSA2 KV layout, one token, batch 1, bf16 caches: picks and
attention in Triton (experiments/csa2/bench/decode_csa2.py times them against PyTorch).

top64(q, keys, weights): one kernel. Each program scores its block of entries into a
scratch row, then takes their top 64 in one tl.topk (packed score and entry index,
the lower entry first at ties); the last program of each group of up to 16 merges
its group's top 64s, and the last group's merger merges the groups' (atomic
counters, left zero).
    CSA2's learned indexer (weights given): sum_h w_h relu(q_h . k) / sqrt(dim);
    ours, exact (no weights): q . k, q the heads' summed query (only its nonzero
    channels read: record 91 pads 96 to 128); blocks > 0: only q's `blocks` aligned
    16-channel blocks of largest sum |q| read (routed channels, csa2_route: q is zero
    outside its kept blocks, so this is exact; 64 bytes per entry at 2 blocks).
prefilter64: csa2_select_decode.prefilter in two kernels. The first scores every
entry on q's aligned 16-channel block (32 bytes per entry) and counts score bins
(histogram, atomics); the second reads only the entries at or above the cut bin,
takes their top 64 by q . k, merges as top64 and clears the histogram.
csa2_attend: CSA2 attention over 64 picks + the window, K = V, a sink logit per
head, then the output's inverse RoPE on the last ROPE channels.
b_attend: record 91 attention (csa2_flex.attention): each head's own window K, V
(heads may share one, MQA) + the picked entries, a pick's value its key turned back
in channel pairs (pair_table: one cos, one sin per pair); q_pick: the picks' logits
use another query (csa2_route's q_routed).
Both attentions load contiguous channels only (no masks, no permuted gathers): at
one token those cost more than the math.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

from nanogpt.csa2_select_fast import _sortable

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

TOP, BINS = 64, 2048  # picks; prefilter bins
GROUP, GROUPS = 16, 16  # programs per first-level merge; most groups
PROGRAMS = 128  # scoring programs per call, about
WARPS = 4  # per attention program (bench/attend_warps.py)
TILE = 8192  # elements per scoring step: rows = TILE // channels read
LEAST = tl.constexpr(-0x7FFFFFFFFFFFFFFF)  # below every packed score
_buffers: dict[torch.device, tuple[Tensor, Tensor]] = {}


@triton.jit
def _pack(score, entry):
    """int64 in score order, then the lower entry first."""
    return (_sortable(score).to(tl.int64) << 32) | (0x7FFFFFFF - entry).to(tl.int64)


@triton.jit
def _bin(score):
    """csa2_select_decode.bins + 1024: the top 11 bits of the sortable score."""
    return (_sortable(score) >> 21) + 1024


@triton.jit
def _keep_best(best, score, rows, TOP: tl.constexpr, SUB: tl.constexpr):
    """best (TOP packed) updated with SUB scores of rows; skipped when none beats
    the TOP-th."""
    packed = _pack(score, rows)
    if tl.max(packed, 0) > tl.min(best, 0):
        if SUB > TOP:
            best = tl.topk(tl.cat(best, tl.topk(packed, TOP), can_reorder=True), TOP)
        else:
            best = tl.topk(tl.cat(best, packed, can_reorder=True), TOP)
    return best


@triton.jit
def _top_blocks(q_ptr, D: tl.constexpr, L: tl.constexpr, KB: tl.constexpr):
    """Channels (KB L,) of q's KB aligned L-channel blocks of largest sum |q|."""
    mass = tl.abs(tl.load(q_ptr + tl.arange(0, D)).to(tl.float32))
    mass = tl.sum(tl.reshape(mass, (D // L, L)), 1)
    b = tl.arange(0, D // L)
    j = tl.arange(0, KB * L)
    c = j
    for r in tl.static_range(KB):
        top = tl.argmax(mass, 0)
        c = tl.where(j // L == r, top * L + j % L, c)
        mass = tl.where(b == top, -1.0, mass)
    return c


@triton.jit
def _first(
    q_ptr,
    k_ptr,
    score_ptr,
    hist_ptr,
    N,
    BLOCK,
    D: tl.constexpr,
    L: tl.constexpr,
    SUB: tl.constexpr,
    BINS: tl.constexpr,
):
    """Program p: entries [p BLOCK, (p + 1) BLOCK) scored on q's aligned L-channel
    block of largest sum |q|, into score; their bins counted into hist."""
    p = tl.program_id(0)
    i = tl.arange(0, D)
    mass = tl.sum(tl.reshape(tl.abs(tl.load(q_ptr + i).to(tl.float32)), (D // L, L)), 1)
    c = tl.argmax(mass, 0) * L + tl.arange(0, L)
    qb = tl.load(q_ptr + c).to(tl.float32)
    hist = tl.zeros((BINS,), tl.int32)
    end = tl.minimum(p * BLOCK + BLOCK, N)
    for s in range(p * BLOCK, end, SUB):
        rows = s + tl.arange(0, SUB)
        ok = rows < end
        k = tl.load(k_ptr + rows[:, None].to(tl.int64) * D + c[None, :], ok[:, None])
        score = tl.sum(k.to(tl.float32) * qb[None, :], 1)
        tl.store(score_ptr + rows, score, ok)
        hist += tl.histogram(_bin(score), BINS, mask=ok)
    tl.atomic_add(hist_ptr + tl.arange(0, BINS), hist, mask=hist > 0)


@triton.jit
def _finish(
    best,
    cand_ptr,
    level_ptr,
    count_ptr,
    pick_ptr,
    hist_ptr,
    blocks,
    TOP: tl.constexpr,
    GP: tl.constexpr,
    GW: tl.constexpr,
    BINS: tl.constexpr,
):
    """Program p's packed top TOP (best) merged across programs: the last program
    of each group of GP merges the group's, the last group's merger merges the GW
    groups' (fewer: padded) and stores entry indices; counters left zero, and
    BINS > 0: the histogram cleared."""
    p = tl.program_id(0)
    i = tl.arange(0, TOP)
    tl.store(cand_ptr + p * TOP + i, best)
    tl.debug_barrier()
    g = p // GP
    size = tl.minimum(blocks - g * GP, GP)
    if tl.atomic_add(count_ptr + 1 + g, 1) == size - 1:
        j = tl.arange(0, GP * TOP)
        mine = j < size * TOP
        x = tl.load(cand_ptr + g * GP * TOP + j, mine, other=0, cache_modifier=".cg")
        best = tl.topk(tl.where(mine, x, LEAST), TOP)
        tl.store(count_ptr + 1 + g, 0)
        groups = tl.cdiv(blocks, GP)
        if GW > 1:
            tl.store(level_ptr + g * TOP + i, best)
            tl.debug_barrier()
            if tl.atomic_add(count_ptr, 1) == groups - 1:
                k = tl.arange(0, GW * TOP)
                real = k < groups * TOP
                y = tl.load(level_ptr + k, real, other=0, cache_modifier=".cg")
                best = tl.topk(tl.where(real, y, LEAST), TOP)
                tl.store(count_ptr, 0)
                tl.store(pick_ptr + i, 0x7FFFFFFF - (best & 0xFFFFFFFF))
                if BINS > 0:
                    tl.store(hist_ptr + tl.arange(0, BINS), 0)
        else:
            tl.store(pick_ptr + i, 0x7FFFFFFF - (best & 0xFFFFFFFF))
            if BINS > 0:
                tl.store(hist_ptr + tl.arange(0, BINS), 0)


@triton.jit
def _major_scores(
    q_ptr, k_ptr, score_ptr, c, N, first, BLOCK: tl.constexpr, SUB: tl.constexpr,
    KB: tl.constexpr,
):  # fmt: skip
    """Scores q . k of entries [first, first + BLOCK) on the blocks of channels c
    (KB blocks of 16) of block-major keys (D // 16, N, 16), SUB rows at a time."""
    i = tl.arange(0, 16)
    j = tl.arange(0, KB * 16)
    for s in range(first, first + BLOCK, SUB):
        rows = s + tl.arange(0, SUB)
        ok = rows < N
        score = tl.zeros((SUB,), tl.float32)
        for r in tl.static_range(KB):
            ch = tl.sum(tl.where(j == r * 16, c, 0), 0)  # the block's first channel
            slab = k_ptr + (ch // 16).to(tl.int64) * N * 16
            k = tl.load(slab + rows[:, None] * 16 + i[None, :], ok[:, None], other=0.0)
            qb = tl.load(q_ptr + ch + i).to(tl.float32)
            score += tl.sum(k.to(tl.float32) * qb[None, :], 1)
        tl.store(score_ptr + rows, tl.where(ok, score, float("-inf")))


@triton.jit
def _block_top(
    q_ptr,
    w_ptr,
    k_ptr,
    score_ptr,
    cand_ptr,
    level_ptr,
    count_ptr,
    pick_ptr,
    N,
    blocks,
    scale,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    TOP: tl.constexpr,
    SUB: tl.constexpr,
    GP: tl.constexpr,
    GW: tl.constexpr,
    KB: tl.constexpr,
    MAJOR: tl.constexpr,
):
    """Program p: entries [p BLOCK, (p + 1) BLOCK) scored on their first C channels
    (KB > 0: on q's top KB 16-channel blocks; H == 0: q . k, else the indexer over H
    heads) SUB rows at a time into score, their packed top TOP in one topk, then
    _finish. MAJOR: keys block-major (D // 16, N, 16), each block's slab read
    contiguously (KB > 0, H == 0)."""
    p = tl.program_id(0)
    if KB > 0:
        c = _top_blocks(q_ptr, D, 16, KB)
    else:
        c = tl.arange(0, D)
    if MAJOR:
        _major_scores(q_ptr, k_ptr, score_ptr, c, N, p * BLOCK, BLOCK, SUB, KB)
    else:
        for s in range(p * BLOCK, p * BLOCK + BLOCK, SUB):
            rows = s + tl.arange(0, SUB)
            ok = rows < N
            at = k_ptr + rows[:, None].to(tl.int64) * D + c[None, :]
            k = tl.load(at, ok[:, None] & (c < C)[None, :], other=0.0).to(tl.float32)
            if H == 0:
                score = tl.sum(k * tl.load(q_ptr + c).to(tl.float32)[None, :], 1)
            else:
                score = tl.zeros((SUB,), tl.float32)
                for h in tl.static_range(H):
                    qh = tl.load(q_ptr + h * D + c).to(tl.float32)
                    dot = tl.sum(k * qh[None, :], 1)
                    score += tl.maximum(dot * scale, 0.0) * tl.load(w_ptr + h)
            tl.store(score_ptr + rows, tl.where(ok, score, float("-inf")))
    tl.debug_barrier()
    rows = p * BLOCK + tl.arange(0, BLOCK)
    best = tl.topk(_pack(tl.load(score_ptr + rows), rows), TOP)
    _finish(best, cand_ptr, level_ptr, count_ptr, pick_ptr, pick_ptr, blocks,
            TOP, GP, GW, 0)  # fmt: skip


@triton.jit
def _rerank(
    q_ptr,
    k_ptr,
    score_ptr,
    hist_ptr,
    ids_ptr,
    cand_ptr,
    level_ptr,
    count_ptr,
    pick_ptr,
    N,
    KEEP,
    blocks,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BINS: tl.constexpr,
    TOP: tl.constexpr,
    SUB: tl.constexpr,
    GP: tl.constexpr,
    GW: tl.constexpr,
):
    """Program p: packed top TOP by q . k (first C channels) of the entries in
    [p BLOCK, (p + 1) BLOCK) whose first score's bin is at or above the KEEP-th
    best's (bins counted in hist); those entries listed in ids first. Then
    _finish, which clears hist."""
    p = tl.program_id(0)
    hist = tl.load(hist_ptr + tl.arange(0, BINS), cache_modifier=".cg")
    at_least = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
    cut = tl.max(tl.where(at_least >= KEEP, tl.arange(0, BINS), -1), 0)
    rows = p * BLOCK + tl.arange(0, BLOCK)
    ok = rows < N
    kept = ok & (_bin(tl.load(score_ptr + rows, ok, other=0.0)) >= cut)
    slot = tl.cumsum(kept.to(tl.int32), 0) - 1
    tl.store(ids_ptr + p * BLOCK + slot, rows, kept)
    tl.debug_barrier()
    count = tl.sum(kept.to(tl.int32), 0)
    c = tl.arange(0, D)
    q = tl.load(q_ptr + c).to(tl.float32)
    best = _pack(tl.full((TOP,), float("-inf"), tl.float32), tl.arange(0, TOP))
    for s in range(0, count, SUB):
        j = s + tl.arange(0, SUB)
        valid = j < count
        e = tl.load(ids_ptr + p * BLOCK + j, valid, other=0)
        at = k_ptr + e[:, None].to(tl.int64) * D + c[None, :]
        k = tl.load(at, valid[:, None] & (c < C)[None, :], other=0.0).to(tl.float32)
        score = tl.where(valid, tl.sum(k * q[None, :], 1), float("-inf"))
        best = _keep_best(best, score, e, TOP, SUB)
    _finish(best, cand_ptr, level_ptr, count_ptr, pick_ptr, hist_ptr, blocks,
            TOP, GP, GW, BINS)  # fmt: skip


def _blocks(n: int, least: int) -> tuple[int, int]:
    """Entries per program (a power of 2, at least `least`) for about PROGRAMS
    programs and at most GROUP * GROUPS, and the programs."""
    most = GROUP * GROUPS
    block = max(least, triton.next_power_of_2(triton.cdiv(n, min(PROGRAMS, most))))
    return block, triton.cdiv(n, block)


def _merge_args(blocks: int, device: torch.device) -> dict:
    """_finish's buffers and sizes: candidates, group results, the zeroed counters
    and histogram (kept per device; each kernel leaves them zero), the picks."""
    if device not in _buffers:
        _buffers[device] = (
            torch.zeros(1 + GROUPS, dtype=torch.int32, device=device),
            torch.zeros(BINS, dtype=torch.int32, device=device),
        )
    gp = min(GROUP, triton.next_power_of_2(blocks))
    groups = triton.cdiv(blocks, gp)
    assert groups <= GROUPS, blocks
    return {
        "cand": torch.empty(blocks * TOP, dtype=torch.int64, device=device),
        "level": torch.empty(groups * TOP, dtype=torch.int64, device=device),
        "count": _buffers[device][0],
        "hist": _buffers[device][1],
        "picks": torch.empty(TOP, dtype=torch.int64, device=device),
        "GP": gp,
        "GW": triton.next_power_of_2(groups),
    }


def top64(
    q: Tensor,
    keys: Tensor,
    weights: Tensor | None = None,
    width: int = 0,
    blocks: int = 0,
) -> Tensor:
    """The 64 best entries of keys (N, D) bf16, as int64 (64,): by q (D,) . k on the
    first `width` channels (0: all; q zero beyond them) or, blocks > 0, on q's
    `blocks` aligned 16-channel blocks of largest sum |q| (keys may then be
    block-major, (D // 16, N, 16)); or by CSA2's indexer when weights (H,) are
    given, q (H, D)."""
    major = keys.dim() == 3  # block-major (dim // 16, N, 16)
    assert not major or (blocks > 0 and weights is None)
    n, dim = (keys.shape[1], keys.shape[0] * 16) if major else keys.shape
    sub = max(TOP, TILE // (blocks * 16 or dim))
    block, programs = _blocks(n, sub)
    m = _merge_args(programs, keys.device)
    score = torch.empty(programs * block, dtype=torch.float32, device=keys.device)
    _block_top[(programs,)](
        q,
        q if weights is None else weights,
        keys,
        score,
        m["cand"],
        m["level"],
        m["count"],
        m["picks"],
        n,
        programs,
        dim**-0.5,
        BLOCK=block,
        D=dim,
        C=width or dim,
        H=0 if weights is None else q.shape[0],
        TOP=TOP,
        SUB=sub,
        GP=m["GP"],
        GW=m["GW"],
        KB=blocks,
        MAJOR=major,
        num_warps=8,
    )
    return m["picks"]


def prefilter64(
    q: Tensor, keys: Tensor, keep: int, dims: int = 16, width: int = 0
) -> Tensor:
    """csa2_select_decode.prefilter for one query q (D,): q . k on its aligned `dims`
    block for every entry, then the top 64 by q . k among the entries at or above
    the keep-th best's bin. keep or fewer entries: top64."""
    n, dim = keys.shape
    if n <= keep:
        return top64(q, keys, width=width)
    first, firsts = _blocks(n, TILE // dims)
    block, blocks = _blocks(n, TOP)
    m = _merge_args(blocks, keys.device)
    score = torch.empty(n, dtype=torch.float32, device=keys.device)
    ids = torch.empty(blocks * block, dtype=torch.int32, device=keys.device)
    _first[(firsts,)](
        q, keys, score, m["hist"], n, first, D=dim, L=dims, SUB=TILE // dims, BINS=BINS
    )
    _rerank[(blocks,)](
        q,
        keys,
        score,
        m["hist"],
        ids,
        m["cand"],
        m["level"],
        m["count"],
        m["picks"],
        n,
        keep,
        blocks,
        BLOCK=block,
        D=dim,
        C=width or dim,
        BINS=BINS,
        TOP=TOP,
        SUB=TOP,
        GP=m["GP"],
        GW=m["GW"],
        num_warps=8,
    )
    return m["picks"]


@triton.jit
def _csa2_attend(
    q_ptr,
    win_ptr,
    kv_ptr,
    pick_ptr,
    sink_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    scale,
    D: tl.constexpr,
    W: tl.constexpr,
    TOP: tl.constexpr,
    ROPE: tl.constexpr,
):
    """Program h: head h over the picks and the window (K = V) with its sink; then
    the inverse RoPE x cos + rotate_half(x) sin on the last ROPE channels, x's
    rotate_half read back from the stored x (a permuted gather of the keys costs
    more than the attention)."""
    h = tl.program_id(0)
    c = tl.arange(0, D)
    rot = c >= D - ROPE
    half = ROPE // 2
    # partner channel of rotate_half: -x[c + half] (first half), x[c - half]
    j = c - (D - ROPE)
    partner = tl.where(rot, D - ROPE + (j + half) % ROPE, c)
    sign = tl.where(j < half, -1.0, 1.0)
    picks = tl.load(pick_ptr + tl.arange(0, TOP))
    q = tl.load(q_ptr + h * D + c).to(tl.float32)
    kp = tl.load(kv_ptr + picks[:, None] * D + c[None, :]).to(tl.float32)
    kw = tl.load(win_ptr + tl.arange(0, W)[:, None] * D + c[None, :]).to(tl.float32)
    sink = tl.load(sink_ptr + h)
    cos = tl.load(cos_ptr + j, rot, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr + j, rot, other=0.0).to(tl.float32)
    lp = tl.sum(kp * q[None, :], 1).to(tl.bfloat16).to(tl.float32) * scale
    lw = tl.sum(kw * q[None, :], 1).to(tl.bfloat16).to(tl.float32) * scale
    top = tl.maximum(tl.maximum(tl.max(lp, 0), tl.max(lw, 0)), sink)
    ep, ew = tl.exp(lp - top), tl.exp(lw - top)
    total = tl.sum(ep, 0) + tl.sum(ew, 0) + tl.exp(sink - top)
    pp = (ep / total).to(tl.bfloat16).to(tl.float32)
    pw = (ew / total).to(tl.bfloat16).to(tl.float32)
    out = (tl.sum(pp[:, None] * kp, 0) + tl.sum(pw[:, None] * kw, 0)).to(tl.bfloat16)
    tl.store(out_ptr + h * D + c, out)
    tl.debug_barrier()
    turned = tl.load(out_ptr + h * D + partner).to(tl.float32)
    tl.debug_barrier()  # every partner read before the overwrite
    out = out.to(tl.float32)
    out = tl.where(rot, out * cos - sign * turned * sin, out)
    tl.store(out_ptr + h * D + c, out.to(tl.bfloat16))


def csa2_attend(
    q: Tensor,
    window_kv: Tensor,
    main_kv: Tensor,
    picks: Tensor,
    sinks: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """CSA2 sparse attention of one token, q (H, D) over main_kv[picks] and the
    window (W, D), K = V, then the inverse RoPE on the last len(cos) channels."""
    heads, dim = q.shape
    out = torch.empty_like(q)
    _csa2_attend[(heads,)](
        q,
        window_kv,
        main_kv,
        picks,
        sinks,
        cos,
        sin,
        out,
        dim**-0.5,
        D=dim,
        W=window_kv.shape[0],
        TOP=picks.shape[0],
        ROPE=cos.shape[0],
        num_warps=WARPS,
    )
    return out


@triton.jit
def _b_attend(
    q_ptr,
    qp_ptr,
    kw_ptr,
    vw_ptr,
    keys_ptr,
    pick_ptr,
    table_ptr,
    out_ptr,
    key_row,
    key_block,
    k_stride,
    k_head,
    v_stride,
    v_head,
    scale,
    QB: tl.constexpr,
    W: tl.constexpr,
    TOP: tl.constexpr,
):
    """Program h: head h over its window (keys 64 + QB channels wide) and the picks
    (128 channels); a pick's value is its key turned back on the first 64 channels,
    as pairs (even x0, odd x1): (cos x0 - sin x1, cos x1 + sin x0), cos and sin
    the pair's from table (N, 64). Window values in the same pairs, so one
    rounding at the end. Unmasked loads of contiguous channels only: masks and
    permuted gathers cost more than the attention."""
    h = tl.program_id(0)
    a = tl.arange(0, 64)
    b = 64 + tl.arange(0, QB)
    p = tl.arange(0, 32)
    picks = tl.load(pick_ptr + tl.arange(0, TOP))
    rows = tl.arange(0, W)[:, None]
    q = q_ptr + h * 128
    q_lo = tl.load(q + a).to(tl.float32)
    q_b = tl.load(q + b).to(tl.float32)
    qp = qp_ptr + h * 128
    qp_lo = tl.load(qp + a).to(tl.float32)
    qp_hi = tl.load(qp + 64 + a).to(tl.float32)
    key = keys_ptr + picks[:, None] * key_row + (a % 16)[None, :]
    lo = tl.load(key + (a // 16 * key_block)[None, :]).to(tl.float32)
    hi = tl.load(key + ((a // 16 + 4) * key_block)[None, :]).to(tl.float32)
    cos = tl.load(table_ptr + picks[:, None] * 64 + p[None, :]).to(tl.float32)
    sin = tl.load(table_ptr + picks[:, None] * 64 + 32 + p[None, :]).to(tl.float32)
    k_at = kw_ptr + rows * k_stride + h * k_head
    kwa = tl.load(k_at + a[None, :]).to(tl.float32)
    kwb = tl.load(k_at + b[None, :]).to(tl.float32)
    v_at = vw_ptr + rows * v_stride + h * v_head
    v_lo = tl.load(v_at + a[None, :]).to(tl.float32)
    v_hi = tl.load(v_at + 64 + a[None, :]).to(tl.float32)
    lp = (tl.sum(lo * qp_lo[None, :], 1) + tl.sum(hi * qp_hi[None, :], 1)) * scale
    lw = (tl.sum(kwa * q_lo[None, :], 1) + tl.sum(kwb * q_b[None, :], 1)) * scale
    top = tl.maximum(tl.max(lp, 0), tl.max(lw, 0))
    ep, ew = tl.exp(lp - top), tl.exp(lw - top)
    total = tl.sum(ep, 0) + tl.sum(ew, 0)
    x0, x1 = tl.split(tl.reshape(lo, (TOP, 32, 2)))
    v0 = (cos * x0 - sin * x1).to(tl.bfloat16).to(tl.float32)
    v1 = (cos * x1 + sin * x0).to(tl.bfloat16).to(tl.float32)
    w0, w1 = tl.split(tl.reshape(v_lo, (W, 32, 2)))
    o0 = (tl.sum(ep[:, None] * v0, 0) + tl.sum(ew[:, None] * w0, 0)) / total
    o1 = (tl.sum(ep[:, None] * v1, 0) + tl.sum(ew[:, None] * w1, 0)) / total
    o_hi = (tl.sum(ep[:, None] * hi, 0) + tl.sum(ew[:, None] * v_hi, 0)) / total
    out = out_ptr + h * 128
    tl.store(out + a, tl.reshape(tl.join(o0, o1), (64,)).to(tl.bfloat16))
    tl.store(out + 64 + a, o_hi.to(tl.bfloat16))


def pair_table(f1: Tensor, f2: Tensor) -> Tensor:
    """b_attend's (N, 64) table from record 91's RoPE rows f1, f2 (N, RR) (pairs
    (2i, 2i + 1) share an angle: f1 = cos, cos; f2 = sin, -sin): the 32 pair
    cosines of the first 64 channels, then their sines. Channels past 64 must not
    turn (record 91 turns 48 of its 96)."""
    n, rr = f1.shape
    assert (f1[:, 64:] == 1).all() and (f2[:, 64:] == 0).all()
    assert (f1[:, 0::2] == f1[:, 1::2]).all() and (f2[:, 0::2] == -f2[:, 1::2]).all()
    f1 = F.pad(f1[:, :64], (0, max(0, 64 - rr)), value=1.0)
    f2 = F.pad(f2[:, :64], (0, max(0, 64 - rr)), value=0.0)
    return torch.cat([f1[:, 0::2], f2[:, 0::2]], 1).contiguous()


def b_attend(
    q: Tensor,
    k_win: Tensor,
    v_win: Tensor,
    keys: Tensor,
    picks: Tensor,
    table: Tensor,
    scale: float,
    q_pick: Tensor | None = None,
) -> Tensor:
    """csa2_flex.attention of one token, q (H, 128): its window k_win (W, H, QK)
    (QK = 96 or 128 real channels; q beyond them ignored), v_win (W, H, 128) (an
    expanded single head is read once, MQA) and keys[picks], keys rotated as
    csa2_flex.rotate by the rows of pair_table(f1, f2) (record 91's bf16 tables);
    values the keys turned back. q_pick (H, 128): the picks' query (default q).
    keys (N, 128) or block-major (8, N, 16)."""
    heads, dim = q.shape
    major = keys.dim() == 3
    key_row, key_block = (16, keys.shape[1] * 16) if major else (128, 16)
    assert dim == 128 and k_win.shape[-1] in (96, 128) and table.shape[1] == 64
    assert k_win.stride(-1) == v_win.stride(-1) == 1
    out = torch.empty_like(q)
    _b_attend[(heads,)](
        q,
        q if q_pick is None else q_pick,
        k_win,
        v_win,
        keys,
        picks,
        table,
        out,
        key_row,
        key_block,
        k_win.stride(0),
        k_win.stride(1),
        v_win.stride(0),
        v_win.stride(1),
        scale,
        QB=k_win.shape[-1] - 64,
        W=k_win.shape[0],
        TOP=picks.shape[0],
        num_warps=WARPS,
    )
    return out
