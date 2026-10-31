"""Triton decode kernels: the hybrid tournament and SparQ, for timing one layer.

One new token per sequence, batch 1. Shapes: q (G, 2, D) (two query heads per KV
head), K and V (G, n, D) row-major, K_T (G, D, n) a transposed copy of K (SparQ's
two-copy layout) for the partial scores.

tournament(): the hybrid of sparse/hybrid_tournament.py, launches:
    1. block_rounds  one program per (KV head, block of BLOCK keys): the first
                     R - GLOBAL_ROUNDS rounds, each block keeping its top half
    2. global_rounds one program per KV head: the last GLOBAL_ROUNDS rounds on the
                     keep * 2^GLOBAL_ROUNDS survivors, global halving
    3. attend_part   exact attention on the winners, CHUNK keys per program,
                     merged in torch (flash-decoding style)
sparq(): sparq_scores (r rows of K_T, every key), torch.topk, then attend_part.
Selection thresholds are exact k-th largest values found by radix select on the
float bits, so winners match the emulation's kthvalue.
"""

import math

import torch
import triton
import triton.language as tl

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

BLOCK = 1024
DIMS = 8
GLOBAL_ROUNDS = 3
CHUNK = 64


@triton.jit
def _lse2(a, b):
    """log(exp(a) + exp(b)), elementwise."""
    top = tl.maximum(a, b)
    return top + tl.log(tl.exp(a - top) + tl.exp(b - top))


@triton.jit
def _sortable(score):
    """uint32 keys whose order is the float order of score."""
    bits = score.to(tl.int32, bitcast=True)
    key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)  # float order as int order
    return (key ^ -2147483648).to(tl.uint32, bitcast=True)  # int order as unsigned


@triton.jit
def _pick_bin(hist, need):
    """Digit value holding the need-th largest entry, and need within it."""
    above = tl.sum(hist, 0) - tl.cumsum(hist, 0)  # entries in higher digit values
    bins = tl.arange(0, 256)
    b = tl.max(tl.where((above < need) & (above + hist >= need), bins, -1), 0)
    return b, need - tl.sum(tl.where(bins == b, above, 0), 0)


@triton.jit
def _top(score, alive, budget):
    """Alive entries at or above the budget-th largest alive score.

    Radix select: the threshold's 4 bytes are fixed high to low, each from a
    histogram of that byte over the entries matching the bytes fixed so far.
    """
    key = _sortable(score)
    floor = tl.full((), 0, tl.uint32)
    match = alive
    need = budget
    for t in tl.static_range(4):
        shift = 24 - 8 * t
        digit = ((key >> shift) & 0xFF).to(tl.int32)
        b, need = _pick_bin(tl.histogram(digit, 256, mask=match), need)
        match = match & (digit == b)
        floor = floor | (b.to(tl.uint32) << shift)
    few = tl.sum(alive.to(tl.int32), 0) <= budget  # no threshold needed
    return alive & ((key >= floor) | few)


@triton.jit
def _add_round(
    q_ptr,
    order_ptr,
    kt_ptr,
    g,
    r,
    pos,
    alive,
    a0,
    a1,
    s_kt_g,
    s_kt_d,
    D: tl.constexpr,
    DIMS: tl.constexpr,
):
    """Partial logits of both query heads plus round r's DIMS dimensions."""
    for j in tl.static_range(DIMS):
        i = r * DIMS + j
        d = tl.load(order_ptr + g * D + i)
        k = tl.load(kt_ptr + g * s_kt_g + d * s_kt_d + pos, mask=alive, other=0.0)
        k = k.to(tl.float32)
        a0 += tl.load(q_ptr + (g * 2) * D + i) * k
        a1 += tl.load(q_ptr + (g * 2 + 1) * D + i) * k
    return a0, a1


@triton.jit
def _round_score(scale_ptr, g, r, a0, a1, alive, NR: tl.constexpr):
    """Group score of round r: logsumexp of the heads' scaled partial logits."""
    s0 = a0 * tl.load(scale_ptr + (g * 2) * NR + r)
    s1 = a1 * tl.load(scale_ptr + (g * 2 + 1) * NR + r)
    return tl.where(alive, _lse2(s0, s1), float("-inf"))


@triton.jit
def _block_rounds(
    q_ptr,
    order_ptr,
    scale_ptr,
    kt_ptr,
    pos_out,
    a0_out,
    a1_out,
    n,
    rounds,
    s_kt_g,
    s_kt_d,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    NR: tl.constexpr,
    B: tl.constexpr,
    QUOTA: tl.constexpr,
):
    g = tl.program_id(0)
    blk = tl.program_id(1)
    pos = blk * B + tl.arange(0, B)
    alive = pos < n
    a0 = tl.zeros((B,), tl.float32)
    a1 = tl.zeros((B,), tl.float32)
    for r in range(rounds):
        a0, a1 = _add_round(
            q_ptr, order_ptr, kt_ptr, g, r, pos, alive, a0, a1, s_kt_g, s_kt_d, D, DIMS
        )
        score = _round_score(scale_ptr, g, r, a0, a1, alive, NR)
        alive = _top(score, alive, B >> (r + 1))
    slot = tl.cumsum(alive.to(tl.int32), 0) - 1
    out = (g * tl.num_programs(1) + blk) * QUOTA + slot
    keep = alive & (slot < QUOTA)
    tl.store(pos_out + out, pos, mask=keep)
    tl.store(a0_out + out, a0, mask=keep)
    tl.store(a1_out + out, a1, mask=keep)


@triton.jit
def _global_rounds(
    q_ptr,
    order_ptr,
    scale_ptr,
    kt_ptr,
    pos_ptr,
    a0_ptr,
    a1_ptr,
    win_ptr,
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
        alive = _top(score, alive, KEEP << (ROUNDS - 1 - t))
    slot = tl.cumsum(alive.to(tl.int32), 0) - 1
    tl.store(win_ptr + g * KEEP + slot, pos, mask=alive & (slot < KEEP))


@triton.jit
def _sparq_scores(
    q_ptr,
    dims_ptr,
    scale_ptr,
    kt_ptr,
    out_ptr,
    n,
    s_kt_g,
    s_kt_d,
    R: tl.constexpr,
    B: tl.constexpr,
):
    g = tl.program_id(0)
    pos = tl.program_id(1) * B + tl.arange(0, B)
    alive = pos < n
    a0 = tl.zeros((B,), tl.float32)
    a1 = tl.zeros((B,), tl.float32)
    for j in tl.static_range(R):
        d = tl.load(dims_ptr + g * R + j)
        k = tl.load(kt_ptr + g * s_kt_g + d * s_kt_d + pos, mask=alive, other=0.0)
        k = k.to(tl.float32)
        a0 += tl.load(q_ptr + (g * 2) * R + j) * k
        a1 += tl.load(q_ptr + (g * 2 + 1) * R + j) * k
    s0 = a0 * tl.load(scale_ptr + g * 2)
    s1 = a1 * tl.load(scale_ptr + g * 2 + 1)
    tl.store(out_ptr + g * n + pos, _lse2(s0, s1), mask=alive)


@triton.jit
def _attend_part(
    q_ptr,
    k_ptr,
    v_ptr,
    idx_ptr,
    m_out,
    l_out,
    o_out,
    scaling,
    s_g,
    s_n,
    K: tl.constexpr,
    CH: tl.constexpr,
    D: tl.constexpr,
):
    g = tl.program_id(0)
    c = tl.program_id(1)
    pos = tl.load(idx_ptr + g * K + c * CH + tl.arange(0, CH))
    ok = pos >= 0
    dd = tl.arange(0, D)
    rows = g * s_g + pos[:, None] * s_n + dd[None, :]
    k = tl.load(k_ptr + rows, mask=ok[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_ptr + rows, mask=ok[:, None], other=0.0).to(tl.float32)
    parts = tl.num_programs(1)
    for h in tl.static_range(2):
        q = tl.load(q_ptr + (g * 2 + h) * D + dd).to(tl.float32)
        s = tl.where(ok, tl.sum(k * q[None, :], 1) * scaling, float("-inf"))
        top = tl.max(s, 0)
        p = tl.where(ok, tl.exp(s - top), 0.0)
        at = (g * 2 + h) * parts + c
        tl.store(m_out + at, top)
        tl.store(l_out + at, tl.sum(p, 0))
        tl.store(o_out + at * D + dd, tl.sum(p[:, None] * v, 0))


def attend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    """(G, 2, D) exact attention of q on the keys idx (G, K) holds; -1 is empty."""
    groups, _, dim = q.shape
    width = idx.shape[1]
    parts = triton.cdiv(width, CHUNK)
    idx = torch.nn.functional.pad(idx, (0, parts * CHUNK - width), value=-1)
    m = torch.empty(groups, 2, parts, device=q.device)
    total = torch.empty_like(m)
    out = torch.empty(groups, 2, parts, dim, device=q.device)
    _attend_part[(groups, parts)](
        q,
        k,
        v,
        idx,
        m,
        total,
        out,
        dim**-0.5,
        k.stride(0),
        k.stride(1),
        parts * CHUNK,
        CHUNK,
        dim,
    )
    weight = (m - m.amax(-1, keepdim=True)).exp()
    return (weight[..., None] * out).sum(2) / (weight * total).sum(2)[..., None]


def order(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dims by group |q|, q in that order, and each round's SparQ temperature."""
    q = q.float()
    groups, _, dim = q.shape
    q_abs = q.abs()
    dims = q_abs.sum(1).argsort(-1, descending=True)
    q_ord = q.gather(-1, dims[:, None].expand(-1, 2, -1)).contiguous()
    seen = q_ord.abs().unflatten(-1, (dim // DIMS, DIMS)).sum(-1).cumsum(-1)
    scale = dim**-0.5 / (seen / q_abs.sum(-1, keepdim=True)).sqrt()
    return dims.int().contiguous(), q_ord, scale.contiguous()


def select(
    q: torch.Tensor, k_t: torch.Tensor, keep: int, block: int = BLOCK
) -> torch.Tensor:
    """(G, keep) winner positions of the hybrid tournament, -1 for empty slots."""
    groups, dim, n = k_t.shape
    total = math.ceil(math.log2(n / keep))
    rounds = total - GLOBAL_ROUNDS
    assert rounds >= 0 and total * DIMS <= dim and keep * 2**total % block == 0
    dims, q_ord, scale = order(q)
    quota = block >> rounds
    blocks = keep * 2**total // block
    width = blocks * quota  # = keep * 2^GLOBAL_ROUNDS
    pos = torch.full((groups, width), -1, dtype=torch.int32, device=q.device)
    a0 = torch.empty(groups, width, device=q.device)
    a1 = torch.empty_like(a0)
    rounds_per = dim // DIMS
    _block_rounds[(groups, blocks)](
        q_ord,
        dims,
        scale,
        k_t,
        pos,
        a0,
        a1,
        n,
        rounds,
        k_t.stride(0),
        k_t.stride(1),
        dim,
        DIMS,
        rounds_per,
        block,
        quota,
    )
    won = torch.full((groups, keep), -1, dtype=torch.int32, device=q.device)
    _global_rounds[(groups,)](
        q_ord,
        dims,
        scale,
        k_t,
        pos,
        a0,
        a1,
        won,
        rounds,
        k_t.stride(0),
        k_t.stride(1),
        dim,
        DIMS,
        rounds_per,
        width,
        keep,
        GLOBAL_ROUNDS,
        num_warps=8,
    )
    return won


def tournament(
    q: torch.Tensor, k: torch.Tensor, k_t: torch.Tensor, v: torch.Tensor, keep: int
) -> torch.Tensor:
    """(G, 2, D) hybrid tournament attention for one decode token."""
    return attend(q, k, v, select(q, k_t, keep))


def sparq(
    q: torch.Tensor,
    k: torch.Tensor,
    k_t: torch.Tensor,
    v: torch.Tensor,
    r: int,
    top: int,
) -> torch.Tensor:
    """(G, 2, D) SparQ attention for one decode token.

    ponytail: the group score is a logsumexp of the heads' logits, not SparQ's
    summed softmax (that needs a global normalizer pass), and the local window is
    left out; both make SparQ slightly cheaper than the paper's.
    """
    groups, dim, n = k_t.shape
    qf = q.float()
    q_abs = qf.abs()
    dims = q_abs.sum(1).topk(r, -1).indices
    picked = dims[:, None].expand(-1, 2, -1)
    q_sel = qf.gather(-1, picked).contiguous()
    frac = q_abs.gather(-1, picked).sum(-1) / q_abs.sum(-1)
    scale = (dim**-0.5 / frac.sqrt()).contiguous()
    score = torch.empty(groups, n, device=q.device)
    _sparq_scores[(groups, triton.cdiv(n, BLOCK))](
        q_sel,
        dims.int().contiguous(),
        scale,
        k_t,
        score,
        n,
        k_t.stride(0),
        k_t.stride(1),
        r,
        BLOCK,
    )
    idx = score.topk(top, -1).indices.int()
    return attend(q, k, v, idx)
