"""Triton decode kernels shared by the tournament and SparQ, for timing one layer.

One new token per sequence, batch 1. Shapes: q (G, 2, D) (two query heads per KV
head), K and V (G, n, D) row-major, K_T (G, D, n) a transposed copy of K (SparQ's
two-copy layout) for the partial scores.

Every round kernel starts with the same reading arguments (reading()): q in the
|q| order, that order, each round's SparQ temperature, K_T and its strides; round
r reads dims r * DIMS to (r + 1) * DIMS of that order. A survivor is an entry: its
position and both query heads' partial dot products (a0, a1).
    _block_rounds   one program per (KV head, block of B keys): rounds inside
                    the block, each keeping its top half
    _global_rounds  one program per KV head: the last rounds, global halving
    attend          exact attention on the winners, CHUNK keys per program
                    (_attend_part), merged in torch (flash-decoding style)
Selection thresholds are exact k-th largest values found by radix select on the
float bits, so winners match the emulation's kthvalue.
"""

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
def _byte(key, t: tl.constexpr):
    """Byte t of the uint32 keys, the high byte first, as int32."""
    return ((key >> (24 - 8 * t)) & 0xFF).to(tl.int32)


@triton.jit
def _radix_floor(key, match, need):
    """The need-th largest of the uint32 keys where match holds, by radix select:
    its 4 bytes fixed high to low, each from a histogram of that byte over the
    keys matching the bytes fixed so far."""
    floor = tl.full((), 0, tl.uint32)
    for t in tl.static_range(4):
        digit = _byte(key, t)
        b, need = _pick_bin(tl.histogram(digit, 256, mask=match), need)
        match = match & (digit == b)
        floor = floor | (b.to(tl.uint32) << (24 - 8 * t))
    return floor


@triton.jit
def _top(score, alive, budget):
    """Alive entries at or above the budget-th largest alive score; every alive
    entry if budget or fewer are. _radix_floor's passes, stopped once the byte
    value picked holds exactly the entries still needed: all of them are kept,
    as the full select keeps them (same entries, 29% faster rounds in
    tournament_fast at stage 3)."""
    key = _sortable(score)
    bins = tl.arange(0, 256)
    match, need = alive, budget
    floor = tl.full((), 0, tl.uint32)
    few = tl.sum(alive.to(tl.int32), 0) <= budget  # no threshold needed
    done = few
    for t in tl.static_range(4):
        if done == 0:
            digit = _byte(key, t)
            hist = tl.histogram(digit, 256, mask=match)
            b, need = _pick_bin(hist, need)
            match = match & (digit == b)
            floor = floor | (b.to(tl.uint32) << (24 - 8 * t))
            done = tl.sum(tl.where(bins == b, hist, 0), 0) == need
    return alive & ((key >= floor) | few)


@triton.jit
def _slots(mask):
    """Each masked entry's place among the masked entries: 0, 1, 2, ..."""
    return tl.cumsum(mask.to(tl.int32), 0) - 1


@triton.jit
def _claim_slots(counter, mask):
    """Slots for the masked entries after those other programs took: the atomic
    counter's old value plus each entry's place."""
    return tl.atomic_add(counter, tl.sum(mask.to(tl.int32), 0)) + _slots(mask)


@triton.jit
def _load_entries(pos_ptr, a0_ptr, a1_ptr, at):
    """Entries at offsets at, alive where the position is not -1."""
    pos = tl.load(pos_ptr + at)
    alive = pos >= 0
    a0 = tl.load(a0_ptr + at, mask=alive, other=0.0)
    a1 = tl.load(a1_ptr + at, mask=alive, other=0.0)
    return pos, alive, a0, a1


@triton.jit
def _store_entries(pos_ptr, a0_ptr, a1_ptr, slot, pos, a0, a1, mask):
    """Store the masked entries at their slots."""
    tl.store(pos_ptr + slot, pos, mask=mask)
    tl.store(a0_ptr + slot, a0, mask=mask)
    tl.store(a1_ptr + slot, a1, mask=mask)


@triton.jit
def _add_round(
    q_ptr,
    order_ptr,
    keys,
    s_kt_d,
    g,
    r,
    pos,
    alive,
    a0,
    a1,
    D: tl.constexpr,
    DIMS: tl.constexpr,
):
    """Partial logits of both query heads plus round r's DIMS dimensions; keys:
    KV head g's K_T."""
    for j in tl.static_range(DIMS):
        i = r * DIMS + j
        d = tl.load(order_ptr + g * D + i)
        k = tl.load(keys + d * s_kt_d + pos, mask=alive, other=0.0)
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
    s_kt_g,
    s_kt_d,
    pos_out,
    a0_out,
    a1_out,
    n,
    rounds,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    NR: tl.constexpr,
    B: tl.constexpr,
    QUOTA: tl.constexpr,
):
    """One program per (KV head, block of B keys): `rounds` rounds, each keeping
    the block's top half, then the first QUOTA survivors' entries."""
    g = tl.program_id(0)
    keys = kt_ptr + g * s_kt_g  # KV head g's K_T
    blk = tl.program_id(1)
    pos = blk * B + tl.arange(0, B)
    alive = pos < n
    a0 = tl.zeros((B,), tl.float32)
    a1 = tl.zeros((B,), tl.float32)
    for r in range(rounds):
        a0, a1 = _add_round(
            q_ptr, order_ptr, keys, s_kt_d, g, r, pos, alive, a0, a1, D, DIMS
        )
        score = _round_score(scale_ptr, g, r, a0, a1, alive, NR)
        alive = _top(score, alive, B >> (r + 1))
    slot = _slots(alive)
    at = (g * tl.num_programs(1) + blk) * QUOTA  # the block's QUOTA slots
    keep = alive & (slot < QUOTA)
    _store_entries(pos_out + at, a0_out + at, a1_out + at, slot, pos, a0, a1, keep)


@triton.jit
def _global_rounds(
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
    first,
    D: tl.constexpr,
    DIMS: tl.constexpr,
    NR: tl.constexpr,
    C: tl.constexpr,
    KEEP: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    """One program per KV head: ROUNDS global halving rounds from round `first` on
    the head's C entries, then the KEEP winners' positions."""
    g = tl.program_id(0)
    keys = kt_ptr + g * s_kt_g  # KV head g's K_T
    pos, alive, a0, a1 = _load_entries(pos_ptr, a0_ptr, a1_ptr, g * C + tl.arange(0, C))
    for t in tl.static_range(ROUNDS):
        r = first + t
        a0, a1 = _add_round(
            q_ptr, order_ptr, keys, s_kt_d, g, r, pos, alive, a0, a1, D, DIMS
        )
        score = _round_score(scale_ptr, g, r, a0, a1, alive, NR)
        alive = _top(score, alive, KEEP << (ROUNDS - 1 - t))
    slot = _slots(alive)
    tl.store(win_ptr + g * KEEP + slot, pos, mask=alive & (slot < KEEP))


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
    """One program per (KV head, CH of its K winners idx): each head's softmax
    partials over them (max, weight sum, weighted values), merged by attend."""
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
        scaling=dim**-0.5,
        s_g=k.stride(0),
        s_n=k.stride(1),
        K=parts * CHUNK,
        CH=CHUNK,
        D=dim,
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


def reading(
    q: torch.Tensor, k_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """The arguments every round kernel starts with: q_ptr, order_ptr, scale_ptr
    (order(q)), kt_ptr and K_T's strides s_kt_g, s_kt_d."""
    dims, q_ord, scale = order(q)
    return q_ord, dims, scale, k_t, k_t.stride(0), k_t.stride(1)


def sizes(dim: int) -> dict[str, int]:
    """The round kernels' size constants: D dims, DIMS per round, NR rounds."""
    return {"D": dim, "DIMS": DIMS, "NR": dim // DIMS}
