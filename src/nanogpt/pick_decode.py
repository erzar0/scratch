"""Decode with union_pick's and sparq_pick's rules: one query per head against a KV
cache, the picks those modules make for the last query of an n-token document, then
exact attention over them. A Triton kernel scores every key; torch selects and
attends (no host syncs: CUDA graphs capture a step).

    scores   SparQ rule: q's top `dims` |q| channels of K transposed;
             union rule: plus, per key, q (top channels zeroed) at the key's own top
             `dims` channels times their values
    picks    each shuffled window of `block` slots (tournament_fast.layouts) keeps
             its best keys, up to `keys` // windows (union_dims.picks' rule)
    attend   softmax(scale * q.k) over the picks on every channel, times v

Caches beside K and V (N, H, D), filled when keys are written (not timed here):

    k_t    (H, D, N)  K transposed: each channel contiguous over keys (SparQ's copy)
    vals   (H, R, N)  each key's top-R |k| channels' values    (union rule only)
    idx    (H, R, N)  their channel numbers, uint8              (union rule only)

Bytes read per key to score: 2 R (SparQ rule), 5 R (union rule); dense: 4 D.
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

from nanogpt.tournament_fast import layouts
from nanogpt.union_dims import Settings

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

KEYS_PER_PROGRAM = 512


@dataclass(frozen=True)
class Cache:
    """K, V (N, H, D); k_t (H, D, N); vals, idx (H, R, N) or None (SparQ rule)."""

    k: Tensor
    v: Tensor
    k_t: Tensor
    vals: Tensor | None
    idx: Tensor | None


def build(k: Tensor, v: Tensor, dims: int, union: bool) -> Cache:
    """The caches for keys k, values v (N, H, D) already written."""
    k_t = k.permute(1, 2, 0).contiguous()
    if not union:
        return Cache(k, v, k_t, None, None)
    channel = k.abs().topk(dims, -1).indices  # top() as indices
    vals = k.gather(-1, channel)
    return Cache(
        k,
        v,
        k_t,
        vals.permute(1, 2, 0).contiguous(),
        channel.to(torch.uint8).permute(1, 2, 0).contiguous(),
    )


@triton.jit
def _scores(
    q_top_ptr,
    dims_ptr,
    q_rest_ptr,
    kt_ptr,
    vals_ptr,
    idx_ptr,
    out_ptr,
    n,
    N,
    D: tl.constexpr,
    R: tl.constexpr,
    UNION: tl.constexpr,
    B: tl.constexpr,
):
    """One program per (head, B keys): the rule's score of each key."""
    h = tl.program_id(0)
    pos = tl.program_id(1) * B + tl.arange(0, B)
    alive = pos < n
    acc = tl.zeros((B,), tl.float32)
    for j in range(R):
        d = tl.load(dims_ptr + h * R + j)
        k = tl.load(kt_ptr + (h * D + d) * N + pos, mask=alive, other=0.0)
        acc += tl.load(q_top_ptr + h * R + j) * k.to(tl.float32)
    if UNION:
        for j in range(R):
            at = (h * R + j) * N + pos
            c = tl.load(idx_ptr + at, mask=alive, other=0).to(tl.int32)
            x = tl.load(vals_ptr + at, mask=alive, other=0.0).to(tl.float32)
            acc += tl.load(q_rest_ptr + h * D + c) * x
    tl.store(out_ptr + h * n + pos, acc, mask=alive)


def scores(q: Tensor, cache: Cache, n: int, dims: int) -> Tensor:
    """(H, n) float32 scores of the first n keys for q (H, D)."""
    heads, dim = q.shape
    qf = q.float()
    channel = qf.abs().topk(dims, -1).indices  # top() as indices, no host sync
    union = cache.vals is not None
    out = torch.empty(heads, n, device=q.device)
    _scores[(heads, triton.cdiv(n, KEYS_PER_PROGRAM))](
        qf.gather(-1, channel).contiguous(),
        channel.int().contiguous(),
        qf.scatter(-1, channel, 0.0),
        cache.k_t,
        cache.vals if union else cache.k_t,
        cache.idx if union else cache.k_t,
        out,
        n,
        cache.k_t.shape[-1],
        D=dim,
        R=dims,
        UNION=union,
        B=KEYS_PER_PROGRAM,
    )
    return out


def picks(score: Tensor, n: int, settings: Settings) -> tuple[Tensor, Tensor]:
    """Positions (H, P) of each window's best keys and which are real keys."""
    s, heads = settings, score.shape[0]
    windows = math.ceil(n / s.block)
    if windows == 1:
        slots = torch.arange(s.block, device=score.device)
    else:
        table = layouts(windows, s.block, str(score.device), True)
        start = s.block * ((windows - 1) * windows // 2 - 1)
        slots = table[start : start + windows * s.block].long()
    alive = slots < n
    ranked = score[:, slots.clamp(max=n - 1)].masked_fill(~alive, -math.inf)
    best, at = ranked.view(heads, windows, s.block).topk(s.keys // windows, -1)
    pos = slots.view(1, windows, s.block).expand(heads, -1, -1).gather(-1, at)
    return pos.flatten(1), (best > -math.inf).flatten(1)


def attend(q: Tensor, cache: Cache, pos: Tensor, real: Tensor, scale: float) -> Tensor:
    """(H, D) softmax(scale * q.k) over the picked positions, times v."""
    head = torch.arange(q.shape[0], device=q.device)[:, None]
    k, v = cache.k[pos, head].float(), cache.v[pos, head].float()
    logits = torch.einsum("hd,hpd->hp", q.float(), k) * scale
    weights = logits.masked_fill(~real, -math.inf).softmax(-1)
    return torch.einsum("hp,hpd->hd", weights, v).to(q.dtype)


def step(q: Tensor, cache: Cache, n: int, scale: float, settings: Settings) -> Tensor:
    """(H, D) attention of q (H, D), the query at position n - 1, on its picks."""
    pos, real = picks(scores(q, cache, n, settings.dims), n, settings)
    return attend(q, cache, pos, real, scale)
