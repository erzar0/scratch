"""Union of top dims in shuffled windows: each query attends at most `keys` keys of
its document prefix, spread evenly over tournament_fast's shuffled windows (blocks
of `block` key slots in layouts()' random order, `keys` // windows per window),
picked and attended on the union of the query's top `dims` |q| channels and each
key's own top `dims` |k| channels:

    s(q, k) = sum over d in top(|q|) | top(|k|) of q_d k_d
            = (q * mq) . k + (q * ~mq) . (k * mk)     one matmul on 2D features

    pass     each window keeps its best keys by s, up to its quota
    rounds   halving: round i of R scores on ceil(i dims / R) channels per side,
             each window keeps max(quota, half its survivors), R = the rounds
             from `block` slots down to the quota

The output is softmax(scale * s) over the picks, times v (every value channel);
dims >= the head size gives exact q.k. Torch, no grad: for checkpoint evaluation
(loss only, no speed). q, k, v (T, H, D) one packed row; cu_seqlens holds the
documents' starts and may repeat T at its end.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from nanogpt.tournament_fast import layouts

CHUNK = 512  # queries per pass: logits (H, CHUNK, T) float32


@dataclass(frozen=True)
class Settings:
    """Key slots per window, keys per query, top channels per side, pass or rounds."""

    block: int = 2048
    keys: int = 1024
    dims: int = 16
    mode: str = "pass"


DEFAULT = Settings()


def top(x: Tensor, dims: int) -> Tensor:
    """x's `dims` largest |channels| (last axis), as a mask."""
    dims = min(dims, x.shape[-1])
    pick = x.abs().topk(dims, -1).indices
    return torch.zeros_like(x, dtype=torch.bool).scatter_(-1, pick, True)


def union_logits(q: Tensor, k: Tensor, dims: int) -> Tensor:
    """(H, Q, S) float32 s(q, k) for q (Q, H, D) and k (S, H, D)."""
    q, k = q.float(), k.float()
    mq = top(q, dims)
    features_q = torch.cat([q * mq, q * ~mq], -1)
    features_k = torch.cat([k, k * top(k, dims)], -1)
    return torch.einsum("qhd,shd->hqs", features_q, features_k)


def rank(score: Tensor, alive: Tensor) -> Tensor:
    """Each alive slot's place in its window (last axis) by score, best first."""
    masked = score.masked_fill(~alive, float("-inf"))
    return torch.argsort(masked, dim=-1, descending=True, stable=True).argsort(-1)


def rounds_of(block: int, quota: int) -> int:
    """Halving rounds from `block` slots down to `quota`."""
    return max(1, math.ceil(math.log2(block / quota)))


@torch.no_grad()
def picks(
    logits: dict[int, Tensor],
    cu_seqlens: Tensor,
    first: int,
    settings: Settings,
) -> Tensor:
    """(H, Q, T) each query's picked keys; logits[d]: (H, Q, T) s on d channels per
    side for the queries at row positions first, first + 1, ... (the dims each
    round needs: dims_of())."""
    heads, count, length = next(iter(logits.values())).shape
    s, device = settings, cu_seqlens.device
    queries = torch.arange(first, first + count, device=device)
    starts = cu_seqlens[torch.searchsorted(cu_seqlens, queries.int(), right=True) - 1]
    n = queries - starts + 1  # keys in each query's prefix
    windows = torch.div(n + s.block - 1, s.block, rounding_mode="floor")
    most = int(windows.max())
    table = layouts(most, s.block, str(device), True)
    keep = torch.zeros(heads, count, length + 1, dtype=torch.bool, device=device)
    for b in range(1, most + 1):
        mine = (windows == b).nonzero().squeeze(1)
        if not len(mine):
            continue
        if b == 1:  # slots in position order
            offset = torch.arange(s.block, device=device)[None]
        else:  # layouts()' table for b windows
            offset = table[s.block * ((b - 1) * b // 2 - 1) :][: b * s.block][None]
        at = starts[mine, None] + offset  # (Q, b * block)
        alive = (offset < n[mine, None]).view(1, len(mine), b, s.block)
        alive = alive.expand(heads, -1, -1, -1)
        gather = at.clamp(max=length - 1)[None].expand(heads, -1, -1)
        quota = s.keys // b
        for d in dims_of(s, b):
            score = logits[d][:, mine].gather(-1, gather).view_as(alive)
            half = (alive.sum(-1, keepdim=True) + 1).div(2, rounding_mode="floor")
            target = half.clamp(min=quota) if s.mode == "rounds" else quota
            alive = alive & (rank(score, alive) < target)
        slot = torch.where(alive.flatten(2), at[None], length)  # dead: column T
        head = torch.arange(heads, device=device)[:, None, None]
        keep[head, mine[None, :, None], slot] = True
    return keep[..., :length]


def dims_of(settings: Settings, windows: int) -> list[int]:
    """Channels per side for each scoring step of a query with `windows` windows."""
    s = settings
    if s.mode == "pass":
        return [s.dims]
    total = rounds_of(s.block, s.keys // windows)
    return [math.ceil(i * s.dims / total) for i in range(1, total + 1)]


@torch.library.custom_op("union_dims::attention", mutates_args=())
def _attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    block: int,
    keys: int,
    dims: int,
    mode: str,
) -> Tensor:
    s = Settings(block, keys, dims, mode)
    length = q.shape[0]
    most = math.ceil(length / s.block)
    needed = {d for b in range(1, most + 1) for d in dims_of(s, b)}
    out = torch.empty_like(q)
    for first in range(0, length, CHUNK):
        part = slice(first, first + CHUNK)
        logits = {d: union_logits(q[part], k, d) for d in needed}
        keep = picks(logits, cu_seqlens, first, s)
        weights = (scale * logits[s.dims]).masked_fill(~keep, float("-inf"))
        attended = torch.einsum("hqs,shd->qhd", weights.softmax(-1), v.float())
        out[part] = attended.to(q.dtype)
    return out


@_attention.register_fake
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    block: int,
    keys: int,
    dims: int,
    mode: str,
) -> Tensor:
    return torch.empty_like(q)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    settings: Settings = DEFAULT,
) -> Tensor:
    """softmax(scale * s) over each query's picks, times v; (T, H, D). No grad."""
    s = settings
    assert s.mode in ("pass", "rounds") and s.keys >= 1, s
    return _attention(q, k, v, cu_seqlens, scale, s.block, s.keys, s.dims, s.mode)
