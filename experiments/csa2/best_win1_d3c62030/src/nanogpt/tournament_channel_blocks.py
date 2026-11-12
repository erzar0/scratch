"""Tournament attention for training with contiguous channel blocks: tournament_fast's
method where each round adds one contiguous block of `dims` channels instead of the
query's next `dims` channels by single-channel |q|.

Round r's block, per head and query: of the windows of `dims` adjacent channels
holding no channel an earlier round read, the one with the largest sum of |q|
(lowest start on ties), if that sum is positive; otherwise the window with the
largest |q| sum over unread channels, its already read channels weighted 0 (no
channel counts twice). Zero channels (record 91 pads q, k from 96 to 128 with
zeros) never win a round while a positive free window exists.

The rounds are tournament_fast's kernels (K transposed, key copies) on this order,
so the winners equal theirs given the same channel order and q values. Reading k
(T, H, D) in place instead was 2.5-3.3x slower: each per-channel load jumps
H * D * 2 bytes per key and misses the cache. Shapes, settings and mask as
tournament_fast.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from nanogpt.tournament_fast import (
    CAP,
    DEFAULT,
    TILE,
    Settings,
    _bits,
    _redo,
    _rounds,
    attend,
    bit_mask,
    doc_tiles,
    key_copies,
    layouts,
    window_words,
)

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)


@triton.jit
def _read(channel, picks, L: tl.constexpr):
    """Whether an earlier round's window [pick, pick + L) holds each channel."""
    gap = channel[:, None] - picks[None, :]
    return tl.sum(((gap >= 0) & (gap < L)).to(tl.int32), 1) > 0


@triton.jit
def _windows(
    q_ptr,
    qo_ptr,
    order_ptr,
    T,
    H,
    D: tl.constexpr,
    L: tl.constexpr,
    R: tl.constexpr,
    RP: tl.constexpr,
):
    """One program per (head, query) row: R blocks of L adjacent channels (module
    docstring), as tournament_fast._order's output: each round's channels and q
    there, 0 where an earlier round read the channel. A free window's channels are
    all unread, so its sum is the round-1 sum; shifted windows are reloads of the
    row (cached), not gathers."""
    row = tl.program_id(0)
    c = tl.arange(0, D)
    i = tl.arange(0, L)
    x_ptr = q_ptr + ((row % T) * H + row // T) * D
    fits = c <= D - L  # window starts
    whole = tl.zeros((D,), tl.float32)  # |q| over each window
    for j in tl.static_range(L):
        x = tl.load(x_ptr + c + j, mask=c + j < D, other=0.0).to(tl.float32)
        whole += tl.abs(x)
    picks = tl.full((RP,), -D, tl.int32)  # earlier rounds' window starts
    unused = fits  # window starts overlapping no earlier round's window
    for r in tl.static_range(R):
        free = tl.where(unused, whole, -1.0)
        first = tl.argmax(free, 0, tie_break_left=True)
        if tl.max(free, 0) <= 0:  # no free window with a positive sum
            window = tl.zeros((D,), tl.float32)  # |q| over the unread channels
            for j in tl.static_range(L):
                y = tl.load(x_ptr + c + j, mask=c + j < D, other=0.0).to(tl.float32)
                window += tl.where(_read(c + j, picks, L), 0.0, tl.abs(y))
            first = tl.argmax(tl.where(fits, window, -1.0), 0, tie_break_left=True)
        channel = first + i
        value = tl.load(x_ptr + channel).to(tl.float32)
        out = row * R * L + r * L + i
        tl.store(order_ptr + out, channel)
        tl.store(qo_ptr + out, tl.where(_read(channel, picks, L), 0.0, value))
        picks = tl.where(tl.arange(0, RP) == r, first, picks)
        unused = unused & (tl.abs(c - first) >= L)


def windows(q: Tensor, settings: Settings = DEFAULT) -> tuple[Tensor, Tensor]:
    """(H, C, channels) q values (0 where re-read) and channel indices of the C
    queries q (C, H, D), round after round."""
    length, heads, dim = q.shape
    assert settings.channels <= dim, settings
    query = torch.empty(heads, length, settings.channels, device=q.device)
    order = torch.empty_like(query, dtype=torch.int32)
    _windows[(heads * length,)](
        q.contiguous(),
        query,
        order,
        length,
        heads,
        D=dim,
        L=settings.dims,
        R=settings.rounds,
        RP=triton.next_power_of_2(max(1, settings.rounds)),
        num_warps=1,
    )
    return query, order


def select(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    longest: int,
    settings: Settings = DEFAULT,
    first: int = 0,
) -> Tensor:
    """tournament_fast.select (arguments, shape) with windows() as the order."""
    length, heads, dim = q.shape
    rows = heads * length
    blocks = triton.cdiv(longest, settings.block)
    query, order = windows(q, settings)
    queries = torch.arange(
        first, first + length, device=q.device, dtype=cu_seqlens.dtype
    )
    starts = cu_seqlens[torch.searchsorted(cu_seqlens, queries, right=True) - 1]
    won = torch.full(
        (heads, length, blocks * settings.winners),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    ties = torch.zeros(settings.rounds + 1, dtype=torch.int32, device=q.device)
    tied = torch.empty(settings.rounds + 1, CAP, 2, dtype=torch.int32, device=q.device)
    device = str(q.device)
    table = layouts(blocks, settings.block, device, True)
    kt = k.permute(1, 2, 0).contiguous()  # (H, D, T)
    off = torch.zeros(1, dtype=torch.int32, device=q.device)
    if longest > settings.block:  # some prefix may span several blocks
        kt, off = key_copies(kt, cu_seqlens, first, length, settings.block, table)
    assert kt.numel() < 2**31, "int32 offsets overflow: pass a chunk"
    args = {
        "q_ptr": query,
        "order_ptr": order,
        "kt_ptr": kt,
        "start_ptr": starts.int(),
        "won_ptr": won,
        "T": kt.shape[-1],
        "Q0": first,
        "C": length,
        "most_blocks": blocks,
        "D": dim,
        "CHANNELS": settings.channels,
        "DIMS": settings.dims,
        "ROUNDS": settings.rounds,
        "BLOCK": settings.block,
        "QUOTA": settings.winners,
        "CAP": CAP,
    }
    for r in range(settings.rounds + 1):
        width = settings.winners << r
        grid = (rows, blocks if width == settings.block else 1)
        _rounds[grid](
            **args,
            off_ptr=off,
            layout_ptr=table,
            ties_ptr=ties[r],
            tied_ptr=tied[r],
            WIDTH=width,
            R=r,
            LOG=width.bit_length() - 1,
        )
    if settings.rounds:  # widths of one slot per winner have no rounds, no ties
        _redo[(CAP, settings.rounds)](
            **args,
            layout_ptr=layouts(blocks, settings.block, device, False),
            ties_ptr=ties,
            tied_ptr=tied,
            num_warps=max(1, min(8, settings.block // 256)),
        )
    return won


@torch.library.custom_op("tournament_channel_blocks::picked", mutates_args=())
def picked(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    low: Tensor,
    longest: int,
    block: int,
    winners: int,
    dims: int,
    first: int = 0,
) -> Tensor:
    """tournament_fast.picked with this module's select()."""
    won = select(q, k, cu_seqlens, longest, Settings(block, winners, dims), first)
    heads, length, most = won.shape
    words = window_words(k.shape[0], longest)
    assert heads * length * words < 2**31, "int32 offsets overflow: pass a chunk"
    out = torch.zeros(heads, length, words, dtype=torch.int32, device=q.device)
    _bits[(heads * length,)](
        won, low, out, length, most, words, TILE=TILE, MP=triton.next_power_of_2(most)
    )
    return out


@picked.register_fake
def _(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    low: Tensor,
    longest: int,
    block: int,
    winners: int,
    dims: int,
    first: int = 0,
) -> Tensor:
    length, heads, _ = q.shape
    words = window_words(k.shape[0], longest)
    return q.new_empty(heads, length, words, dtype=torch.int32)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    longest: int | None = None,
    settings: Settings = DEFAULT,
    chunk: int | None = None,
) -> Tensor:
    """tournament_fast.attention on this module's winners."""
    length, heads, _ = q.shape
    step = chunk or length
    assert length % TILE == 0 and step % TILE == 0, (length, step)
    bound = length if longest is None else min(longest, length)
    low, count, index = doc_tiles(cu_seqlens, length, heads)
    outs = []
    for first in reversed(range(0, length, step)):  # as tournament_fast.attention
        end = min(first + step, length)
        tiles = slice(first // TILE, end // TILE)
        words = picked(
            q[first:end].detach(),
            k.detach(),
            cu_seqlens,
            low[tiles],
            bound,
            settings.block,
            settings.winners,
            settings.dims,
            first,
        )
        mask_mod = bit_mask(words, low[tiles])
        part = q[first:end], k, v, scale, count[..., tiles], index[..., tiles, :]
        outs.append(attend(*part, mask_mod))
    return outs[0] if len(outs) == 1 else torch.cat(outs[::-1])
