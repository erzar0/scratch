"""Tournament attention for training, fast: nanogpt/tournament.py's method at any
settings, the same winners at its settings.

Each query runs the tournament on its own document's prefix: blocks of `block` key
slots in the emulation's random slot order, each sending `winners` keys to
attention; each round adds the query's next `dims` channels by |q| and every block
keeps its top half. attention() is exact attention on the winners (FlexAttention).

Differences from tournament.py, none of which change a winner:
- one kernel orders each query's channels by |q| and keeps only those rounds read;
- rounds run on a vector that halves each round (survivors packed); a block that
  ties at a round's threshold is listed and rerun at full width afterwards;
- the widest launch spans the blocks of `longest`, the caller's bound on document
  length, instead of the whole row;
- winners become bit words over each 128-query tile's document window instead of
  the whole row, and flex's block mask comes from the document starts instead of
  a scan of every bit;
- a query whose prefix spans several blocks reads a copy of its document's keys in
  its layout's slot order (contiguous loads, not one per slot).
Triton kernels autotune on the device (warps, threshold method per round width).
"""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    FlexKernelOptions,
    flex_attention,
)

from attention.triton.common import DIMS, _slots, _top
from nanogpt.tournament import BLOCK, QUOTA, _growing_rounds, _slot_positions

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

TILE = 128  # flex block-mask tile, queries and keys
CAP = 1024  # tied blocks rerun per launch width (measured: at most 8 per call)
REPEATS = 25  # autotune timings per config


def _bench(
    kernel_call: Callable[[], object], quantiles: Sequence[float]
) -> list[float]:
    """Autotune timing as the kernels run in select(): after ~0.2 s of GPU work (the
    GPU idles down while a config compiles, up to a minute) and on a warm L2 (each
    launch rereads what the one before read; triton's do_bench flushes L2)."""
    kernel_call()  # compiles
    a = torch.empty(4096, 4096, device="cuda")
    for _ in range(50):
        a @ a
    events = [
        [torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(REPEATS)
    ]
    for start, end in events:
        start.record()
        kernel_call()
        end.record()
    torch.cuda.synchronize()
    times = torch.tensor([start.elapsed_time(end) for start, end in events])
    return times.quantile(torch.tensor(quantiles)).tolist()


@dataclass(frozen=True)
class Settings:
    """Key slots per block, keys each block sends to attention, channels each
    round adds; all powers of 2."""

    block: int = BLOCK
    winners: int = QUOTA
    dims: int = DIMS

    @property
    def rounds(self) -> int:
        """Halving rounds of a full block."""
        return (self.block // self.winners).bit_length() - 1

    @property
    def channels(self) -> int:
        """Channels the rounds of a full block read."""
        return self.rounds * self.dims


DEFAULT = Settings()


@cache
def layouts(blocks: int, block: int, device: str, ordered: bool) -> Tensor:
    """Slot -> document position tables for 2 to `blocks` blocks of `block` slots,
    concatenated: the emulation's random order (as tournament.layouts), each
    block's slots in position order when `ordered` (the same keys meet)."""
    tables = []
    for b in range(2, max(blocks, 2) + 1):
        perm = torch.randperm(block * b, generator=torch.Generator().manual_seed(0))
        tables.append(
            perm.view(b, block).sort(-1).values.flatten() if ordered else perm
        )
    return torch.cat(tables).int().to(device)


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (1, 2, 4)],
    key=["T", "CHANNELS"],
    cache_results=True,
    do_bench=_bench,
)
@triton.jit
def _order(q_ptr, qo_ptr, order_ptr, T, H, D: tl.constexpr, CHANNELS: tl.constexpr):
    """One program per (head, query) row: its first CHANNELS channels by
    descending |q| (ties: lower channel first, as a stable argsort) and q there."""
    row = tl.program_id(0)
    c = tl.arange(0, D)
    x = tl.load(q_ptr + ((row % T) * H + row // T) * D + c).to(tl.float32)
    size = tl.abs(x).to(tl.int32, bitcast=True).to(tl.int64)  # int order = |q| order
    ranked = tl.sort((size << 32) | (D - 1 - c), descending=True)
    channel = (D - 1 - (ranked & 0xFFFFFFFF)).to(tl.int32)
    first = c < CHANNELS
    tl.store(order_ptr + row * CHANNELS + c, channel, mask=first)
    tl.store(qo_ptr + row * CHANNELS + c, tl.gather(x, channel, 0), mask=first)


@triton.jit
def _top_sort(score, alive, budget, WIDTH: tl.constexpr):
    """_top by sorting: alive entries at or above the budget-th largest alive score;
    every alive entry if budget or fewer are."""
    s = tl.where(alive, score, float("-inf"))
    ranked = tl.sort(s, descending=True)
    floor = tl.sum(tl.where(tl.arange(0, WIDTH) == budget - 1, ranked, 0.0), 0)
    few = tl.sum(alive.to(tl.int32), 0) <= budget
    return alive & ((s >= floor) | few)


@triton.jit
def _pack(keep, pos, dot, WIDTH: tl.constexpr, LOG: tl.constexpr):
    """The first WIDTH / 2 kept entries in slot order (binary search on the running
    count): whether each exists, its position and partial logit; and how many
    entries were kept."""
    count = tl.cumsum(keep.to(tl.int32), 0)
    want = tl.arange(0, WIDTH // 2) + 1
    at = tl.zeros((WIDTH // 2,), tl.int32)  # slots whose count is below want
    for t in tl.static_range(LOG):
        step = WIDTH >> (t + 1)
        at = tl.where(tl.gather(count, at + (step - 1), 0) < want, at + step, at)
    total = tl.sum(keep.to(tl.int32), 0)
    return want <= total, tl.gather(pos, at, 0), tl.gather(dot, at, 0), total


def _distinct(
    configs: list[triton.Config], named_args: dict, **kwargs: object
) -> list[triton.Config]:
    """Configs that differ at this launch width: 1 to 16 slots per thread (or one
    warp), and sort thresholds up to twice the width (all larger sort every round)."""
    width = cast(int, {**named_args, **kwargs}["WIDTH"])
    seen, out = set(), []
    for config in configs:
        slots = width // (32 * config.num_warps)
        sort_below = min(config.kwargs["SORT_BELOW"], 2 * width)
        fits = (slots >= 1 or config.num_warps == 1) and slots <= 16
        if fits and (config.num_warps, sort_below) not in seen:
            seen.add((config.num_warps, sort_below))
            out.append(config)
    return out


@triton.autotune(
    configs=[
        triton.Config({"SORT_BELOW": s}, num_warps=w)
        for w in (1, 2, 4, 8)
        for s in (256, 512, 1024, 2048)
    ],
    key=["C", "WIDTH", "BLOCK", "QUOTA", "DIMS"],
    prune_configs_by={"early_config_prune": _distinct},
    reset_to_zero=["ties_ptr"],
    cache_results=True,
    do_bench=_bench,
)
@triton.jit
def _rounds(
    q_ptr,
    order_ptr,
    kt_ptr,
    off_ptr,
    start_ptr,
    layout_ptr,
    won_ptr,
    ties_ptr,
    tied_ptr,
    T,
    Q0,
    C,
    most_blocks,
    D: tl.constexpr,
    CHANNELS: tl.constexpr,
    DIMS: tl.constexpr,
    ROUNDS: tl.constexpr,
    BLOCK: tl.constexpr,
    QUOTA: tl.constexpr,
    CAP: tl.constexpr,
    WIDTH: tl.constexpr,
    R: tl.constexpr,
    LOG: tl.constexpr,
    SORT_BELOW: tl.constexpr,
):
    """One program per (head and query, block) whose block has WIDTH slots, for the
    C queries at row positions Q0 on of a row of T keys: its R rounds on a halving
    vector (threshold by radix select, or by sort below SORT_BELOW slots), then its
    QUOTA winners. A block that tied at a threshold lists itself for _redo. Keys:
    kt (H, D, T) from the query's document start, or with several blocks from its
    copy in slot order at off (key_copies())."""
    row = tl.program_id(0)  # head * C + query - Q0
    blk = tl.program_id(1)
    head = row // C
    start = tl.load(start_ptr + row % C)
    n = Q0 + row % C - start + 1  # keys in the query's cache
    blocks = tl.cdiv(n, BLOCK)
    rounds = _growing_rounds(n, blocks, QUOTA, ROUNDS)
    if (blk < blocks) & ((QUOTA << rounds) == WIDTH):
        pos = _slot_positions(layout_ptr, blocks, blk, WIDTH, BLOCK)
        alive = pos < n
        copy = blocks > 1
        here = start
        if WIDTH == BLOCK:  # pos: slots in the copy until the rounds end
            here = tl.where(copy, tl.load(off_ptr + row % C, mask=copy, other=0), start)
            pos = tl.where(copy, blk * BLOCK + tl.arange(0, WIDTH), pos)
        keys = kt_ptr + head * D * T + here
        dot = tl.zeros((WIDTH,), tl.float32)  # partial logits
        ties = 0
        for r in tl.static_range(R):
            for j in tl.static_range(DIMS):
                i = r * DIMS + j
                d = tl.load(order_ptr + row * CHANNELS + i)
                k = tl.load(keys + pos + d * T, mask=alive, other=0.0).to(tl.float32)
                dot += tl.load(q_ptr + row * CHANNELS + i) * k
            score = tl.where(alive, dot, float("-inf"))
            if (WIDTH >> r) >= SORT_BELOW:
                keep = _top(score, alive, WIDTH >> (r + 1))
            else:
                keep = _top_sort(score, alive, WIDTH >> (r + 1), WIDTH >> r)
            alive, pos, dot, total = _pack(keep, pos, dot, WIDTH >> r, LOG - r)
            ties += (total > (WIDTH >> (r + 1))).to(tl.int32)
        if WIDTH == BLOCK:  # slots in the copy back to document positions
            table = layout_ptr + BLOCK * ((blocks - 1) * blocks // 2 - 1)
            pos = tl.where(copy, tl.load(table + pos, mask=alive & copy, other=0), pos)
        out = won_ptr + (row * most_blocks + blk) * QUOTA + tl.arange(0, QUOTA)
        tl.store(out, start + pos, mask=alive)
        if ties > 0:
            at = tl.atomic_add(ties_ptr, 1)
            tl.store(tied_ptr + 2 * at, row, mask=at < CAP)
            tl.store(tied_ptr + 2 * at + 1, blk, mask=at < CAP)


@triton.jit
def _redo(
    q_ptr,
    order_ptr,
    kt_ptr,
    start_ptr,
    layout_ptr,
    won_ptr,
    ties_ptr,
    tied_ptr,
    T,
    Q0,
    C,
    most_blocks,
    D: tl.constexpr,
    CHANNELS: tl.constexpr,
    DIMS: tl.constexpr,
    ROUNDS: tl.constexpr,
    BLOCK: tl.constexpr,
    QUOTA: tl.constexpr,
    CAP: tl.constexpr,
):
    """One program per listed (row, block) of launch width QUOTA << (axis 1 + 1):
    its winners again by tournament._query_rounds' full-width rounds, on the
    emulation's slot order (unsorted layout), as tournament.py picks them."""
    item = tl.program_id(0)
    launch = tl.program_id(1) + 1  # the launch of width QUOTA << launch
    if item < tl.minimum(tl.load(ties_ptr + launch), CAP):
        listed = tied_ptr + (launch * CAP + item) * 2
        row = tl.load(listed)
        blk = tl.load(listed + 1)
        head = row // C
        start = tl.load(start_ptr + row % C)
        n = Q0 + row % C - start + 1
        blocks = tl.cdiv(n, BLOCK)
        rounds = _growing_rounds(n, blocks, QUOTA, ROUNDS)
        pos = _slot_positions(layout_ptr, blocks, blk, BLOCK, BLOCK)
        alive = pos < n  # a narrower block is a single block: slots = positions
        keys = kt_ptr + head * D * T + start + pos
        dot = tl.zeros((BLOCK,), tl.float32)
        for r in range(rounds):
            for j in tl.static_range(DIMS):
                i = r * DIMS + j
                d = tl.load(order_ptr + row * CHANNELS + i)
                k = tl.load(keys + d * T, mask=alive, other=0.0).to(tl.float32)
                dot += tl.load(q_ptr + row * CHANNELS + i) * k
            score = tl.where(alive, dot, float("-inf"))
            alive = _top(score, alive, (QUOTA << rounds) >> (r + 1))
        rank = _slots(alive)
        out = won_ptr + (row * most_blocks + blk) * QUOTA
        tl.store(out + tl.arange(0, QUOTA), -1)
        tl.debug_barrier()
        tl.store(out + rank, start + pos, mask=alive & (rank < QUOTA))


def select(
    q: Tensor,
    k: Tensor,
    cu_seqlens: Tensor,
    longest: int,
    settings: Settings = DEFAULT,
    first: int = 0,
) -> Tensor:
    """Each query's winners, row positions, -1 none.

    Args:
        q: (C, H, D) the queries at row positions first to first + C - 1.
        k: (T, H, D) one packed row.
        cu_seqlens: document starts, may repeat T at its end.
        longest: bound on every document's length, in tokens.
        settings: block, winners and channels per round.
        first: row position of the first query.

    Returns:
        (H, C, winners * ceil(longest / block)) int32.
    """
    length, heads, dim = q.shape
    assert settings.channels <= dim, settings
    rows = heads * length
    blocks = triton.cdiv(longest, settings.block)
    query = torch.empty(heads, length, settings.channels, device=q.device)
    order = torch.empty_like(query, dtype=torch.int32)
    _order[(rows,)](
        q.contiguous(), query, order, length, heads, D=dim, CHANNELS=settings.channels
    )
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


def key_copies(
    kt: Tensor, cu_seqlens: Tensor, first: int, length: int, block: int, table: Tensor
) -> tuple[Tensor, Tensor]:
    """kt (H, D, T) followed by one copy of a document's keys in slot order for each
    (document, block count of 2 or more) among the queries first to first + length
    - 1 (table: layouts()' ordered tables); and each query's copy offset in the
    result (0 if none). A block of a copy is contiguous; in kt its slots are about
    `blocks` keys apart, one 32-byte sector per slot from 16 blocks on."""
    rows = kt.shape[-1]
    bounds = sorted({*cu_seqlens.tolist(), rows})
    parts, off, at = [kt], torch.zeros(length, dtype=torch.int32), rows
    for start, end in itertools.pairwise(bounds):
        low, high = max(start, first), min(end, first + length)  # queries here
        least = max(2, triton.cdiv(low - start + 1, block))
        for blocks in range(least, triton.cdiv(high - start, block) + 1):
            a = max(low, start + (blocks - 1) * block)  # queries with `blocks` blocks
            z = min(high, start + blocks * block)
            if a < z:
                slots = table[block * ((blocks - 1) * blocks // 2 - 1) :]
                pos = (start + slots[: blocks * block]).clamp(max=rows - 1)
                parts.append(kt[:, :, pos])
                off[a - first : z - first] = at
                at += blocks * block
    return torch.cat(parts, -1) if len(parts) > 1 else kt, off.to(kt.device)


def window_words(length: int, longest: int) -> int:
    """Bit words per query: a 128-query tile's keys start at most longest + 254
    before its end (its first query's document start, rounded down to a tile)."""
    return min(triton.cdiv(longest + 2 * TILE, 32), triton.cdiv(length, 32))


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (1, 2, 4, 8)],
    key=["T", "MP"],
    cache_results=True,
    do_bench=_bench,
)
@triton.jit
def _bits(
    won_ptr,
    low_ptr,
    words_ptr,
    T,
    M,
    WORDS,
    TILE: tl.constexpr,
    MP: tl.constexpr,
):
    """One program per (head, query) row of zeroed words: bit s % 32 of word s // 32
    is set when key base + s is a winner, base = the first key of the row's tile's
    window. One atomic or per winner (rerunning sets the same bits)."""
    row = tl.program_id(0)
    base = tl.load(low_ptr + (row % T) // TILE) * TILE
    m = tl.arange(0, MP)
    key = tl.load(won_ptr + row * M + m, mask=m < M, other=-1)
    at = key - base
    bit = (1 << (at % 32)).to(tl.int32)
    word = words_ptr + row * WORDS + at // 32
    tl.atomic_or(word, bit, mask=key >= 0, sem="relaxed")


# An opaque op: built inside a compiled graph, a scattered mask reached flex's
# kernel all False (tournament.py)
@torch.library.custom_op("tournament_fast::picked", mutates_args=())
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
    """(H, C, window_words) int32: select()'s winners for the C queries q from row
    position first, as bits over each query tile's window, which starts at key tile
    low[tile] (low: those queries' tiles)."""
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


# Opaque too: computed in the graph, the mask's low reaches flex's lowering without a
# fixed layout, which fails in no-grad graphs (torch 2.10: validation)
@torch.library.custom_op("tournament_fast::tiles", mutates_args=())
def tiles(cu_seqlens: Tensor, length: int) -> tuple[Tensor, Tensor, Tensor]:
    """Each 128-query tile's first key tile (its first query's document start),
    its KV tile count and its KV tile indices (tiles, tiles): every tile from that
    one to the diagonal, all partial."""
    n = length // TILE
    tile = torch.arange(n, device=cu_seqlens.device)
    first = (TILE * tile).to(cu_seqlens.dtype)
    start = cu_seqlens[torch.searchsorted(cu_seqlens, first, right=True) - 1]
    low = (start // TILE).long()
    index = (low[:, None] + tile[None]).clamp(max=n - 1).int()
    return low, (tile - low + 1).int(), index


@tiles.register_fake
def _(cu_seqlens: Tensor, length: int) -> tuple[Tensor, Tensor, Tensor]:
    n = length // TILE
    return (
        cu_seqlens.new_empty(n, dtype=torch.long),
        cu_seqlens.new_empty(n, dtype=torch.int32),
        cu_seqlens.new_empty(n, n, dtype=torch.int32),
    )


def doc_tiles(cu_seqlens: Tensor, length: int, heads: int) -> tuple[Tensor, ...]:
    """tiles() with the counts and indices broadcast to flex's (1, H, tiles) and
    (1, H, tiles, tiles)."""
    low, count, index = tiles(cu_seqlens, length)
    n = length // TILE
    return low, count.expand(1, heads, n), index.expand(1, heads, n, n)


SWEPT = {  # fastest forward, then backward, at stage 3 on an RTX 4070 Ti SUPER
    "fwd_BLOCK_M": 128,
    "fwd_BLOCK_N": 32,
    "fwd_num_warps": 8,
    "fwd_num_stages": 3,
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 32,
    "bwd_BLOCK_M2": 32,
    "bwd_BLOCK_N2": 32,
    "bwd_num_warps": 4,
    "bwd_num_stages": 2,
}


def flex_options() -> FlexKernelOptions:
    """Flex kernel options for this GPU: on Ada (sm_89) the swept tiles (forward
    + backward 1.14 -> 0.84 ms at stage 3); elsewhere torch's defaults, with a
    2-stage forward where the 3-stage default (112 KB) exceeds the shared memory
    per block."""
    props = torch.cuda.get_device_properties()
    small = props.shared_memory_per_block_optin < 112 * 1024  # type: ignore[attr-defined]
    ada = (props.major, props.minor) == (8, 9)
    # the typed options lack fwd_ keys, which flex reads
    options = SWEPT if ada else {"fwd_num_stages": 2} if small else {}
    return cast(FlexKernelOptions, options)


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
    """Each query's exact attention on its select() winners.

    Args:
        q, k, v: (T, H, D) one packed row, T a multiple of 128.
        cu_seqlens: document starts, may repeat T at its end.
        scale: logit scale.
        longest: bound on every document's length; None: T.
        settings: block, winners and channels per round.
        chunk: queries selected and attended at a time, a multiple of 128 (memory
            for the winners scales with chunk * T, not T^2); None: T. Same output.

    Returns:
        (T, H, D) attention output.
    """
    length, heads, _ = q.shape
    step = chunk or length
    assert length % TILE == 0 and step % TILE == 0, (length, step)
    bound = length if longest is None else min(longest, length)
    low, count, index = doc_tiles(cu_seqlens, length, heads)
    outs = []
    # last chunk first: its key copies are the largest, later (smaller) ones reuse
    # the cached memory; in increasing size each chunk reserved a new segment
    # (13 GiB reserved for 1.2 GB used at T = 131072, then CUDA out of memory)
    for first in reversed(range(0, length, step)):
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


MaskMod = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]  # flex mask_mod


def bit_mask(words: Tensor, low: Tensor) -> MaskMod:
    """Flex mask_mod: the key is one of the query's winners in words (picked())."""

    def winner(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        at = kv_idx - low[q_idx // TILE] * TILE
        return ((words[h, q_idx, at // 32] >> (at % 32)) & 1) != 0

    return winner


def attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    count: Tensor,
    index: Tensor,
    mask_mod: MaskMod,
) -> Tensor:
    """Flex attention of the queries q (C, H, D) on the packed row k, v (T, H, D)
    over the KV tiles index (count per query tile, doc_tiles()) where mask_mod
    holds."""
    lengths = q.shape[0], k.shape[0]
    mask = BlockMask.from_kv_blocks(
        count, index, BLOCK_SIZE=TILE, mask_mod=mask_mod, seq_lengths=lengths
    )
    q, k, v = (x.transpose(0, 1)[None] for x in (q, k, v))
    out = flex_attention(
        q, k, v, block_mask=mask, scale=scale, kernel_options=flex_options()
    )
    return cast(Tensor, out)[0].transpose(0, 1)
