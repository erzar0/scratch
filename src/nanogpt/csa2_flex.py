"""CSA2's KV layout (DeepSeek-V4.1 CSA2) in record 91's attention, with csa2_select's
parameter-free picks. Each query attends in one softmax to:
- its own layer's keys within WINDOW tokens of its document (the local window);
- top_k entries of a global KV shared by a group of layers: one K=V head, entry e a
  gated pool of tokens [e m, (e + 1) m) (the archived Csa2Compressor, no norm
  weight), picked by csa2_select from the layer's own q.
Entry keys are the entries rotated by record 91's RoPE (its yarn tables; pairs
(2i, 2i + 1), y = f1 x + f2 x_flip) to the entry's first token. Values are the
unrotated entries, so a cache keeps one D-channel vector per entry.

attention() is flex attention (tournament_fast.attend) over the row's keys followed
by the entries. Each 128-query tile reads its own row tile, the tile before it, and
the tiles of entries its documents have finished.
Shapes: q, k, v (T, H, D) for one packed row, T a multiple of 128; entries (N, D).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from nanogpt.tournament_fast import TILE, MaskMod, attend

WINDOW = 128  # CSA2's sliding window


def starts(cu_seqlens: Tensor, length: int) -> Tensor:
    """(length,) int64 row position of each token's document start."""
    cu = cu_seqlens.long()
    at = torch.searchsorted(cu, torch.arange(length, device=cu.device), right=True)
    return cu[at - 1]


def compress(x: Tensor, w_kv: Tensor, w_gate: Tensor, m: int) -> Tensor:
    """(T // m, D) entries of x (T, C), w (D, C): rms_norm of the softmax(x w_gate)
    weighted sum of x w_kv over each window of m tokens."""
    n = x.shape[0] // m
    x = x[: n * m]
    kv = F.linear(x, w_kv).view(n, m, -1)
    gate = F.linear(x, w_gate).view(n, m, -1).softmax(1, dtype=torch.float32)
    return F.rms_norm((kv * gate.type_as(kv)).sum(1), (kv.shape[-1],))


def rotate(entries: Tensor, m: int, factor1: Tensor, factor2: Tensor) -> Tensor:
    """entries (N, D) rotated to token e m by record 91's RoPE tables (rows =
    positions) on the first factor1.shape[1] channels; the rest unchanged."""
    n, width = entries.shape[0], factor1.shape[1]
    x = entries[:, :width].float()
    flip = x.view(n, -1, 2).flip(-1).view(n, width)
    f1, f2 = factor1[: n * m : m].float(), factor2[: n * m : m].float()
    return torch.cat([(f1 * x + f2 * flip).type_as(entries), entries[:, width:]], 1)


def words(entry: Tensor, valid: Tensor, entries: int) -> Tensor:
    """(T, entries / 32) int32: bit e % 32 of word e // 32 set per valid pick e."""
    out = entry.new_zeros(entry.shape[0], -(-entries // 32), dtype=torch.int32)
    one = torch.ones_like(entry, dtype=torch.int32)
    bit = (one << (entry % 32).int()) * valid  # distinct bits: the sum is their or
    return out.scatter_add_(1, entry // 32, bit)


@torch.library.custom_op("csa2_flex::mask_inputs", mutates_args=())
def mask_inputs(
    entry: Tensor, valid: Tensor, cu_seqlens: Tensor, length: int, entries: int
) -> tuple[Tensor, Tensor]:
    """starts() and words() as one opaque op: under torch.compile, flex's mask_mod
    cannot index buffers computed in the graph (inductor: FlexibleLayout)."""
    return starts(cu_seqlens, length), words(entry, valid, entries)


@mask_inputs.register_fake
def _(
    entry: Tensor, valid: Tensor, cu_seqlens: Tensor, length: int, entries: int
) -> tuple[Tensor, Tensor]:
    return (
        entry.new_empty(length),
        entry.new_empty(entry.shape[0], -(-entries // 32), dtype=torch.int32),
    )


def tiles(begin: Tensor, length: int, n: int, m: int) -> tuple[Tensor, Tensor]:
    """flex's (1, 1, tiles) KV tile counts and (1, 1, tiles, KV tiles) indices
    (from_kv_blocks reads the last size as the KV tile count): query tile i reads
    row tiles i - 1 and i, then the tiles holding entries from its first query's
    document start to its last query's finished entries."""
    rows = length // TILE
    i = torch.arange(rows, device=begin.device)
    near = 1 + (i > 0).long()
    lo = -(-begin[i * TILE] // m)
    hi = torch.clamp((i + 1) * TILE // m, max=n)
    first = rows + lo // TILE
    far = torch.where(hi > lo, rows + (hi - 1) // TILE - first + 1, 0)
    slot = torch.arange(rows + -(-n // TILE), device=begin.device)
    index = torch.where(
        slot < near[:, None],
        (i - near + 1)[:, None] + slot,
        first[:, None] + slot - near[:, None],
    )
    count = near + far
    index = torch.where(slot < count[:, None], index, 0)
    return count[None, None].int(), index[None, None].int()


def mask(begin: Tensor, bits: Tensor, length: int) -> MaskMod:
    """Flex mask_mod: a row key in the query's window of its document, or a picked
    entry (key length + e)."""

    def keep(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        near = (kv_idx <= q_idx) & (q_idx - kv_idx < WINDOW) & (kv_idx >= begin[q_idx])
        e = (kv_idx - length).clamp(min=0)
        bit = (bits[q_idx, e // 32] >> (e % 32)) & 1
        return near | ((kv_idx >= length) & (bit != 0))

    return keep


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    keys: Tensor,
    values: Tensor,
    entry: Tensor,
    valid: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    m: int = 4,
) -> Tensor:
    """(T, H, D) softmax over each query's window keys and picked entries.

    Args:
        q, k, v: (T, H, D) one packed row, T a multiple of 128.
        keys, values: (N, D) rotated and unrotated entries (one head, all heads).
        entry, valid: (T, top_k) csa2_select picks.
        cu_seqlens: document starts, may repeat T at its end.
        scale: logit scale.
        m: tokens per entry.
    """
    length, heads, _ = q.shape
    n = keys.shape[0]
    pad = -n % TILE
    keys, values = (F.pad(x, (0, 0, 0, pad)) for x in (keys, values))
    begin, bits = mask_inputs(entry, valid, cu_seqlens, length, n + pad)
    count, index = tiles(begin, length, n, m)
    k = torch.cat([k, keys[:, None].expand(-1, heads, -1)])
    v = torch.cat([v, values[:, None].expand(-1, heads, -1)])
    return attend(q, k, v, scale, count, index, mask(begin, bits, length))
