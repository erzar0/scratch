"""Plain PyTorch decode steps, for timing the versions without a Triton kernel.

One decode token, batch 1, as the Triton kernels: q (G, 2, D) and a KV cache of K
and V (G, n, D). State a decoder keeps at cache-write time (K transposed, block
means, PCA-rotated keys, the sum of values) is built on first use, outside the
timing (KV.once). A step returns the output (G, 2, D) and the winner positions
(G, M), -1 for none (None: every key), so tests check them against the emulation.
Partial scores read only the survivors' channels (a gather), not all of K.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn.functional as F

from attention import loki
from attention.bandit import BoundBandit
from attention.block_topk import BlockTopK, summary_keys
from attention.tournament.blocks import BlockTournament
from attention.tournament.channel_groups import GROUP, GroupOrder, permutation
from attention.tournament.general import Tournament, halving_rounds, rotation

Tensor = torch.Tensor


@dataclass
class KV:
    """A KV cache and the state its writes keep."""

    k: Tensor
    v: Tensor
    memo: dict[str, Any] = field(default_factory=dict)

    def once[T](self, name: str, make: Callable[[], T]) -> T:
        """make() the first time name is asked for, the stored value after."""
        if name not in self.memo:
            self.memo[name] = make()
        return cast(T, self.memo[name])

    @property
    def k_t(self) -> Tensor:
        """(G, D, n) K transposed (SparQ's second copy)."""
        return self.once("k_t", lambda: self.k.transpose(1, 2).contiguous())


type Step = Callable[[Tensor, KV], tuple[Tensor, Tensor | None]]


def attend(q: Tensor, k: Tensor, v: Tensor, idx: Tensor) -> Tensor:
    """(G, 2, D) exact attention of q on the keys idx (G, M) holds; -1 is none."""
    rows = idx.clamp(min=0).long()[..., None].expand(-1, -1, k.shape[-1])
    k_sel, v_sel = k.gather(1, rows)[:, None], v.gather(1, rows)[:, None]
    mask = (idx >= 0)[:, None, None]  # (G, 1, 1, M)
    out = F.scaled_dot_product_attention(
        q[:, :, None], k_sel, v_sel, attn_mask=mask, enable_gqa=True
    )
    return out[:, :, 0]


def dense(q: Tensor, kv: KV) -> tuple[Tensor, None]:
    """Every key: SDPA."""
    groups, size, dim = q.shape
    out = F.scaled_dot_product_attention(
        q.reshape(1, groups * size, 1, dim), kv.k[None], kv.v[None], enable_gqa=True
    )
    return out.reshape(groups, size, dim), None


def lse_scores(q: Tensor, keys: Tensor) -> Tensor:
    """(G, M) logsumexp over the group of the scaled logits on keys (G, M, D)."""
    return ((q @ keys.mT).float() * q.shape[-1] ** -0.5).logsumexp(1)


def exact_topk(top: int) -> Step:
    """Every key scored exactly; the top by summed softmax attended."""

    def step(q: Tensor, kv: KV) -> tuple[Tensor, Tensor]:
        logits = (q @ kv.k.mT).float() * q.shape[-1] ** -0.5
        idx = logits.softmax(-1).sum(1).topk(top, -1).indices
        return attend(q, kv.k, kv.v, idx), idx

    return step


def summed_query_topk(top: int) -> Step:
    """Every key scored by the heads' summed query; the top attended. (The nanogpt
    decode's fused Triton top-k takes one KV head per call: at 8 heads it is slower
    than this.)"""

    def step(q: Tensor, kv: KV) -> tuple[Tensor, Tensor]:
        idx = (kv.k @ q.sum(1)[..., None])[..., 0].topk(top, -1).indices
        return attend(q, kv.k, kv.v, idx), idx

    return step


def loki_step(d: int, top: int) -> Step:
    """Loki: keys stored in their PCA basis, the first d channels scored."""

    def step(q: Tensor, kv: KV) -> tuple[Tensor, Tensor]:
        axes = kv.once("pca", lambda: loki.basis(kv.k.float()).to(kv.k.dtype))
        axes = axes[..., :d]
        k_pca = kv.once(f"pca{d}", lambda: (kv.k @ axes).contiguous())
        approx = ((q @ axes) @ k_pca.mT).float() * q.shape[-1] ** -0.5
        idx = approx.softmax(-1).sum(1).topk(top, -1).indices
        return attend(q, kv.k, kv.v, idx), idx

    return step


def superblock_candidates(
    m: BlockTopK, q: Tensor, kv: KV, ids: Tensor, means: Tensor
) -> tuple[Tensor, Tensor]:
    """(G, B') the blocks of the kept_superblocks best superblocks (by their
    summaries) plus those of the unfinished last one, and (G, B', D) their means;
    ids (G, B) and means (G, B, D) are every block's."""
    dim = kv.k.shape[-1]
    size, n_super = m.superblock, ids.shape[1] // m.superblock
    span = size * m.block
    summary = kv.once(
        f"{m.coarse}{m.block}x{size}",
        lambda: summary_keys(
            kv.k[:, : n_super * span].unflatten(1, (-1, span)), m.coarse
        ),
    )  # (G, n_super, r, D)
    if m.coarse == "bound":
        mid, half = summary.unbind(-2)
        upper = (q @ mid.mT + q.abs() @ half.mT).float() * dim**-0.5
        score = upper.logsumexp(1)
    else:
        score = lse_scores(q, summary[:, :, 0])
    kept = score.topk(min(m.kept_superblocks, n_super), -1).indices
    within = torch.arange(size, device=q.device)
    own = ids[:, n_super * size :]  # the unfinished superblock's blocks
    ids = torch.cat([(kept[..., None] * size + within).flatten(1), own], 1)
    return ids, means.gather(1, ids[..., None].expand(-1, -1, dim))


def block_keys(blocks: Tensor, block: int) -> Tensor:
    """(G, B * block) every position of the blocks (G, B) of block keys."""
    within = torch.arange(block, device=blocks.device)
    return (blocks[..., None] * block + within).flatten(1)


def block_topk(m: BlockTopK) -> Step:
    """BlockTopK: block means (and superblock summaries) kept as K is written."""

    def step(q: Tensor, kv: KV) -> tuple[Tensor, Tensor]:
        groups, n, dim = kv.k.shape
        n_blocks = n // m.block
        means = kv.once(
            f"means{m.block}",
            lambda: kv.k[:, : n_blocks * m.block].unflatten(1, (-1, m.block)).mean(2),
        )
        start = max(n - m.window + m.block - 1, 0) // m.block  # blocks before window
        ids = torch.arange(n_blocks, device=q.device).expand(groups, -1)
        if not m.superblock:  # the allowed blocks are a prefix: slice, no mask kernel
            ids, means = ids[:, :start], means[:, :start]
        else:
            ids, means = superblock_candidates(m, q, kv, ids, means)
        top = min((m.keep - m.window) // m.block, ids.shape[1])
        score = lse_scores(q, means)
        if m.superblock:
            score = score.masked_fill(ids >= start, float("-inf"))
        blocks = ids.gather(1, score.topk(top, -1).indices)
        recent = torch.arange(n - m.window, n, device=q.device).expand(groups, -1)
        twice = (recent[..., None] // m.block == blocks[:, None]).any(-1)
        idx = torch.cat([block_keys(blocks, m.block), recent.masked_fill(twice, -1)], 1)
        return attend(q, kv.k, kv.v, idx), idx

    return step


def channel_order(m: Tournament, q: Tensor) -> Tensor:
    """(G, D) channels in the order m reads them (q: (G, 2, D) float32)."""
    groups, _, dim = q.shape
    if isinstance(m, GroupOrder):  # groups of GROUP stored channels, by group |q|
        stored = permutation(m.grouping, dim, str(q.device)).argmax(0)
        energy = (q.abs() @ permutation(m.grouping, dim, str(q.device))).sum(1)
        place = energy.unflatten(-1, (-1, GROUP)).sum(-1).argsort(-1, descending=True)
        return stored.view(-1, GROUP)[place].flatten(-2)
    if m.order == "query":
        return q.abs().sum(1).argsort(-1, descending=True)
    assert m.order == "random"  # the emulation's layer 0 permutation
    return rotation("random", 0, groups, dim, str(q.device)).argmax(-2)


def pick(m: Tournament, score: Tensor, r: int, budget: int, blocks: int) -> Tensor:
    """(G, M') positions in the current list (G, M) that go on from round r;
    blocks: fixed blocks of the original slots, for halving inside blocks."""
    if m.rounds == "bracket":
        pairs = score.unflatten(-1, (-1, 2))
        return pairs.argmax(-1) + 2 * torch.arange(pairs.shape[1], device=score.device)
    if not m.in_blocks(r):
        return score.topk(budget, -1).indices
    per = score.unflatten(-1, (blocks, -1))
    top = per.topk(per.shape[-1] // 2, -1).indices
    offset = per.shape[-1] * torch.arange(blocks, device=score.device)[:, None]
    return (top + offset).flatten(1)


def read_channels(k_t: Tensor, channels: Tensor, slots: Tensor) -> Tensor:
    """(G, w, M) the entries of K (k_t: (G, D, n) transposed) on channels (G, w)
    at slots (G, M)."""
    head = torch.arange(k_t.shape[0], device=k_t.device)[:, None, None]
    return k_t[head, channels[..., None], slots[:, None]]


def keep_slots(go: Tensor, slots: Tensor, partial: Tensor) -> tuple[Tensor, Tensor]:
    """The survivors go (G, M') of slots (G, M) and of their partial dot products
    (G, 2, M)."""
    rows = go[:, None].expand(-1, partial.shape[1], -1)
    return slots.gather(1, go), partial.gather(2, rows)


def tournament(m: Tournament) -> Step:
    """Tournament and its subclasses: halving (global or in blocks) or bracket,
    channels by |q|, random or in groups, losers dropped, approximate or mean."""
    assert m.local == 0 and m.losers in ("drop", "approx", "meanv")

    def step(q: Tensor, kv: KV) -> tuple[Tensor, Tensor]:
        groups, n, dim = kv.k.shape
        total = halving_rounds(n, m.keep)
        slots = m.keep << total if total else n
        block = m.block if isinstance(m, BlockTournament) else slots
        blocks = slots // min(block, slots)
        k_t = kv.once(
            f"k_t{slots}",
            lambda: F.pad(kv.k, (0, 0, 0, slots - n)).transpose(1, 2).contiguous(),
        )
        qf = q.float()
        dims = channel_order(m, qf)
        q_ord = qf.gather(-1, dims[:, None].expand(-1, 2, -1))
        q_sum = qf.abs().sum(-1)
        idx = torch.arange(slots, device=q.device).expand(groups, -1)
        partial = torch.zeros(groups, 2, slots, device=q.device)
        approx = torch.full_like(partial, float("-inf"))
        for r in range(1, total + 1):
            low, high = m.read_by(r - 1), m.read_by(r)
            rows = read_channels(k_t, dims[:, low:high], idx)  # (G, w, M)
            partial = partial + q_ord[..., low:high] @ rows.float()
            seen = q_ord[..., :high].abs().sum(-1) / q_sum
            current = partial * (dim**-0.5 / seen.sqrt())[..., None]
            if m.losers != "drop":  # winners' entries are replaced at the end
                approx.scatter_(-1, idx[:, None].expand(-1, 2, -1), current)
            score = current.logsumexp(1).masked_fill(idx >= n, float("-inf"))
            go = pick(m, score, r, m.keep << (total - r), blocks)
            idx, partial = keep_slots(go, idx, partial)
        won = idx.masked_fill(idx >= n, -1)
        if m.losers == "drop":
            return attend(q, kv.k, kv.v, won), won
        return approximate_losers(m, q, kv, idx, approx), won

    return step


def approximate_losers(
    m: Tournament, q: Tensor, kv: KV, won: Tensor, approx: Tensor
) -> Tensor:
    """Winners exact, each loser its last partial logit capped at the lowest winner
    logit; one softmax over every key. approx: every value read; meanv: the
    losers' weight goes to their mean value (from the running sum of values).

    Args:
        won: (G, M) winner slots; slots from n on are padding.
        approx: (G, 2, slots) each slot's last scaled partial logit.
    """
    n, dim = kv.k.shape[1:]
    ok = (won < n)[:, None]  # (G, 1, M)
    rows = won.clamp(max=n - 1)[..., None].expand(-1, -1, dim)
    k_win, v_win = kv.k.gather(1, rows).float(), kv.v.gather(1, rows).float()
    exact = ((q.float() @ k_win.mT) * dim**-0.5).masked_fill(~ok, float("-inf"))
    floor = exact.masked_fill(~ok, float("inf")).amin(-1, keepdim=True)
    at = won[:, None].expand(-1, 2, -1)
    logits = approx.minimum(floor).scatter(-1, at, exact)[..., :n]
    if m.losers == "approx":
        return (logits.softmax(-1) @ kv.v.float()).to(q.dtype)
    weight = logits.softmax(-1)
    win = weight.gather(-1, at.clamp(max=n - 1)).masked_fill(~ok, 0)
    v_sum = kv.once("v_sum", lambda: kv.v.float().sum(1))
    v_lost = v_sum - (v_win * ok[:, 0, :, None]).sum(1)
    lost = v_lost / (n - ok[:, 0].sum(-1, keepdim=True))
    share = 1 - win.sum(-1, keepdim=True)
    return (win @ v_win + share * lost[:, None]).to(q.dtype)


def bandit(m: BoundBandit) -> Step:
    """BoundBandit with masks over every key: PyTorch cannot shrink the survivor
    set inside a CUDA graph, so every round reads `dims` channels of all keys.

    ponytail: an upper bound on its time; a kernel would compact the survivors.
    """

    def step(q: Tensor, kv: KV) -> tuple[Tensor, Tensor]:
        groups, n, dim = kv.k.shape
        mu = kv.once("mu", lambda: kv.k.float().mean(1, keepdim=True))
        c_t = kv.once("c_t", lambda: (kv.k.float() - mu).transpose(1, 2).contiguous())
        total_sq = kv.once("c_sq", lambda: c_t.square().sum(1))  # (G, n)
        var = kv.once("c_var", lambda: c_t.square().mean(-1))  # (G, D)
        qf = q.float()
        dims = qf.abs().sum(1).argsort(-1, descending=True)
        q_ord = qf.gather(-1, dims[:, None].expand(-1, 2, -1))
        offset = qf @ mu.mT  # (G, 2, 1)
        partial = torch.zeros(groups, 2, n, device=q.device)
        read_sq = torch.zeros(groups, n, device=q.device)
        alive = torch.ones(groups, n, dtype=torch.bool, device=q.device)
        scale = dim**-0.5
        for r in range(dim // m.dims):
            now = dims[:, r * m.dims : (r + 1) * m.dims]
            rows = c_t.gather(1, now[..., None].expand(-1, -1, n))
            partial = partial + q_ord[..., r * m.dims : (r + 1) * m.dims] @ rows
            q_left = q_ord[..., (r + 1) * m.dims :]
            if m.bound == "exact":
                read_sq = read_sq + rows.square().sum(1)
                c_left = (total_sq - read_sq).clamp(min=0).sqrt()[:, None]
                radius = q_left.norm(dim=-1, keepdim=True) * c_left
            else:
                left = var.gather(1, dims[:, (r + 1) * m.dims :])[:, None]
                radius = (q_left.square() * left).sum(-1, keepdim=True).sqrt()
            est = (offset + partial) * scale
            radius = radius * (m.width * scale)
            lower = (est - radius).logsumexp(1).masked_fill(~alive, float("-inf"))
            upper = (est + radius).logsumexp(1)
            cutoff = lower.topk(min(m.keep, n), -1).values[..., -1:]
            alive = alive & (upper >= cutoff)
        idx = lower.topk(min(m.keep, n), -1).indices
        return attend(q, kv.k, kv.v, idx), idx

    return step
