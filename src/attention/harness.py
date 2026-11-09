"""Decode-like sparse attention inside any Hugging Face model, and its measures.

attach(model, method, scored) registers a transformers attention function: the
prefix runs exact SDPA, and each of the last `scored` queries runs `method` against
its whole causal past, CHUNK queries at a time. Methods are emulated with dense
float32 math; their cost is counted, not timed: elements of K and V read per query
and KV head (dense reads 2 n d), as in SparQ (arXiv 2312.04985).

Grouped-query attention: the g query heads of a KV head share one selection, made
from the sum of their attention scores (SparQ's rule).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AttentionInterface, PreTrainedModel

NAME = "sparse_decode"
CHUNK = 128
TOP = 128  # exact top-k that recall is measured against


@dataclass(frozen=True)
class Chunk:
    """One chunk of scored queries of one layer, float32, batch 1.

    Attributes:
        query: (G, g, Q, D) post-rotary queries: G KV heads of g query heads each.
        key: (G, N, D) keys of every position up to the chunk's last query.
        value: (G, N, D) values of the same positions.
        logits: (G, g, Q, N) exact scaled logits, -inf on keys after the query.
        valid: (Q, N) whether key n is in query q's causal past.
        layer: Index of the layer.
        scaling: Logit scale, 1 / sqrt(D).
    """

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    logits: torch.Tensor
    valid: torch.Tensor
    layer: int
    scaling: float


@dataclass(frozen=True)
class Outcome:
    """A method's result on one chunk.

    Attributes:
        output: (G, g, Q, D) attention output.
        selected: (G, Q, N) keys whose exact logits the output used.
        reads: (G, Q) elements of K and V read.
        extra: Name -> (G, Q) further costs to sum (the tournament's sectors).
    """

    output: torch.Tensor
    selected: torch.Tensor
    reads: torch.Tensor
    extra: dict[str, torch.Tensor] = field(default_factory=dict)


type Method = Callable[[Chunk], Outcome]

LAYOUTS = {  # K layout (bf16) -> (keys per 32-byte sector, dimensions per sector)
    "sectors_rows": (1, 16),
    "sectors_chunks": (2, 8),
    "sectors_transpose": (16, 1),
}


def blocks_max(values: torch.Tensor, size: int) -> torch.Tensor:
    """(..., ceil(N / size)) the largest value in each block of size keys."""
    return (
        F.pad(values, (0, -values.shape[-1] % size)).unflatten(-1, (-1, size)).amax(-1)
    )


def k_sectors(
    rank: torch.Tensor, levels: torch.Tensor, level: torch.Tensor
) -> dict[str, torch.Tensor]:
    """(G, Q) 32-byte sectors of K read, per layout.

    Layouts: LAYOUTS (rows (n, d), 8-dimension chunks (d / 8, n, 8), transpose
    (d, n)), and sectors_copies: two copies of K as in SparQ, the partial reads
    from the transpose and the exact pass from rows.

    Args:
        rank: (G, Q, D) each dimension's place in the reading order.
        levels: (L,) dimensions read at each level, 0 first, D (the exact pass) last.
        level: (G, Q, N) the level each key reached.
    """
    prefix = rank[..., None, :] < levels[:, None]  # (G, Q, L, D)
    level = level.long()
    out = {}
    for name, (keys, per_sector) in LAYOUTS.items():
        sectors = prefix.unflatten(-1, (-1, per_sector)).any(-1).sum(-1)
        out[name] = sectors.gather(-1, blocks_max(level, keys)).sum(-1)
    exact = len(levels) - 1
    partial = blocks_max(level.clamp(max=exact - 1), 16)
    rows = (level == exact).sum(-1) * (rank.shape[-1] // 16)
    out["sectors_copies"] = prefix.sum(-1).gather(-1, partial).sum(-1) + rows
    return out


def group_score(logits: torch.Tensor) -> torch.Tensor:
    """(G, Q, N) attention scores summed over each group's query heads."""
    return logits.softmax(-1).sum(1)


def top_mask(score: torch.Tensor, k: int, valid: torch.Tensor) -> torch.Tensor:
    """(G, Q, N) the k highest-scoring valid keys of each query."""
    top = score.topk(min(k, score.shape[-1]), -1).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, top, True) & valid


def attend(chunk: Chunk, selected: torch.Tensor) -> torch.Tensor:
    """(G, g, Q, D) exact softmax attention over the selected keys only."""
    logits = chunk.logits.masked_fill(~selected[:, None], float("-inf"))
    return logits.softmax(-1) @ chunk.value[:, None]


def query_rank(q: torch.Tensor) -> torch.Tensor:
    """(G, Q, D) each dimension's place in the |q| order (0: largest |q| summed
    over the group's heads), for q (G, g, Q, D)."""
    return q.abs().sum(1).argsort(-1, descending=True).argsort(-1)


def recent_keys(valid: torch.Tensor, local: int) -> torch.Tensor:
    """(Q, N) each query's last `local` valid keys; none for local 0."""
    count = valid.sum(-1, keepdim=True)
    return valid & (torch.arange(valid.shape[-1], device=valid.device) >= count - local)


def temperature(
    q_abs: torch.Tensor, read: torch.Tensor, scaling: float
) -> torch.Tensor:
    """(G, g, Q) SparQ's logit scale for dot products over the dimensions read
    (G, Q, D): scaling / sqrt(f), f = |q[read]|_1 / |q|_1."""
    seen = (q_abs * read[:, None]).sum(-1)
    return scaling / (seen / q_abs.sum(-1)).sqrt()


def capped_logits(
    chunk: Chunk, won: torch.Tensor, approx: torch.Tensor
) -> torch.Tensor:
    """(G, g, Q, N) winners' exact logits; each loser its approximate logit, capped
    at the lowest winner logit (won: (G, Q, N), approx: (G, g, Q, N))."""
    winner = won[:, None]
    floor = chunk.logits.masked_fill(~winner, float("inf")).amin(-1, keepdim=True)
    return torch.where(winner, chunk.logits, approx.minimum(floor))


@dataclass
class Installed:
    """The registered attention function and the per-layer sums it keeps.

    Attributes:
        method: The method of the scored queries.
        scored: How many of the last queries run it.
        sums: Layer -> name -> sum over scored queries and KV heads: count, recall
            (of the exact top-TOP keys), mass (exact attention weight on the
            selected keys), error (relative to exact output), reads, dense reads,
            and the method's extra costs.
    """

    method: Method
    scored: int
    sums: dict[int, dict[str, float]] = field(default_factory=dict)

    @torch.no_grad()
    def __call__(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        """An HF attention function: (1, S, H, D) output, no weights."""
        assert query.shape[0] == 1 and attention_mask is None
        heads, seq_len, dim = query.shape[1:]
        groups = key.shape[1]
        start = max(seq_len - self.scored, 0)
        out = torch.empty_like(query)
        if start:
            out[:, :, :start] = F.scaled_dot_product_attention(
                query[:, :, :start],
                key[:, :, :start],
                value[:, :, :start],
                is_causal=True,
                scale=scaling,
                enable_gqa=True,
            )
        q = query[0].float().view(groups, heads // groups, seq_len, dim)
        k, v = key[0].float(), value[0].float()
        positions = torch.arange(seq_len, device=query.device)
        for s in range(start, seq_len, CHUNK):
            e = min(s + CHUNK, seq_len)
            valid = positions[None, :e] <= positions[s:e, None]
            logits = (q[:, :, s:e] @ k[:, None, :e].mT) * scaling
            chunk = Chunk(
                q[:, :, s:e],
                k[:, :e],
                v[:, :e],
                logits.masked_fill(~valid, float("-inf")),
                valid,
                cast(int, module.layer_idx),
                scaling,
            )
            outcome = self.method(chunk)
            out[0, :, s:e] = outcome.output.reshape(heads, e - s, dim).to(out.dtype)
            self.measure(chunk, outcome)
        return out.transpose(1, 2).contiguous(), None

    def measure(self, chunk: Chunk, outcome: Outcome) -> None:
        """Add the chunk's recall, mass, error and costs to sums."""
        probs = chunk.logits.softmax(-1)
        exact = probs @ chunk.value[:, None]
        mean = probs.mean(1)
        top = top_mask(mean, TOP, chunk.valid)
        selected = outcome.selected
        error = (outcome.output - exact).norm(dim=-1) / exact.norm(dim=-1)
        n = chunk.valid.sum(-1)
        dim = chunk.query.shape[-1]
        values = {
            "count": float(selected.shape[0] * selected.shape[1]),
            "recall": (selected & top).sum(-1) / top.sum(-1),
            "mass": (mean * selected).sum(-1),
            "error": error.mean(1),
            "reads": outcome.reads,
            "dense_reads": (2 * dim * n).expand_as(outcome.reads),
            "dense_sectors": (n * dim // 16).expand_as(outcome.reads),
        } | outcome.extra
        sums = self.sums.setdefault(chunk.layer, {})
        for name, value in values.items():
            total = value if isinstance(value, float) else float(value.double().sum())
            sums[name] = sums.get(name, 0.0) + total


def attach(model: PreTrainedModel, method: Method, scored: int) -> Installed:
    """Run method on the last scored queries of every attention of model."""
    state = Installed(method, scored)
    AttentionInterface.register(NAME, state)
    model.set_attn_implementation(NAME)
    return state
