"""CSA2 as DeepSeek's reference code builds it, trainable on record 91's packed rows.

Source: DeepSeek-V4.1-Flash-1017 inference/model.py (vendored in
src/deepseek4.1-1017/inference/, revision 1bae0e10): Attention, Compressor, Indexer,
select_candidate_blocks and SharedAttentionRuntime, step for step. TensorRT-LLM's
CSA2 backend (NVIDIA/TensorRT-LLM, commit b38f901) follows the same numerics; its
cache formats are fake_quant(). Parameter names are model.py's, so a layer's
state_dict loads into model.py's Attention (tests/nanogpt/test_csa2_ref.py).

SMALL is V4.1-Flash-1017's layout scaled to record 91's 10 attention layers:
    layer   0  1  2  3  4  5  6  7  8  9
    ratio   -  2  2  2  2  1  1  1  1  1    (tokens per entry; - = window only)
    KV         S           S                (kv_source_layers; others reuse)
    picks      S           S           S    (index_source_layers; others reuse)
    blocks                 C           c    (layer 9 scores only the 256 blocks of
                                             8 entries layer 5 kept)
Every layer also reads its own 128-token window: one K=V row per token.

Not in model.py, each marked so in the code:
- documents: one packed row (B = 1), cu_seqlens at document starts. A query reads
  window rows and entries of its own document; an entry straddling a start is read
  by neither document. Candidate blocks count from the document's first entry.
- training: no FP8/FP4 rounding (fake_quant, env CSA2_REF_FP4=1, is an eval
  variant); the indexer learns by DeepSeek-V3.2's KL loss on detached inputs (V4.1
  publishes no training code), as the old baseline (experiments/csa2/a).
- attention: flex attention (window as one GQA head + picked entries, the sink via
  the log-sum-exp) instead of the gather kernel sparse_attn.
"""

import os
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention

from nanogpt import csa2_flex
from nanogpt.csa2_flex import mask, mask_inputs, starts, tiles
from nanogpt.tournament_fast import TILE, flex_options

FP4 = bool(int(os.environ.get("CSA2_REF_FP4", 0)))  # eval: TensorRT-LLM's caches
CHUNK = 1024  # indexer queries per pass (4096: no faster, profile_csa2_ref_sync)


@dataclass(frozen=True)
class Config:
    """model.py ModelArgs' attention fields (no YaRN: original_seq_len 0)."""

    dim: int = 768
    n_heads: int = 6
    head_dim: int = 128
    rope_head_dim: int = 16
    q_lora_rank: int = 192
    o_groups: int = 2
    o_lora_rank: int = 192
    norm_eps: float = 1e-20
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 2, 2, 2, 2, 1, 1, 1, 1, 1)
    kv_source_layers: tuple[int, ...] = (1, 5)
    index_source_layers: tuple[int, ...] = (1, 5, 9)
    candidate_source_layer: int = 5
    candidate_topk_blocks: int = 256
    candidate_block_size: int = 8
    index_n_heads: int = 6
    index_head_dim: int = 32
    index_topk: int = 64
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0


SMALL = Config()


class RMSNorm(nn.Module):
    """model.py's RMSNorm: fp32 inside, input dtype out."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        y = F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps)
        return y.to(x.dtype)


def rotate_tail(
    x: Tensor, positions: Tensor, rd: int, theta: float, inverse: bool = False
) -> Tensor:
    """model.py's apply_rotary_emb on x's last rd channels: adjacent pairs as complex
    numbers times exp(i positions freqs), in fp32. x (T, ..., D), positions (T,)."""
    freqs = theta ** -(torch.arange(0, rd, 2, device=x.device) / rd)
    angle = positions.float()[:, None] * freqs
    view = (-1,) + (1,) * (x.ndim - 2) + (rd // 2,)
    cos, sin = angle.cos().view(view), angle.sin().view(view)
    sin = -sin if inverse else sin
    a, b = x[..., -rd:].float().unflatten(-1, (-1, 2)).unbind(-1)
    y = torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)
    return torch.cat((x[..., :-rd], y.to(x.dtype)), -1)


def e2m1(x: Tensor) -> Tensor:
    """x rounded to FP4 E2M1 (|x| <= 6), nearest, ties to even, as TensorRT-LLM's
    quantization.pack_rows."""
    levels = x.new_tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6))
    mid = (levels[:-1] + levels[1:]) / 2
    size = x.abs().clamp(max=6).contiguous()
    code = torch.bucketize(size, mid)
    code = code + ((size == mid[code.clamp(max=6)]) & (code % 2 == 1)).long()
    return levels[code] * torch.where(torch.signbit(x), -1.0, 1.0)


def fake_quant(x: Tensor, fmt: str) -> Tensor:
    """x rounded through TensorRT-LLM's CSA2 cache row (quantization.pack_rows then
    unpack_rows): main = E2M1 + E4M3 scale per 16 channels; index = E2M1 + UE8M0 per
    32; swa = E4M3 + UE8M0 per 32."""
    group = 16 if fmt == "main" else 32
    blocks = x.float().unflatten(-1, (-1, group))
    amax = blocks.abs().amax(-1, keepdim=True)
    if fmt == "main":
        scale = (amax.clamp(min=6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        div, floor = (448, 1e-4) if fmt == "swa" else (6, 6 * 2**-126)
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp(min=floor) / div)))
    y = blocks / scale
    if fmt == "swa":
        y = y.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    else:
        y = e2m1(y)
    return (y * scale).flatten(-2).to(x.dtype)


def _picks(
    qi: Tensor,
    w: Tensor,
    k: Tensor,
    begin: Tensor,
    m: int,
    top_k: int,
    cand: Tensor,
    mode: int,
    blocks: int,
    size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """model.py Indexer.forward's top-k, CHUNK queries at a time (module docstring)."""
    length = qi.shape[0]
    entry = torch.zeros(length, top_k, dtype=torch.long, device=qi.device)
    valid = torch.zeros(length, top_k, dtype=torch.bool, device=qi.device)
    kept = torch.full((length, blocks), -1, dtype=torch.long, device=qi.device)
    heads = begin[::CHUNK].tolist()  # one host sync per call, not per chunk
    for a, head in zip(range(0, length, CHUNK), heads, strict=True):
        b = min(a + CHUNK, length)
        lo, hi = -(-head // m), min(b // m, k.shape[0])  # starts ascend
        if hi <= lo:
            continue
        # not in model.py: a query's own document; its first entry starts at or after
        # the document start
        first = -(-begin[a:b] // m)
        e = torch.arange(lo, hi, device=qi.device)
        t = torch.arange(a, b, device=qi.device)[:, None]
        allowed = (e >= first[:, None]) & (e < (t + 1) // m)
        logits = torch.einsum("qhd,ed->qhe", qi[a:b].float(), k[lo:hi].float())
        score = (logits.relu() * w[a:b].float()[..., None]).sum(1)
        score = score.masked_fill(~allowed, float("-inf"))
        if mode:  # candidate blocks, numbered from the document's first entry
            rel = (e - first[:, None]).clamp(min=0) // size
            n_blocks = -(-(hi - lo) // size)
            if mode == 1:  # select_candidate_blocks
                best = score.new_full((b - a, n_blocks), float("-inf"))
                best = best.scatter_reduce(1, rel, score, "amax")
                newest = (allowed.sum(1) - 1).div(size, rounding_mode="floor")
                at = torch.arange(n_blocks, device=qi.device)
                best = best.masked_fill(at == newest[:, None], float("inf"))
                best_top, best_at = best.topk(min(blocks, n_blocks), 1)
                ids = torch.where(best_top > float("-inf"), best_at, -1)
                kept[a:b, : ids.shape[1]] = ids
            else:  # uses_candidates: only inside the source's blocks
                ids = cand[a:b].clamp(max=n_blocks)
                ids = torch.where(ids < 0, n_blocks, ids)
                keep = e.new_zeros(b - a, n_blocks + 1, dtype=torch.bool)
                keep = keep.scatter(1, ids, True)[:, :n_blocks]
                score = score.masked_fill(~keep.gather(1, rel), float("-inf"))
        top, index = score.topk(min(top_k, hi - lo), -1)
        entry[a:b, : top.shape[1]] = index + lo
        valid[a:b, : top.shape[1]] = top > float("-inf")
    return entry, valid, kept


@torch.library.custom_op("csa2_ref::picks", mutates_args=())
def picks(
    qi: Tensor,
    w: Tensor,
    k: Tensor,
    begin: Tensor,
    m: int,
    top_k: int,
    cand: Tensor,
    mode: int,
    blocks: int,
    size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Each query's top_k entries by model.py's indexer score
    sum_h relu(qi_h . k_e) w_h: entry, valid (T, top_k) and, mode 1 (candidate
    source), the kept blocks (T, blocks), -1 for none; mode 2 scores only inside
    cand's blocks. qi (T, Hi, Di), w (T, Hi), k (N, Di), begin (T,) document
    starts. Opaque to torch.compile; no grad."""
    with torch.no_grad():
        return _picks(qi, w, k, begin, m, top_k, cand, mode, blocks, size)


@picks.register_fake
def _(
    qi: Tensor,
    w: Tensor,
    k: Tensor,
    begin: Tensor,
    m: int,
    top_k: int,
    cand: Tensor,
    mode: int,
    blocks: int,
    size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    length = qi.shape[0]
    return (
        qi.new_empty((length, top_k), dtype=torch.long),
        qi.new_empty((length, top_k), dtype=torch.bool),
        qi.new_empty((length, blocks), dtype=torch.long),
    )


def attend(
    q: Tensor,
    kv: Tensor,
    entries: Tensor,
    entry: Tensor,
    valid: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    m: int,
    sink: Tensor,
) -> tuple[Tensor, Tensor]:
    """model.py's sparse_attn on a packed row: softmax of q (T, H, D) over its window
    rows kv (T, D) and picked entries (N, D) (keys = values), plus a per-head sink
    logit (H,) with no value. Returns the output (T, H, D) and the log-sum-exp
    without the sink (T, H)."""
    length, n = q.shape[0], entries.shape[0]
    pad = -n % TILE
    width = max(n + pad, 32)  # words() needs one word for window-only layers
    begin, bits = mask_inputs(entry, valid, cu_seqlens, length, width)
    count, index = tiles(begin, length, n, m)
    rows = torch.cat([kv, F.pad(entries, (0, 0, 0, pad))])[None, None]
    block_mask = BlockMask.from_kv_blocks(
        count,
        index,
        BLOCK_SIZE=TILE,
        mask_mod=mask(begin, bits, length),
        seq_lengths=(length, rows.shape[2]),
    )
    out, aux = cast(
        tuple[Tensor, Any],
        flex_attention(
            q.transpose(0, 1)[None],
            rows,
            rows,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=True,
            kernel_options=flex_options(),
            return_aux=AuxRequest(lse=True),
        ),
    )
    lse = aux.lse[0].transpose(0, 1)
    # the sink adds exp(sink) to each head's softmax sum
    out = out[0].transpose(0, 1) * torch.sigmoid(lse - sink)[..., None]
    return out.to(q.dtype), lse


class _AttachLoss(torch.autograd.Function):
    """Identity on the output; the backward gives the loss gradient 1, as if the loss
    were added to the training loss (the old baseline's deepseek/csa2.py)."""

    @staticmethod
    def forward(ctx: Any, output: Tensor, loss: Tensor) -> Tensor:
        ctx.loss_like = (loss.dtype, loss.device)
        return output.view_as(output)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, Tensor]:
        dtype, device = ctx.loss_like
        return grad, torch.ones((), dtype=dtype, device=device)


class Compressor(nn.Module):
    """model.py's Compressor: ratio tokens -> one latent (pre-RoPE)."""

    def __init__(self, cfg: Config, ratio: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        # TensorRT-LLM's choice: input-dtype projections (model.py: fp32 for ratio > 1)
        self.wkv = nn.Linear(cfg.dim, cfg.head_dim, bias=False)
        if ratio > 1:
            self.wgate = nn.Linear(cfg.dim, cfg.head_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """(T // ratio, D) latents of x (T, C): softmax-gated sum per group, normed."""
        if self.ratio == 1:
            return cast(Tensor, self.norm(self.wkv(x)))
        n = x.shape[0] // self.ratio
        x = x[: n * self.ratio]
        kv = self.wkv(x).unflatten(0, (n, self.ratio))
        score = self.wgate(x).unflatten(0, (n, self.ratio)).float()
        pooled = (kv.float() * score.softmax(1)).sum(1).to(kv.dtype)  # bf16 in autocast
        return cast(Tensor, self.norm(pooled))


class Indexer(nn.Module):
    """model.py's Indexer weights; the scoring is picks()."""

    def __init__(self, cfg: Config, layer: int) -> None:
        super().__init__()
        heads, dim = cfg.index_n_heads, cfg.index_head_dim
        self.owns_k = layer in cfg.kv_source_layers
        self.wq_b = nn.Linear(cfg.q_lora_rank, heads * dim, bias=False)
        self.weights_proj = nn.Linear(cfg.dim, heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(cfg.head_dim, dim, bias=False)
            self.k_norm = RMSNorm(dim, cfg.norm_eps)


class Csa2RefAttention(nn.Module):
    """model.py's Attention on one packed row: layer(x, cu_seqlens, shared) -> out.

    x (1, T, dim), T a multiple of 128; shared: one dict per model forward (model.py's
    SharedAttentionRuntime: entries and index keys from KV sources, picks from index
    sources, candidate blocks from the candidate source).
    """

    def __init__(self, layer: int, cfg: Config = SMALL) -> None:
        super().__init__()
        assert set(cfg.kv_source_layers) <= set(cfg.index_source_layers)
        assert cfg.window_size == csa2_flex.WINDOW
        self.layer, self.cfg = layer, cfg
        self.ratio = cfg.compress_ratios[layer]
        self.theta = cfg.compress_rope_theta if self.ratio else cfg.rope_theta
        dim, heads, hd = cfg.dim, cfg.n_heads, cfg.head_dim
        self.attn_sink = nn.Parameter(torch.zeros(heads))
        self.wq_a = nn.Linear(dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, heads * hd, bias=False)
        self.wkv = nn.Linear(dim, hd, bias=False)
        self.kv_norm = RMSNorm(hd, cfg.norm_eps)
        groups = cfg.o_groups
        self.wo_a = nn.Linear(
            heads * hd // groups, groups * cfg.o_lora_rank, bias=False
        )
        self.wo_b = nn.Linear(groups * cfg.o_lora_rank, dim, bias=False)
        self.compressor = (
            Compressor(cfg, self.ratio) if layer in cfg.kv_source_layers else None
        )
        self.indexer = Indexer(cfg, layer) if layer in cfg.index_source_layers else None
        self.mode = (
            1 if layer == cfg.candidate_source_layer
            else 2 if 0 <= cfg.candidate_source_layer < layer
            else 0
        )  # fmt: skip
        # not in model.py (inference only): the old baseline's initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: Tensor, cu_seqlens: Tensor, shared: dict) -> Tensor:
        cfg, rd, theta, m = self.cfg, self.cfg.rope_head_dim, self.theta, self.ratio
        x = x[0]
        length = x.shape[0]
        pos = torch.arange(length, device=x.device)
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (cfg.n_heads, cfg.head_dim))
        q = rotate_tail(q, pos, rd, theta)
        kv = rotate_tail(self.kv_norm(self.wkv(x)), pos, rd, theta)
        kv = fake_quant(kv, "swa") if FP4 else kv
        scale = cfg.head_dim**-0.5
        loss = None
        if not m:  # window only
            none = torch.zeros(length, 1, dtype=torch.long, device=x.device)
            entries = kv[:0]
            out, lse = attend(q, kv, entries, none, none.bool(), cu_seqlens, scale, 1,
                              self.attn_sink)  # fmt: skip
        else:
            if self.compressor is not None:
                assert self.indexer is not None
                latent = self.compressor(x)
                at = torch.arange(latent.shape[0], device=x.device) * m
                # not in model.py: detached (V3.2), so only the indexer loss trains wk
                k = self.indexer.k_norm(self.indexer.wk(latent.detach()))
                k = rotate_tail(k, at, rd, theta)
                shared["index_k"] = fake_quant(k, "index") if FP4 else k
                entries = rotate_tail(latent, at, rd, theta)
                shared["entries"] = fake_quant(entries, "main") if FP4 else entries
            entries = shared["entries"]
            if self.indexer is not None:
                idx, heads = self.indexer, cfg.index_n_heads
                # not in model.py: detached inputs (V3.2's indexer training)
                qi = idx.wq_b(qr.detach()).unflatten(-1, (heads, cfg.index_head_dim))
                qi = rotate_tail(qi, pos, rd, theta)
                qi = fake_quant(qi, "index") if FP4 else qi
                w = idx.weights_proj(x.detach()) * (
                    cfg.index_head_dim**-0.5 * heads**-0.5
                )
                index_k = shared["index_k"]
                begin = starts(cu_seqlens, length)
                cand = shared["blocks"] if self.mode == 2 else begin[:0]
                entry, valid, kept = picks(
                    qi.detach(), w.detach(), index_k.detach(), begin, m,
                    cfg.index_topk, cand, self.mode,
                    cfg.candidate_topk_blocks if self.mode == 1 else 0,
                    cfg.candidate_block_size,
                )  # fmt: skip
                if self.mode == 1:
                    shared["blocks"] = kept
                shared["picks"] = entry, valid
            entry, valid = shared["picks"]
            out, lse = attend(q, kv, entries, entry, valid, cu_seqlens, scale, m,
                              self.attn_sink)  # fmt: skip
            if self.indexer is not None and torch.is_grad_enabled():
                loss = indexer_loss(q, entries, entry, valid, lse, self.attn_sink,
                                    scale, qi, w, index_k)  # fmt: skip
        out = rotate_tail(out, pos, rd, theta, inverse=True)
        groups = cfg.o_groups
        wo_a = self.wo_a.weight.view(groups, cfg.o_lora_rank, -1)
        o = torch.einsum("tgd,grd->tgr", out.view(length, groups, -1), wo_a)
        y = self.wo_b(o.flatten(1))[None]
        return _AttachLoss.apply(y, loss) if loss is not None else y


def indexer_loss(
    q: Tensor,
    entries: Tensor,
    entry: Tensor,
    valid: Tensor,
    lse: Tensor,
    sink: Tensor,
    scale: float,
    qi: Tensor,
    w: Tensor,
    index_k: Tensor,
) -> Tensor:
    """Not in model.py: DeepSeek-V3.2's indexer loss (the old baseline's): cross-
    entropy from the attention's weight on the picks, summed over heads and
    renormalized, to the softmax of the indexer's scores of the picks."""
    with torch.no_grad():
        logits = torch.einsum("thd,tkd->thk", q, entries[entry]).float() * scale
        lse = torch.logaddexp(lse.float(), sink.float())  # the sink's share too
        prob = (logits - lse[..., None]).exp().sum(1).masked_fill(~valid, 0)
        target = prob / prob.sum(-1, keepdim=True).clamp_min(1e-9)
    logits = torch.einsum("thd,tkd->thk", qi, index_k[entry]).float()
    score = (logits.relu() * w.float()[..., None]).sum(1)
    # -1e9, not -inf: a query with no valid pick gets a finite log-softmax, times 0
    log_p = score.masked_fill(~valid, -1e9).log_softmax(-1)
    return -(target * log_p).sum(-1).mean()
