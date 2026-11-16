"""DeepSeek V4.1-Flash-1017's CSA2 (csa2_ref: layout, compressor, K=V window, sink,
low-rank q and grouped o, partial RoPE) trained with our parameter-free picks in
place of its learned indexer. At each index source layer each query keeps its top
index_topk allowed entries by

    score(t, e) = (sum_h q_h(t)) . entry_e

on attention's own rotated q and entries, in one Triton launch (csa2_select_fast;
env CSA2_SELECT: exact, or tournament = halving rounds of 16 channels). The other
layers reuse the picks, as csa2_ref. No indexer weights, index keys or indexer loss.
No candidate blocks: layer 9 ranks every entry (csa2_ref's layer 9 ranks only layer
5's 256 blocks of 8 = every entry of a 2048-token training document).
Cache formats stay eval variants: under csa2_ref.FP4 the picks read the FP4 main
entries attention reads; the query stays bf16.
"""

import os

import torch
from torch import Tensor

from nanogpt import csa2_ref, csa2_select_fast
from nanogpt.csa2_flex import starts

RULE = getattr(csa2_select_fast, os.environ.get("CSA2_SELECT", "exact"))


class Csa2OursAttention(csa2_ref.Csa2RefAttention):
    """csa2_ref.Csa2RefAttention without the indexer: layer(x, cu_seqlens, shared)."""

    def __init__(self, layer: int, cfg: csa2_ref.Config = csa2_ref.SMALL) -> None:
        super().__init__(layer, cfg)
        self.picks = self.indexer is not None
        self.indexer = None

    def forward(self, x: Tensor, cu_seqlens: Tensor, shared: dict) -> Tensor:
        # csa2_ref's module attributes, so its eval knobs (FP4, rotate_tail) apply
        ref, cfg, rd, theta = csa2_ref, self.cfg, self.cfg.rope_head_dim, self.theta
        x = x[0]
        length = x.shape[0]
        pos = torch.arange(length, device=x.device)
        q = self.wq_b(self.q_norm(self.wq_a(x))).unflatten(-1, (cfg.n_heads, -1))
        q = ref.rotate_tail(q, pos, rd, theta)
        kv = ref.rotate_tail(self.kv_norm(self.wkv(x)), pos, rd, theta)
        kv = ref.fake_quant(kv, "swa") if ref.FP4 else kv
        none = torch.zeros(length, 1, dtype=torch.long, device=x.device)
        entries, entry, valid, m = kv[:0], none, none.bool(), self.ratio or 1
        if self.ratio:
            if self.compressor is not None:
                latent = self.compressor(x)
                at = torch.arange(latent.shape[0], device=x.device) * m
                entries = ref.rotate_tail(latent, at, rd, theta)
                shared["entries"] = (
                    ref.fake_quant(entries, "main") if ref.FP4 else entries
                )
            entries = shared["entries"]
            if self.picks:
                begin = starts(cu_seqlens, length)
                shared["picks"] = RULE(q, entries, begin, m, cfg.index_topk)
            entry, valid = shared["picks"]
        out, _ = ref.attend(q, kv, entries, entry, valid, cu_seqlens,
                            cfg.head_dim**-0.5, m, self.attn_sink)  # fmt: skip
        out = ref.rotate_tail(out, pos, rd, theta, inverse=True)
        wo_a = self.wo_a.weight.view(cfg.o_groups, cfg.o_lora_rank, -1)
        o = torch.einsum("tgd,grd->tgr", out.view(length, cfg.o_groups, -1), wo_a)
        return self.wo_b(o.flatten(1))[None]
