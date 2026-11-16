"""DeepSeek V4.1-Flash-1017's CSA2 attention with no parameters of its own: only the
projections every attention has remain (csa2_ref's low-rank q, the K=V projection
wkv with kv_norm, grouped o). From csa2_ours (summed-query picks, no indexer):

    entries  model.py's compressor with its weights tied to the attention's own and
             a uniform gate: entry e of a KV source layer = kv_norm(mean of wkv(x)
             over tokens [e m, (e + 1) m)), RoPE at token e m; at m = 1 the
             window's own rows
    sink     none (SINK = -inf): one softmax over the window rows and the picks;
             csa2_pf_sink sets a constant
    picks    csa2_ours.RULE (read at call time, so eval/rule_eval.py's rule swaps
             apply) among entries wholly older than the query's window, so no row
             is attended twice (at m = 1 entries are window rows)

A csa2_ref or csa2_ours checkpoint loads with its compressor, indexer and sink
dropped (eval/swap_eval.py). The cache holds the source layers' entries and every
layer's window; at m = 1 the window rows are entries, stored once.
"""

import torch
from torch import Tensor

from nanogpt import csa2_flex, csa2_ours, csa2_ref
from nanogpt.csa2_flex import starts


def pool(latent: Tensor, m: int) -> Tensor:
    """(T // m, D) mean of each aligned group of m rows of latent (T, D), in fp32."""
    n = latent.shape[0] // m
    return latent[: n * m].unflatten(0, (n, m)).float().mean(1).to(latent.dtype)


def window_picks(
    q: Tensor, entries: Tensor, begin: Tensor, m: int, top_k: int
) -> tuple[Tensor, Tensor]:
    """csa2_ours.RULE's picks among entries that end before each query's window:
    the rule run on the queries from WINDOW on, so query t ranks the entries below
    (t + 1 - WINDOW) // m as the rule ranks query t - WINDOW's."""
    w, length = csa2_flex.WINDOW, q.shape[0]
    entry = torch.zeros(min(w, length), top_k, dtype=torch.long, device=q.device)
    valid = entry.bool()
    if length <= w:
        return entry, valid
    later, ok = csa2_ours.RULE(q[w:], entries, begin[w:], m, top_k)
    return torch.cat([entry, later]), torch.cat([valid, ok])


class Csa2PfAttention(csa2_ours.Csa2OursAttention):
    """csa2_ours without the compressor and the sink: layer(x, cu_seqlens, shared)."""

    SINK = float("-inf")  # every head's sink logit (a constant, not a parameter)

    def __init__(self, layer: int, cfg: csa2_ref.Config = csa2_ref.SMALL) -> None:
        super().__init__(layer, cfg)
        self.source = self.compressor is not None
        self.compressor = None
        del self.attn_sink

    def forward(self, x: Tensor, cu_seqlens: Tensor, shared: dict) -> Tensor:
        # csa2_ref's module attributes, so its eval knobs (FP4, rotate_tail) apply
        ref, cfg, rd, theta = csa2_ref, self.cfg, self.cfg.rope_head_dim, self.theta
        x = x[0]
        length = x.shape[0]
        pos = torch.arange(length, device=x.device)
        q = self.wq_b(self.q_norm(self.wq_a(x))).unflatten(-1, (cfg.n_heads, -1))
        q = ref.rotate_tail(q, pos, rd, theta)
        latent = self.wkv(x)
        kv = ref.rotate_tail(self.kv_norm(latent), pos, rd, theta)
        kv = ref.fake_quant(kv, "swa") if ref.FP4 else kv
        none = torch.zeros(length, 1, dtype=torch.long, device=x.device)
        entries, entry, valid, m = kv[:0], none, none.bool(), self.ratio or 1
        if self.ratio:
            if self.source:
                at = torch.arange(length // m, device=x.device) * m
                entries = ref.rotate_tail(self.kv_norm(pool(latent, m)), at, rd, theta)
                shared["entries"] = (
                    ref.fake_quant(entries, "main") if ref.FP4 else entries
                )
            entries = shared["entries"]
            if self.picks:
                begin = starts(cu_seqlens, length)
                shared["picks"] = window_picks(q, entries, begin, m, cfg.index_topk)
            entry, valid = shared["picks"]
        sink = q.new_full((cfg.n_heads,), self.SINK, dtype=torch.float32)
        out, _ = ref.attend(q, kv, entries, entry, valid, cu_seqlens,
                            cfg.head_dim**-0.5, m, sink)  # fmt: skip
        out = ref.rotate_tail(out, pos, rd, theta, inverse=True)
        wo_a = self.wo_a.weight.view(cfg.o_groups, cfg.o_lora_rank, -1)
        o = torch.einsum("tgd,grd->tgr", out.view(length, cfg.o_groups, -1), wo_a)
        return self.wo_b(o.flatten(1))[None]


@torch.no_grad()
def drop_in(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    span: int,
    m: int,
    top_k: int,
    chunk: int = 512,
    sinks: int = 0,
) -> Tensor:
    """The block in place of any model's causal attention kernel (eval, loss only):
    q (T, H, D), k (T, Hk, D), v (T, Hk, Dv) post-RoPE, H a multiple of Hk, packed
    documents cu_seqlens. Each query attends to its newest WINDOW keys and its top_k
    entries by (sum of its group's q) . entry among those that end before the window
    and start within span keys back (the host's own window; T for none). Entries are
    mean-pooled aligned blocks of m keys and values, the key rescaled to its members'
    mean norm (host keys rotate before pooling, unlike the model layer). sinks > 0
    also keeps each document's first `sinks` keys exact (StreamingLLM's attention
    sinks, for hosts that park attention on their first token); entries then start
    after them. Returns (T, H, Dv) in v's dtype."""
    length, heads, _ = q.shape
    groups = heads // k.shape[1]
    n = length // m
    blocks = k[: n * m].float().unflatten(0, (n, m))  # (n, m, Hk, D)
    k_mean = blocks.mean(1)
    norm = blocks.norm(dim=-1).mean(1)[..., None]  # (n, Hk, 1) members' mean norm
    k_mean = k_mean * norm / k_mean.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v_mean = v[: n * m].float().unflatten(0, (n, m)).mean(1)
    begin = starts(cu_seqlens, length)
    w, e = csa2_flex.WINDOW, torch.arange(n, device=q.device)
    out = torch.empty(length, heads, v.shape[-1], dtype=v.dtype, device=v.device)
    for a in range(0, length, chunk):
        b = min(a + chunk, length)
        t, lo = torch.arange(a, b, device=q.device)[:, None], max(a - w + 1, 0)
        j = torch.arange(lo, b, device=q.device)
        qc = q[a:b].float().unflatten(1, (-1, groups))  # (C, Hk, g, D)
        near = (j <= t) & (t - j < w) & (j >= begin[a:b, None]) & (t - j <= span)
        logits = torch.einsum("ckgd,jkd->kgcj", qc, k[lo:b].float()) * scale
        logits = logits.masked_fill(~near, float("-inf"))
        far = torch.einsum("ckgd,ekd->kgce", qc, k_mean) * scale  # (Hk, g, C, n)
        allowed = ((e + 1) * m - 1 <= t - w) & (e * m >= begin[a:b, None] + sinks)
        allowed &= e * m >= t - span
        score = far.sum(1).masked_fill(~allowed, float("-inf"))  # summed query
        top = score.topk(min(top_k, n), -1).indices
        picked = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, top, True)
        far = far.masked_fill(~(picked & allowed)[:, None], float("-inf"))
        at = (begin[a:b, None] + torch.arange(sinks, device=q.device)).clamp(max=b - 1)
        keep = (t - at >= w) & (t - at <= span)  # not in the window, within span
        first = torch.einsum("ckgd,cskd->kgcs", qc, k[at].float()) * scale
        first = first.masked_fill(~keep, float("-inf"))
        p = torch.cat([logits, far, first], -1).softmax(-1)
        y = p[..., : j.numel()] @ v[lo:b].float().transpose(0, 1)[:, None]
        y = y + p[..., j.numel() : j.numel() + n] @ v_mean.transpose(0, 1)[:, None]
        y = y + torch.einsum("kgcs,cskd->kgcd", p[..., j.numel() + n :], v[at].float())
        out[a:b] = y.flatten(0, 1).transpose(0, 1).to(v.dtype)
    return out
