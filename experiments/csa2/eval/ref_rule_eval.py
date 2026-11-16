"""packed_val.py with the V4.1-Flash-1017 reference checkpoint (b/train_csa2_ref.py)
picking by our parameter-free summed query instead of its learned indexer, at its
own pick layers (1, 5, 9; the others reuse their picks), over the same entries:
    score(t, e) = (sum_h q_h(t)) . entry_e   (attention's own rotated q and entries)
exact       top index_topk allowed entries.
block POOL  blocks of candidate_block_size entries counted from the document's first
            entry; complete blocks scored by their mean entry score (= summed query
            . mean entry), the newest block always kept (as select_candidate_blocks);
            top POOL * index_topk blocks; top index_topk allowed entries inside them.
            Layer 9 searches its own blocks, not layer 5's.
own         its learned indexer (no rewiring; for the env knobs below).
The indexer is rewired so forward() is unchanged: its query projection becomes
attention's wq_b at index_head_dim 128 (qi = q), its key path the identity
(index_k = entries); its weights and head weights w are not used.

Env knobs (eval only):
YARN=S        record 91's Yarn.apply(2048, 2048 * S) on the entry layers' RoPE
              (theta 160000; layer 0's window theta is untouched): channels with
              under 1 turn over 2048 tokens slow by S, over 32 turns unchanged.
YARN_TEMP=1   with YARN: record 91's attention scale x (1 + 0.2 ln S), all layers.
CSA2_REF_FP4  TensorRT-LLM's caches; under exact/block our picks read the FP4 main
              entries (the query stays bf16; no index cache).
SHARE=1       layer 9 reuses layer 5's picks (no pick pass at layer 9).
TOPK=N        N picks per query instead of index_topk 64.

python ref_rule_eval.py SCRIPT RUN ROWS OUT.csv own | exact | block POOL
"""

import dataclasses
import math
import os
import sys

import packed_val
import torch
from nanogpt_old import model as nm
from torch import Tensor, nn

script, run, rows, out, rule, *sizes = sys.argv[1:]
POOL = int(sizes[0]) if rule == "block" else 0
nm.SCRIPT = script
nm.module()  # imports nanogpt.csa2_ref as the script does
from nanogpt import csa2_ref  # noqa: E402

NEG = float("-inf")
YARN = float(os.environ.get("YARN", 0))
TOPK = int(os.environ.get("TOPK", 0))
SHARE = bool(int(os.environ.get("SHARE", 0)))
quant = csa2_ref.fake_quant


def yarn_tail(
    x: Tensor, positions: Tensor, rd: int, theta: float, inverse: bool = False
) -> Tensor:
    """csa2_ref.rotate_tail with YARN's frequencies (module docstring)."""
    freqs = theta ** -(torch.arange(0, rd, 2, device=x.device) / rd)
    if theta == csa2_ref.SMALL.compress_rope_theta:
        turns = 2048 * freqs / (2 * math.pi)
        weight = ((turns - 1) / 31).clamp(0, 1)
        freqs = freqs * (1 / YARN + weight * (1 - 1 / YARN))
    angle = positions.float()[:, None] * freqs
    view = (-1,) + (1,) * (x.ndim - 2) + (rd // 2,)
    cos, sin = angle.cos().view(view), angle.sin().view(view)
    sin = -sin if inverse else sin
    a, b = x[..., -rd:].float().unflatten(-1, (-1, 2)).unbind(-1)
    y = torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)
    return torch.cat((x[..., :-rd], y.to(x.dtype)), -1)


def _summed(
    qi: Tensor, k: Tensor, begin: Tensor, m: int, top_k: int, blocks: int, size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """csa2_ref._picks with the summed-query score (module docstring)."""
    length, chunk = qi.shape[0], csa2_ref.CHUNK
    entry = torch.zeros(length, top_k, dtype=torch.long, device=qi.device)
    valid = torch.zeros(length, top_k, dtype=torch.bool, device=qi.device)
    kept = torch.full((length, blocks), -1, dtype=torch.long, device=qi.device)
    qs = qi.float().sum(1)
    k = quant(k, "main") if csa2_ref.FP4 else k  # the FP4 entries attention reads
    for a, head in zip(range(0, length, chunk), begin[::chunk].tolist(), strict=True):
        b = min(a + chunk, length)
        lo, hi = -(-head // m), min(b // m, k.shape[0])
        if hi <= lo:
            continue
        first = -(-begin[a:b] // m)
        e = torch.arange(lo, hi, device=qi.device)
        t = torch.arange(a, b, device=qi.device)[:, None]
        allowed = (e >= first[:, None]) & (e < (t + 1) // m)
        score = (qs[a:b] @ k[lo:hi].float().T).masked_fill(~allowed, NEG)
        if POOL:
            n_blocks = -(-(hi - lo) // size)
            rel = e - first[:, None]
            rel = torch.where(
                rel >= 0, rel // size, n_blocks
            )  # spare bin: earlier docs
            mean = score.new_full((b - a, n_blocks + 1), NEG)
            mean = mean.scatter_reduce(1, rel, score, "mean", include_self=False)
            newest = (allowed.sum(1) - 1).div(size, rounding_mode="floor")
            at = torch.arange(n_blocks + 1, device=qi.device)
            mean = mean.masked_fill(at == newest[:, None], float("inf"))
            mean[:, n_blocks] = NEG  # older blocks are complete; later ones -inf
            best, ids = mean.topk(min(POOL * top_k, n_blocks), 1)
            ids = torch.where(best > NEG, ids, n_blocks)
            keep = torch.zeros_like(mean, dtype=torch.bool).scatter(1, ids, True)
            keep[:, n_blocks] = False
            score = score.masked_fill(~keep.gather(1, rel), NEG)
        top, index = score.topk(min(top_k, hi - lo), -1)
        entry[a:b, : top.shape[1]] = index + lo
        valid[a:b, : top.shape[1]] = top > NEG
    return entry, valid, kept


@torch.library.custom_op("csa2_ref_eval::summed_picks", mutates_args=())
def summed_picks(
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
    """csa2_ref.picks' signature; w and cand unused."""
    with torch.no_grad():
        return _summed(qi, k, begin, m, top_k, blocks, size)


@summed_picks.register_fake
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


def rewired(path):  # type: ignore[no-untyped-def]
    """nm.load, then each pick layer's indexer gives qi = q and index_k = entries
    (not for own), and the SHARE / TOPK knobs."""
    model, config = load(path)
    for layer in model.modules():
        if not isinstance(layer, csa2_ref.Csa2RefAttention):
            continue
        if TOPK:
            layer.cfg = dataclasses.replace(layer.cfg, index_topk=TOPK)
        if SHARE and layer.layer == 9:
            layer.indexer = None
        if rule != "own" and layer.indexer is not None:
            layer.cfg = dataclasses.replace(
                layer.cfg, index_head_dim=layer.cfg.head_dim
            )
            layer.indexer.wq_b = layer.wq_b
            if layer.indexer.owns_k:
                layer.indexer.wk = layer.indexer.k_norm = nn.Identity()
    return model, config


load, nm.load = nm.load, rewired
if rule != "own":
    csa2_ref.picks = summed_picks
    # no index cache: the query stays bf16 (the entries are quantized in _summed)
    csa2_ref.fake_quant = lambda x, fmt: x if fmt == "index" else quant(x, fmt)
if YARN:
    csa2_ref.rotate_tail = yarn_tail
    if int(os.environ.get("YARN_TEMP", 0)):
        attend, gain = csa2_ref.attend, 0.2 * math.log(YARN) + 1
        csa2_ref.attend = lambda *a: attend(*a[:6], a[6] * gain, *a[7:])
print("picks:", rule, *sizes, "fp4", csa2_ref.FP4, "yarn", YARN, "share", SHARE,
      "topk", TOPK or 64, flush=True)  # fmt: skip
packed_val.main(script, run, rows, out)
