"""Block-mean picks for one decode token in one Triton launch (batch 1, bf16 or FP8).

The rule (eval/ref_rule_eval.py block 1, bench/decode_ref_picks.py): entries in
aligned blocks of SIZE, each complete block summarized by its mean entry (kept as
entries are written); the 64 blocks of best q . mean, then the 64 entries of best
q . entry among those blocks' 64 * SIZE entries and the incomplete newest block's.
q is the heads' summed query.

decode_ref_picks.py runs it as top64 on the means, index arithmetic, a gather,
top64 on the gathered entries and an index: 6 launches. Here the programs score the
means and merge their top 64 blocks as csa2_decode.top64 does (groups of up to 16,
atomic counters left zero); the program that finishes the merge then scores the
chosen blocks' entries itself, through their block ids, SUB rows at a time.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from nanogpt import csa2_decode as cd
from nanogpt.csa2_decode import LEAST, _keep_best, _merged, _pack

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

SIZE = 8  # entries per block (V4.1: candidate_block_size 8)


@triton.jit
def _fused(
    q_ptr,
    mean_ptr,
    k_ptr,
    score_ptr,
    cand_ptr,
    level_ptr,
    count_ptr,
    pick_ptr,
    N,
    NB,
    programs,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    TOP: tl.constexpr,
    SUB: tl.constexpr,
    GP: tl.constexpr,
    GW: tl.constexpr,
    SIZE: tl.constexpr,
):
    """Program p: means [p BLOCK, (p + 1) BLOCK) of NB scored, their top TOP merged
    across programs; the last merger scores the chosen blocks' entries and the
    entries past NB SIZE (of N), and stores the top TOP entry indices."""
    p = tl.program_id(0)
    c = tl.arange(0, D)
    q = tl.load(q_ptr + c).to(tl.float32)
    for s in range(p * BLOCK, p * BLOCK + BLOCK, SUB):
        rows = s + tl.arange(0, SUB)
        ok = rows < NB
        m = tl.load(mean_ptr + rows[:, None].to(tl.int64) * D + c[None, :], ok[:, None])
        score = tl.sum(m.to(tl.float32) * q[None, :], 1)
        tl.store(score_ptr + rows, tl.where(ok, score, float("-inf")))
    tl.debug_barrier()
    rows = p * BLOCK + tl.arange(0, BLOCK)
    best = tl.topk(_pack(tl.load(score_ptr + rows), rows), TOP)
    i = tl.arange(0, TOP)
    tl.store(cand_ptr + p * TOP + i, best)
    tl.debug_barrier()
    g = p // GP
    size = tl.minimum(programs - g * GP, GP)
    if tl.atomic_add(count_ptr + 1 + g, 1) == size - 1:
        best = _merged(cand_ptr + g * GP * TOP, size * TOP, GP * TOP, TOP)
        tl.store(count_ptr + 1 + g, 0)
        groups = tl.cdiv(programs, GP)
        last = True
        if GW > 1:
            tl.store(level_ptr + g * TOP + i, best)
            tl.debug_barrier()
            last = tl.atomic_add(count_ptr, 1) == groups - 1
            if last:
                best = _merged(level_ptr, groups * TOP, GW * TOP, TOP)
                tl.store(count_ptr, 0)
        if last:
            # the chosen blocks; padding (rows >= NB, LEAST) when NB < TOP: -1
            blk = 0x7FFFFFFF - (best & 0xFFFFFFFF)
            tl.store(pick_ptr + i, tl.where(blk < NB, blk, -1))
            tl.debug_barrier()
            top = tl.full((TOP,), LEAST, tl.int64)
            for s in range(0, TOP * SIZE, SUB):
                e = s + tl.arange(0, SUB)
                b = tl.load(pick_ptr + e // SIZE, cache_modifier=".cg")
                ok = b >= 0
                ent = tl.where(ok, b * SIZE + e % SIZE, 0).to(tl.int32)  # _pack: >= 0
                at = k_ptr + ent[:, None].to(tl.int64) * D + c[None, :]
                kk = tl.load(at, ok[:, None], other=0.0).to(tl.float32)
                score = tl.where(ok, tl.sum(kk * q[None, :], 1), float("-inf"))
                top = _keep_best(top, score, ent, TOP, SUB)
            ent = NB * SIZE + tl.arange(0, SUB)  # the incomplete newest block
            ok = ent < N
            at = k_ptr + ent[:, None].to(tl.int64) * D + c[None, :]
            kk = tl.load(at, ok[:, None], other=0.0).to(tl.float32)
            score = tl.where(ok, tl.sum(kk * q[None, :], 1), float("-inf"))
            top = _keep_best(top, score, ent, TOP, SUB)
            tl.debug_barrier()
            tl.store(pick_ptr + i, 0x7FFFFFFF - (top & 0xFFFFFFFF))


def picks(q: Tensor, keys: Tensor, means: Tensor) -> Tensor:
    """The 64 picks (int64 (64,)) of the rule above: q (D,) the summed query, keys
    (N, D) bf16 entries, means (N // SIZE, D) bf16 their complete blocks' means.
    N >= 64 entries (fewer: invalid slots). Either cache may be FP8 E4M3 instead."""
    nb, dim = means.shape
    assert nb > 0 and keys.shape[0] - nb * SIZE < SIZE, (nb, keys.shape)
    sub = cd.TILE // dim
    block, programs = cd._blocks(nb, max(cd.TOP, sub))
    m = cd._merge_args(programs, keys.device)
    score = torch.empty(programs * block, dtype=torch.float32, device=keys.device)
    _fused[(programs,)](
        q,
        means,
        keys,
        score,
        m["cand"],
        m["level"],
        m["count"],
        m["picks"],
        keys.shape[0],
        nb,
        programs,
        BLOCK=block,
        D=dim,
        TOP=cd.TOP,
        SUB=sub,
        GP=m["GP"],
        GW=m["GW"],
        SIZE=SIZE,
        num_warps=8,
    )
    return m["picks"]
