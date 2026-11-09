"""Triton decode: blocks then global (attention/tournament/blocks_then_global.py).

Launches common._block_rounds (the first R - GLOBAL_ROUNDS rounds, inside blocks),
common._global_rounds (the last ones, global halving), then common.attend."""

import torch

from attention.tournament.general import halving_rounds
from attention.triton.common import (
    BLOCK,
    DIMS,
    GLOBAL_ROUNDS,
    _block_rounds,
    _global_rounds,
    attend,
    reading,
    sizes,
)


def select(
    q: torch.Tensor, k_t: torch.Tensor, keep: int, block: int = BLOCK
) -> torch.Tensor:
    """(G, keep) winner positions of the hybrid tournament, -1 for empty slots."""
    groups, dim, n = k_t.shape
    total = halving_rounds(n, keep)
    rounds = total - GLOBAL_ROUNDS
    slots = keep << total
    assert rounds >= 0 and total * DIMS <= dim and slots % block == 0
    read = reading(q, k_t)
    quota = block >> rounds
    blocks = slots // block
    width = blocks * quota  # = keep * 2^GLOBAL_ROUNDS
    pos = torch.full((groups, width), -1, dtype=torch.int32, device=q.device)
    a0 = torch.empty(groups, width, device=q.device)
    a1 = torch.empty_like(a0)
    _block_rounds[(groups, blocks)](
        *read,
        pos_out=pos,
        a0_out=a0,
        a1_out=a1,
        n=n,
        rounds=rounds,
        **sizes(dim),
        B=block,
        QUOTA=quota,
    )
    won = torch.full((groups, keep), -1, dtype=torch.int32, device=q.device)
    _global_rounds[(groups,)](
        *read,
        pos_ptr=pos,
        a0_ptr=a0,
        a1_ptr=a1,
        win_ptr=won,
        first=rounds,
        **sizes(dim),
        C=width,
        KEEP=keep,
        ROUNDS=GLOBAL_ROUNDS,
        num_warps=8,
    )
    return won


def tournament(
    q: torch.Tensor, k: torch.Tensor, k_t: torch.Tensor, v: torch.Tensor, keep: int
) -> torch.Tensor:
    """(G, 2, D) hybrid tournament attention for one decode token."""
    return attend(q, k, v, select(q, k_t, keep))
