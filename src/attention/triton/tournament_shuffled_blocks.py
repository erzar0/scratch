"""Triton decode kernel: the parallel tournament, every round inside blocks.

Keys are stored in a random order (attention/tournament/shuffled_blocks.py), so
each block of K is a random sample of the context. Random keys time exactly that
layout, so the kernel does no permutation. Launches:
    1. common._block_rounds  one program per (KV head, block): all R rounds, each
                             block keeping its top half, then its keep / blocks
                             winners
    2. common.attend         exact attention on the winners
No global cutoff and no sync across blocks: the sampled kernel's cutoff launches
and exact last rounds (about 60 of 146 µs at 32768 tokens) are gone.
"""

import torch

from attention.tournament.general import halving_rounds
from attention.triton.common import DIMS, _block_rounds, attend, reading, sizes

BLOCKS = 32  # per KV head: keep / 32 winners per block at every length


def select(
    q: torch.Tensor, k_t: torch.Tensor, keep: int, blocks: int = BLOCKS
) -> torch.Tensor:
    """(G, keep) winner positions, keep / blocks per block, -1 for empty slots."""
    groups, dim, n = k_t.shape
    total = halving_rounds(n, keep)
    quota = keep // blocks
    block = quota << total  # a power of 2 when keep and blocks are
    assert quota * blocks == keep and total * DIMS <= dim
    pos = torch.full((groups, keep), -1, dtype=torch.int32, device=q.device)
    a0 = torch.empty(groups, keep, device=q.device)
    a1 = torch.empty_like(a0)
    _block_rounds[(groups, blocks)](
        *reading(q, k_t),
        pos_out=pos,
        a0_out=a0,
        a1_out=a1,
        n=n,
        rounds=total,
        **sizes(dim),
        B=block,
        QUOTA=quota,
        num_warps=8,  # measured: beats 4 and 16 at 32768 and 131072 keys
    )
    return pos


def tournament(
    q: torch.Tensor,
    k: torch.Tensor,
    k_t: torch.Tensor,
    v: torch.Tensor,
    keep: int,
    blocks: int = BLOCKS,
) -> torch.Tensor:
    """(G, 2, D) parallel tournament attention for one decode token."""
    return attend(q, k, v, select(q, k_t, keep, blocks))
