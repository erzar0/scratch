"""Tile selection (src/nanogpt/tile_select.py): picks() keeps each query tile's own
key blocks plus its best earlier blocks by pooled score up to the budget (float64
reference, fixed and growing budgets); attention() is the softmax over each query's
document prefix inside its tile's picks, forward and backward (a budget covering
every document: dense), and compiles in a no-grad model. Needs CUDA:
    uv run python -m tests.nanogpt.test_tile_select
"""

import torch
from torch import Tensor

from nanogpt.tile_select import Settings, attention, picks
from nanogpt.tournament_fast import TILE
from tests.nanogpt.test_tournament import packed
from tests.nanogpt.test_tournament_fast import matches_masked_softmax
from tests.nanogpt.test_tournament_local import local_mask


def reference(q: Tensor, k: Tensor, cu: Tensor, s: Settings) -> Tensor:
    """(H, tiles, key blocks) picked, float64, one tile at a time."""
    length, heads, _ = q.shape
    n, nb, own = length // TILE, length // s.block, TILE // s.block
    tile_q = q.double().view(n, TILE, heads, -1).mean(1)
    block_k = k.double().view(nb, s.block, heads, -1).mean(1)
    out = torch.zeros(heads, n, nb, dtype=torch.bool, device=q.device)
    for i in range(n):
        start = max(c for c in cu.tolist() if c <= i * TILE)
        earlier = [j for j in range(nb) if (j + 1) * s.block > start and j < i * own]
        budget = s.keys * -(-((i + 1) * TILE - start) // 2048) if s.grow else s.keys
        extra = min(budget // s.block - own, len(earlier))
        out[:, i, i * own : (i + 1) * own] = True
        for h in range(heads):
            score = block_k[earlier, h] @ tile_q[i, h]
            top = score.topk(extra).indices.tolist()
            out[h, i, [earlier[t] for t in top]] = True
    return out


def blocks_of(q: Tensor, k: Tensor, cu: Tensor, s: Settings) -> Tensor:
    """(H, tiles, key blocks) picks()'s first `count` indices of each tile."""
    count, index = picks(q, k, cu, s.block, s.keys, s.grow)
    nb = index.shape[-1]
    keep = torch.arange(nb, device=q.device) < count[0, ..., None]
    return torch.zeros_like(keep).scatter_(-1, index[0].long(), keep)


def picks_own_plus_best(s: Settings, docs: tuple[int, ...]) -> None:
    """Same counts as float64 math; picks differ at most on float32 near-ties."""
    q, k, _, cu = packed(docs)
    got, want = blocks_of(q, k, cu, s), reference(q, k, cu, s)
    assert torch.equal(got.sum(-1), want.sum(-1)), (s, docs)
    assert int((got & want).sum()) >= 0.999 * int(want.sum()), (s, docs)


def mask_of(s: Settings):
    """(H, T, T) document prefix inside the tile's picked blocks."""

    def mask(q: Tensor, k: Tensor, cu: Tensor) -> Tensor:
        blocks = blocks_of(q, k, cu, s)
        keys = blocks.repeat_interleave(TILE, 1).repeat_interleave(s.block, 2)
        return keys & local_mask(q, cu, q.shape[0])

    return mask


def is_the_softmax_over_the_picks(s: Settings) -> None:
    compiled = torch.compile(attention, dynamic=False)  # settings: constants
    matches_masked_softmax(
        lambda q, k, v, cu: compiled(q, k, v, cu, 0.12, s), mask_of(s), s
    )


def compiles_in_a_no_grad_model() -> None:
    """Two layers on one cu_seqlens, compiled without grad, as record 91 validates."""
    x, _, _, cu = packed((300, 1200, 548), torch.bfloat16)

    def model(x: Tensor) -> Tensor:
        for _ in range(2):
            x = x + attention(x, x, x, cu, 0.12)
        return x

    with torch.no_grad():
        got, want = torch.compile(model, fullgraph=True)(x), model(x)
    error = float((got - want).float().norm() / want.float().norm())
    assert error < 0.01, error


if __name__ == "__main__":
    for s in (Settings(16, 256), Settings(128, 384), Settings(32, 128, True)):
        for docs in ((300, 1200, 548), (5, 1500, 4639)):
            picks_own_plus_best(s, docs)
    print("picks: own blocks plus the best earlier ones, fixed and growing budgets")
    for s in (Settings(16, 256), Settings(128, 384), Settings(16, 2048)):
        is_the_softmax_over_the_picks(s)
    print("attention: softmax over the picks (budget 2048: dense), forward, backward")
    compiles_in_a_no_grad_model()
    print("compiles in a no-grad model")
