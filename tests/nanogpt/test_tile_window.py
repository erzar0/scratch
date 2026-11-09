"""Tile selection inside a sliding window (src/nanogpt/tile_window.py): picks() keeps
each query tile's own key blocks plus its best earlier blocks within reach of the
window (float64 reference); a window past every document picks as tile_select;
attention() is the softmax over each query's last window + 1 document keys inside
its tile's picks, forward and backward (budget 2048: sliding-window dense), and
compiles in a no-grad model. Needs CUDA:
    uv run python -m tests.nanogpt.test_tile_window
"""

import torch
from torch import Tensor

from nanogpt import tile_select
from nanogpt.tile_select import Settings
from nanogpt.tile_window import attention, picks
from nanogpt.tournament_fast import TILE
from tests.nanogpt.test_tournament import packed
from tests.nanogpt.test_tournament_fast import matches_masked_softmax
from tests.nanogpt.test_tournament_local import local_mask


def reference(q: Tensor, k: Tensor, cu: Tensor, s: Settings, window: int) -> Tensor:
    """(H, tiles, key blocks) picked, float64, one tile at a time."""
    length, heads, _ = q.shape
    n, nb, own = length // TILE, length // s.block, TILE // s.block
    tile_q = q.double().view(n, TILE, heads, -1).mean(1)
    block_k = k.double().view(nb, s.block, heads, -1).mean(1)
    out = torch.zeros(heads, n, nb, dtype=torch.bool, device=q.device)
    for i in range(n):
        doc = max(c for c in cu.tolist() if c <= i * TILE)
        start = max(doc, i * TILE - window)
        earlier = [j for j in range(nb) if (j + 1) * s.block > start and j < i * own]
        extra = min(s.keys // s.block - own, len(earlier))
        out[:, i, i * own : (i + 1) * own] = True
        for h in range(heads):
            score = block_k[earlier, h] @ tile_q[i, h]
            top = score.topk(extra).indices.tolist()
            out[h, i, [earlier[t] for t in top]] = True
    return out


def blocks_of(q: Tensor, k: Tensor, cu: Tensor, s: Settings, window: int) -> Tensor:
    """(H, tiles, key blocks) picks()'s first `count` indices of each tile."""
    count, index = picks(q, k, cu, s.block, s.keys, window)
    keep = torch.arange(index.shape[-1], device=q.device) < count[0, ..., None]
    return torch.zeros_like(keep).scatter_(-1, index[0].long(), keep)


def picks_own_plus_best(s: Settings, window: int, docs: tuple[int, ...]) -> None:
    """Same counts as float64 math; picks differ at most on float32 near-ties."""
    q, k, _, cu = packed(docs)
    got, want = blocks_of(q, k, cu, s, window), reference(q, k, cu, s, window)
    assert torch.equal(got.sum(-1), want.sum(-1)), (s, window, docs)
    assert int((got & want).sum()) >= 0.999 * int(want.sum()), (s, window, docs)


def wide_window_is_tile_select(s: Settings) -> None:
    q, k, _, cu = packed((5, 1500, 4639))
    got = picks(q, k, cu, s.block, s.keys, q.shape[0])
    want = tile_select.picks(q, k, cu, s.block, s.keys, False)
    assert all(map(torch.equal, got, want))


def is_the_softmax_over_the_picks(s: Settings, window: int) -> None:
    compiled = torch.compile(attention, dynamic=False)  # settings: constants

    def mask(q: Tensor, k: Tensor, cu: Tensor) -> Tensor:
        blocks = blocks_of(q, k, cu, s, window)
        keys = blocks.repeat_interleave(TILE, 1).repeat_interleave(s.block, 2)
        return keys & local_mask(q, cu, window + 1)

    matches_masked_softmax(
        lambda q, k, v, cu: compiled(q, k, v, cu, 0.12, window, s), mask, (s, window)
    )


def compiles_in_a_no_grad_model() -> None:
    """Two layers on one cu_seqlens, compiled without grad, as record 91 validates."""
    x, _, _, cu = packed((300, 1200, 548), torch.bfloat16)

    def model(x: Tensor) -> Tensor:
        for window in (768, 1500):
            x = x + attention(x, x, x, cu, 0.12, window)
        return x

    with torch.no_grad():
        got, want = torch.compile(model, fullgraph=True)(x), model(x)
    error = float((got - want).float().norm() / want.float().norm())
    assert error < 0.01, error


if __name__ == "__main__":
    for s, window in ((Settings(32, 256), 768), (Settings(16, 512), 300)):
        for docs in ((300, 1200, 548), (5, 1500, 4639)):
            picks_own_plus_best(s, window, docs)
    wide_window_is_tile_select(Settings(32, 512))
    print("picks: own blocks plus the best earlier ones within the window")
    for s, window in ((Settings(32, 256), 768), (Settings(16, 2048), 300)):
        is_the_softmax_over_the_picks(s, window)
    print("attention: softmax over the picks (budget 2048: sliding window), backward")
    compiles_in_a_no_grad_model()
    print("compiles in a no-grad model")
