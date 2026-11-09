"""Tile keys (src/nanogpt/tile_keys.py): picks() keeps each query tile's own keys
plus its best earlier single keys by score (float64 reference); attention() is the
softmax over each query's document prefix inside its tile's picks, forward and
backward, at query tiles of 128 and 32 and in query chunks, and compiles in a
no-grad model. Needs CUDA:
    uv run python -m tests.nanogpt.test_tile_keys
"""

import torch
from torch import Tensor

from nanogpt.tile_keys import Settings, attention, picks
from tests.nanogpt.test_tournament import packed
from tests.nanogpt.test_tournament_fast import matches_masked_softmax
from tests.nanogpt.test_tournament_local import local_mask


def reference(q: Tensor, k: Tensor, cu: Tensor, s: Settings) -> Tensor:
    """(H, tiles, T) picked, float64, one tile at a time."""
    length, heads, _ = q.shape
    n = length // s.tile
    tile_q = q.double().view(n, s.tile, heads, -1).mean(1)
    out = torch.zeros(heads, n, length, dtype=torch.bool, device=q.device)
    for i in range(n):
        begin = i * s.tile
        start = max(c for c in cu.tolist() if c <= begin)
        out[:, i, begin : begin + s.tile] = True
        extra = min(s.keys - s.tile, begin - start)
        for h in range(heads if extra else 0):
            score = k[start:begin, h].double() @ tile_q[i, h]
            out[h, i, start + score.topk(extra).indices] = True
    return out


def keys_of(q: Tensor, k: Tensor, cu: Tensor, s: Settings) -> Tensor:
    """(H, tiles, T) picks()'s used slots."""
    length, heads, _ = q.shape
    pos = picks(q, k, cu, 0, s.tile, s.keys)[0].view(heads, length // s.tile, -1)
    out = torch.zeros(heads, length // s.tile, length + 1, dtype=torch.bool)
    return out.to(q.device).scatter_(-1, pos.long(), True)[..., :length]


def picks_own_plus_best(s: Settings, docs: tuple[int, ...]) -> None:
    """Same counts as float64 math; picks differ at most on float32 near-ties."""
    q, k, _, cu = packed(docs)
    got, want = keys_of(q, k, cu, s), reference(q, k, cu, s)
    assert torch.equal(got.sum(-1), want.sum(-1)), (s, docs)
    assert int((got & want).sum()) >= 0.999 * int(want.sum()), (s, docs)


def is_the_softmax_over_the_picks(s: Settings, chunk: int | None = None) -> None:
    compiled = torch.compile(attention, dynamic=False)  # settings: constants

    def mask(q: Tensor, k: Tensor, cu: Tensor) -> Tensor:
        keys = keys_of(q, k, cu, s).repeat_interleave(s.tile, 1)
        return keys & local_mask(q, cu, q.shape[0])

    matches_masked_softmax(
        lambda q, k, v, cu: compiled(q, k, v, cu, 0.12, s, chunk), mask, (s, chunk)
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
    for s in (Settings(128, 384), Settings(32, 256), Settings(128, 1024)):
        for docs in ((300, 1200, 548), (5, 1500, 4639)):
            picks_own_plus_best(s, docs)
    print("picks: own keys plus the best earlier ones")
    for s, chunk in (
        (Settings(128, 384), None),
        (Settings(32, 256), None),
        (Settings(32, 256), 512),
        (Settings(128, 2048), None),
    ):
        is_the_softmax_over_the_picks(s, chunk)
    print("attention: softmax over the picks (2048 keys: dense), chunks, backward")
    compiles_in_a_no_grad_model()
    print("compiles in a no-grad model")
