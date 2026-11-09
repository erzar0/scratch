"""Fast training tournament (src/nanogpt/tournament_fast.py): at tournament.py's
settings it picks tournament.py's keys; at other settings what the emulation
(shuffled blocks with that block size, winners per block and channels per round)
picks on each query's document prefix; the bit words hold exactly the winners; and
attention() is the softmax over the winners, forward and backward, and compiles in
a no-grad model (validation); queries in chunks get the same bits. The kernel
cases need CUDA (one selection config per width, not the autotuned ones):
    uv run python -m tests.nanogpt.test_tournament_fast
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import Tensor

from attention import harness
from attention.tournament.shuffled_blocks import ShuffledBlocks
from nanogpt import tournament, tournament_fast
from nanogpt.tournament_fast import (
    DEFAULT,
    TILE,
    Settings,
    _distinct,
    attention,
    doc_tiles,
    layouts,
    picked,
    select,
    window_words,
)
from tests.nanogpt.test_tournament import DOCS, packed

OTHER = (Settings(512, 8, 4), Settings(2048, 64, 16))


def test_default_settings_are_tournament_py() -> None:
    settings = (DEFAULT.block, DEFAULT.winners, DEFAULT.rounds)
    assert settings == (tournament.BLOCK, tournament.QUOTA, tournament.ROUNDS)
    ordered, shuffled = (layouts(3, 1024, "cpu", o) for o in (True, False))
    assert torch.equal(shuffled, tournament.layouts(3, "cpu"))
    for low, high in ((0, 2048), (2048, 5120)):  # 2 blocks, then 3
        blocks = shuffled[low:high].view(-1, 1024)
        assert torch.equal(ordered[low:high].view(-1, 1024), blocks.sort(-1).values)


def test_window_holds_every_query_prefix() -> None:
    """Keys from each query's document start to the query lie in its tile's window
    and in flex's key tiles for that tile."""
    length, longest = 6144, 2048
    cu = torch.tensor([0, *accumulate((100, 2048, 1500, 2048, 448))])
    low, count, _ = doc_tiles(cu, length, 1)
    query = torch.arange(length)
    start = cu[torch.searchsorted(cu, query, right=True) - 1]
    base = low[query // TILE] * TILE
    assert bool((start >= base).all())
    assert int((query - base).max()) < 32 * window_words(length, longest)
    assert torch.equal(low + count[0, 0] - 1, torch.arange(length // TILE))


def winner_mask(won: Tensor, length: int) -> Tensor:
    """(H, T, T) bool from winner lists (H, T, M), -1 none."""
    mask = torch.zeros(*won.shape[:2], length + 1, dtype=torch.bool, device=won.device)
    mask.scatter_(2, torch.where(won < 0, length, won).long(), True)
    return mask[..., :length]


@dataclass(frozen=True)
class Growing:
    """The emulation at these settings: winners per block, blocks grow with n."""

    settings: Settings

    def __call__(self, chunk: harness.Chunk) -> harness.Outcome:
        """Shuffled block halving at this chunk's length."""
        block, winners, dims = (
            self.settings.block,
            self.settings.winners,
            self.settings.dims,
        )
        blocks = -(-chunk.key.shape[1] // block)
        return ShuffledBlocks(block=block, keep=winners * blocks, dims=dims)(chunk)


def picks_tournament_py_keys() -> None:
    for dtype in (torch.float32, torch.bfloat16):
        q, k, _, cu = packed(DOCS, dtype)
        length = q.shape[0]
        mine = winner_mask(select(q, k, cu, max(DOCS)), length)
        assert torch.equal(mine, winner_mask(tournament.select(q, k, cu), length))


def picks_the_emulation_keys(settings: Settings, docs: tuple[int, ...]) -> None:
    """Queries at the block edges (n near 1, block, 2 block) and every 97th. Float32
    sums in another order may swap keys tied to 1e-8."""
    q, k, v, cu = packed(docs)
    got = winner_mask(select(q, k, cu, max(docs), settings), q.shape[0])
    block = settings.block
    edges = {*range(1, 40), *range(block - 25, block + 25)}
    edges |= {*range(2 * block - 15, 2 * block + 15)}
    emulation = Growing(settings)
    shared = total = 0
    for start, end in zip(cu[:-2].tolist(), cu[1:-1].tolist(), strict=True):
        for t in range(start, end):
            n = t - start + 1
            if n not in edges and t % 97:
                continue
            query = q[t, :, None, None]  # (H, 1 head, 1 query, D)
            key, value = (x[start : t + 1].transpose(0, 1) for x in (k, v))
            logits = query @ key[:, None].mT * 128**-0.5
            valid = torch.ones(1, n, dtype=torch.bool, device="cuda")
            chunk = harness.Chunk(query, key, value, logits, valid, 0, 128**-0.5)
            want = emulation(chunk).selected[:, 0]
            mine = got[:, t, start : t + 1]
            assert int(got[:, t].sum()) == int(mine.sum()), t  # only its prefix
            assert torch.equal(mine.sum(-1), want.sum(-1)), (settings, t, n)
            shared += int((mine & want).sum())
            total += int(want.sum())
    assert shared >= 0.999 * total, (settings, shared, total)


def bits_are_the_winners(settings: Settings) -> None:
    docs = (300, 1200, 548)
    q, k, _, cu = packed(docs)
    length = q.shape[0]
    low, _, _ = doc_tiles(cu, length, q.shape[1])
    longest = max(docs)
    blocks, winners, dims = settings.block, settings.winners, settings.dims
    words = picked(q, k, cu, low, longest, blocks, winners, dims)
    bit = torch.arange(32 * words.shape[-1], device="cuda")
    keys = low[:, None] * TILE + bit  # (tiles, window) key of each bit
    keys = keys.repeat_interleave(TILE, 0)  # per query
    on = ((words[..., bit // 32] >> (bit % 32)) & 1) == 1
    won = torch.where(on & (keys < length), keys, -1)
    want = winner_mask(select(q, k, cu, longest, settings), length)
    assert torch.equal(winner_mask(won, length), want), settings


def matches_masked_softmax(
    run: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    mask_of: Callable[[Tensor, Tensor, Tensor], Tensor],
    label: object,
) -> None:
    """run(q, k, v, cu) in bf16 on documents (300, 1200, 548), forward and
    gradients, within 1% of float32 softmax attention (scale 0.12) where
    mask_of(q, k, cu) (H, T, T) holds."""
    q, k, v, cu = packed((300, 1200, 548), torch.bfloat16)
    for x in (q, k, v):
        x.requires_grad_()
    out = run(q, k, v, cu)
    grad = torch.randn_like(out)
    out.backward(grad)
    mask = mask_of(q.detach(), k.detach(), cu)
    exact = [x.detach().float().requires_grad_() for x in (q, k, v)]
    logits = torch.einsum("thd,shd->hts", exact[0], exact[1]) * 0.12
    weights = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    want = torch.einsum("hts,shd->thd", weights, exact[2])
    want.backward(grad.float())
    for name, a, b in zip(
        ("output", "q grad", "k grad", "v grad"),
        (out.detach(), q.grad, k.grad, v.grad),
        (want.detach(), *(x.grad for x in exact)),
        strict=True,
    ):
        assert a is not None and b is not None
        error = float((a.float() - b).norm() / b.norm())
        assert error < 0.01, (label, name, error)


def attention_is_the_softmax_over_the_winners(settings: Settings) -> None:
    """bf16 forward and gradients within 1% of float32 math on the same winners."""
    compiled = torch.compile(attention)
    matches_masked_softmax(
        lambda q, k, v, cu: compiled(q, k, v, cu, 0.12, 2048, settings),
        lambda q, k, cu: winner_mask(select(q, k, cu, 2048, settings), q.shape[0]),
        settings,
    )


def chunks_are_the_whole_row(settings: Settings) -> None:
    """Queries in chunks of 512 (starting mid-document): picked() the whole row's
    bit words; attention() the softmax over them, forward and backward."""
    q, k, _, cu = packed((300, 1200, 548))
    length, heads, _ = q.shape
    low, _, _ = doc_tiles(cu, length, heads)
    run = settings.block, settings.winners, settings.dims
    whole = picked(q, k, cu, low, 2048, *run)
    for first in range(0, length, 512):
        tiles = slice(first // TILE, (first + 512) // TILE)
        part = picked(q[first : first + 512], k, cu, low[tiles], 2048, *run, first)
        assert torch.equal(part, whole[:, first : first + 512]), (settings, first)
    compiled = torch.compile(attention)
    matches_masked_softmax(
        lambda q, k, v, cu: compiled(q, k, v, cu, 0.12, 2048, settings, 512),
        lambda q, k, cu: winner_mask(select(q, k, cu, 2048, settings), q.shape[0]),
        (settings, "chunks of 512"),
    )


def compiles_in_a_no_grad_model() -> None:
    """Two layers on one cu_seqlens, compiled without grad, as record 91 validates:
    document tiles computed in the graph once broke flex's lowering there."""
    docs = (300, 1200, 548)
    x, _, _, cu = packed(docs, torch.bfloat16)

    def model(x: Tensor) -> Tensor:
        for _ in range(2):
            x = x + attention(x, x, x, cu, 0.12, None)
        return x

    with torch.no_grad():
        got, want = torch.compile(model, fullgraph=True)(x), model(x)
    error = float((got - want).float().norm() / want.float().norm())
    assert error < 0.01, error


def first_config(configs: list, named_args: dict, **kwargs: object) -> list:
    """The first distinct config per width (radix select from 256 slots, sort
    below): exactness does not depend on the config, and one compiles in minutes."""
    return _distinct(configs, named_args, **kwargs)[:1]


if __name__ == "__main__":
    tournament_fast._rounds.early_config_prune = first_config
    picks_tournament_py_keys()
    print("default settings: tournament.py's keys, float32 and bf16")
    picks_the_emulation_keys(OTHER[0], DOCS)
    picks_the_emulation_keys(OTHER[1], (5, 1500, 4500))
    print("other settings: the emulation's keys on each document prefix")
    for settings in (DEFAULT, *OTHER):
        bits_are_the_winners(settings)
        attention_is_the_softmax_over_the_winners(settings)
        chunks_are_the_whole_row(settings)
    print("bit words = winners; attention = softmax over them, forward and backward")
    print("queries in chunks of 512: the same bit words and attention")
    compiles_in_a_no_grad_model()
    print("compiles in a no-grad model")
