"""Channel-block training tournament (src/nanogpt/tournament_channel_blocks.py): the
blocks follow the rule (a greedy reference, with zero channels, and a case where no
free window is left); the winners are tournament_fast's kernels' on the same channel
order, exactly; the bit words hold the winners; attention() is the softmax over
them, forward and backward, in chunks too, and compiles in a no-grad model. Needs
CUDA (one selection config per width, not the autotuned ones):
    uv run python -m tests.nanogpt.test_tournament_channel_blocks
"""

from unittest import mock

import torch
from torch import Tensor

from nanogpt import tournament_channel_blocks as blocks
from nanogpt import tournament_fast
from nanogpt.tournament_fast import DEFAULT, Settings
from tests.nanogpt import test_tournament_fast as fast_tests
from tests.nanogpt.test_tournament import DOCS, packed
from tests.nanogpt.test_tournament_fast import first_config, winner_mask

RUN1, RUN2 = Settings(2048, 32, 16), Settings(2048, 64, 16)
REAL = 96  # record 91's real q, k channels of 128


def reference(q: Tensor, settings: Settings) -> tuple[Tensor, Tensor]:
    """The rule round by round in torch: windows() for q (C, H, D)."""
    width, rounds = settings.dims, settings.rounds
    x = q.float().transpose(0, 1)  # (H, C, D)
    size, dim = x.abs(), x.shape[-1]
    starts = dim - width + 1
    read = torch.zeros_like(size, dtype=torch.int32)
    channel_ = torch.arange(dim, device=q.device)
    values, orders = [], []
    for _ in range(rounds):
        unread = torch.where(read > 0, 0.0, size)
        window = torch.zeros_like(size[..., :starts])
        seen = torch.zeros_like(read[..., :starts])
        for j in range(width):
            window = window + unread[..., j : j + starts]
            seen = seen + read[..., j : j + starts]
        free = torch.where(seen == 0, window, -1.0)
        first = torch.where(free.amax(-1) > 0, free.argmax(-1), window.argmax(-1))
        channel = first[..., None] + torch.arange(width, device=q.device)
        fresh = read.gather(-1, channel) == 0
        values.append(torch.where(fresh, x.gather(-1, channel), 0.0))
        orders.append(channel)
        inside = (channel_ >= first[..., None]) & (channel_ < first[..., None] + width)
        read = torch.where(inside, 1, read)
    return torch.cat(values, -1), torch.cat(orders, -1).int()


def windows_follow_the_rule() -> None:
    """windows() = reference() on random q with record 91's 32 zero channels and
    on q with 40 real channels (from round 3 no free window has a positive sum, so
    rounds re-read channels at weight 0); no channel weighted twice."""
    q = torch.randn(300, 2, 128, device="cuda", dtype=torch.bfloat16)
    for real in (REAL, 40):
        x = q.clone()
        x[..., real:] = 0
        for settings in (DEFAULT, RUN1, RUN2):
            got, want = blocks.windows(x, settings), reference(x, settings)
            assert torch.equal(got[1], want[1]), (real, settings)
            assert torch.equal(got[0], want[0]), (real, settings)
            weighted = torch.where(got[0] != 0, got[1], -1).sort(-1).values
            twice = (weighted[..., 1:] == weighted[..., :-1]) & (weighted[..., 1:] >= 0)
            assert not bool(twice.any()), (real, settings)
            assert real > 40 or bool((got[0] == 0).any()), settings  # re-reads


def picks_tournament_fast_keys(settings: Settings, docs: tuple[int, ...]) -> None:
    """select() = tournament_fast.select with _order writing windows()' order,
    float32 and bf16, all 128 channels real and record 91's 96."""

    def window_order(grid: tuple[int]):  # tournament_fast._order[grid](...)
        def launch(q, query, order, length, heads, D, CHANNELS):  # noqa: N803
            query[:], order[:] = blocks.windows(q, settings)

        return launch

    for dtype in (torch.float32, torch.bfloat16):
        q, k, _, cu = packed(docs, dtype)
        for real in (128, REAL):
            q[..., real:], k[..., real:] = 0, 0
            length, longest = q.shape[0], max(docs)
            got = winner_mask(blocks.select(q, k, cu, longest, settings), length)
            order = mock.MagicMock()
            order.__getitem__.side_effect = window_order
            with mock.patch.object(tournament_fast, "_order", order):
                want = tournament_fast.select(q, k, cu, longest, settings)
            assert torch.equal(got, winner_mask(want, length)), (settings, dtype, real)


if __name__ == "__main__":
    for module in (tournament_fast, blocks):
        module._rounds.early_config_prune = first_config
    windows_follow_the_rule()
    print("blocks follow the rule (zero channels; re-reads at weight 0 when needed)")
    picks_tournament_fast_keys(DEFAULT, DOCS)
    picks_tournament_fast_keys(RUN1, (5, 1500, 4500))
    picks_tournament_fast_keys(RUN2, (5, 1500, 4500))
    print("winners = tournament_fast's kernels on the same channel order")
    names = ("attention", "picked", "select")
    with mock.patch.multiple(fast_tests, **{n: getattr(blocks, n) for n in names}):
        for settings in (DEFAULT, RUN2):
            fast_tests.bits_are_the_winners(settings)
            fast_tests.attention_is_the_softmax_over_the_winners(settings)
            fast_tests.chunks_are_the_whole_row(settings)
        print("bit words = winners; attention = softmax over them; chunks = whole row")
        fast_tests.compiles_in_a_no_grad_model()
    print("compiles in a no-grad model")
