"""csa2_exact_picks: the whole window plus the keep older keys of highest summed
logit, none of them inside the window."""

import torch

from attention import harness
from attention.csa2_exact_picks import Csa2ExactPicks
from tests.attention.test_harness import make_chunk


def test_window_plus_summed_query_top_of_older_keys() -> None:
    chunk = make_chunk(keys=203)
    selected = Csa2ExactPicks(keep=32, window=16)(chunk).selected  # (G, Q, N)
    window = harness.recent_keys(chunk.valid, 16).expand_as(selected)
    picks = selected & ~window
    assert bool((selected & window).sum(-1).eq(16).all())
    assert bool(picks.sum(-1).eq(32).all())
    score = chunk.logits.sum(1)
    unpicked = chunk.valid & ~window & ~picks
    lowest = score.masked_fill(~picks, float("inf")).amin(-1)
    assert bool((lowest >= score.masked_fill(~unpicked, -torch.inf).amax(-1)).all())
