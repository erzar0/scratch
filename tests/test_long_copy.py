"""Copy test at long T (src/experiments/long_copy.py): chunked = full-logit losses."""

import pytest
import torch
import torch.nn.functional as F

from deepseek import csa2
from deepseek.double_indexing import DoubleIndexing
from experiments import long_copy
from experiments.selector_forward import selecting
from tests.test_csa2 import MODES, TINY


def test_chunked_losses_match_full_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: chunks of 8 queries and positions on a 48-token window
    monkeypatch.setattr(long_copy, "CHUNK", 8)
    torch.manual_seed(0)
    model = csa2.build(MODES, **TINY).eval()
    piece = torch.randint(0, TINY["vocab_size"], (1, 24))
    seq = torch.cat([piece, piece, piece[:, :1]], 1)

    for route in (None, DoubleIndexing(2, 1, "bound")):
        # Act
        hits = long_copy.CopyHits(long_copy.chunked(model, route), 24)
        text, copy = long_copy.window_losses(model, piece, hits)

        # Assert
        with torch.no_grad(), selecting(model, route):
            logits = model(input_ids=seq[:, :-1], use_cache=False).logits[0]
        nll = F.cross_entropy(logits, seq[0, 1:], reduction="none")
        torch.testing.assert_close(text, float(nll[:23].mean()))
        torch.testing.assert_close(copy, float(nll[24:].mean()))
        assert sorted(hits.counts) == [0, 2]  # the Full and Reindex layers
        assert all(0 <= h <= n == 24 for h, n in hits.counts.values())
