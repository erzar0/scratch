"""SparQ (src/attention/sparq.py)."""

import torch

from attention.exact_topk import ExactTopK
from attention.sparq import SparQ
from tests.attention.test_harness import make_chunk


def test_sparq_with_every_dimension_picks_the_exact_top_k() -> None:
    chunk = make_chunk()
    sparq = SparQ(r=16, k=4, local=0)(chunk)
    exact = ExactTopK(4)(chunk)
    assert torch.equal(sparq.selected, exact.selected)
