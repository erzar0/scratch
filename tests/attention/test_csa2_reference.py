"""csa2_reference: with no sink and no RoPE turn, the output is exact attention over
the picks and the window, K = V (the suite's fast_pytorch.attend)."""

import torch

from attention.csa2_reference import Csa2Reference
from attention.fast_pytorch import KV, attend


def test_attention_is_exact_over_picks_and_window() -> None:
    torch.manual_seed(0)
    csa2 = Csa2Reference(keep=32, window=16)
    q, k = torch.randn(3, 2, 128), torch.randn(3, 200, 128)
    kv = KV(k, torch.full_like(k, float("nan")))  # values unread: K = V
    state = csa2.weights(q, 200)
    state["sinks"].fill_(float("-inf"))
    state["cos"].fill_(1.0)
    state["sin"].fill_(0.0)
    kv.memo["csa2_reference"] = state
    out, picks = csa2(q, kv)
    window = torch.arange(184, 200).expand(3, -1)
    torch.testing.assert_close(out, attend(q, k, k, torch.cat([picks, window], 1)))
