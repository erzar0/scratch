"""csa2_pf with a fixed attention sink: one constant logit SINK = ln 128 per head, no
parameter. csa2_ref.attend scales each head's output by sigmoid(lse - SINK), as if a
window's worth (128) of keys with logit 0 and value 0 were present, so a head can
attend to nothing. csa2_pf (K = V, RMS-normed values, no sink) cannot; the learned
sinks of the reference-layout runs take 2-62% of attention per layer, most at early
positions (experiments/csa2/eval/sink_share.py), where csa2_pf loses to them.
"""

import math

from nanogpt import csa2_pf


class Csa2PfSinkAttention(csa2_pf.Csa2PfAttention):
    """csa2_pf.Csa2PfAttention with sink logit ln 128 in every head."""

    SINK = math.log(128)
