"""csa2_pf_sink = csa2_pf with sink logit ln 128: no parameters beyond csa2_pf's, and
every layer's output = csa2_ours's with its compressor tied to wkv / kv_norm (mean
pooling), its sink filled with ln 128 and csa2_pf's picks (test_csa2_pf's check).
Needs CUDA:
    uv run python -m tests.nanogpt.test_csa2_pf_sink
"""

import math

from nanogpt import csa2_pf_sink
from tests.nanogpt.test_csa2_pf import test_same_as_ours


def test_same_as_ours_with_sink() -> None:
    assert csa2_pf_sink.Csa2PfSinkAttention.SINK == math.log(128)
    test_same_as_ours(csa2_pf_sink.Csa2PfSinkAttention)


if __name__ == "__main__":
    test_same_as_ours_with_sink()
