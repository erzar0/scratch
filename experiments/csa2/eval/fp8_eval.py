"""An eval script (packed_val.py, rule_eval.py, ref_rule_eval.py) with every CSA2 cache
stored as plain FP8 E4M3 (no scales): csa2_ref.FP4 on and csa2_ref.fake_quant swapped
for an E4M3 round trip, the format of bench/decode_ref_picks.py's "FP8 caches" rows.
csa2_ref, csa2_ours and csa2_pf read both at call time. From experiments/csa2:
    python eval/fp8_eval.py eval/rule_eval.py SCRIPT RUN ROWS OUT.csv RULE ...
"""

import runpy
import sys

import torch
from torch import Tensor

from nanogpt import csa2_ref


def e4m3(x: Tensor, fmt: str, to: torch.dtype = torch.float8_e4m3fn) -> Tensor:
    """x rounded through float8_e4m3fn (every cache format). No globals: compiled
    under runpy, whose __main__ swap breaks torch.compile's guards on this module."""
    return x.to(to).to(x.dtype)


csa2_ref.FP4, csa2_ref.fake_quant = True, e4m3
sys.argv[0] = sys.argv.pop(1)  # the eval script, then its own arguments
runpy.run_path(sys.argv[0], run_name="__main__")
