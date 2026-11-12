"""packed_val.py with a B checkpoint picking by another rule at eval: exact |
tournament (csa2_select_fast), or csa2_select_decode.prefilter with keep and dims.
CSA2_SELECT must name the script's own rule (it reads it at import). Env EVAL_WINDOW
shrinks csa2_flex's window at eval (a probe; tiles assume it is at most 128).

python rule_eval.py SCRIPT RUN ROWS OUT.csv RULE [KEEP DIMS]
"""

import functools
import os
import sys

import packed_val
from nanogpt_old import model as nm

script, run, rows, out, rule, *sizes = sys.argv[1:]
nm.SCRIPT = script
tg = nm.module()
from nanogpt import (  # noqa: E402  (script's path)
    csa2_flex,
    csa2_select_decode,
    csa2_select_fast,
)

if "EVAL_WINDOW" in os.environ:
    csa2_flex.WINDOW = int(os.environ["EVAL_WINDOW"])
    assert 0 < csa2_flex.WINDOW <= 128

if rule == "prefilter":
    keep, dims = map(int, sizes)
    tg.CSA2_RULE = functools.partial(csa2_select_decode.prefilter, keep=keep, dims=dims)
else:
    tg.CSA2_RULE = getattr(csa2_select_fast, rule)
print("picks:", rule, *sizes, "window", csa2_flex.WINDOW, flush=True)
packed_val.main(script, run, rows, out)
