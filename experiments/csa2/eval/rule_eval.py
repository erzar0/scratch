"""packed_val.py with a B checkpoint picking by another rule at eval: exact |
tournament (csa2_select_fast), csa2_select_decode.prefilter with keep and dims, or
csa2_select_block.block with size, keep (blocks) and summary, csa2_select_reuse.reuse
with size, keep and the first layer's rule (block | exact), csa2_select_stale.stale
with size, keep and every, or csa2_select_head_max.head_max.
CSA2_SELECT must name the script's own rule (it reads it at import); for
b/train_csa2_ours*.py the rule replaces csa2_ours.RULE. Env EVAL_WINDOW
shrinks csa2_flex's window at eval (a probe; tiles assume it is at most 128).

python rule_eval.py SCRIPT RUN ROWS OUT.csv RULE [KEEP DIMS | SIZE KEEP SUMMARY |
    SIZE KEEP FIRST | SIZE KEEP EVERY]
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
    csa2_select_block,
    csa2_select_decode,
    csa2_select_fast,
    csa2_select_head_max,
    csa2_select_reuse,
    csa2_select_stale,
)

if "EVAL_WINDOW" in os.environ:
    csa2_flex.WINDOW = int(os.environ["EVAL_WINDOW"])
    assert 0 < csa2_flex.WINDOW <= 128

if rule == "prefilter":
    keep, dims = map(int, sizes)
    tg.CSA2_RULE = functools.partial(csa2_select_decode.prefilter, keep=keep, dims=dims)
elif rule == "block":
    size, keep, summary = int(sizes[0]), int(sizes[1]), sizes[2]
    tg.CSA2_RULE = functools.partial(
        csa2_select_block.block, size=size, keep=keep, summary=summary
    )
elif rule == "reuse":  # layer 9 inside layer 5's kept blocks
    size, keep, first = int(sizes[0]), int(sizes[1]), sizes[2]
    tg.CSA2_RULE = functools.partial(
        csa2_select_reuse.reuse, size=size, keep=keep, first=first
    )
elif rule == "stale":  # blocks ranked every `every` queries
    size, keep, every = map(int, sizes)
    tg.CSA2_RULE = functools.partial(
        csa2_select_stale.stale, size=size, keep=keep, every=every
    )
elif rule == "head_max":
    tg.CSA2_RULE = csa2_select_head_max.head_max
else:
    tg.CSA2_RULE = getattr(csa2_select_fast, rule)
if hasattr(tg, "csa2_ours"):  # b/train_csa2_ours*.py: the reference with our picks
    tg.csa2_ours.RULE = tg.CSA2_RULE
print("picks:", rule, *sizes, "window", csa2_flex.WINDOW, flush=True)
packed_val.main(script, run, rows, out)
