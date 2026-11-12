"""Writes OUT next to this file: BASE (a B script picking in each group's first layer,
e.g. train_tournament_kv_win1_pick4_fast.py) picking only in LAYERS (comma-separated
attention layers, e.g. 0 or 0,4); every other layer reuses the latest picks, across
groups too (IndexCache, arXiv:2603.12201: layers reuse an earlier layer's top-k).
Entry j holds tokens 4j..4j+3 in every group, so a pick list indexes any group's
entries. Same parameters as BASE: an eval-only variant for its checkpoints.

    python make_pick_share_script.py LAYERS BASE OUT
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
layers, base, out = sys.argv[1:]
picks = tuple(int(a) for a in layers.split(","))
assert picks[0] == 0, "the first attention layer must pick"
src = (HERE / base).read_text()
old = "        if layer % 4 == 0:\n            begin = csa2_flex.starts(seqlens, T)\n"
new = old.replace("layer % 4 == 0", f"layer in {picks}")
assert src.count(old) == 1, "pick condition not found once"
(HERE / out).write_text(src.replace(old, new))
print("wrote", HERE / out)
