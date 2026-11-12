"""Writes OUT next to this file: BASE (a B script with CSA2's entry cache, e.g.
train_tournament_kv_win1_pick4_share04_fast.py) with entries rounded to FP8 (e4m3,
no scale: entries are rms-normed, |x| <= sqrt(128) < 448), an eval-only variant.
MODE picks: only the rotated keys the pick rule scores; MODE all: the entries
themselves, so picks and attention both read them, as an FP8 entry cache would.

    python make_fp8_script.py MODE BASE OUT
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
mode, base, out = sys.argv[1:]
src = (HERE / base).read_text()
keys = "        keys = csa2_flex.rotate(entries, CSA2_M, yarn.factor1, yarn.factor2)\n"
fp8 = ".to(torch.float8_e4m3fn).to({0}.dtype)"
old, new = {
    "picks": (
        "CSA2_RULE(q[0], keys, begin,",
        f"CSA2_RULE(q[0], keys{fp8.format('keys')}, begin,",
    ),
    "all": (keys, f"        entries = entries{fp8.format('entries')}  # FP8\n" + keys),
}[mode]
assert src.count(old) == 1, f"not found once: {old!r}"
(HERE / out).write_text(src.replace(old, new))
print("wrote", HERE / out)
