"""Median ms per step in each training stage of a record 91 log (the stage's first
5 steps skipped). Stages split the schedule in thirds, then the extension.

    python stage_times.py LOG [LOG ...]
"""

import itertools
import re
import statistics
import sys

for path in sys.argv[1:]:
    text = open(path).read()
    rows = re.findall(r"^step:(\d+)/(\d+) train_time:(\d+)ms", text, re.M)
    time = {int(s): int(t) for s, _, t in rows}
    total = int(rows[-1][1])
    iters = round(total / 1.0333) if total > 1000 else total - 10  # ext 160 or 10
    edges = (0, iters // 3, 2 * iters // 3, iters)
    out = []
    for a, b in itertools.pairwise(edges):
        steps = [time[s] - time[s - 1] for s in range(a + 6, b) if s - 1 in time]
        out.append(f"{statistics.median(steps):.0f}")
    print(path, "ms/step by stage:", " / ".join(out))
