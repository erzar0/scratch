"""Writes train_tournament_kv_w<W>.py next to this file: train_tournament_kv.py (run
make_kv_script.py first) with a W-token local window (csa2_flex.WINDOW) in place of
CSA2's 128, record 91's six K and V heads kept. A decode cache keeps W window tokens
per layer, so W = 32 brings the 6-head window to 0.82 MiB (CSA2's: 0.31).

    python make_window_script.py 32
"""

# ruff: noqa: E501  (patch string must match the generated script's line)
import sys
from pathlib import Path

window = int(sys.argv[1])
assert 0 < window <= 128  # csa2_flex reads each tile's own row tile and the one before
HERE = Path(__file__).parent
src = (HERE / "train_tournament_kv.py").read_text()
old = "CSA2_M, CSA2_TOP = 4, 64  # tokens per entry, entries per query (CSA2 nanogpt sizes)\n"
assert src.count(old) == 1, old
src = src.replace(old, f"{old}csa2_flex.WINDOW = {window}  # local window, tokens\n")
out = HERE / f"train_tournament_kv_w{window}.py"
out.write_text(src)
print("wrote", out.name)
