"""Writes <script>_fast.py next to a generated B script: picks from csa2_select_fast
(one Triton program per query) instead of csa2_select; its code goes in the log.

    python make_fast_script.py train_tournament_kv.py
"""

# ruff: noqa: E501  (patch strings must match the generated script's lines)
import sys
from pathlib import Path

script = Path(__file__).parent / sys.argv[1]
src = script.read_text()
patches = (
    (
        "from nanogpt import csa2_flex, csa2_select\n",
        "from nanogpt import block_topk_fast, csa2_flex, csa2_select, csa2_select_fast\n",
    ),
    (
        "CSA2_RULE = getattr(csa2_select, ",
        "CSA2_RULE = getattr(csa2_select_fast, ",
    ),
    (
        "for _module in (csa2_flex, csa2_select, ",
        "for _module in (csa2_flex, csa2_select, csa2_select_fast, block_topk_fast, ",
    ),
)
for old, new in patches:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
out = script.with_name(script.stem + "_fast.py")
out.write_text(src)
print("wrote", out.name)
