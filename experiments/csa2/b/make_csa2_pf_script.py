"""Writes train_csa2_pf[_VARIANT].py next to this file: train_csa2_ours[_VARIANT].py
(make_csa2_ours_script.py) with scratch src/nanogpt/csa2_pf.py's layers: the same
layout and summed-query picks (env CSA2_SELECT) with no parameters beyond q, wkv +
kv_norm and o: entries are mean-pooled kv, no sink, picks avoid the window.
    (none)  csa2_ref.SMALL: ratio 2 at layers 1-4, ratio 1 at layers 5-9
    smooth  the same with the stage-switch fix and train-loss logging
    python make_csa2_ours_script.py smooth && python make_csa2_pf_script.py smooth
    b/run_b.sh tournament train_csa2_pf_smooth.py
"""

# ruff: noqa: E501  (patch strings must match the source lines)
import sys
from pathlib import Path

HERE = Path(__file__).parent
variant = sys.argv[1] if len(sys.argv) > 1 else ""
suffix = "_" * bool(variant) + variant
src = (HERE / f"train_csa2_ours{suffix}.py").read_text()
edits = [
    (
        """from nanogpt import csa2_flex, csa2_ours, csa2_ref, csa2_select_fast
for _module in (csa2_ref, csa2_flex, csa2_ours, csa2_select_fast):""",
        """from nanogpt import csa2_flex, csa2_ours, csa2_pf, csa2_ref, csa2_select_fast
for _module in (csa2_ref, csa2_flex, csa2_ours, csa2_pf, csa2_select_fast):""",
    ),
    (
        """print(f"CSA2 reference layers, our picks""",
        """print(f"parameter-free CSA2 layers (pooled kv, no sink), our picks""",
    ),
    (
        "csa2_ours.Csa2OursAttention(i, config)",
        "csa2_pf.Csa2PfAttention(i, config)",
    ),
]
for old, new in edits:
    assert src.count(old) == 1, f"not found once: {old[:80]!r}"
    src = src.replace(old, new)
out = HERE / f"train_csa2_pf{suffix}.py"
out.write_text(src)
print("wrote", out)
