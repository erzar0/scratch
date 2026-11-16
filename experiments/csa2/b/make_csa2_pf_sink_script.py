"""Writes train_csa2_pf_sink.py next to this file: train_csa2_pf_smooth.py
(make_csa2_pf_script.py smooth) with scratch src/nanogpt/csa2_pf_sink.py's layers:
the parameter-free block plus a constant sink logit ln 128 per head.
    python make_csa2_pf_sink_script.py
    b/run_b.sh tournament train_csa2_pf_sink.py
"""

# ruff: noqa: E501  (patch strings must match the source lines)
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "train_csa2_pf_smooth.py").read_text()
edits = [
    (
        """from nanogpt import csa2_flex, csa2_ours, csa2_pf, csa2_ref, csa2_select_fast
for _module in (csa2_ref, csa2_flex, csa2_ours, csa2_pf, csa2_select_fast):""",
        """from nanogpt import csa2_flex, csa2_ours, csa2_pf, csa2_pf_sink, csa2_ref, csa2_select_fast
for _module in (csa2_ref, csa2_flex, csa2_ours, csa2_pf, csa2_pf_sink, csa2_select_fast):""",
    ),
    (
        """print(f"parameter-free CSA2 layers (pooled kv, no sink), our picks""",
        """print(f"parameter-free CSA2 layers (pooled kv, fixed sink ln 128), our picks""",
    ),
    (
        "csa2_pf.Csa2PfAttention(i, config)",
        "csa2_pf_sink.Csa2PfSinkAttention(i, config)",
    ),
]
for old, new in edits:
    assert src.count(old) == 1, f"not found once: {old[:80]!r}"
    src = src.replace(old, new)
out = HERE / "train_csa2_pf_sink.py"
out.write_text(src)
print("wrote", out)
