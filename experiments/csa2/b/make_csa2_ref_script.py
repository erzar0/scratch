"""Writes train_csa2_ref.py next to this file: r91's train_csa2.py (read only; the old
CSA2 baseline's script) with scratch src/nanogpt/csa2_ref.py's layers (DeepSeek
V4.1-Flash-1017 model.py's attention, its layout scaled to 10 layers) in the 10
attention layers instead of the old port of the V4.1 report. Same stripping, data,
schedule and optimizer split (NorMuon for matrices, o projection per group; Adam
for norms, sinks and the indexer's head weights). csa2_ref.py and csa2_flex.py are
appended to the logged code. Run with run_b.sh (its CSA2_SELECT is unused):
    b/run_b.sh exact train_csa2_ref.py
"""

# ruff: noqa: E501  (patch strings must match record 91's source lines)
from pathlib import Path

R91 = Path.home() / "repos/modded-nanogpt-r91"
HERE = Path(__file__).parent
src = (R91 / "train_csa2.py").read_text()
edits = [
    (
        """# CSA2 attention from the scratch repo: deepseek/csa2.py, nanogpt/csa2_varlen.py
sys.path.insert(0, os.environ.get("SCRATCH_SRC", os.path.expanduser("~/repos/scratch/src")))
from deepseek import csa2
from nanogpt.csa2_varlen import VarlenCsa2Attention
from nanogpt.csa2_library import LibraryCsa2Attention
from nanogpt.probe import SIZES as CSA2_SIZES
""",
        """# CSA2 attention as DeepSeek V4.1-Flash-1017's model.py: scratch nanogpt/csa2_ref.py
sys.path.insert(0, os.path.expanduser("~/repos/scratch/src"))
from nanogpt import csa2_flex, csa2_ref
for _module in (csa2_ref, csa2_flex):
    with open(_module.__file__) as f:
        code += f"\\n\\n{'-'*40}\\n# scratch {_module.__name__}\\n{'-'*40}\\n\\n" + f.read()
print(f"CSA2 reference layers, FP4 caches {csa2_ref.FP4}", flush=True)
""",
    ),
    (
        """        # CSA2 in the 10 attention layers (layer 6 has none), sized as nanogpt/probe.py's
        # base: V4.1's modes, main KV shared by 4 layers, top-k recomputed every second layer
        num_attn_layers = num_layers - 1
        modes = ((csa2.FULL, csa2.REUSE, csa2.REINDEX, csa2.REUSE) * 3)[:num_attn_layers]
        config = csa2.small_config(modes, **CSA2_SIZES)
        # SWEEP_CSA2_LIBRARY=1: same layers, attending as transformers' DeepSeek-V4 does
        # (dense masked; scratch nanogpt/csa2_library.py)
        layer = LibraryCsa2Attention if _env("SWEEP_CSA2_LIBRARY", int, 0) else VarlenCsa2Attention
        self.csa2 = nn.ModuleList(layer(config, i) for i in range(num_attn_layers))
        # NorMuon orthogonalizes each projection matrix, o_a_proj per output group; norms,
        # sinks and the indexer's head weights go to Adam
        for name, param in self.csa2.named_parameters():
            if param.ndim == 2 and "weights_proj" not in name:
                groups = config.o_groups if "o_a_proj" in name else 1
                param.reshape = (groups, param.shape[0] // groups, param.shape[1])""",
        """        # CSA2 in the 10 attention layers (layer 6 has none): csa2_ref.SMALL, V4.1-Flash-1017's
        # layout (window-only layer 0, ratio 2 then 1, shared KV / picks / candidate blocks)
        num_attn_layers = num_layers - 1
        config = csa2_ref.SMALL
        self.csa2 = nn.ModuleList(csa2_ref.Csa2RefAttention(i, config) for i in range(num_attn_layers))
        # NorMuon orthogonalizes each projection matrix, wo_a per output group; norms,
        # sinks and the indexer's head weights go to Adam
        for name, param in self.csa2.named_parameters():
            if param.ndim == 2 and "weights_proj" not in name:
                groups = config.o_groups if "wo_a" in name else 1
                param.reshape = (groups, param.shape[0] // groups, param.shape[1])""",
    ),
    (
        """        csa2_shared = csa2.SharedState()  # main KV and top-k handed between CSA2 layers""",
        """        csa2_shared = {}  # entries, index keys, picks, blocks handed between CSA2 layers""",
    ),
]
for old, new in edits:
    assert src.count(old) == 1, f"not found once: {old[:80]!r}"
    src = src.replace(old, new)
(HERE / "train_csa2_ref.py").write_text(src)
print("wrote", HERE / "train_csa2_ref.py")
