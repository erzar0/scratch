#!/bin/bash
# Rebuilds csa2_src/ for the CSA2 model with our picks (train_csa2.py via
# r91's run_csa2.sh, SCRATCH_SRC=<this dir>/csa2_src/src): the CSA2 sources of commit
# 0fc71ef, the repo's current nanogpt modules on the package path, and glue that
# swaps the learned indexer for csa2_select picks (env CSA2_SELECT=exact|tournament;
# CSA2_FAST=1: csa2_select_fast's kernel);
# eval probes CSA2_PICK_LIMIT, CSA2_FULL_LAYER.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
rm -rf "$D/csa2_src" && mkdir -p "$D/csa2_src"
git -C "$HOME/repos/scratch" archive 0fc71ef src | tar -x -C "$D/csa2_src"
C=$D/csa2_src/src/nanogpt
cat >> "$C/__init__.py" <<'EOF'

# scratch shim: also the repo's current nanogpt modules (csa2_select, tournament_*)
# and, last on sys.path, its other packages (attention: tournament_fast's kernels)
import os as _os
import sys as _sys

__path__ = [_os.path.dirname(__file__), _os.path.expanduser("~/repos/scratch/src/nanogpt")]
_sys.path.append(_os.path.expanduser("~/repos/scratch/src"))
EOF
cat >> "$C/csa2_varlen.py" <<'EOF'


# ---- scratch glue (not in the repo): CSA2 with parameter-free picks --------------------
import os  # noqa: E402

from nanogpt import csa2_select, csa2_select_fast  # noqa: E402

RULE = os.environ.get("CSA2_SELECT", "")


class SelectCsa2Attention(VarlenCsa2Attention):
    """VarlenCsa2Attention whose Full and Reindex layers pick by (sum_h q_h) . main KV
    with csa2_select's rule CSA2_SELECT (exact | tournament): no indexer weights, no
    indexer loss, no indexer K."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.picks_here = self.indexer is not None
        self.indexer = None
        if self.compressor is not None:
            self.compressor.index_k_proj = nn.Identity()  # indexer K unused
        fast = os.environ.get("CSA2_FAST")  # csa2_select_fast: same picks, faster
        self.pick = getattr(csa2_select_fast if fast else csa2_select, RULE)
        self.compress_rate = (config.compress_rates or {})[csa2.csa.CSA]
        self.top_k = config.index_topk

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        csa2_shared: csa2.SharedState,
    ) -> torch.Tensor:
        """VarlenCsa2Attention.forward with csa2_select's picks."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        position_ids = torch.arange(input_shape[1], device=hidden_states.device)[None]
        starts = document_starts(cu_seqlens, input_shape[1])
        cos, sin = self.rotary_emb(
            hidden_states, position_ids, layer_type=self.rope_layer_type
        )

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = apply_rotary_pos_emb(self.q_b_norm(q), cos, sin)
        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape)
        window_kv = apply_rotary_pos_emb(kv.transpose(1, 2), cos, sin)[:, 0]

        if self.compressor is not None:
            csa2_shared.global_kv[self.layer_idx] = self.compressor(hidden_states)
        main_kv, _ = csa2_shared.global_kv[self.kv_source]
        if self.picks_here:
            entry, valid = self.pick(
                q[0].transpose(0, 1), main_kv[0], starts, self.compress_rate, self.top_k
            )
            csa2_shared.picks[self.layer_idx] = csa2.Picks(entry[None], valid[None], None)
        picks = csa2_shared.picks[self.picks_source]

        attn_output, _ = self.attend_documents(q, window_kv, main_kv, picks, starts)
        attn_output = apply_rotary_pos_emb(
            attn_output.transpose(1, 2), cos, -sin
        ).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        return self.o_b_proj(self.o_a_proj(grouped).flatten(2))


if RULE:
    VarlenCsa2Attention = SelectCsa2Attention  # type: ignore[misc]  # train_csa2.py's name
    print(f"csa2_varlen: {RULE} picks, no indexer", flush=True)
EOF
python3 - "$C/csa2_varlen.py" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
a = 'RULE = os.environ.get("CSA2_SELECT", "")\n'
b = a + '# eval probe only: picks limited to entries within LIMIT tokens of the query\nLIMIT = int(os.environ.get("CSA2_PICK_LIMIT", 0))\n'
c = """        if self.picks_here:
            entry, valid = self.pick(
                q[0].transpose(0, 1), main_kv[0], starts, self.compress_rate, self.top_k
            )"""
d = """        if self.picks_here:
            near = starts
            if LIMIT:
                t = torch.arange(len(starts), device=starts.device)
                near = torch.maximum(starts, t - LIMIT + 1)
            entry, valid = self.pick(
                q[0].transpose(0, 1), main_kv[0], near, self.compress_rate, self.top_k
            )"""
for x, y in ((a, b), (c, d)):
    assert s.count(x) == 1, x
    s = s.replace(x, y)
open(p, "w").write(s)
EOF
python3 - "$C/csa2_varlen.py" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
a = 'LIMIT = int(os.environ.get("CSA2_PICK_LIMIT", 0))\n'
b = a + 'FULL = int(os.environ.get("CSA2_FULL_LAYER", -1))  # probe: this layer unlimited\n'
c = "            if LIMIT:\n"
d = "            if LIMIT and self.layer_idx != FULL:\n"
for x, y in ((a, b), (c, d)):
    assert s.count(x) == 1, x
    s = s.replace(x, y)
open(p, "w").write(s)
EOF
echo "built $D/csa2_src"
