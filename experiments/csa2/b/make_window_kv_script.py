"""Writes OUT next to this file: BASE (a B script: train_tournament_kv.py or its route
version) with CSA2's window (src/nanogpt/csa2_window.py): per token G shared K=V
vectors (G = 1: CSA2's one head; 2: heads 0-2 / 3-5; 6: one per head) replace the
heads' own window K and V. Their weights are G more kv_bank slices per layer (same
normuon setup as the entry compressor). The heads' k and v go unused: v is no longer
computed (no XSA, value embeddings or DC attention in B); k stays in the fused QK
norm (ponytail: its 576 columns are wasted, split the q projection if time matters).
Record 91's key offset on some layers does not apply to the shared window.

    python make_window_kv_script.py G BASE OUT && python make_fast_script.py OUT
"""

# ruff: noqa: E501  (patch strings must match the generated script's lines)
import sys
from pathlib import Path

HERE = Path(__file__).parent
groups, base, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
assert 6 % groups == 0
src = (HERE / base).read_text()
patches = (
    (
        "CSA2_M, CSA2_TOP = 4, 64  # tokens per entry, entries per query (CSA2 nanogpt sizes)\n",
        "CSA2_M, CSA2_TOP = 4, 64  # tokens per entry, entries per query (CSA2 nanogpt sizes)\n"
        f"from nanogpt import csa2_window  # CSA2's window: {groups} shared K=V vector(s) per token\n",
    ),
    (
        "for _module in (csa2_flex, csa2_select, ",
        "for _module in (csa2_flex, csa2_select, csa2_window, ",
    ),
    (
        """            qkv_weight = sa_lambdas[0] * torch.cat((qk_w, v_w)).type_as(x)
            qkv = F.linear(x, qkv_weight)
            qk, v = qkv.split((self.qk_proj_dim, self.hdim), dim=-1)
            qk = qk.view(B, T, 2 * self.num_heads, self.qk_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)
""",
        """            # shared K=V window: the heads' own v is never read, so not computed
            qk = F.linear(x, sa_lambdas[0] * qk_w.type_as(x))
            qk = qk.view(B, T, 2 * self.num_heads, self.qk_dim)
            v = None
""",
    ),
    ("(w_kv, w_gate), state", "(w_kv, w_gate), state, w_win"),
    (
        '        entries = state["entries"]\n',
        '        entries = state["entries"]\n'
        "        # CSA2's window (csa2_window): shared K=V vectors replace the heads' k, v\n"
        "        k_win, v_win = csa2_window.shared_window(\n"
        "            x[0], tuple(w.type_as(x) for w in w_win), self.num_heads,\n"
        "            yarn.factor1, yarn.factor2,\n"
        "        )\n",
    ),
    ("k[0], v[0], keys, entries", "k_win, v_win, keys, entries"),
    (
        "        num_kv = 2 * -(-num_attn_layers // 4)  # CSA2 compressor W_kv, W_gate per group\n",
        "        num_kv = 2 * -(-num_attn_layers // 4)  # CSA2 compressor W_kv, W_gate per group\n"
        f"        num_kv += {groups} * num_attn_layers  # + shared K=V window weights per layer\n",
    ),
    (
        "        kv_all = self.kv_bank.unbind(0)\n",
        "        kv_all = self.kv_bank.unbind(0)\n"
        "        first_win = 2 * -(-self._num_attn_layers // 4)\n"
        f"        win_all = [kv_all[first_win + {groups} * i : first_win + {groups} * (i + 1)]\n"
        "                   for i in range(self._num_attn_layers)]\n",
    ),
    (
        "kv_all[2 * group : 2 * group + 2], csa2_state",
        "kv_all[2 * group : 2 * group + 2], csa2_state, win_all[attn_idx]",
    ),
)
for old, new in patches:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
(HERE / out).write_text(src)
print("wrote", out)
