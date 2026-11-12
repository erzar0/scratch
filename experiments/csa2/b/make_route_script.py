"""Writes train_tournament_kv_route.py next to this file: train_tournament_kv.py (run
make_kv_script.py first) with routed channels for the entries (src/nanogpt/
csa2_route.py): per attention layer a router (route_bank, Adam) keeps 2 of 6
blocks of 16 of q's 96 channels per token; picks and entry logits use only those,
gate-weighted; the window keeps every channel. Run with CSA2_SELECT=exact.

    python make_route_script.py && python make_fast_script.py train_tournament_kv_route.py
"""

# ruff: noqa: E501  (patch strings must match the generated script's lines)
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "train_tournament_kv.py").read_text()
patches = (
    (
        "CSA2_M, CSA2_TOP = 4, 64  # tokens per entry, entries per query (CSA2 nanogpt sizes)\n",
        "CSA2_M, CSA2_TOP = 4, 64  # tokens per entry, entries per query (CSA2 nanogpt sizes)\n"
        "from nanogpt import csa2_route  # routed channels for entries: 2 of 6 blocks\n",
    ),
    (
        "                nanogpt.tournament_channel_blocks, attention.triton.common):\n",
        "                nanogpt.tournament_channel_blocks, attention.triton.common, csa2_route):\n",
    ),
    (
        "        layer, (w_kv, w_gate), state = csa\n",
        "        layer, (w_kv, w_gate), state, w_route = csa\n",
    ),
    (
        """        if layer % 2 == 0:
            begin = csa2_flex.starts(seqlens, T)
            state["picks"] = CSA2_RULE(q[0], keys, begin, CSA2_M, CSA2_TOP)
        y = csa2_flex.attention(
            q[0], k[0], v[0], keys, entries, *state["picks"], seqlens,
            yarn.attn_scale, CSA2_M,
        )""",
        """        # routed channels: picks and entry logits on 2 of 6 blocks of q, gated
        q_routed = csa2_route.route(x[0], w_route.type_as(x), q[0], self.qk_dim)
        if layer % 2 == 0:
            begin = csa2_flex.starts(seqlens, T)
            state["picks"] = CSA2_RULE(q_routed, keys, begin, CSA2_M, CSA2_TOP)
        y = csa2_route.attention(
            q[0], q_routed, k[0], v[0], keys, entries, *state["picks"], seqlens,
            yarn.attn_scale, CSA2_M,
        )""",
    ),
    (
        "        self.kv_bank.reshape = (num_kv, head_dim, model_dim)\n",
        "        self.kv_bank.reshape = (num_kv, head_dim, model_dim)\n"
        "        self.route_bank = nn.Parameter(torch.empty(num_attn_layers, 6, model_dim))\n",
    ),
    (
        "            self.kv_bank.uniform_(-bound, bound)\n",
        "            self.kv_bank.uniform_(-bound, bound)\n"
        "            self.route_bank.uniform_(-bound, bound)\n",
    ),
    (
        "        kv_all = self.kv_bank.unbind(0)\n",
        "        kv_all = self.kv_bank.unbind(0)\n"
        "        route_all = self.route_bank.unbind(0)\n",
    ),
    (
        "                csa = (attn_idx, kv_all[2 * group : 2 * group + 2], csa2_state)\n",
        "                csa = (attn_idx, kv_all[2 * group : 2 * group + 2], csa2_state,\n"
        "                       route_all[attn_idx])\n",
    ),
    (
        """            "kv_bank":        {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
""",
        """            "kv_bank":        {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
            "route_bank":     {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.1,  "wd_mul": 0.0},
""",
    ),
    (
        """            "post_lambdas", "resid_lambdas",  # Small, fast""",
        """            "post_lambdas", "resid_lambdas", "route_bank",  # Small, fast""",
    ),
    (
        "model.kv_bank.data = model.kv_bank.data.bfloat16()\n",
        "model.kv_bank.data = model.kv_bank.data.bfloat16()\n"
        "model.route_bank.data = model.route_bank.data.bfloat16()\n",
    ),
)
for old, new in patches:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
(HERE / "train_tournament_kv_route.py").write_text(src)
print("wrote train_tournament_kv_route.py")
