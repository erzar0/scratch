"""Writes train_tournament_kv.py next to this file: r91's train_tournament.py (read
only) with CSA2's KV layout (scratch src/nanogpt/csa2_flex.py): per group of 4
attention layers one compressed K=V head (m = 4), picks (csa2_select, env
CSA2_SELECT) by layers 0, 2, 4, 6, 8, reused by the odd ones, window 128.
Also copies r91's triton_kernels.py and dc_triton_kernels.py, which the script reads
from its own directory."""

# ruff: noqa: E501  (patch strings must match record 91's source lines)
import shutil
from pathlib import Path

R91 = Path.home() / "repos/modded-nanogpt-r91"
HERE = Path(__file__).parent
src = (R91 / "train_tournament.py").read_text()
edits = [
    (  # imports: csa2_flex / csa2_select instead of tournament.py
        """import attention.triton.common
import nanogpt.tournament
from nanogpt.tournament import attention as tournament_attention
for _module in (nanogpt.tournament, attention.triton.common):""",
        """import attention.triton.common
import nanogpt.tournament_channel_blocks
import nanogpt.tournament_fast
from nanogpt import csa2_flex, csa2_select
CSA2_M, CSA2_TOP = 4, 64  # tokens per entry, entries per query (CSA2 nanogpt sizes)
CSA2_RULE = getattr(csa2_select, os.environ["CSA2_SELECT"])  # exact | tournament
print(f"CSA2 KV layout, {os.environ['CSA2_SELECT']} picks", flush=True)
for _module in (csa2_flex, csa2_select, nanogpt.tournament_fast,
                nanogpt.tournament_channel_blocks, attention.triton.common):""",
    ),
    (  # attention signature
        """        qkv_fp8=None, dc_w: tuple[Tensor, Tensor] | None = None,
    ):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1\"""",
        """        qkv_fp8=None, dc_w: tuple[Tensor, Tensor] | None = None, csa=None,
    ):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1\"""",
    ),
    (  # attention: window 128 + picked entries of the group's compressed KV
        """        y = tournament_attention(q[0], k[0], v[0], seqlens, yarn.attn_scale)""",
        """        # CSA2 KV layout: the group's first layer compresses its input (one K=V
        # head, m tokens per entry); even layers pick with their own q, odd reuse
        layer, (w_kv, w_gate), state = csa
        if layer % 4 == 0:
            state["entries"] = csa2_flex.compress(
                x[0], w_kv.type_as(x), w_gate.type_as(x), CSA2_M
            )
        entries = state["entries"]
        keys = csa2_flex.rotate(entries, CSA2_M, yarn.factor1, yarn.factor2)
        if layer % 2 == 0:
            begin = csa2_flex.starts(seqlens, T)
            state["picks"] = CSA2_RULE(q[0], keys, begin, CSA2_M, CSA2_TOP)
        y = csa2_flex.attention(
            q[0], k[0], v[0], keys, entries, *state["picks"], seqlens,
            yarn.attn_scale, CSA2_M,
        )""",
    ),
    (  # compressor weights: W_kv, W_gate per group of 4 attention layers
        """        self.vo_bank.reshape = (num_vo_padded, hdim, model_dim)
""",
        """        self.vo_bank.reshape = (num_vo_padded, hdim, model_dim)
        num_kv = 2 * -(-num_attn_layers // 4)  # CSA2 compressor W_kv, W_gate per group
        self.kv_bank = nn.Parameter(torch.empty(num_kv, head_dim, model_dim))
        self.kv_bank.reshape = (num_kv, head_dim, model_dim)
""",
    ),
    (
        """            self.vo_bank[num_vo_real:].zero_()
""",
        """            self.vo_bank[num_vo_real:].zero_()
            self.kv_bank.uniform_(-bound, bound)
""",
    ),
    (
        """        v_per_layer = vo_all[0::2]
""",
        """        v_per_layer = vo_all[0::2]
        kv_all = self.kv_bank.unbind(0)
        csa2_state = {}  # entries and picks handed to later layers
""",
    ),
    (
        """                attn_out = attn(attn_in_normed, attn_args, qk_w, v_w, o_w, qkv_fp8, dc_w)""",
        """                group = attn_idx // 4
                csa = (attn_idx, kv_all[2 * group : 2 * group + 2], csa2_state)
                attn_out = attn(
                    attn_in_normed, attn_args, qk_w, v_w, o_w, qkv_fp8, dc_w, csa
                )""",
    ),
    (
        """            "vo_bank":        {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
""",
        """            "vo_bank":        {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
            "kv_bank":        {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
""",
    ),
    (
        """            "qk_bank", "vo_bank", "mlp_bank",  # Large""",
        """            "kv_bank", "qk_bank", "vo_bank", "mlp_bank",  # Large""",
    ),
    (
        """model.vo_bank.data = model.vo_bank.data.bfloat16()
""",
        """model.vo_bank.data = model.vo_bank.data.bfloat16()
model.kv_bank.data = model.kv_bank.data.bfloat16()
""",
    ),
]
for old, new in edits:
    assert src.count(old) == 1, old[:80]
    src = src.replace(old, new)
assert "tournament_attention" not in src
(HERE / "train_tournament_kv.py").write_text(src)
for name in ("triton_kernels.py", "dc_triton_kernels.py"):
    shutil.copy(R91 / name, HERE / name)
print("wrote train_tournament_kv.py")
