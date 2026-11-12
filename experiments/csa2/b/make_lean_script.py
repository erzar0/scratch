"""Writes OUT next to this file: BASE (a B script with csa2_window's 1 shared K=V window
head, e.g. train_tournament_kv_win1_pick4_fast.py) without work whose results are never
read, same math: the QK projection computes only q (the first heads x qk_dim rows of
qk_w; the parameter keeps its shape, so the optimizer sees the same matrix with zero k
rows as before) normed and rotated by csa2_lean.q_norm_rope_pad, and attention reads
the one window head through flex's GQA mapping (csa2_flex_gqa) instead of 6 copies.

    python make_lean_script.py BASE OUT
"""

# ruff: noqa: E501  (patch strings must match the generated script's lines)

import sys
from pathlib import Path

HERE = Path(__file__).parent
base, out = sys.argv[1:]
src = (HERE / base).read_text()
REPLACE = [
    (
        "from nanogpt import csa2_window  #",
        "from nanogpt import csa2_flex_gqa, csa2_lean, csa2_window  #",
    ),
    (
        "for _module in (csa2_flex, ",
        "for _module in (csa2_flex, csa2_flex_gqa, csa2_lean, ",
    ),
    (
        """            # shared K=V window: the heads' own v is never read, so not computed
            qk = F.linear(x, sa_lambdas[0] * qk_w.type_as(x))
            qk = qk.view(B, T, 2 * self.num_heads, self.qk_dim)
            v = None
            q, k = QKNormRoPEPad(
                qk[0], yarn.factor1[:T], yarn.factor2[:T], self.num_heads,
                self.attn_qk_dim, self.paired, key_offset,
            )
            q, k = q[None], k[None]
""",
        """            # shared K=V window: the heads' own k and v are never read, so not
            # computed (csa2_lean: q only, same norm + RoPE as QKNormRoPEPad)
            w_q = qk_w[: self.num_heads * self.qk_dim]
            q = F.linear(x[0], sa_lambdas[0] * w_q.type_as(x))
            q = csa2_lean.q_norm_rope_pad(
                q.view(T, self.num_heads, self.qk_dim),
                yarn.factor1[:T], yarn.factor2[:T], self.attn_qk_dim,
            )[None]
            k = v = None
""",
    ),
    (
        """        k_win, v_win = csa2_window.shared_window(
            x[0], tuple(w.type_as(x) for w in w_win), self.num_heads,
""",
        """        k_win, v_win = csa2_window.shared_window(  # 1 head, read by all (GQA)
            x[0], tuple(w.type_as(x) for w in w_win), 1,
""",
    ),
    (
        "        y = csa2_flex.attention(\n            q[0], k_win, v_win,",
        "        y = csa2_flex_gqa.attention(\n            q[0], k_win, v_win,",
    ),
]
for old, new in REPLACE:
    assert src.count(old) == 1, f"not found once: {old[:60]!r}"
    src = src.replace(old, new)
(HERE / out).write_text(src)
print("wrote", HERE / out)
