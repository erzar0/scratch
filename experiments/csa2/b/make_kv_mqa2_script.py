"""Writes train_tournament_kv_mqa2.py next to this file: train_tournament_kv.py (run
make_kv_script.py first) with a one-head window per layer, so a decode cache keeps
one K and one V vector per window token instead of record 91's six.
As make_kv_mqa_script.py (window key = mean of the six heads' keys after QK norm and
RoPE, normed again; window value = mean of the six values), except that gated XSA
still removes each head's own value: only attention reads the head means."""

# ruff: noqa: E501  (patch strings must match the generated script's lines)
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "train_tournament_kv.py").read_text()
patches = (
    (
        """        # Tournament attention instead of FA2's window (bm_size goes unused): each query
""",
        """        # One K and one V head per layer (MQA window): head means; the key normed again
        # on its qk_dim real channels with the QK norm's eps (RoPE keeps norms). XSA
        # below keeps each head's own v.
        k_mean = k[..., : self.qk_dim].mean(2, keepdim=True)
        k_mean = F.rms_norm(k_mean, (self.qk_dim,), eps=1.1920928955078125e-7)
        k = F.pad(k_mean, (0, k.size(-1) - self.qk_dim)).expand_as(k)
        v_mean = v.mean(2, keepdim=True).expand_as(v)

        # Tournament attention instead of FA2's window (bm_size goes unused): each query
""",
    ),
    (
        "            q[0], k[0], v[0], keys, entries, *state[\"picks\"], seqlens,\n",
        "            q[0], k[0], v_mean[0], keys, entries, *state[\"picks\"], seqlens,\n",
    ),
)
for old, new in patches:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
(HERE / "train_tournament_kv_mqa2.py").write_text(src)
print("wrote train_tournament_kv_mqa2.py")
