"""Writes train_tournament_kv_mqa.py next to this file: train_tournament_kv.py (run
make_kv_script.py first) with a one-head window per layer, so a decode cache keeps
one K and one V vector per window token instead of record 91's six.
Each layer's window key is the mean of its six heads' keys after QK norm and RoPE,
normed again over the qk_dim real channels as the QK norm does (RoPE keeps norms);
its window value is the mean of the six values. Every head's projection still gets
gradients."""

# ruff: noqa: E501  (patch strings must match the generated script's lines)
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "train_tournament_kv.py").read_text()
old = """        # Tournament attention instead of FA2's window (bm_size goes unused): each query
"""
new = (
    """        # One K and one V head per layer (MQA window): head means; the key normed again
        # on its qk_dim real channels with the QK norm's eps (RoPE keeps norms).
        k_mean = k[..., : self.qk_dim].mean(2, keepdim=True)
        k_mean = F.rms_norm(k_mean, (self.qk_dim,), eps=1.1920928955078125e-7)
        k = F.pad(k_mean, (0, k.size(-1) - self.qk_dim)).expand_as(k)
        v = v.mean(2, keepdim=True).expand_as(v)

"""
    + old
)
assert src.count(old) == 1
(HERE / "train_tournament_kv_mqa.py").write_text(src.replace(old, new))
print("wrote train_tournament_kv_mqa.py")
