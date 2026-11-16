"""packed_val.py with a flash-attention model's attention kernel replaced by
csa2_pf.drop_in: the parameter-free CSA2 block on each layer's own q, k, v (window
128 + top_k mean-pooled entries of m keys by summed query, within the layer's own
window; the first SINKS tokens of each document kept exact). For record-91-style
scripts (flash_attn_varlen_func). Run eager: the block loops over query chunks.

python pf_swap_eval.py SCRIPT RUN ROWS OUT.csv M TOP_K [SINKS]
"""

import sys
import types

import packed_val
from nanogpt_old import model as nm

script, run, rows, out, m, top_k = sys.argv[1:7]
sinks = int(sys.argv[7]) if len(sys.argv) > 7 else 0
nm.SCRIPT = script
tg = nm.module()
from nanogpt import csa2_pf  # noqa: E402  (script's path)


def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, window_size, **kw):
    """flash_attn_varlen_func's causal call as the scripts make it."""
    assert kw["causal"] and cu_seqlens_q is cu_seqlens_k
    span = window_size[0] if window_size[0] >= 0 else q.shape[0]
    return csa2_pf.drop_in(
        q, k, v, cu_seqlens_q, softmax_scale, span, int(m), int(top_k), sinks=sinks
    )


tg.flash_attn_interface = types.SimpleNamespace(flash_attn_varlen_func=varlen)
print(f"parameter-free CSA2 drop-in: m {m}, top {top_k}, window 128, sinks {sinks}")
packed_val.main(script, run, rows, out)
