"""rule_eval.py on a csa2_ref / csa2_ours checkpoint with parts of the
parameter-free block (src/nanogpt/csa2_pf.py) put in after loading:
    SINK=<value>  every attention sink set to this logit (-inf: none; 0: fixed)
    POOL=1        each compressor made mean pooling of its attention's own kv: its
                  wkv and norm tied to the attention's wkv and kv_norm, gate zero
Same arguments as rule_eval.py:
python pf_parts_eval.py SCRIPT RUN ROWS OUT.csv RULE [...]
"""

import os

import torch
from nanogpt_old import model as nm

loaded = nm.load


def load(path):
    model, config = loaded(path)
    sinks = torch.stack([layer.attn_sink.float() for layer in model.csa2])
    print("learned sinks (layer x head):", sinks.cpu().numpy().round(2), flush=True)
    with torch.no_grad():
        for layer in model.csa2:
            if "SINK" in os.environ:
                layer.attn_sink.fill_(float(os.environ["SINK"]))
            if int(os.environ.get("POOL", 0)) and layer.compressor is not None:
                layer.compressor.wkv.weight.copy_(layer.wkv.weight)
                layer.compressor.norm.weight.copy_(layer.kv_norm.weight)
                if layer.ratio > 1:
                    layer.compressor.wgate.weight.zero_()
    print("sink", os.environ.get("SINK", "learned"), "pool", os.environ.get("POOL", 0))
    return model, config


nm.load = load
import rule_eval  # noqa: E402, F401  (runs packed_val with its rule)
