"""packed_val.py with non-strict checkpoint loading, so a CSA2 checkpoint runs under
our picks (its indexer weights dropped; the dropped keys are printed).
python swap_eval.py SCRIPT RUN ROWS OUT.csv"""

import sys

import packed_val
import torch

strict_load = torch.nn.Module.load_state_dict


def load(self, state, strict=True, assign=False):
    out = strict_load(self, state, strict=False, assign=assign)
    print("missing", out.missing_keys, "dropped", out.unexpected_keys, flush=True)
    return out


torch.nn.Module.load_state_dict = load
packed_val.main(*sys.argv[1:5])
