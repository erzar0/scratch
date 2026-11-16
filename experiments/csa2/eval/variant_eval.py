"""An eval script (rule_eval.py) with decode variants, each an env knob (1 = on),
applied when packed_val.main starts (after rule_eval.py set csa2_ours.RULE):
WINDOW_PICKS  csa2_ours.RULE picks only among entries that end before each query's
              window (as csa2_pf.window_picks): no picks inside the window.
SHARE9        layer 9 keeps layer 5's picks (no pick pass at layer 9), as
              ref_rule_eval.py's SHARE, for b/train_csa2_ours*.py and _pf*.py.
MEANS_QUANT   the block rule's "mean" summary is the block's mean entry rounded as
              the entry cache (csa2_ref.fake_quant "main"): FP4 under CSA2_REF_FP4=1,
              E4M3 under fp8_eval.py; needs one of them.
HEAD_CLIP     lm_head weights clipped to the training head's FP8 range (448 w_s =
              1.6): training rounds them to E4M3 with saturation (triton_kernels
              FusedSoftcappedCrossEntropy), the eval head uses them raw, and 0.3-1.3%
              of them drift past 1.6 (up to 6) in every record 91 checkpoint.
The compiled pick wrapper reads no module globals: runpy swaps __main__, which breaks
torch.compile's guards on this module (eval/fp8_eval.py). From experiments/csa2:
    SHARE9=1 python eval/variant_eval.py eval/rule_eval.py SCRIPT RUN ROWS OUT.csv \
        tournament
FP8 caches: python eval/fp8_eval.py eval/variant_eval.py eval/rule_eval.py ...
"""

import os
import runpy
import sys
from collections.abc import Callable

import packed_val
from nanogpt_old import model as nm

KNOBS = {k: bool(int(os.environ.get(k, 0)))
         for k in ("WINDOW_PICKS", "SHARE9", "MEANS_QUANT", "HEAD_CLIP")}  # fmt: skip
Rule = Callable[..., tuple]


def windowed(rule: Rule, w: int) -> Rule:
    """rule's picks among the entries that end before each query's w-token window."""
    import torch  # a closure variable of pick, not a global

    def pick(q, entries, begin, m, top_k):  # type: ignore[no-untyped-def]
        entry = torch.zeros(min(w, q.shape[0]), top_k, dtype=torch.long,
                            device=q.device)  # fmt: skip
        valid = entry.bool()
        if q.shape[0] <= w:
            return entry, valid
        later, ok = rule(q[w:], entries, begin[w:], m, top_k)
        return torch.cat([entry, later]), torch.cat([valid, ok])

    return pick


def main(
    script: str, run: str, rows: str, out: str,
    base: Callable[..., None] = packed_val.main,
) -> None:  # fmt: skip
    """packed_val.main with the knobs on; fails if a knob never fired."""
    from nanogpt import csa2_flex, csa2_ours, csa2_ref, csa2_select_block

    fired = {k: 0 for k, on in KNOBS.items() if on}
    if KNOBS["WINDOW_PICKS"]:
        tg = sys.modules["__main__"].tg  # rule_eval.py's training script
        assert tg.csa2_ours is csa2_ours
        csa2_ours.RULE = windowed(csa2_ours.RULE, csa2_flex.WINDOW)
        fired["WINDOW_PICKS"] = 1
    if KNOBS["SHARE9"]:
        load = nm.load

        def shared(path):  # type: ignore[no-untyped-def]
            model, config = load(path)
            for layer in model.modules():
                if isinstance(layer, csa2_ours.Csa2OursAttention) and layer.layer == 9:
                    assert layer.picks
                    layer.picks = False
                    fired["SHARE9"] += 1
            return model, config

        nm.load = shared
    if KNOBS["MEANS_QUANT"]:
        assert csa2_ref.FP4, "MEANS_QUANT needs CSA2_REF_FP4=1 or fp8_eval.py"
        scores = csa2_select_block.block_scores

        def quant_means(qs, blocks, summary):  # type: ignore[no-untyped-def]
            if summary != "mean":
                return scores(qs, blocks, summary)
            fired["MEANS_QUANT"] += 1  # inside the custom op: never compiled
            return qs @ csa2_ref.fake_quant(blocks.mean(1), "main").T

        csa2_select_block.block_scores = quant_means
    if KNOBS["HEAD_CLIP"]:
        unclipped = nm.load

        def clipped(path):  # type: ignore[no-untyped-def]
            model, config = unclipped(path)
            cap = 448 * model.lm_head.w_s
            model.lm_head.weight.data.clamp_(-cap, cap)
            fired["HEAD_CLIP"] += 1
            return model, config

        nm.load = clipped
    print("variant:", fired, flush=True)
    base(script, run, rows, out)
    print("variant fired:", fired, flush=True)
    assert all(fired.values()), fired
    assert fired.get("SHARE9", 1) == 1, fired
    assert fired.get("HEAD_CLIP", 1) == 1, fired


packed_val.main = main
sys.argv[0] = sys.argv.pop(1)  # the eval script, then its own arguments
runpy.run_path(sys.argv[0], run_name="__main__")
