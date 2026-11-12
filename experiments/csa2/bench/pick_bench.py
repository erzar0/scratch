"""ms per csa2_select call at B's training shapes (one row of a micro-batch: 2048,
4096, 6144 tokens per stage; 6 heads x 128; m 4; top 64; documents capped at 2048,
lengths uniform 64-2048, seed 0). Per step: 8 micro-batches x 5 pick layers = 40
calls; compare ms_per_step with the run's step_avg.
python pick_bench.py"""

import torch
import triton

from nanogpt import csa2_select

M, TOP, HEADS, DIM = 4, 64, 6, 128
g = torch.Generator().manual_seed(0)
print("tokens,rule,ms_per_call,ms_per_step")
for tokens in (2048, 4096, 6144):
    lengths, total = [], 0
    while total < tokens:
        n = min(int(torch.randint(64, 2049, (1,), generator=g)), tokens - total)
        lengths.append(n)
        total += n
    first = torch.tensor([0, *lengths[:-1]]).cumsum(0)
    starts = first.repeat_interleave(torch.tensor(lengths)).cuda()
    q = torch.randn(tokens, HEADS, DIM, generator=g).cuda().bfloat16()
    kv = torch.randn(tokens // M, DIM, generator=g).cuda().bfloat16()
    for rule in ("exact", "tournament"):
        call = getattr(csa2_select, rule)
        ms = triton.testing.do_bench(
            lambda call=call, q=q, kv=kv, s=starts: call(q, kv, s, M, TOP), rep=200
        )
        print(f"{tokens},{rule},{ms:.3f},{40 * ms:.1f}", flush=True)
