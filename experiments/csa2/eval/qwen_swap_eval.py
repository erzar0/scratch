"""Qwen3-0.6B-Base with every attention replaced by csa2_pf.drop_in (the
parameter-free CSA2 block on each layer's own q, k, v: window 128 + top_k
mean-pooled entries of m keys by summed query over each KV head's 2 query heads),
or its own attention (M = 0). Text: the cached Paul Graham essays, joined with EOS,
cut into ROWS rows of 8192 tokens (one document per row). Per-token loss by
position as packed_val.py: OUT.csv and OUT_tokens.pt, gated by paired.py.
From the repo root:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python \
        experiments/csa2/eval/qwen_swap_eval.py ROWS OUT.csv M TOP_K [SINKS]
SINKS: each row's first SINKS tokens kept exact (drop_in's sink tokens).
"""

import csv
import sys
from itertools import pairwise

import datasets
import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

from nanogpt import csa2_pf

MODEL, ROW = "Qwen/Qwen3-0.6B-Base", 8192
EDGES = (0, 128, 512, 1024, 2048, 2560, 4096, ROW)


def drop_in(module, query, key, value, attention_mask, scaling, **kwargs):
    """An HF attention function: (1, S, H, D) output, no weights."""
    q, k, v = (x[0].transpose(0, 1) for x in (query, key, value))  # (S, H, D)
    cu = torch.tensor([0, q.shape[0]], device=q.device)
    out = csa2_pf.drop_in(q, k, v, cu, scaling, q.shape[0], M, TOP_K, sinks=SINKS)
    return out[None], None


@torch.no_grad()
def main(rows: int, out: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    texts = datasets.load_dataset("baber/paul_graham_essays")["train"]["text"]
    ids = [
        i for t in texts for i in tokenizer(t)["input_ids"] + [tokenizer.eos_token_id]
    ]
    tokens = torch.tensor(ids[: rows * ROW + 1])
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    if M:
        AttentionInterface.register("csa2_pf", drop_in)
        model.set_attn_implementation("csa2_pf")
    idx = torch.arange(ROW)
    kept = []
    for r in range(rows):
        buf = tokens[r * ROW : (r + 1) * ROW + 1].cuda()
        hidden = model.model(buf[None, :-1]).last_hidden_state[0]
        losses = torch.cat([
            F.cross_entropy(model.lm_head(h).float(), y, reduction="none")
            for h, y in zip(hidden.split(1024), buf[1:].split(1024), strict=True)
        ]).double().cpu()  # fmt: skip
        kept.append(torch.stack([losses, idx.double(), idx.double(), 0 * losses]))
    kept_t = torch.stack(kept)
    torch.save(kept_t, out.removesuffix(".csv") + "_tokens.pt")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "m", "top_k", "start", "end", "tokens", "loss"])
        for a, b in (*pairwise(EDGES), (0, ROW)):
            loss = kept_t[:, 0, a:b]
            w.writerow(
                ["qwen3-0.6b", M, TOP_K, a, b, loss.numel(), f"{loss.mean():.5f}"]
            )
    print(open(out).read(), flush=True)


if __name__ == "__main__":
    M, TOP_K = int(sys.argv[3]), int(sys.argv[4])
    SINKS = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    main(int(sys.argv[1]), sys.argv[2])
