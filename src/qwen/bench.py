"""Cost of CSA2-Qwen against Qwen.

layer  a Qwen-sized CSA2 Full layer against Qwen3-sized exact attention (16 query
       heads, 8 KV heads of 128, SDPA FlashAttention): prefill, decode step and
       cache per sequence, 1k-128k tokens (bench/layer.py)
       -> runs/qwen/bench_layer.csv
model  whole-model prefill (one sequence, no logits) and its peak memory, for
       the qwen and csa2 rows at 4k-32k -> runs/qwen/bench_model.csv

uv run python -m qwen.bench layer|model
"""

import logging
import sys

import torch
from transformers import Qwen3Config

from bench import layer as bench_layer
from bench.timing import measure
from experiments.results import Results
from qwen import evaluate
from qwen import model as qm

logger = logging.getLogger(__name__)

MODEL_LENGTHS = (4096, 8192, 16384, 32768)


def layer() -> None:
    """bench/layer.py at Qwen's sizes."""
    qwen = Qwen3Config.from_pretrained(qm.QWEN)
    exact = bench_layer.Exact(
        qwen.hidden_size,
        qwen.num_attention_heads,
        qwen.num_key_value_heads or qwen.num_attention_heads,
        qwen.head_dim,
    )
    bench_layer.main(qm.csa2_config(qwen), exact, evaluate.OUT / "bench_layer.csv")


@torch.no_grad()
def model() -> None:
    """Whole-model prefill time and peak memory per row and length."""
    results = Results(evaluate.OUT / "bench_model.csv", {})
    for row in ("qwen", "csa2"):
        lm, selector = evaluate.load_row(row)
        with evaluate.scoped(lm, selector):
            for seq_len in MODEL_LENGTHS:
                tokens = torch.randint(0, 1000, (1, seq_len), device="cuda")
                timing = measure(lambda t=tokens, m=lm: m.model(input_ids=t), reps=3)
                entry = {"row": row, "T": seq_len, "prefill_ms": timing.ms}
                results.add(entry | {"peak_gib": timing.peak_gib})
                logger.info("%s", entry)
        del lm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"layer": layer, "model": model}[sys.argv[1]]()
