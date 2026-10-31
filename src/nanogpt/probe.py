"""Can one CSA2 layer stand in for a record 91 attention layer?

qwen/probe.py's layer study on record 91. Every attention layer gets a Full CSA2
student, trained alone on that layer's own input and output (the stage-1 loss) on
2048-token windows of single long FineWeb documents; one teacher forward feeds all
ten students. Qwen3-0.6B's layers stayed at 0.26-0.56 held-out error trained alone; a
layer that can stand in should reach about 0.1.

    PYTHONPATH=src $R91/.venv/bin/python -m nanogpt.probe CKPT [VARIANT...]

with R91 = ../modded-nanogpt-r91.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from deepseek import csa, csa2
from experiments.results import Results
from model import train
from nanogpt import model as nm
from qwen import model as qm

logger = logging.getLogger(__name__)

RUNS = Path(__file__).resolve().parents[2] / "runs" / "nanogpt"
STEPS = 600
BATCH = 4  # windows per step: 8192 tokens, as qwen/probe.py
HELD_OUT = 8  # validation-shard windows
# V4-Flash attention scaled to width 768 as qwen/model.py SIZES is to 1024; heads and
# value width are record 91's
SIZES: dict[str, Any] = {
    "hidden_size": 768,
    "num_attention_heads": 6,
    "head_dim": 128,
    "partial_rotary_factor": 0.125,
    "q_lora_rank": 192,
    "o_groups": 2,
    "o_lora_rank": 192,
    "index_n_heads": 6,
    "index_head_dim": 32,
    "index_topk": 64,
    "sliding_window": 128,
    "max_position_embeddings": nm.SEQ,
}
VARIANTS: dict[str, dict[str, Any]] = {
    "base": {},
    # qwen/probe.py's "all": key/value 512 wide, full-rank query and output, every
    # channel rotated, no pooling
    "all": {
        "head_dim": 512,
        "index_head_dim": 512,
        "q_lora_rank": 768,
        "o_lora_rank": 768,
        "partial_rotary_factor": 1.0,
        "compress_rates": {csa.CSA: 1, "heavily_compressed_attention": 128},
    },
}


@torch.no_grad()
def teacher_io(
    model: nn.Module, config: Any, window: torch.Tensor
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Every attention layer's input and output on one window."""
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def keep(_: nn.Module, args: tuple, output: torch.Tensor) -> None:
        calls.append((args[0], output))

    attentions = [model.get_submodule(n) for n in ("attn", "attn_paired")]
    handles = [m.register_forward_hook(keep) for m in attentions]
    try:
        nm.token_losses(model, config, window)
    finally:
        for handle in handles:
            handle.remove()
    return dict(zip(nm.ATTENTION_LAYERS, calls, strict=True))


def relative_error(
    student: qm.QwenCsa2Attention, x: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """qwen/probe.py's error: squared error over the target's mean square."""
    student.current.reset()
    with torch.autocast("cuda", torch.bfloat16):
        output, _ = student(x)
    target = target.float()
    return (output.float() - target).pow(2).mean() / target.pow(2).mean()


def probe(checkpoint: Path, names: list[str]) -> None:
    """The named VARIANTS (all if none), errors every 50 steps to probe_layer.csv."""
    torch.set_float32_matmul_precision("high")
    model, config = nm.load(checkpoint)
    windows = nm.long_documents("fineweb_train_000001.bin")
    held_out = [
        teacher_io(model, config, w)
        for w in nm.long_documents("fineweb_val_*.bin")[:HELD_OUT]
    ]
    logger.info("%d training windows", len(windows))
    settings = train.TrainConfig(
        seq_len=nm.SEQ, batch=BATCH, steps=STEPS, peak_lr=1e-3, warmup=60
    )
    for variant in names or VARIANTS:
        results = Results(RUNS / "probe_layer.csv", {"variant": variant})
        torch.manual_seed(0)
        modes = (csa2.FULL,) * 11
        csa2_config = csa2.small_config(modes, **SIZES | VARIANTS[variant])
        students = {
            i: qm.QwenCsa2Attention(csa2_config, i, qm.Current()).cuda()
            for i in nm.ATTENTION_LAYERS
        }
        trainable = nn.ModuleList(students.values())
        optimizer = train.make_optimizer(trainable, settings)  # type: ignore[arg-type]
        params = sum(p.numel() for p in students[0].parameters())
        rng = np.random.default_rng(0)
        start = time.time()
        for step in range(1, STEPS + 1):
            errors = dict.fromkeys(students, 0.0)
            for index in rng.integers(0, len(windows), BATCH):
                io = teacher_io(model, config, windows[index])
                # one backward per layer: ten graphs at once overflow 16 GB
                for i, student in students.items():
                    part = relative_error(student, *io[i]) / BATCH
                    part.backward()
                    errors[i] += float(part)
            for student in students.values():
                kept = [
                    p for n, p in student.named_parameters() if not train.is_indexer(n)
                ]
                torch.nn.utils.clip_grad_norm_(kept, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = train.learning_rate(step, settings)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 50:
                continue
            with torch.no_grad():
                for i, student in students.items():
                    held = [relative_error(student, *io[i]) for io in held_out]
                    results.add(
                        {"layer": i, "variant": variant, "params": params}
                        | {"step": step, "train_error": errors[i]}
                        | {"val_error": float(torch.stack(held).mean())}
                    )
            logger.info(
                "[probe %s] step %d: %s (%.0f s, peak %.1f GiB)",
                variant,
                step,
                " ".join(f"{i}:{e:.2f}" for i, e in errors.items()),
                time.time() - start,
                torch.cuda.max_memory_allocated() / 2**30,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    probe(Path(sys.argv[1]), sys.argv[2:])
