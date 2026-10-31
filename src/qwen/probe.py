"""Why CSA2-Qwen trails Qwen: the sparsity pattern, or the CSA2 layer itself?

uv run python -m qwen.probe mask
    Qwen's own attention, untrained, restricted to what a CSA2 query can see
    (local window, first tokens, oracle top-k blocks): WikiText val loss on
    LIMIT (8) windows of SEQ -> runs/qwen/probe_mask.csv
[LAYERS=3,7] uv run python -m qwen.probe layer [<variant>...]
    CSA2 layers (default 0 and 4) trained alone as Full layers with the transfer
    loss on Qwen's inputs, under bigger sizes (VARIANTS)
    -> runs/qwen/probe_layer.csv (probe_layer_3-7.csv for other layers)
"""

import functools
import logging
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from transformers import AttentionInterface, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, repeat_kv

from deepseek import csa, csa2, csa2_sink
from experiments.results import Results
from model import train
from qwen import data
from qwen import evaluate as qe
from qwen import model as qm
from qwen import train as qt

logger = logging.getLogger(__name__)

SEQ = 4096
WINDOW = 128  # SIZES["sliding_window"]
BLOCK = 4  # tokens per main KV entry (m)
TOP_K = 64  # SIZES["index_topk"]
# row -> (first tokens always visible, oracle blocks, whether the first tokens
# keep their values: False leaves them only soaking up weight, like CSA2's sink
# logits); None is plain Qwen
PATTERNS: dict[str, tuple[int, int, bool] | None] = {
    "qwen": None,
    "window": (0, 0, True),
    "window+sink": (4, 0, True),
    "window+sink+oracle": (4, TOP_K, True),
    "window+sink weight": (4, 0, False),
    "window+sink weight+oracle": (4, TOP_K, False),
}
# layers trained alone, each as a Full layer (own block KV and indexer); default:
# the best and worst Full layer in the transfer stage
LAYERS = tuple(int(i) for i in os.environ.get("LAYERS", "0,4").split(","))
LAYER_CSV = (
    "probe_layer.csv"
    if LAYERS == (0, 4)
    else "probe_layer_{}.csv".format("-".join(map(str, LAYERS)))
)
STEPS = 600
# index_head_dim must hold the rotated channels: head_dim x partial_rotary_factor
VARIANTS: dict[str, dict[str, object]] = {
    "base": {},
    "wide": {"head_dim": 512, "index_head_dim": 128},
    "full-rank": {"q_lora_rank": 1024, "o_lora_rank": 1024},
    "rope": {"partial_rotary_factor": 1.0, "index_head_dim": 128},
    "no-pool": {"compress_rates": {csa.CSA: 1, "heavily_compressed_attention": 128}},
}
VARIANTS["all"] = {k: v for o in VARIANTS.values() for k, v in o.items()} | {
    "index_head_dim": 512
}
# the same sizes, every query also reading the first tokens (deepseek/csa2_sink.py)
VARIANTS["sinks"] = VARIANTS["base"]
VARIANTS["sinks+all"] = VARIANTS["all"]


class QwenCsa2SinkAttention(csa2_sink.SinkAttend, qm.QwenCsa2Attention):
    """QwenCsa2Attention whose queries also read the first csa2_sink.SINKS tokens."""


def allowed(seq_len: int, sinks: int, device: torch.device) -> torch.Tensor:
    """(S, S) causal mask of the last WINDOW tokens and the first `sinks` tokens."""
    q = torch.arange(seq_len, device=device)[:, None]
    k = torch.arange(seq_len, device=device)[None]
    return (k <= q) & ((q - k < WINDOW) | (k < sinks))


def restricted(
    module: Qwen3Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    *,
    sinks: int,
    top_k: int,
    sink_values: bool,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Qwen's attention over allowed(), plus each query's top_k finished blocks by
    its causal attention weight summed over heads (batch 1, no padding). Without
    sink_values the first `sinks` tokens are read as zero vectors."""
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    if not sink_values:
        value = value.clone()
        value[..., :sinks, :] = 0
    logits = (query @ key.transpose(2, 3)).float() * scaling  # (B, H, S, S)
    seq_len = logits.shape[-1]
    visible = allowed(seq_len, sinks, logits.device).expand(logits.shape[0], -1, -1)
    if top_k:
        causal = allowed(seq_len, seq_len, logits.device)
        weight = logits.masked_fill(~causal, float("-inf")).softmax(-1).sum(1)
        n = seq_len // BLOCK
        blocks = weight[..., : n * BLOCK].unflatten(-1, (n, BLOCK)).sum(-1)
        positions = torch.arange(seq_len, device=logits.device)[None]
        picks = csa2.top_picks(blocks, positions, BLOCK, top_k)
        chosen = torch.zeros_like(blocks, dtype=torch.bool)
        chosen.scatter_(-1, picks.entry, picks.valid)
        visible = visible.clone()
        visible[..., : n * BLOCK] |= chosen.repeat_interleave(BLOCK, -1)
    probs = logits.masked_fill(~visible[:, None], float("-inf")).softmax(-1)
    return (probs.to(value.dtype) @ value).transpose(1, 2).contiguous(), None


@torch.no_grad()
def mask() -> None:
    """WikiText loss of Qwen under every PATTERNS row."""
    model = qm.load()
    windows = list(
        train.windows(data.load("validation"), SEQ, 1, qe.sample_limit() or 8)
    )
    for row, pattern in PATTERNS.items():
        model.set_attn_implementation("sdpa")
        if pattern is not None:
            name = f"probe_{row}"
            sinks, top_k, sink_values = pattern
            fn = functools.partial(
                restricted, sinks=sinks, top_k=top_k, sink_values=sink_values
            )
            AttentionInterface.register(name, fn)
            model.set_attn_implementation(name)
        results = Results(qt.RUNS / "probe_mask.csv", {"row": row})
        for i, batch in enumerate(windows):
            ce = float(qe.token_losses(model, batch).mean())
            results.add({"row": row, "window": i, "ce": ce})
        logger.info("[probe mask] %s done", row)


@torch.no_grad()
def teacher_io(
    model: Qwen3ForCausalLM, tokens: torch.Tensor
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Qwen's attention input and output at LAYERS."""
    captured = {}

    def keep(layer: int):
        def hook(_: nn.Module, args: tuple, kwargs: dict, output: tuple) -> None:
            captured[layer] = (kwargs["hidden_states"], output[0])

        return hook

    handles = [
        model.model.layers[layer].self_attn.register_forward_hook(
            keep(layer), with_kwargs=True
        )
        for layer in LAYERS
    ]
    try:
        model.model(input_ids=tokens)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def relative_error(
    student: qm.QwenCsa2Attention, x: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """train.transfer_errors' loss for one Full layer."""
    student.current.reset()
    with qt.autocast():
        output, _ = student(x)
    target = target.float()
    return (output.float() - target).pow(2).mean() / target.pow(2).mean()


def layer(names: list[str]) -> None:
    """The named VARIANTS rows (all if none): LAYERS trained alone for STEPS,
    errors every 50 steps."""
    torch.set_float32_matmul_precision("high")
    model = qm.load().requires_grad_(False)
    mix = data.mix()
    split = int(len(mix) * 0.9)
    held_out = {  # Qwen's attention inputs and outputs on held-out windows
        name: [teacher_io(model, b[:, :-1]) for b in train.windows(source, SEQ, 1, 4)]
        for name, source in (
            ("val_mix", mix[split:]),
            ("val_wikitext", data.load("validation")),
        )
    }
    config = train.TrainConfig(
        seq_len=SEQ, batch=2, steps=STEPS, peak_lr=1e-3, warmup=60
    )
    for variant in names or VARIANTS:
        overrides = VARIANTS[variant]
        results = Results(qt.RUNS / LAYER_CSV, {"variant": variant})
        torch.manual_seed(0)
        csa2_config = qm.csa2_config(model.config, **overrides)
        csa2_config.csa2_modes = [csa2.FULL] * len(csa2_config.csa2_modes)
        kind = (
            QwenCsa2SinkAttention
            if variant.startswith("sinks")
            else qm.QwenCsa2Attention
        )
        students = {i: kind(csa2_config, i, qm.Current()).cuda() for i in LAYERS}
        trainable = nn.ModuleList(students.values())
        optimizer = train.make_optimizer(trainable, config)  # type: ignore[arg-type]
        params = sum(p.numel() for p in next(iter(students.values())).parameters())
        rng = np.random.default_rng(0)
        start = time.time()
        for step in range(1, STEPS + 1):
            starts = rng.integers(0, split - SEQ, config.batch)
            tokens = train.to_device(mix[starts[:, None] + np.arange(SEQ)])
            # one sequence per backward: head_dim 512 overflows 16 GB at batch 2
            errors = dict.fromkeys(LAYERS, 0.0)
            for sequence in tokens.split(1):
                io = teacher_io(model, sequence)
                parts = {i: relative_error(students[i], *io[i]) for i in LAYERS}
                (torch.stack(list(parts.values())).sum() / config.batch).backward()
                for i, part in parts.items():
                    errors[i] += float(part) / config.batch
            for student in students.values():
                kept = [
                    p for n, p in student.named_parameters() if not train.is_indexer(n)
                ]
                torch.nn.utils.clip_grad_norm_(kept, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = train.learning_rate(step, config)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 50:
                continue
            with torch.no_grad():
                for i in LAYERS:
                    row = {"layer": i, "variant": variant, "params": params}
                    row |= {"step": step, "train_error": errors[i]}
                    for name, ios in held_out.items():
                        held = [relative_error(students[i], *x[i]) for x in ios]
                        row[name] = float(torch.stack(held).mean())
                    results.add(row)
            logger.info(
                "[probe layer %s] step %d: %s (%.0f s, peak %.1f GiB)",
                variant,
                step,
                ", ".join(f"layer {i} {e:.3f}" for i, e in errors.items()),
                time.time() - start,
                torch.cuda.max_memory_allocated() / 2**30,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if sys.argv[1] == "mask":
        mask()
    else:
        layer(sys.argv[2:])
