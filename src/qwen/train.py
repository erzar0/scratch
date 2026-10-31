"""Train CSA2-Qwen, or the dense control, with the rest of Qwen frozen.

Stages (each starts from the previous stage's weights):
    transfer  each CSA2 layer gets the input Qwen's attention got and learns its
              output: squared error relative to the output's mean square, plus the
              indexer loss. A Full layer and the layers sharing its KV run as one
              group with its own backward.
    e2e       next-token cross-entropy plus KL(Qwen || model), from the final hidden
              states in chunks of CHUNK positions, so full logits never exist.
    long      e2e on longer windows (SEQ=16384), run only if PG-19 loss rises with
              position past the training length.

Training windows come from the web + books mix (data.mix); validation from
WikiText-103 val. (A first run trained on WikiText-103 alone failed on books:
runs/qwen/wikitext_only.)

    SEQ=4096 BATCH=2 STAGE=transfer STEPS=1500 PEAK_LR=1e-3 \
        uv run python -m qwen.train csa2
    SEQ=4096 BATCH=2 STAGE=e2e STEPS=2000 PEAK_LR=3e-4 uv run python -m qwen.train csa2
    SEQ=4096 BATCH=2 STAGE=e2e STEPS=2000 PEAK_LR=3e-5 uv run python -m qwen.train dense
Logs go to runs/qwen/<model>_<stage>.csv, trainable weights to
runs/qwen/<model>_<stage>.pt.
"""

import csv
import itertools
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3ForCausalLM

from model import train
from qwen import data
from qwen import model as qm

logger = logging.getLogger(__name__)

RUNS = Path("runs/qwen")
CHUNK = 512
PREVIOUS = {"e2e": "transfer", "long": "e2e"}  # stage -> stage it starts from


def weights_path(model_name: str, stage: str) -> Path:
    """Where a stage's trainable weights go."""
    return RUNS / f"{model_name}_{stage}.pt"


def final_stage(model_name: str) -> str:
    """The last stage trained: long if its weights exist, else e2e."""
    return "long" if weights_path(model_name, "long").exists() else "e2e"


def build(model_name: str, stage: str | None = None) -> Qwen3ForCausalLM:
    """Qwen with trainable csa2 or dense attention, loaded from stage if given.

    Raises:
        ValueError: If the checkpoint has keys the model lacks.
    """
    model = qm.load()
    qm.convert(model, qm.csa2_config(model.config) if model_name == "csa2" else None)
    if stage is not None:
        state = torch.load(weights_path(model_name, stage), weights_only=True)
        unexpected = model.load_state_dict(state, strict=False).unexpected_keys
        if unexpected:
            raise ValueError(f"checkpoint keys not in the model: {unexpected[:3]}")
    return model


def autocast() -> torch.autocast:
    """bfloat16 autocast: float32 trainable weights meet bfloat16 Qwen activations."""
    return torch.autocast("cuda", torch.bfloat16)


@torch.no_grad()
def teacher_io(
    model: Qwen3ForCausalLM, tokens: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Each layer's attention input and output under Qwen's own attention."""
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def keep(_: nn.Module, args: tuple, kwargs: dict, output: tuple) -> None:
        captured.append((kwargs["hidden_states"], output[0]))

    handles = [
        t.register_forward_hook(keep, with_kwargs=True) for t in qm.teachers(model)
    ]
    try:
        with qm.teacher(model):
            model.model(input_ids=tokens)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def transfer_errors(
    model: Qwen3ForCausalLM, io: list[tuple[torch.Tensor, torch.Tensor]]
) -> list[torch.Tensor]:
    """Each CSA2 layer's relative squared error on the teacher's inputs.

    With gradient enabled, back-propagates the mean over layers group by group.
    """
    attentions = qm.csa2_students(model)
    errors: list[torch.Tensor] = []
    for _, group in itertools.groupby(attentions, key=lambda a: a.kv_source):
        attentions[0].current.reset()
        loss = torch.zeros((), device=io[0][0].device)
        for attention in group:
            x, target = io[attention.layer_idx]
            with autocast():
                output, _ = attention(x)
            target = target.float()
            error = (output.float() - target).pow(2).mean() / target.pow(2).mean()
            errors.append(error.detach())
            loss = loss + error / len(attentions)
        if torch.is_grad_enabled():
            loss.backward()
    return errors


def chunk_losses(
    head: nn.Module, hidden: torch.Tensor, teacher: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Summed cross-entropy and KL(teacher || model) of one chunk of positions."""
    log_p = head(hidden).float().log_softmax(-1)
    with torch.no_grad():
        log_q = head(teacher).float().log_softmax(-1)
    ce = -log_p.gather(-1, targets.unsqueeze(-1)).sum()
    kl = (log_q.exp() * (log_q - log_p)).sum()
    return torch.stack([ce, kl])


def lm_losses(model: Qwen3ForCausalLM, tokens: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy and mean KL(Qwen || model) per token of a (B, T + 1) batch."""
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    with torch.no_grad(), qm.teacher(model):
        teacher = model.model(input_ids=inputs).last_hidden_state
    with autocast():
        hidden = model.model(input_ids=inputs).last_hidden_state
    total = torch.zeros(2, device=tokens.device)
    for s in range(0, targets.shape[1], CHUNK):
        part = (model.lm_head, hidden[:, s : s + CHUNK], teacher[:, s : s + CHUNK])
        if torch.is_grad_enabled():
            losses = checkpoint(
                chunk_losses, *part, targets[:, s : s + CHUNK], use_reentrant=False
            )
        else:
            losses = chunk_losses(*part, targets[:, s : s + CHUNK])
        total = total + losses
    return total / targets.numel()


def step_losses(
    model: Qwen3ForCausalLM, stage: str, tokens: torch.Tensor
) -> torch.Tensor:
    """The stage's losses on a batch; backward already run when gradient is on.

    transfer: each layer's relative error. e2e and long: cross-entropy, KL.
    """
    if stage == "transfer":
        return torch.stack(transfer_errors(model, teacher_io(model, tokens[:, :-1])))
    losses = lm_losses(model, tokens)
    if torch.is_grad_enabled():
        losses.sum().backward()
    return losses.detach()


@torch.no_grad()
def validate(
    model: Qwen3ForCausalLM, stage: str, val: np.ndarray, config: train.TrainConfig
) -> np.ndarray:
    """Mean step_losses over the first eval_batches validation windows."""
    model.eval()
    batches = train.windows(val, config.seq_len, config.batch, config.eval_batches)
    losses = [step_losses(model, stage, b).cpu().numpy() for b in batches]
    model.train()
    return np.mean(losses, axis=0)


def columns(stage: str, n_layers: int) -> list[str]:
    """Names of a stage's losses."""
    if stage == "transfer":
        return [f"error_{i}" for i in range(n_layers)]
    return ["ce", "kl"]


def main(model_name: str, stage: str, config: train.TrainConfig) -> None:
    """Train one stage and save the trainable weights.

    Args:
        model_name: csa2 or dense.
        stage: transfer, e2e or long.
        config: Training settings; eval_every and eval_batches as in model/train.py.

    Raises:
        FloatingPointError: If the loss diverges.
    """
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    train_data, val_data = data.mix(), data.load("validation")
    previous = PREVIOUS.get(stage)
    if model_name == "dense" and previous == "transfer":
        previous = None  # the dense control starts from Qwen's own attention
    model = build(model_name, previous).train()
    if stage != "transfer":
        model.gradient_checkpointing_enable({"use_reentrant": False})
    optimizer = train.make_optimizer(model, config)
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    clipped = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not train.is_indexer(n)
    ]
    names = columns(stage, len(model.model.layers))
    logger.info(
        "[%s %s] %.1fM trainable params, %d steps x %d tokens, from %s",
        model_name,
        stage,
        sum(p.numel() for p in trainable) / 1e6,
        config.steps,
        config.batch * config.seq_len,
        previous or "Qwen",
    )
    rng = np.random.default_rng(config.seed)
    RUNS.mkdir(parents=True, exist_ok=True)
    start = time.time()
    path = weights_path(model_name, stage).with_suffix(".csv")
    with path.open("w", newline="") as log:
        fields = ["step", "tokens", "seconds", "grad_norm"]
        fields += [f"train_{n}" for n in names] + [f"val_{n}" for n in names]
        writer = csv.DictWriter(log, fields)
        writer.writeheader()
        for step in range(1, config.steps + 1):
            starts = rng.integers(0, len(train_data) - config.seq_len - 1, config.batch)
            batch = train.to_device(
                train_data[starts[:, None] + np.arange(config.seq_len + 1)]
            )
            losses = step_losses(model, stage, batch)
            norm = torch.nn.utils.clip_grad_norm_(clipped, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = train.learning_rate(step, config)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 50:
                continue
            values = losses.cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError(
                    f"[{model_name} {stage}] diverged at step {step}"
                )
            row = {
                "step": step,
                "tokens": step * config.batch * config.seq_len,
                "seconds": f"{time.time() - start:.1f}",
                "grad_norm": f"{float(norm):.4g}",
            }
            row |= {
                f"train_{n}": f"{v:.5g}" for n, v in zip(names, values, strict=True)
            }
            note = ""
            if step % config.eval_every == 0 or step == config.steps:
                val = validate(model, stage, val_data, config)
                row |= {f"val_{n}": f"{v:.5g}" for n, v in zip(names, val, strict=True)}
                note = f", val {val.mean():.4f}"
            writer.writerow(row)
            log.flush()
            logger.info(
                "[%s %s] step %d: train %.4f%s",
                model_name,
                stage,
                step,
                values.mean(),
                note,
            )
    torch.save(qm.trainable_state(model), weights_path(model_name, stage))
    logger.info("[%s %s] done in %.0f s", model_name, stage, time.time() - start)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(sys.argv[1], os.environ.get("STAGE", "e2e"), train.TrainConfig.from_env())
