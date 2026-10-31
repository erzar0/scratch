"""Train DeepSeek-V4.1 CSA2 (or the V4 CSA baseline) on WikiText-103 GPT-2 tokens.

Prep once (downloads ~320 MB of parquet, writes data/wt103_*.bin):
    uv run --extra prep python -m model.train prep
Train one model, csa2 (deepseek/csa2.py) or csa (HF DeepSeek-V4, untrained indexer):
    SEQ=2048 BATCH=2 STEPS=24000 uv run python -m model.train csa2
Logs go to runs/train/<model>_seq<SEQ>.csv, weights to runs/train/<model>_seq<SEQ>.pt;
SEED=n (default 0) changes init and data order and appends _s<n> to both names.
Loss is nats per BPE token: perplexities are per BPE token, not the word-level
perplexities papers report on WikiText-103.
"""

import logging
import math
import os
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from deepseek import csa, csa2

logger = logging.getLogger(__name__)

DATA = Path("data")
RUNS = Path("runs/train")
SPLITS = {"train": 2, "validation": 1, "test": 1}  # split -> parquet shards
MODELS: dict[str, Callable[[], PreTrainedModel]] = {
    "csa": csa.build,
    "csa2": csa2.build,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class TrainConfig:
    """Training settings; SEQ, BATCH, STEPS, PEAK_LR and SEED come from the environment.

    Attributes:
        seq_len: Tokens per training window.
        batch: Windows per step.
        steps: Optimizer steps.
        peak_lr: Learning rate after warmup.
        warmup: Linear warmup steps.
        eval_every: Steps between validation estimates.
        eval_batches: Batches per validation estimate.
        seed: Init and data-order seed.
    """

    seq_len: int = 512
    batch: int = 8
    steps: int = 8000
    peak_lr: float = 1e-3
    warmup: int = 300
    eval_every: int = 500
    eval_batches: int = 40
    seed: int = 0

    @classmethod
    def from_env(cls) -> "TrainConfig":
        """Read SEQ, BATCH, STEPS, PEAK_LR and SEED; unset ones keep the defaults."""
        env = os.environ
        return cls(
            seq_len=int(env.get("SEQ", cls.seq_len)),
            batch=int(env.get("BATCH", cls.batch)),
            steps=int(env.get("STEPS", cls.steps)),
            peak_lr=float(env.get("PEAK_LR", cls.peak_lr)),
            seed=int(env.get("SEED", cls.seed)),
        )


def wikitext_lines(split: str) -> list[str]:
    """Download one WikiText-103 split's raw lines."""
    import urllib.request

    import pyarrow as pa
    import pyarrow.parquet as pq

    base = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/"
    shards = SPLITS[split]
    lines = []
    for shard in range(shards):
        url = f"{base}{split}-{shard:05d}-of-{shards:05d}.parquet"
        with urllib.request.urlopen(url) as response:
            table = pq.read_table(pa.BufferReader(response.read()))
        lines += table.column("text").to_pylist()
    return lines


def prep() -> None:
    """Download WikiText-103 and write GPT-2 token ids as uint16 files."""
    import tiktoken

    encoding = tiktoken.get_encoding("gpt2")
    DATA.mkdir(exist_ok=True)
    for split in SPLITS:
        encoded = encoding.encode_ordinary_batch(wikitext_lines(split))
        ids = np.concatenate([np.array(t, np.uint16) for t in encoded if t])
        ids.tofile(DATA / f"wt103_{split}.bin")
        logger.info("%s: %s tokens", split, f"{len(ids):,}")


def load(split: str) -> np.ndarray:
    """Memory-mapped token ids of one split."""
    return np.memmap(DATA / f"wt103_{split}.bin", dtype=np.uint16, mode="r")


def to_device(tokens: np.ndarray) -> torch.Tensor:
    """int64 token ids on DEVICE."""
    return torch.from_numpy(tokens.astype(np.int64)).to(DEVICE, non_blocking=True)


def windows(
    data: np.ndarray, seq_len: int, batch: int, limit: int | None = None
) -> Iterator[torch.Tensor]:
    """Consecutive non-overlapping (batch, seq_len + 1) windows, full batches only.

    Args:
        data: Token ids.
        seq_len: Inputs per window (each window holds one more token as target).
        batch: Windows per yielded batch.
        limit: Yield at most this many batches.

    Yields:
        Token batches on DEVICE.
    """
    n_batches = (len(data) - 1) // seq_len // batch
    for start in range(0, min(n_batches, limit or n_batches) * batch, batch):
        rows = np.arange(start, start + batch)[:, None] * seq_len
        yield to_device(data[rows + np.arange(seq_len + 1)])


def loss_fn(model: PreTrainedModel, batch: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy of a (B, T + 1) token batch, bf16 on CUDA."""
    with torch.autocast(DEVICE.type, torch.bfloat16, enabled=DEVICE.type == "cuda"):
        logits = model(input_ids=batch[:, :-1], use_cache=False).logits
    return F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten())


def learning_rate(step: int, config: TrainConfig) -> float:
    """Linear warmup, then cosine decay to 10% of peak."""
    if step < config.warmup:
        return config.peak_lr * step / config.warmup
    progress = min(step / config.steps, 1.0)
    return config.peak_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def is_indexer(name: str) -> bool:
    """CSA2 indexer weights, trained by their own loss only (deepseek/csa2.py)."""
    return ".indexer." in name or "index_k_proj" in name


def make_optimizer(model: PreTrainedModel, config: TrainConfig) -> torch.optim.AdamW:
    """AdamW (b2 = 0.95) over trainable weights, weight decay 0.1 on matrices only,
    LR set per step."""
    matrices: list[torch.nn.Parameter] = []
    others: list[torch.nn.Parameter] = []
    for param in model.parameters():
        if param.requires_grad:
            (matrices if param.ndim == 2 else others).append(param)
    groups: list[dict[str, Any]] = [
        {"params": matrices, "weight_decay": 0.1},
        {"params": others},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=DEVICE.type == "cuda",
    )


def train_step(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One optimizer step.

    Global-norm clipping at 1 skips the indexer: its loss has a much larger gradient
    scale and would shrink every other gradient; Adam normalizes it anyway.

    Args:
        model: The model, in train mode.
        optimizer: From make_optimizer.
        batch: (B, T + 1) tokens.
        lr: Learning rate of this step.

    Returns:
        Loss and the gradient norm before clipping, as device scalars.
    """
    loss = loss_fn(model, batch)
    loss.backward()
    lm_params = [p for n, p in model.named_parameters() if not is_indexer(n)]
    norm = torch.nn.utils.clip_grad_norm_(lm_params, 1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach(), norm


@torch.no_grad()
def evaluate(
    model: PreTrainedModel,
    data: np.ndarray,
    config: TrainConfig,
    limit: int | None = None,
) -> float:
    """Mean loss over consecutive windows (all of them when limit is None)."""
    batches = windows(data, config.seq_len, config.batch, limit)
    return float(np.mean([float(loss_fn(model, b)) for b in batches]))


def seed_suffix(seed: int) -> str:
    """File-name suffix of a model seed: none for seed 0, the original runs."""
    return f"_s{seed}" if seed else ""


def run_name(model_name: str, config: TrainConfig) -> Path:
    """Path stem of a run's log and weights."""
    return RUNS / f"{model_name}_seq{config.seq_len}{seed_suffix(config.seed)}"


def load_model(model_name: str, path: Path) -> PreTrainedModel:
    """A trained model on DEVICE, in eval mode."""
    with torch.device(DEVICE):
        model = MODELS[model_name]()
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model.eval()


def main(model_name: str, config: TrainConfig) -> None:
    """Train one model, then report full validation and test loss.

    Args:
        model_name: A key of MODELS.
        config: Training settings.

    Raises:
        FloatingPointError: If the loss diverges.
    """
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    train_data, val_data = load("train"), load("validation")
    with torch.device(DEVICE):
        model = MODELS[model_name]().train()
    for layer in model.model.layers:  # one graph per layer: 3x fewer kernel launches
        layer.compile()
    optimizer = make_optimizer(model, config)
    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = config.batch * config.seq_len
    logger.info(
        "[%s] %.1fM params, %d steps x %d tokens",
        model_name,
        n_params / 1e6,
        config.steps,
        tokens_per_step,
    )
    rng = np.random.default_rng(config.seed)  # same data order for every model
    run = run_name(model_name, config)
    run.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    with run.with_suffix(".csv").open("w") as log:
        log.write("step,tokens,train_loss,val_loss,seconds\n")
        for step in range(1, config.steps + 1):
            starts = rng.integers(0, len(train_data) - config.seq_len - 1, config.batch)
            rows = starts[:, None] + np.arange(config.seq_len + 1)
            batch = to_device(train_data[rows])
            loss, norm = train_step(
                model, optimizer, batch, learning_rate(step, config)
            )
            if step % 100:
                continue
            loss_value, elapsed = float(loss), time.time() - start_time
            if not math.isfinite(loss_value):
                logger.error("[%s] loss diverged at step %d", model_name, step)
                raise FloatingPointError(f"[{model_name}] diverged at step {step}")
            seen = step * tokens_per_step
            logger.info(
                "[%s] step %d: loss %.3f |g| %.2f %s tok/s",
                model_name,
                step,
                loss_value,
                float(norm),
                f"{seen / elapsed:,.0f}",
            )
            val_loss = ""
            if step % config.eval_every == 0:
                model.eval()
                estimate = evaluate(model, val_data, config, config.eval_batches)
                model.train()
                logger.info("[%s] step %d: val loss %.3f", model_name, step, estimate)
                val_loss = f"{estimate}"
            log.write(f"{step},{seen},{loss_value:.4f},{val_loss},{elapsed:.1f}\n")
            log.flush()
    torch.save(model.state_dict(), run.with_suffix(".pt"))
    model.eval()
    final_val = evaluate(model, val_data, config)
    final_test = evaluate(model, load("test"), config)
    logger.info(
        "[%s] final val %.4f (ppl %.2f), test %.4f (ppl %.2f), %.0fs",
        model_name,
        final_val,
        math.exp(final_val),
        final_test,
        math.exp(final_test),
        time.time() - start_time,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if sys.argv[1] == "prep":
        prep()
    else:
        main(sys.argv[1], TrainConfig.from_env())
