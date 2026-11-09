"""Tasks: one tiny Llama trained once on every task with dense attention, then each
version swapped into every layer at the answer positions (the harness's HF hook).

Model: LlamaForCausalLM, 4 layers, width 512, 4 query and 2 KV heads of dim 128.
Training: transformers.Trainer on all `tasks` mixed (copy forms the lookup circuit
fast; MQAR and needle reuse it, and alone they stall at chance). Each item is a
batch of TOKENS tokens: sequences of one task and length, cycling through tasks
and `train_lengths`, so short lengths come many to a batch; loss on the answers
only, averaged over the answers of `accumulate` batches, so answer-dense copy
dominates (that bootstraps the circuit; weighing every batch the same stalled at
chance), logged to MLflow (experiment tasks-training); the model is saved to
runs/tasks/model/. Evaluation: per task, length and seed,
`sequences` fresh sequences; the last ANSWERS answers of each are scored, and the
version runs on every query from the first of them on (the prefix runs dense
SDPA). Accuracy: share of answers predicted exactly.
"""

import os
from pathlib import Path
from typing import Any

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from attention import harness
from attention.registry import Version
from benchmarks import synthetic
from common import tracking

ROOT = Path("runs/tasks")
ANSWERS = 128
TOKENS = 16384  # per training batch
MODEL: dict[str, Any] = {
    "vocab_size": synthetic.VOCAB,
    "hidden_size": 512,
    "intermediate_size": 1408,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "max_position_embeddings": 16384,
}


class Batches(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Item i: TOKENS // n sequences of task tasks[i % T] and length n, the task's
    next length (lengths below the task's shortest skipped), seeded by i."""

    def __init__(
        self, tasks: list[str], lengths: list[int], size: int, seed: int
    ) -> None:
        self.lengths = {
            task: [n for n in lengths if n >= synthetic.SHORTEST[task]]
            for task in tasks
        }
        self.tasks, self.size, self.seed = tasks, size, seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        task = self.tasks[i % len(self.tasks)]
        lengths = self.lengths[task]
        n = lengths[i // len(self.tasks) % len(lengths)]
        count = max(1, TOKENS // n)
        pairs = [synthetic.sequence(task, n, gen) for _ in range(count)]
        ids, answer = (torch.stack(part) for part in zip(*pairs, strict=True))
        return {"input_ids": ids, "labels": ids.masked_fill(~answer, -100)}


def first(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate: the items are batches already."""
    return items[0]


def train(config: dict[str, Any]) -> None:
    """Train the model unless runs/tasks/model/ holds one."""
    out = ROOT / "model"
    if (out / "config.json").exists():
        return
    os.environ["MLFLOW_TRACKING_URI"] = tracking.URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "tasks-training"
    tracking.experiment("tasks-training")
    steps, accumulate = config["train_steps"], config["accumulate"]
    args = TrainingArguments(
        output_dir=str(ROOT / "trainer"),
        run_name="train-" + "-".join(config["tasks"]),
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=accumulate,
        learning_rate=1e-3,
        warmup_steps=steps // 20,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        bf16=True,
        logging_steps=25,
        save_strategy="no",
        report_to=["mlflow"],
        dataloader_num_workers=2,
        disable_tqdm=True,
        seed=0,
    )
    model = LlamaForCausalLM(LlamaConfig(**MODEL))
    data = Batches(config["tasks"], config["train_lengths"], steps * accumulate, 1)
    Trainer(model=model, args=args, train_dataset=data, data_collator=first).train()
    model.save_pretrained(out)


@torch.no_grad()
def accuracy(
    model: LlamaForCausalLM,
    version: Version,
    task: str,
    n: int,
    seed: int,
    sequences: int,
) -> tuple[float, float]:
    """Share of the scored answers right, and of K and V elements read."""
    right = total = 0
    reads = dense_reads = 0.0
    for i in range(sequences):
        gen = torch.Generator().manual_seed(10_000 + 1000 * seed + i)  # not training's
        ids, answer = synthetic.sequence(task, n, gen)
        at = answer.nonzero()[-ANSWERS:, 0]
        assert version.emulation is not None, "latency only"
        state = harness.attach(model, version.emulation, scored=n - int(at[0]) + 1)
        logits = model(input_ids=ids[None].to(model.device)).logits[0]
        hit = logits[at - 1].argmax(-1).cpu() == ids[at]
        right, total = right + int(hit.sum()), total + len(at)
        reads += sum(s["reads"] for s in state.sums.values())
        dense_reads += sum(s["dense_reads"] for s in state.sums.values())
    return right / total, reads / dense_reads


def run(
    version: Version,
    config: dict[str, Any],
    device: str = "cuda",
    model: LlamaForCausalLM | None = None,
) -> list[dict[str, Any]]:
    """One row per task, length and seed; the model defaults to runs/tasks/model/."""
    if model is None:
        model = LlamaForCausalLM.from_pretrained(ROOT / "model", dtype=torch.float32)
    model = model.to(device).eval()  # type: ignore[arg-type]
    rows = []
    for task in config["tasks"]:
        for n in config["lengths"]:
            for seed in config["seeds"]:
                acc, reads = accuracy(
                    model, version, task, n, seed, config["sequences"]
                )
                rows.append(
                    {
                        "task": task,
                        "n": n,
                        "seed": seed,
                        "accuracy": acc,
                        "reads": reads,
                    }
                )
                print(f"  {task} {n} seed {seed}: accuracy {acc:.3f}", flush=True)
    return rows
