"""Evaluate Qwen, the dense control and CSA2-Qwen.

Rows are named <model>[:<selector>]: qwen (the original), dense (control), csa2,
and csa2 with a selector in its Full and Reindex layers (SELECTORS: window only,
random, oracle; double indexing mean, bound, learned, mixed; learned and mixed
need qwen/route.py's routers). Trained attentions run in bfloat16, without
autocast. qwen:<row> runs a sparse decode attention of experiments/sparse_qwen.py on
the last SPARSE_SCORED (128) queries.

    uv run python -m qwen.evaluate lm <row>...
        WikiText-103 val windows of SEQ (4096) tokens: cross-entropy, KL to Qwen,
        each CSA2 layer's error on Qwen's inputs, the picks' share of the attention
        weight -> runs/qwen/lm.csv, runs/qwen/layers.csv
        PG-19 test, first LONG tokens of each book: loss by position
        -> runs/qwen/pg19.csv
    uv run python -m qwen.evaluate long <row>...
        RULER, BABILong and HELMET ICL samples (qwen/tasks.py), teacher-forced
        answers
        -> runs/qwen/longctx.csv
    uv run python -m qwen.evaluate short <row>...
        lm-eval-harness: MMLU, commonsense, LAMBADA, SWDE, FDA, SQuAD completion
        -> runs/qwen/short.csv
    uv run python -m qwen.evaluate check qwen
        real generation on 4096-token samples, for the teacher-forcing check
        -> runs/qwen/generation_check.csv
Rows of the given model rows are replaced.
"""

import logging
import os
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedTokenizerBase, Qwen3ForCausalLM

from deepseek import csa2, reference
from deepseek.double_indexing import DoubleIndexing
from experiments import eval_route, sparse_qwen
from experiments.results import Results
from experiments.selector_forward import reads_per_seq, selecting
from model import train
from qwen import data, route, tasks
from qwen import model as qm
from qwen import train as qt
from sparse import attend

logger = logging.getLogger(__name__)

SEQ = 4096
LONG = 32768
BUCKETS = (0, 1024, 2048, 4096, 8192, 16384, 32768)  # PG-19 position buckets
OUT = qt.RUNS


def own_indexer(
    attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
) -> csa2.Picks:
    """The layer's own indexer: pure CSA2's picks."""
    assert attention.indexer is not None
    return attention.indexer(
        inputs.hidden_states, inputs.q_residual, inputs.position_ids, inputs.index_k
    )


@dataclass(frozen=True)
class Chunked:
    """A selector over chunk queries at a time: the same picks without any
    (B, S, heads, n) tensor of every query at once (17 GB at 32k tokens)."""

    inner: csa2.Selector
    chunk: int = 1024

    @torch.no_grad()
    def __call__(
        self, attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
    ) -> csa2.Picks:
        """inner on each chunk of queries, concatenated."""
        hidden, q_residual, positions, query, main_kv, index_k = inputs
        parts = [
            self.inner(
                attention,
                csa2.SelectorInputs(
                    hidden[:, s : s + self.chunk],
                    q_residual[:, s : s + self.chunk],
                    positions[:, s : s + self.chunk],
                    query[:, :, s : s + self.chunk],
                    main_kv,
                    index_k,
                ),
            )
            for s in range(0, hidden.shape[1], self.chunk)
        ]
        return csa2.Picks(*(torch.cat(p, 1) for p in zip(*parts, strict=True)))


@dataclass(frozen=True)
class PerLayer:
    """One selector per layer (entries of Reuse layers unused)."""

    layers: tuple[csa2.Selector | None, ...]

    def __call__(
        self, attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
    ) -> csa2.Picks:
        """The layer's own selector."""
        chosen = self.layers[attention.layer_idx]
        assert chosen is not None
        return chosen(attention, inputs)


@torch.no_grad()
def window_only(
    attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
) -> csa2.Picks:
    """No picks: each query reads its sliding window and the sink only."""
    batch, seq_len = inputs.hidden_states.shape[:2]
    shape = (batch, seq_len, 1)
    device = inputs.index_k.device
    entry = torch.zeros(shape, dtype=torch.long, device=device)
    valid = torch.zeros(shape, dtype=torch.bool, device=device)
    return csa2.Picks(entry, valid, torch.zeros(shape, device=device))


@dataclass
class MassShare:
    """Wraps a selector; sums, per layer, the dense attention weight on its picks,
    on the oracle's top-k, and on every finished entry (SEQ-length inputs only).

    Attributes:
        inner: The selector whose picks are measured.
        sums: Layer -> [weight on picks, weight on oracle top-k, total weight].
    """

    inner: csa2.Selector
    sums: dict[int, np.ndarray] = field(default_factory=dict)

    @torch.no_grad()
    def __call__(
        self, attention: csa2.Csa2Attention, inputs: csa2.SelectorInputs
    ) -> csa2.Picks:
        """The inner picks, after adding their weight to sums."""
        picks = self.inner(attention, inputs)
        weight = reference.attention_weights(attention, inputs)
        picked = weight.gather(-1, picks.entry) * picks.valid
        k = picks.entry.shape[-1]
        best = weight.topk(min(k, weight.shape[-1]), dim=-1).values
        totals = [picked.sum(), best.sum(), weight.sum()]
        row = np.array([float(t) for t in totals])
        self.sums[attention.layer_idx] = self.sums.get(attention.layer_idx, 0) + row
        return picks


def mixed(model: Qwen3ForCausalLM) -> PerLayer:
    """Learned (S 16, s 4) in most layers; bound (S 16, s 16) in "needle" layers,
    those whose output the learned route moves most (experiments/eval_route.py)."""
    route.load(model)
    narrow = DoubleIndexing(route.SIZE, 4, "learned")
    tokens = data.mix()[: 2 * SEQ].reshape(2, SEQ)
    needle = eval_route.needle_layers(model, train.to_device(tokens), narrow)
    wide = DoubleIndexing(route.SIZE, 16, "bound")
    layers = range(len(model.model.layers))
    return PerLayer(tuple(wide if i in needle else narrow for i in layers))


def learned(model: Qwen3ForCausalLM) -> DoubleIndexing:
    """Double indexing with the trained routers, superblocks of 16, keep 8."""
    route.load(model)
    return DoubleIndexing(route.SIZE, 8, "learned")


# row suffix -> the selector of the row's Full and Reindex layers
SELECTORS: dict[str, Callable[[Qwen3ForCausalLM], csa2.Selector]] = {
    "": lambda _: own_indexer,
    "window": lambda _: window_only,
    "random": lambda _: reference.RandomEntries(qm.SIZES["index_topk"]),
    "oracle": lambda _: reference.Oracle(qm.SIZES["index_topk"]),
    "mean": lambda _: DoubleIndexing(route.SIZE, 8, "mean"),
    "bound": lambda _: DoubleIndexing(route.SIZE, 8, "bound"),
    "learned": learned,
    "mixed": mixed,
}


def load_row(row: str) -> tuple[Qwen3ForCausalLM, csa2.Selector | None]:
    """A row's model in eval mode, and the selector of its CSA2 layers (None for
    rows without CSA2)."""
    base, _, selector = row.partition(":")
    if base == "qwen":
        model = qm.load()
        sparse = sparse_qwen.method(selector) if selector else None
        if (
            sparse is not None
        ):  # experiments/sparse_qwen.py rows, e.g. qwen:sparq-32-128
            attend.attach(model, sparse, int(os.environ.get("SPARSE_SCORED", 128)))
        return model, None
    model = qt.build(base, qt.final_stage(base)).eval()
    for param in model.parameters():  # parameters only: RoPE buffers stay float32
        param.data = param.data.to(torch.bfloat16)
    return model, SELECTORS[selector](model) if base == "csa2" else None


def scoped(
    model: Qwen3ForCausalLM, selector: csa2.Selector | None
) -> AbstractContextManager[None]:
    """selecting(model, Chunked(selector)) for CSA2 rows; nothing for the others."""
    return nullcontext() if selector is None else selecting(model, Chunked(selector))


@torch.no_grad()
def token_losses(model: Qwen3ForCausalLM, tokens: torch.Tensor) -> torch.Tensor:
    """(T,) next-token cross-entropy of a (1, T + 1) window, CHUNK logits at a time."""
    hidden = model.model(input_ids=tokens[:, :-1]).last_hidden_state[0]
    losses = [
        F.cross_entropy(
            model.lm_head(hidden[s : s + qt.CHUNK]).float(),
            tokens[0, s + 1 : s + 1 + qt.CHUNK],
            reduction="none",
        )
        for s in range(0, hidden.shape[0], qt.CHUNK)
    ]
    return torch.cat(losses)


def sample_limit() -> int | None:
    """LIMIT from the environment: samples per task and length, windows or books
    (smoke runs); None runs everything."""
    return int(os.environ["LIMIT"]) if "LIMIT" in os.environ else None


@torch.no_grad()
def argmax_hits(model: Qwen3ForCausalLM, ids: list[int], first: int) -> torch.Tensor:
    """(len(ids) - first,) whether each token from first on is the argmax given the
    tokens before it, in one forward; logits only for those positions."""
    tokens = torch.tensor([ids], device=model.lm_head.weight.device)
    hidden = model.model(input_ids=tokens[:, :-1]).last_hidden_state
    return model.lm_head(hidden[0, first - 1 :]).argmax(-1) == tokens[0, first:]


def greedy_matches(
    model: Qwen3ForCausalLM, prompt: list[int], answer: list[int]
) -> bool:
    """Whether greedy decoding after prompt starts with answer."""
    return bool(argmax_hits(model, prompt + answer, len(prompt)).all())


def answer_score(
    model: Qwen3ForCausalLM, tok: PreTrainedTokenizerBase, sample: dict[str, Any]
) -> float:
    """Best answer's share of values whose every token is the greedy argmax, the
    answer before them teacher-forced (qwen/tasks.py)."""
    best = 0.0
    context = sample["context"]
    for text, values in sample["answers"]:
        encoded = tok(
            context + text, add_special_tokens=False, return_offsets_mapping=True
        )
        offsets = encoded["offset_mapping"]
        spans, cursor = [], len(context)
        for value in values:
            start = (context + text).index(value, cursor)
            cursor = start + len(value)
            spans.append(
                [i for i, (a, b) in enumerate(offsets) if a < cursor and b > start]
            )
        first = spans[0][0]
        hits = argmax_hits(model, encoded["input_ids"], first).tolist()
        found = [all(hits[i - first] for i in span) for span in spans]
        best = max(best, sum(found) / len(found))
    return best


def long_tasks(row: str) -> None:
    """Every qwen/tasks.py task (TASKS: comma-separated subset), shortest first
    -> runs/qwen/longctx.csv, rows of row and task replaced."""
    limit = sample_limit()
    model, selector = load_row(row)
    tok = data.tokenizer()
    every = (*tasks.RULER, *tasks.BABILONG, *tasks.ICL)
    names = os.environ.get("TASKS", ",".join(every))
    with scoped(model, selector):
        for task in names.split(","):
            results = Results(OUT / "longctx.csv", {"row": row, "task": task})
            for sample in sorted(tasks.load(task), key=lambda s: s["length"]):
                if limit is not None and sample["index"] >= limit:
                    continue
                score = answer_score(model, tok, sample)
                keys = {k: sample[k] for k in ("task", "length", "index")}
                results.add({"row": row} | keys | {"score": score})
            logger.info("[%s] %s done", row, task)


# tasks and lengths of the generation check: needles, variable tracking, BABILong qa1
CHECKED = (*tasks.NEEDLES, "ruler_vt", "babilong_qa1")


def generation_check(row: str) -> None:
    """Real greedy generation, scored by lm-eval, on the first LIMIT (25) 4096-token
    samples of CHECKED; cached generation, so for rows without CSA2 only
    -> runs/qwen/generation_check.csv, to compare with longctx.csv."""
    import lm_eval

    model, selector = load_row(row)
    assert selector is None, "generation with a KV cache: rows without CSA2 only"
    limit = sample_limit() or 25
    results = Results(OUT / "generation_check.csv", {"row": row})
    lm_model = HFLM(pretrained=model, tokenizer=data.tokenizer(), batch_size=1)
    for task in CHECKED:
        length = tasks.LENGTHS[0]
        metadata: dict[str, Any] = {"max_seq_lengths": [length], "tokenizer": qm.QWEN}
        if task in tasks.BABILONG:
            metadata = {"max_seq_lengths": f"{length // 1024}k"}
        output = lm_eval.simple_evaluate(
            model=lm_model, tasks=[task], limit=limit, metadata=metadata
        )
        assert output is not None
        for index, sample in enumerate(output["samples"][task]):
            score = sample.get(str(length), sample.get("acc"))
            results.add(
                {"row": row, "task": task, "length": length, "index": index}
                | {"generated": score}
            )
        logger.info("[%s] %s checked", row, task)


class UncachedHFLM(HFLM):
    """lm-eval's HF model, generating without a KV cache (CSA2 has none)."""

    def _model_generate(
        self, context: torch.Tensor, max_length: int, stop: list[str], **kwargs: Any
    ) -> torch.Tensor:
        """HFLM._model_generate with use_cache=False."""
        generate = self.model.generate
        self.model.generate = lambda **kw: generate(**(kw | {"use_cache": False}))
        try:
            return super()._model_generate(context, max_length, stop, **kwargs)
        finally:
            del self.model.generate


# lm-eval task groups: tasks, shots, batch size, samples per task (None: all).
# Generation runs one sequence at a time (CSA2 ignores padding masks and HFLM pads
# generation on the left) and uncached, so its tasks are subsampled.
SHORT: tuple[tuple[tuple[str, ...], int, int, int | None], ...] = (
    (("mmlu",), 5, 4, None),  # 5-shot prompts: 4 x 2k tokens x 152k logits
    (("hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande"), 0, 16, None),
    (("lambada_openai",), 0, 16, None),
    (("swde", "fda", "squad_completion"), 0, 1, 500),
)


def short(row: str) -> None:
    """lm-eval-harness tasks (SHORT; LIMIT=n samples per task for smoke runs)
    -> runs/qwen/short.csv, rows of row and task replaced."""
    import lm_eval

    model, selector = load_row(row)
    smoke = sample_limit()
    with scoped(model, selector):
        for names, shots, batch, limit in SHORT:
            lm_model = UncachedHFLM(
                pretrained=model, tokenizer=data.tokenizer(), batch_size=batch
            )
            output = lm_eval.simple_evaluate(
                model=lm_model,
                tasks=list(names),
                num_fewshot=shots,
                limit=smoke or limit,
            )
            assert output is not None
            for task, metrics in output["results"].items():
                results = Results(OUT / "short.csv", {"row": row, "task": task})
                for key, value in metrics.items():
                    metric, _, kind = key.partition(",")
                    if metric.endswith("_stderr") or not isinstance(value, float):
                        continue
                    stderr = metrics.get(f"{metric}_stderr,{kind}")
                    results.add(
                        {"row": row, "task": task, "metric": metric, "shots": shots}
                        | {"value": value, "stderr": stderr}
                    )
            logger.info("[%s] %s done", row, ", ".join(names))


@torch.no_grad()
def lm(row: str) -> None:
    """WikiText windows and PG-19 books for one row; see the module docstring."""
    model, selector = load_row(row)
    measured = None if selector is None else MassShare(selector)
    errors = np.zeros(len(model.model.layers))
    windows = Results(OUT / "lm.csv", {"row": row})
    limit = sample_limit()
    reads = {}  # indexer reads per sequence (flat for selectors other than routes)
    if selector is not None:
        routes = selector.layers if isinstance(selector, PerLayer) else selector
        for length in (SEQ, LONG):
            reads[f"reads_{length // 1024}k"] = reads_per_seq(model, length, routes)
    for i, batch in enumerate(train.windows(data.load("validation"), SEQ, 1, limit)):
        entry: dict[str, object] = {"row": row, "window": i} | reads
        with scoped(model, measured):
            if row == "qwen":
                entry |= {"ce": float(token_losses(model, batch).mean()), "kl": 0.0}
            else:
                ce, kl = qt.lm_losses(model, batch).tolist()
                entry |= {"ce": ce, "kl": kl}
        if selector is not None:
            with scoped(model, selector):
                io = qt.teacher_io(model, batch[:, :-1])
                errors += torch.stack(qt.transfer_errors(model, io)).cpu().numpy()
        windows.add(entry)
    if measured is not None:
        layers = Results(OUT / "layers.csv", {"row": row})
        for layer, (picked, best, total) in sorted(measured.sums.items()):
            share = {"picks_share": picked / total, "oracle_share": best / total}
            layers.add({"row": row, "layer": layer} | share)
        for layer, error in enumerate(errors / (i + 1)):
            layers.add({"row": row, "layer": layer, "error": error})
    logger.info("[%s] wikitext done", row)
    books = Results(OUT / "pg19.csv", {"row": row})
    with scoped(model, selector):
        long_books = [b for b in data.books() if len(b) > LONG][:limit]
        for i, book in enumerate(long_books):
            losses = token_losses(model, train.to_device(book[None, : LONG + 1]))
            for start, end in zip(BUCKETS, BUCKETS[1:], strict=False):
                mean = float(losses[start:end].mean())
                books.add(
                    {"row": row, "book": i, "start": start, "end": end, "ce": mean}
                )
    logger.info("[%s] pg19 done", row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_float32_matmul_precision("high")
    study, rows = sys.argv[1], sys.argv[2:]
    studies = {"lm": lm, "long": long_tasks, "short": short, "check": generation_check}
    for name in rows:
        studies[study](name)
