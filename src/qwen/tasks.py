"""Long-context samples for the Qwen rows, from lm-eval-harness tasks.

A sample is lm-eval's prompt plus answer texts that follow it, each with the values
inside it that are scored (evaluate.answer_score). The answer copies the format
of the task's own examples:

    RULER needles   " v1, v2, ..."  after "... is" / "... are"; every value scored
    RULER vt        " v1 v2 ..."; cwe " 1. w1 2. w2 ..."; fwe " w1, w2, w3"
    RULER qa        one answer per accepted string; the best one counts
    BABILong qa1-3  the examples' sentence up to the location, then " location."
    HELMET ICL      " label" and the demonstrations' separator; label and separator
                    scored, so a longer label (12 for 1) is wrong

Multi-value answers keep RULER's gold order (approximation: a model listing the
same values in another order scores lower; the same for every row).

HELMET many-shot in-context learning follows HELMET's data.py (prompt, labels
mapped to shuffled integers per sample, label-balanced demonstrations), except:
demonstrations fill each length instead of a fixed count per length, the shorter
lengths' demonstrations are the first of the longer ones', and Banking77 comes from
the mteb/banking77 copy. HELMET's fifth task, NLU, is skipped: its dataset is no
longer on the Hugging Face Hub.

    uv run python -m qwen.tasks    -> data/longctx/<task>.jsonl (CPU only)
"""

import hashlib
import json
import logging
import math
import random
import re
from collections import defaultdict
from itertools import accumulate
from pathlib import Path
from typing import Any

from model import train
from qwen import data
from qwen.model import QWEN

logger = logging.getLogger(__name__)

LENGTHS = (4096, 8192, 16384, 32768)
PER_LENGTH = 50
OUT = train.DATA / "longctx"
NEEDLES = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
)
RULER = (
    *NEEDLES,
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_squad",
    "ruler_qa_hotpot",
)
# BABILong question -> the examples' answer sentence up to the location
BABILONG = {
    "babilong_qa1": (r"Where is (.+)\?", "The most recent location of {0} is"),
    "babilong_qa2": (r"Where is the (.+)\?", "The {0} is in the"),
    "babilong_qa3": (
        r"Where was the (.+) before the (.+)\?",
        "Before the {1} the {0} was in the",
    ),
}

# HELMET ICL: dataset, revision, config, label field, test split
ICL = {
    "icl_trec_coarse": ("CogComp/trec", "refs/convert/parquet", None, "coarse_label"),
    "icl_trec_fine": ("CogComp/trec", "refs/convert/parquet", None, "fine_label"),
    "icl_banking77": ("mteb/banking77", None, None, "label"),
    "icl_clinic150": ("clinc/clinc_oos", None, "plus", "intent"),
}
ICL_TEST = {"icl_clinic150": "validation"}  # HELMET's test split, if not "test"
ICL_HEADER = (
    "Use the provided mapping from the text to label to assign a label to the "
    'text. Only output "label: {label}" and nothing else. '
)
SEED = 42

type Answer = tuple[str, list[str]]  # answer text, the values in it that are scored


def ruler_answers(task: str, values: list[str]) -> list[Answer]:
    """Answer texts of a RULER sample in the format of the task's examples."""
    if task.startswith("ruler_qa"):
        return [(f" {v}", [v]) for v in dict.fromkeys(values)]
    if task == "ruler_vt":
        text = " " + " ".join(values)
    elif task == "ruler_cwe":
        text = " " + " ".join(f"{i}. {w}" for i, w in enumerate(values, 1))
    else:
        text = " " + ", ".join(values)
    return [(text, values)]


def babilong_answers(task: str, question: str, target: str) -> list[Answer]:
    """The answer sentence of a BABILong sample, scored on its location."""
    pattern, sentence = BABILONG[task]
    match = re.search(pattern, question)
    assert match is not None, question
    return [(f" {sentence.format(*match.groups())} {target}.", [target])]


def build(task: str, length: int) -> list[dict[str, Any]]:
    """PER_LENGTH samples of one task at one length."""
    from lm_eval.tasks import TaskManager, get_task_dict

    if task in BABILONG:
        metadata: dict[str, Any] = {"max_seq_lengths": f"{length // 1024}k"}
    else:
        metadata = {"max_seq_lengths": [length], "tokenizer": QWEN}
    lm_task = get_task_dict([task], TaskManager(metadata=metadata))[task]
    lm_task.build_all_requests(limit=PER_LENGTH, rank=0, world_size=1)
    samples = []
    for index, instance in enumerate(lm_task.instances):
        doc = instance.doc
        if task in BABILONG:
            answers = babilong_answers(task, doc["question"], doc["target"])
        else:
            answers = ruler_answers(task, list(doc["outputs"]))
        samples.append(
            {
                "task": task,
                "length": length,
                "index": index,
                "context": instance.args[0],
                "answers": answers,
            }
        )
    return samples


def balanced(
    items: list[dict[str, Any]], field: str, count: int, rand: random.Random
) -> list[dict[str, Any]]:
    """count items in rounds of one per label, each round shuffled (HELMET's
    balance_labels; a label short of items repeats some)."""
    by_label = defaultdict(list)
    for item in items:
        by_label[item[field]].append(item)
    rounds = math.ceil(count / len(by_label))
    drawn = [
        rand.sample(group, min(rounds, len(group)))
        + rand.choices(group, k=max(0, rounds - len(group)))
        for group in by_label.values()
    ]
    ordered = []
    for i in range(rounds):
        block = [labelled[i] for labelled in drawn]
        rand.shuffle(block)
        ordered += block
    return ordered[:count]


def icl(task: str) -> list[dict[str, Any]]:
    """PER_LENGTH label-balanced test texts at every length, each after as many
    demonstrations as fit (HELMET's prompt; see the module docstring)."""
    import datasets

    name, revision, config, field = ICL[task]
    splits = datasets.load_dataset(name, config, revision=revision)
    demos = list(splits["train"])
    n_labels = len(set(d[field] for d in demos))
    test = list(splits[ICL_TEST.get(task, "test")])
    random.Random(SEED).shuffle(test)
    tok = data.tokenizer()
    texts = [d["text"] for d in demos]
    text_tokens = dict(zip(texts, map(len, tok(texts)["input_ids"]), strict=True))
    samples = []
    for index, item in enumerate(
        balanced(test, field, PER_LENGTH, random.Random(SEED))
    ):
        digest = int(hashlib.sha256(item["text"].encode()).hexdigest(), 16)
        rand = random.Random((digest + SEED) % 2**31)
        mapping = list(range(n_labels))
        rand.shuffle(mapping)
        chosen = balanced(demos, field, LENGTHS[-1] // 4, rand)
        lines = [f"{d['text']}\nlabel: {mapping[d[field]]}" for d in chosen]
        cost = [text_tokens[d["text"]] + 6 for d in chosen]  # + label, separator
        label = mapping[item[field]]
        question = f"\n\n{item['text']}\nlabel:"
        for length in LENGTHS:
            budget = length - 64 - len(tok(ICL_HEADER + question)["input_ids"])
            shots = sum(1 for total in accumulate(cost) if total <= budget)
            while True:
                context = ICL_HEADER + "\n\n" + "\n\n".join(lines[:shots]) + question
                if len(tok(context)["input_ids"]) <= length - 16:
                    break
                shots = int(shots * 0.95)
            answers = [(f" {label}\n\n", [f"{label}\n\n"])]
            samples.append(
                {"task": task, "length": length, "index": index, "shots": shots}
                | {"context": context, "answers": answers}
            )
    return samples


def load(task: str) -> list[dict[str, Any]]:
    """The cached samples of a task, every length."""
    with (OUT / f"{task}.jsonl").open() as lines:
        return [json.loads(line) for line in lines]


def main() -> None:
    """Write every task's samples, skipping tasks already cached."""
    OUT.mkdir(parents=True, exist_ok=True)
    for task in (*RULER, *BABILONG, *ICL):
        path = OUT / f"{task}.jsonl"
        if path.exists():
            continue
        if task in ICL:
            samples = icl(task)
        else:
            samples = [s for length in LENGTHS for s in build(task, length)]
        partial = Path(f"{path}.partial")
        partial.write_text("".join(json.dumps(s) + "\n" for s in samples))
        partial.rename(path)
        logger.info("%s: %d samples", task, len(samples))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
