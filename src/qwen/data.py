"""Token files for the Qwen runs, in Qwen's tokenizer.

    data/qwen_wt103_{train,validation,test}.bin   uint32 ids, lines joined
    data/qwen_pg19_test.npz                       uint32 ids, one array per book
    data/qwen_mix_train.bin                       uint32 ids, training mix:
        MIX_TOKENS of FineWeb-Edu (web pages, streamed) and as many of PG-19 train
        (books, one ~300 MB shard), documents shuffled, each ended by Qwen's EOS

Prep once (downloads ~320 MB of WikiText parquet, the 100 PG-19 test books, a
PG-19 train shard and streams FineWeb-Edu):
    uv run --extra prep python -m qwen.data [mix]
"""

import logging
import sys

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model import train
from qwen.model import QWEN

logger = logging.getLogger(__name__)

PG19 = ("emozilla/pg19-test", "data/test-00000-of-00001-29a571947c0b5ccc.parquet")
PG19_TRAIN = ("emozilla/pg19", "data/train-00000-of-00023-5263dc4323e881d2.parquet")
FINEWEB = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
MIX_TOKENS = 30_000_000  # per source


def tokenizer() -> PreTrainedTokenizerBase:
    """Qwen's tokenizer, without a length warning on whole books."""
    tok = AutoTokenizer.from_pretrained(QWEN)
    tok.model_max_length = 10**9
    return tok


def encode(tok: PreTrainedTokenizerBase, texts: list[str]) -> list[np.ndarray]:
    """uint32 ids of each text, no special tokens."""
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    return [np.array(t, np.uint32) for t in ids]


def prep() -> None:
    """Write the WikiText-103 splits and the PG-19 test books."""
    tok = tokenizer()
    train.DATA.mkdir(exist_ok=True)
    for split in train.SPLITS:
        ids = np.concatenate(
            [t for t in encode(tok, train.wikitext_lines(split)) if t.size]
        )
        ids.tofile(train.DATA / f"qwen_wt103_{split}.bin")
        logger.info("wikitext %s: %s tokens", split, f"{len(ids):,}")
    path = hf_hub_download(*PG19, repo_type="dataset")
    books = encode(tok, pl.read_parquet(path)["text"].to_list())
    np.savez(train.DATA / "qwen_pg19_test.npz", *books)
    logger.info(
        "pg19 test: %d books, %s tokens", len(books), f"{sum(map(len, books)):,}"
    )


def capped(docs: list[np.ndarray]) -> list[np.ndarray]:
    """The first docs holding MIX_TOKENS tokens."""
    total = np.cumsum([len(d) for d in docs])
    return docs[: int(np.searchsorted(total, MIX_TOKENS)) + 1]


def prep_mix() -> None:
    """Write the training mix: web pages and books, documents shuffled."""
    import datasets

    tok = tokenizer()
    web: list[np.ndarray] = []
    stream = datasets.load_dataset(*FINEWEB, split="train", streaming=True)
    for batch in stream.iter(batch_size=1000):
        web += encode(tok, batch["text"])
        if sum(map(len, web)) >= MIX_TOKENS:
            break
    path = hf_hub_download(*PG19_TRAIN, repo_type="dataset")
    books = encode(tok, pl.read_parquet(path)["text"].to_list())
    docs = capped(web) + capped(books)
    np.random.default_rng(0).shuffle(docs)  # type: ignore[arg-type]
    eos = np.array([tok.eos_token_id], np.uint32)
    ids = np.concatenate([part for doc in docs for part in (doc, eos)])
    ids.tofile(train.DATA / "qwen_mix_train.bin")
    logger.info("mix: %d documents, %s tokens", len(docs), f"{len(ids):,}")


def mix() -> np.ndarray:
    """Memory-mapped ids of the training mix."""
    return np.memmap(train.DATA / "qwen_mix_train.bin", dtype=np.uint32, mode="r")


def load(split: str) -> np.ndarray:
    """Memory-mapped WikiText-103 ids of one split."""
    return np.memmap(train.DATA / f"qwen_wt103_{split}.bin", dtype=np.uint32, mode="r")


def books() -> list[np.ndarray]:
    """PG-19 test books, as ids."""
    with np.load(train.DATA / "qwen_pg19_test.npz") as archive:
        return [archive[name] for name in archive.files]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if sys.argv[1:] == ["mix"]:
        prep_mix()
    else:
        prep()
