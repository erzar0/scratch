"""Token files for the Qwen runs, in Qwen's tokenizer.

    data/qwen_wt103_{train,validation,test}.bin   uint32 ids, lines joined
    data/qwen_pg19_test.npz                       uint32 ids, one array per book

Prep once (downloads ~320 MB of WikiText parquet and the 100 PG-19 test books):
    uv run --extra prep python -m qwen.data
"""

import logging

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model import train
from qwen.model import QWEN

logger = logging.getLogger(__name__)

PG19 = ("emozilla/pg19-test", "data/test-00000-of-00001-29a571947c0b5ccc.parquet")


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


def load(split: str) -> np.ndarray:
    """Memory-mapped WikiText-103 ids of one split."""
    return np.memmap(train.DATA / f"qwen_wt103_{split}.bin", dtype=np.uint32, mode="r")


def books() -> list[np.ndarray]:
    """PG-19 test books, as ids."""
    with np.load(train.DATA / "qwen_pg19_test.npz") as archive:
        return [archive[name] for name in archive.files]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prep()
