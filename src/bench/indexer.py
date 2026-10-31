"""Long-context cost of CSA2's select step: the flat indexer (scores every main KV
entry) and double indexing (deepseek/double_indexing.py, its pick fast path).

Indexer sized like the trained models (4 heads x 32, m = 4, top_k 8), bf16 inputs,
float32 scores as the HF scorer computes them, random queries, weights and keys (the
cost does not depend on them); every pick function torch.compile'd, decode timed as
a CUDA graph replay (bench/timing.py).

    decode   picks for one query per sequence, 64 sequences at once, each with its
             own cached indexer keys (and summary keys) of T tokens. Query
             projections are excluded: the same for all but the learned router.
    prefill  picks for all T queries, CHUNK queries at a time, so flat is O(T)
             memory too: the fair baseline, not a (T, T/m) score matrix. Summary
             keys are computed inside.
    kib      indexer keys read per decoded token, bf16.

    uv run python -m bench.indexer  -> runs/bench/indexer.csv
"""

import logging
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from bench.timing import graphed, measure
from deepseek import csa, csa2
from deepseek.double_indexing import (
    DoubleIndexing,
    Router,
    pick,
    query_reads,
    summary_keys,
    superblocks,
)
from experiments.results import Results

logger = logging.getLogger(__name__)

OUT = Path("runs/bench/indexer.csv")
HEADS, DIM, TOP_K = csa.SMALL["index_n_heads"], csa.SMALL["index_head_dim"], 8
M = csa.SMALL["compress_rates"][csa.CSA]
BATCH, CHUNK, SCALE = 64, 1024, DIM**-0.5
LENGTHS = [4096, 16384, 32768, 65536, 131072]
type Method = DoubleIndexing | None
METHODS: list[Method] = [
    None,
    DoubleIndexing(16, 4, "learned"),
    DoubleIndexing(16, 32, "bound"),
    DoubleIndexing(64, 4, "learned"),
]
DEVICE = torch.device("cuda")


def name_of(method: Method) -> str:
    """CSV label of a method."""
    if method is None:
        return "flat"
    return f"{method.coarse} S={method.superblock_size} s={method.kept_superblocks}"


def flat_pick(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    finished: torch.Tensor,
) -> torch.Tensor:
    """The flat indexer's top-k entries: every finished entry scored."""
    per_head = F.relu(torch.einsum("bthc,bnc->bthn", q.float(), index_k.float()))
    score = torch.einsum("bth,bthn->btn", weights, per_head) * SCALE
    future = torch.arange(index_k.shape[1], device=q.device) >= finished
    return score.masked_fill(future, float("-inf")).topk(TOP_K, dim=-1).indices


def routed_pick(method: DoubleIndexing) -> Callable[..., torch.Tensor]:
    """pick's entries for method, from precomputed summary keys."""

    def run(
        q: torch.Tensor,
        weights: torch.Tensor,
        index_k: torch.Tensor,
        finished: torch.Tensor,
        summaries: torch.Tensor,
    ) -> torch.Tensor:
        found = pick(q, weights, index_k, summaries, finished, TOP_K, method, SCALE)
        return found.entry

    return run


def summaries_of(
    method: DoubleIndexing, index_k: torch.Tensor, router: Router | None
) -> torch.Tensor:
    """Summary keys of index_k (B, n, c), bf16."""
    blocks = superblocks(index_k, method.superblock_size)
    return summary_keys(blocks, method.coarse, router).to(index_k.dtype)


def router_for(method: Method) -> Router | None:
    """A fresh learned router, bf16 (its pooling costs the same once trained)."""
    if method is None or method.coarse != "learned":
        return None
    router = Router(csa2.small_config(), method.superblock_size)
    return router.to(DEVICE, torch.bfloat16)


def draw(*shape: int) -> torch.Tensor:
    """Random bf16 tensor on the GPU."""
    return torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)


@torch.no_grad()
def decode_us(seq_len: int, method: Method) -> float:
    """µs per decode step's picks for BATCH sequences."""
    n_entries = seq_len // M
    q, weights = draw(BATCH, 1, HEADS, DIM), draw(BATCH, 1, HEADS).float()
    index_k = draw(BATCH, n_entries, DIM)
    finished = torch.full((BATCH, 1, 1), n_entries, device=DEVICE)
    if method is None:
        fn = graphed(torch.compile(flat_pick), q, weights, index_k, finished)
        return measure(fn).ms * 1e3
    summaries = summaries_of(method, index_k, router_for(method))
    fn = graphed(
        torch.compile(routed_pick(method)), q, weights, index_k, finished, summaries
    )
    return measure(fn).ms * 1e3


@torch.no_grad()
def prefill(seq_len: int, method: Method) -> tuple[float, float]:
    """(ms, peak GiB) of the picks of all seq_len queries."""
    n_entries = seq_len // M
    q, weights = draw(1, seq_len, HEADS, DIM), draw(1, seq_len, HEADS).float()
    index_k = draw(1, n_entries, DIM)
    router = router_for(method)
    chunk_pick = torch.compile(flat_pick if method is None else routed_pick(method))

    def run() -> None:
        extra = () if method is None else (summaries_of(method, index_k, router),)
        for start in range(0, seq_len, CHUNK):
            rows = slice(start, start + CHUNK)
            positions = torch.arange(start, start + CHUNK, device=DEVICE)
            finished = ((positions + 1) // M)[None, :, None]
            chunk_pick(q[:, rows], weights[:, rows], index_k, finished, *extra)

    return tuple(measure(run))  # type: ignore[return-value]


def kib_per_token(seq_len: int, method: Method) -> float:
    """Indexer keys read for the last token of a seq_len context, in KiB."""
    keys = query_reads(torch.tensor(seq_len // M), method)
    return float(keys) * DIM * 2 / 1024


def main() -> None:
    """Time every method at every length."""
    torch._dynamo.config.cache_size_limit = 64  # one graph per (T, method)
    results = Results(OUT, {})
    for seq_len in LENGTHS:
        for method in METHODS:
            us = decode_us(seq_len, method)
            ms, gib = prefill(seq_len, method)
            row = {"method": name_of(method), "T": seq_len, "decode_us_batch64": us}
            row |= {"prefill_index_ms": ms, "prefill_peak_gib": gib}
            row |= {"indexer_kib_per_token": kib_per_token(seq_len, method)}
            logger.info("%s", row)
            results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
