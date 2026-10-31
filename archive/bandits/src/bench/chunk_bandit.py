"""Forward time of the chunk bandit (attention/dense/chunk_bandit.py chunk_sparse, split
kernel) vs cuDNN flash: one layer of 8 heads, head dim 64, bf16. One head is too
small a call to time reliably under WSL.

    uv run python -m bench.chunk_bandit  -> runs/bench/chunk_bandit.csv
"""

import logging
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from attention.dense.chunk_bandit import ChunkBanditConfig, chunk_sparse
from bench.timing import measure
from experiments.results import Results

logger = logging.getLogger(__name__)

OUT = Path("runs/bench/chunk_bandit.csv")
N_HEADS, HEAD_DIM = 8, 64
LENGTHS = [4096, 8192, 16384, 32768]
SHAPES = [(32, 8), (64, 4), (64, 8), (64, 16)]  # (chunk, refined_chunks)
KEY = jax.random.PRNGKey(0)


def flash(q: jax.Array, k: jax.Array, v: jax.Array, _: jax.Array) -> jax.Array:
    """(heads, T, d) causal attention via cuDNN."""
    out = jax.nn.dot_product_attention(
        *(a.transpose(1, 0, 2)[None] for a in (q, k, v)),
        is_causal=True,
        implementation="cudnn",
    )
    return out[0].transpose(1, 0, 2)


def main() -> None:
    """Time flash and every chunk config at every length."""
    fns = {"flash": flash}
    for chunk, refined in SHAPES:
        for selector in ("bandit", "random"):
            config = ChunkBanditConfig(selector, chunk, refined, impl="split")  # type: ignore[arg-type]
            head = partial(chunk_sparse, config=config)
            fns[f"{selector} c{chunk} r{refined}"] = jax.vmap(head, (0, 0, 0, None))
    results = Results(OUT, {})
    for seq_len in LENGTHS:
        q, k, v = jax.random.normal(KEY, (3, N_HEADS, seq_len, HEAD_DIM), jnp.bfloat16)
        for name, fn in fns.items():
            timing = measure(fn, q, k, v, KEY)
            row = {
                "config": name,
                "T": seq_len,
                "ms": timing.ms,
                "temp_gib": timing.temp_gib,
            }
            logger.info("%s", row)
            results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
