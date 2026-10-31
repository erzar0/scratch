"""E4: the trained softmax LM with every head's attention swapped for the chunk
bandit (attention/dense/chunk_bandit.py chunk_attn), against oracle and random chunk
picks and (T = 4096) sliding windows: val loss and KL(exact model || swapped model) over
next-token distributions, on the first EVAL_WINDOWS validation windows. delta_se is
the paired standard error over batches; reads_ratio counts q·k dot products per query
(attention/dense/chunk_bandit.py chunk_reads) relative to exact causal attention.

    SEQ=1024 BATCH=4 uv run python -m experiments.eval_chunk_bandit [smoke]
    SEQ=4096 BATCH=1 uv run python -m experiments.eval_chunk_bandit
    -> runs/chunk_bandit/chunk_loss.csv (rows for SEQ replaced)
"""

import dataclasses
import logging
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from attention.csa import Params
from attention.dense.chunk_bandit import ChunkBanditConfig, chunk_attn, chunk_reads
from attention.dense.key_bandit import exact_causal
from experiments.results import Results, paired_delta
from model import train
from model.transformer import ForwardOptions, forward

logger = logging.getLogger(__name__)

OUT = Path("runs/chunk_bandit/chunk_loss.csv")
EVAL_WINDOWS = 64
SHAPES = {  # (chunk, refined_chunks)
    1024: [(16, 2), (16, 4), (16, 8), (32, 2), (32, 4), (32, 8)],
    4096: [(32, 4), (32, 8), (32, 16), (64, 4), (64, 8), (64, 16)],
}
VARIANT_SHAPE = {1024: (32, 4), 4096: (64, 8)}
PULLS = [
    (1, 4, 1.0),
    (2, 8, 0.0),
    (2, 8, 2.0),
    (4, 16, 1.0),
]  # (first, second, ucb_width)
WINDOWS = (256, 512, 1024)


@dataclasses.dataclass(frozen=True)
class ChunkHead:
    """chunk_attn as a head attention (fixed key), hashable for jit."""

    config: ChunkBanditConfig

    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        return chunk_attn(q, k, v, jax.random.PRNGKey(0), self.config)


@dataclasses.dataclass(frozen=True)
class WindowHead:
    """Exact softmax over the last `size` tokens."""

    size: int

    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        heads = (a[None, :, None] for a in (q, k, v))
        out = jax.nn.dot_product_attention(
            *heads,
            is_causal=True,
            local_window_size=(self.size - 1, 0),
            implementation="xla",
        )
        return out[0, :, 0]


def exact_head(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    return exact_causal(q, k, v)


def token_kl(exact: jax.Array, approx: jax.Array, tok_emb: jax.Array) -> jax.Array:
    """KL(exact || approx) of next-token distributions per position, 512 at a time."""

    def one(pair: tuple[jax.Array, jax.Array]) -> jax.Array:
        log_p = jax.nn.log_softmax(pair[0] @ tok_emb.T)
        log_q = jax.nn.log_softmax(pair[1] @ tok_emb.T)
        return (jnp.exp(log_p) * (log_p - log_q)).sum(-1)

    dim = exact.shape[-1]
    return jax.lax.map(one, (exact.reshape(-1, 512, dim), approx.reshape(-1, 512, dim)))


@partial(jax.jit, static_argnames="options")
def loss_and_kl(
    params: Params, batch: jax.Array, options: ForwardOptions
) -> tuple[jax.Array, jax.Array]:
    """Mean loss of the swapped model and mean KL from the exact one."""
    options = dataclasses.replace(options, return_hidden=True)
    hidden_of = jax.vmap(lambda o, t: forward(params, t, o), in_axes=(None, 0))
    exact = hidden_of(
        dataclasses.replace(options, head_attention=exact_head), batch[:, :-1]
    )
    approx = hidden_of(options, batch[:, :-1])
    loss = train.token_nll(approx, params["tok_emb"], batch[:, 1:]).mean()
    return loss, token_kl(exact, approx, params["tok_emb"]).mean()


def heads(seq_len: int) -> dict[str, object]:
    """Name -> head attention for every row of this T."""
    found: dict[str, object] = {"exact": exact_head}
    if seq_len >= 4096:
        found |= {f"window {w}": WindowHead(w) for w in WINDOWS}
    for chunk, refined in SHAPES[seq_len]:
        for selector in ("bandit", "oracle", "random"):
            config = ChunkBanditConfig(selector, chunk, refined)  # type: ignore[arg-type]
            found[f"{selector} c{chunk} r{refined}"] = ChunkHead(config)
    chunk, refined = VARIANT_SHAPE[seq_len]
    for first, second, width in PULLS:
        config = ChunkBanditConfig(
            "bandit", chunk, refined, None, first, second, None, width
        )
        found[f"bandit c{chunk} r{refined} pulls {first}+{second} z{width:g}"] = (
            ChunkHead(config)
        )
    return found


def describe(head: object) -> dict[str, object]:
    """Result columns of a head attention."""
    if isinstance(head, ChunkHead):
        c = head.config
        return {
            "selector": c.selector,
            "chunk": c.chunk,
            "refined_chunks": c.refined_chunks,
            "first_pulls": c.first_pulls,
            "second_pulls": c.second_pulls,
            "ucb_width": c.ucb_width,
        }
    if isinstance(head, WindowHead):
        return {"selector": "window", "window": head.size}
    return {"selector": "exact"}


def reads_ratio(head: object, seq_len: int) -> float:
    """q·k dot products per query relative to exact causal attention's (T + 1) / 2."""
    exact = (seq_len + 1) / 2
    if isinstance(head, ChunkHead):
        return chunk_reads(seq_len, head.config) / exact
    if isinstance(head, WindowHead):
        return float(np.minimum(np.arange(1, seq_len + 1), head.size).mean()) / exact
    return 1.0


def main(run: train.TrainConfig, smoke: bool) -> None:
    """Evaluate every head attention; see the module docstring."""
    params = train.load_params(train.run_name("softmax", run).with_suffix(".pkl"))
    val = train.load("validation")
    limit = 1 if smoke else EVAL_WINDOWS // run.batch
    found = heads(run.seq_len)
    if smoke:
        found = dict(list(found.items())[:2])
    results = Results(OUT, {"T": run.seq_len})
    exact_losses = None  # per batch; "exact" is the first head
    for name, head in found.items():
        options = ForwardOptions(n_heads=run.n_heads, head_attention=head)  # type: ignore[arg-type]
        pairs = np.asarray(
            [
                loss_and_kl(params, batch, options)
                for batch in train.windows(val, run.seq_len, run.batch, limit)
            ],
            np.float64,
        )
        if exact_losses is None:
            exact_losses = pairs[:, 0]
        delta, delta_se = paired_delta(pairs[:, 0], exact_losses)
        loss, kl = pairs.mean(0)
        row = {
            "T": run.seq_len,
            "name": name,
            **describe(head),
            "reads_ratio": reads_ratio(head, run.seq_len),
            "val_loss": loss,
            "delta_vs_exact": delta,
            "delta_se": delta_se,
            "kl": kl,
        }
        logger.info("%s", row)
        results.add(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(train.TrainConfig.from_env(), "smoke" in sys.argv)
