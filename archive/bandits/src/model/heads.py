"""Real attention heads of the trained softmax LM (runs/train/softmax_seq1024.pkl)."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from model.train import RUNS, load, load_params
from model.transformer import ForwardOptions, add_mlp, embed, layer_norm, mix

SEQ_LEN, N_HEADS = 1024, 8
LAYERS, HEADS = (0, 3, 7), (0, 2, 4, 6)
WEIGHTS = RUNS / "softmax_seq1024.pkl"
TOKEN_OFFSET = 100_000  # validation tokens skipped before the sample


class Head(NamedTuple):
    """One head on one validation window, in float64.

    Attributes:
        layer: Layer index.
        head: Head index.
        q: (T, d) queries.
        k: (T, d) keys.
        v: (T, d) values.
    """

    layer: int
    head: int
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray


def get_heads() -> list[Head]:
    """Heads HEADS of layers LAYERS on SEQ_LEN validation tokens.

    Returns:
        One Head per (layer, head).

    Raises:
        FileNotFoundError: If WEIGHTS or the validation split is missing.
    """
    params = load_params(WEIGHTS)
    tokens = np.asarray(load("validation")[TOKEN_OFFSET : TOKEN_OFFSET + SEQ_LEN])
    options = ForwardOptions(n_heads=N_HEADS)
    x = embed(params, jnp.asarray(tokens, jnp.int32))
    heads = []
    for index, layer in enumerate(params["layers"]):
        normed = layer_norm(layer["ln1"], x)
        if index in LAYERS:
            qkv = np.asarray(normed @ layer["qkv"], np.float64)
            q, k, v = (a.reshape(SEQ_LEN, N_HEADS, -1) for a in np.split(qkv, 3, -1))
            heads += [Head(index, h, q[:, h], k[:, h], v[:, h]) for h in HEADS]
        x = add_mlp(layer, x + mix(layer, normed, options))
    return heads
