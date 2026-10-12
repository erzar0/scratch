"""Real attention heads (S = scaled logits, V) from the trained softmax LM runs/train/softmax_seq1024.pkl."""
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from transformer import layer_norm, mix

T, H, LAYERS, HEADS = 1024, 8, (0, 3, 7), (0, 2, 4, 6)


def get_heads(with_qk=False):
    """-> list of (layer, head, S (T,T) f64, V (T,64) f64[, Q, K])."""
    params = pickle.load(open("runs/train/softmax_seq1024.pkl", "rb"))
    toks = np.fromfile("data/wt103_validation.bin", np.uint16, T, offset=2 * 100_000).astype(np.int32)
    x = params["tok_emb"][toks] + params["pos_emb"][:T]
    heads = []
    for l, p in enumerate(params["layers"]):
        a = layer_norm(p["ln1"], x)
        if l in LAYERS:
            q, k, v = (np.asarray(t, np.float64).reshape(T, H, -1) for t in jnp.split(a @ p["qkv"], 3, -1))
            heads += [(l, h, q[:, h] @ k[:, h].T / np.sqrt(q.shape[-1]), v[:, h]) + ((q[:, h], k[:, h]) if with_qk else ())
                      for h in HEADS]
        x = x + mix(p, a, H)
        x = x + jax.nn.gelu(layer_norm(p["ln2"], x) @ p["fc1"]) @ p["fc2"]
    return heads
