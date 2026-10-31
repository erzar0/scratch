"""Cut from model/transformer.py when the bandits were archived (frozen, not run).

The "bandit" mixer: softmax weights, with each head's softmax replaced by the per-key
bandit (attention/dense/key_bandit.py). transformer.py had:

    MIXERS = ("softmax", "csa", "bandit")
    ForwardOptions.head_attention: HeadAttention | None = None
    init_params(mixer="bandit"): every layer gets an empty "bandit" dict as a marker
    mix(): head = _bandit_head if "bandit" in layer else options.head_attention;
           if head is not None: jax.checkpoint(approx_attention, ...)(layer, x,
           options.n_heads, head)

and attention/softmax.py had the per-head hook:

    type HeadAttention = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]

    def approx_attention(params, x, n_heads, head_attention):
        q, k, v = split_heads(params, x, n_heads)
        return _merge_heads(params, jax.vmap(head_attention)(q, k, v))
"""

import jax

from attention.dense.key_bandit import KeyBanditConfig, key_bandit


def _bandit_head(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """The bandit mixer's head: fixed channel order, so every step reads alike."""
    return key_bandit(q, k, v, jax.random.PRNGKey(0), KeyBanditConfig(ucb_width=2.0))
