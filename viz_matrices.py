"""Attention matrices of each method on one real head (v = identity returns the matrix itself)."""
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
from heads import get_heads
from attn_methods_jax import METHODS, exact_causal
L, H, S, V, Q, K = get_heads(with_qk=True)[5]
n = 256
q, k = jnp.asarray(Q[:n]), jnp.asarray(K[:n]); I = jnp.eye(n); key = jax.random.PRNGKey(0)
cfg = dict(sampled=dict(m=64), nystrom=dict(m=64), lowrank=dict(m=64), ppr=dict(), sparse_softmax=dict(),
           stationary=dict(), bandit=dict(z=2.0), bitonic=dict(topk=16, tau=0.1, pool=64), active=dict())
P0 = np.asarray(exact_causal(q, k, I))
mats = {"exact": P0} | {m: np.asarray(f(q, k, I, key, **cfg[m])) for m, f in METHODS.items()}
fig, axs = plt.subplots(2, 5, figsize=(20, 8.4))
for ax, (m, P) in zip(axs.ravel(), mats.items()):
    e = np.linalg.norm(P - P0) / np.linalg.norm(P0)
    ax.imshow(np.log10(np.maximum(P, 1e-6)), cmap="viridis", vmin=-6, vmax=0)
    ax.set_title(m + ("" if m == "exact" else f"  err {e:.2f}"), fontsize=11); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"attention matrices, layer {L} head {H}, first {n} tokens, log10 weight"); fig.tight_layout(h_pad=2.5, rect=(0, 0, 1, 0.96))
fig.savefig("runs/approx/matrices.png", dpi=110)
