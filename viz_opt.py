"""Before/after wall time of the optimised methods (before = values from runs/approx/methods_timing.csv)."""
import time, jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from heads import get_heads
from attn_methods_jax import stationary, bandit, active
_, _, S, V, Q, K = get_heads(with_qk=True)[5]
key = jax.random.PRNGKey(0)
def ms(f, *a, n=30):
    jax.block_until_ready(f(*a)); t0 = time.perf_counter()
    for _ in range(n): o = f(*a)
    jax.block_until_ready(o); return (time.perf_counter() - t0) / n * 1e3
before = {("stationary", 1024): 2.83, ("stationary", 4096): 12.1, ("active", 1024): 2.63, ("active", 4096): 38.7,
          ("bandit", 1024): 0.36, ("bandit", 4096): 5.49}
after = {}
for T in (1024, 4096):
    q, k, v = (jnp.tile(jnp.asarray(x, jnp.float32), (T // 1024, 1)) for x in (Q, K, V))
    for name, f in dict(stationary=stationary, bandit=bandit, active=active).items():
        after[name, T] = ms(jax.jit(f), q, k, v, key)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, T in zip(axs, (1024, 4096)):
    x = np.arange(3)
    ax.bar(x - .2, [before[m, T] for m in ("stationary", "bandit", "active")], .4, label="before")
    ax.bar(x + .2, [after[m, T] for m in ("stationary", "bandit", "active")], .4, label="after")
    ax.set_xticks(x, ["stationary", "bandit", "active"]); ax.set_yscale("log"); ax.set_title(f"T={T}"); ax.set_ylabel("ms")
    for i, m in enumerate(("stationary", "bandit", "active")):
        ax.text(i + .2, after[m, T], f"{before[m, T] / after[m, T]:.1f}x", ha="center", va="bottom", fontsize=9)
axs[0].legend(); fig.tight_layout(); fig.savefig("runs/approx/optimized.png", dpi=120)
print({k: round(v, 2) for k, v in after.items()})
