"""Forward time of the chunk-level methods vs flash: one layer of 8 heads, head dim 64, bf16.
One head is too small a call (0.1-0.5 ms) to time reliably under WSL.
    .env/bin/python bench_fast.py   -> runs/approx/fast_timing.csv"""
import csv, os, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from functools import partial
import jax, jax.numpy as jnp
from attn_fast import chunk_attn

key = jax.random.PRNGKey(0)


def bench(f, *args, reps=7):
    """-> (min ms over reps, temp GiB). Runs ~0.3 s first: the idle GPU clocks down to 210 MHz and takes a while to ramp."""
    fn = jax.jit(f).lower(*args).compile()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3: fn(*args).block_until_ready()
    t0 = time.perf_counter(); fn(*args).block_until_ready()
    n = max(20, int(0.05 / (time.perf_counter() - t0)))  # ~50 ms of queued calls per timing: short calls are noisy under WSL
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n): out = fn(*args)
        out.block_until_ready(); ts.append((time.perf_counter() - t0) / n)
    return min(ts) * 1e3, fn.memory_analysis().temp_size_in_bytes / 2**30
H = 8
FNS = {"flash": lambda q, k, v, key: jax.nn.dot_product_attention(
    *(a.transpose(1, 0, 2)[None] for a in (q, k, v)), is_causal=True, implementation="cudnn")[0].transpose(1, 0, 2)}
for c, nr in ((32, 4), (32, 8), (32, 16), (64, 4), (64, 8)):
    for m in ("hier", "bandit", "active", "stationary"):
        FNS[f"{m} c{c} r{nr}"] = jax.vmap(partial(chunk_attn, method=m, chunk=c, n_ref=nr), (0, 0, 0, None))
if __name__ == "__main__":
    rows = []
    for T in map(int, os.environ.get("TS", "1024 2048 4096 8192 16384 32768").split()):
        q, k, v = jax.random.normal(key, (3, H, T, 64), jnp.bfloat16)
        for name, f in FNS.items():
            try: ms, gib = bench(f, q, k, v, key)
            except Exception as e: ms, gib = float("nan"), float("nan"); print(name, T, type(e).__name__)
            rows.append((name, T, ms, gib)); print(f"T={T:6d} {name:22s} {ms:8.3f} ms", flush=True)
    with open(os.environ.get("OUT", "runs/approx/fast_timing.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("config", "T", "ms", "temp_gib")] + rows)
