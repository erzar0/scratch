"""Forward-pass time and memory of one attention layer vs sequence length (1k-128k tokens):
CSA (DeepSeek-V4) vs hierarchical chunk attention, with exact causal attention (cuDNN flash) as reference.

Sized to fit a 16 GB card: batch 1, bf16, d_model 64, 1 head, forward only. The binding limit is not the
total (~11 GB usable under WSL) but WSL's cap on one GPU buffer, just under 4 GiB: XLA allocates a program's
whole working memory ("temp") as one buffer. The attention code is unchanged. CSA materializes an O(T²) score
matrix whose size doesn't depend on width: the indexer scores T x T/block_size. Hier computes its
T x T/chunk_size summary logits one block of query chunks at a time, so its memory is O(T). CSA with the
reference block size 4 needs 12 GiB at 64k, so it also runs with block size 32, the same
granularity as hier (both then pick 8 chunks of 32 tokens per query).
    .env/bin/python bench_attention.py      -> table on stdout, runs/bench/bench_attention.csv
"""
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from csa import csa_attention, init_csa  # noqa: E402
from hier import hier_attention  # noqa: E402
from transformer import init_params  # noqa: E402

D_MODEL, N_HEADS, DTYPE = 64, 1, jnp.bfloat16
LENGTHS = [2**i for i in range(10, 18)]  # 1k ... 128k
MEMORY_BUDGET = 3.9 * 2**30  # temp is one buffer, and WSL fails single allocations of 4 GiB or more
REPEATS = 3  # timing runs; the median is reported

key = jax.random.PRNGKey(0)
to_dtype = lambda tree: jax.tree.map(lambda a: a.astype(DTYPE), tree)
p_csa4 = to_dtype(init_csa(key, D_MODEL, N_HEADS, block_size=4, n_groups=1))
p_csa32 = to_dtype(init_csa(key, D_MODEL, N_HEADS, block_size=32, n_groups=1))
p_att = to_dtype(init_params(key, 16, D_MODEL, N_HEADS, n_layers=1)["layers"][0])


def flash(p, x):
    """Exact causal attention with the same q/k/v/out weights as hier, via cuDNN's fused kernel."""
    T, D = x.shape
    q, k, v = (a.reshape(1, T, N_HEADS, D // N_HEADS) for a in jnp.split(x @ p["qkv"], 3, axis=-1))
    out = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")
    return out.reshape(T, D) @ p["proj"]


MIXERS = {
    "csa block 4": lambda x: csa_attention(p_csa4, x),
    "csa block 32": lambda x: csa_attention(p_csa32, x),
    # chunk 32; window 64 = the original design (previous + current chunk), 512 and 1024 = longer sliding windows
    "hier": lambda x: hier_attention(p_att, x, N_HEADS),
    **{f"hier + window {w}": lambda x, w=w: hier_attention(p_att, x, N_HEADS, window=w) for w in [512, 1024]},
    "exact (flash)": lambda x: flash(p_att, x),
}


def measure(f, T):
    """-> (median ms, temp GiB), or (None, GiB or None) if it doesn't fit."""
    x = jax.random.normal(key, (T, D_MODEL), DTYPE)
    try:
        fn = jax.jit(f).lower(x).compile()  # autotuning can OOM already here
    except jax.errors.JaxRuntimeError:
        return None, None
    temp = fn.memory_analysis().temp_size_in_bytes
    if temp > MEMORY_BUDGET:
        return None, temp / 2**30
    def per_call(n):
        # n calls queued back to back, one sync at the end: a sync costs ~1 ms under WSL, which would
        # otherwise swamp the short runs
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn(x)
        out.block_until_ready()
        return (time.perf_counter() - t0) / n

    try:
        per_call(1)  # warm-up
        n = min(200, max(3, int(0.3 / per_call(3))))  # enough calls for ~0.3 s per timing run
        times = sorted(per_call(n) for _ in range(REPEATS))
    except jax.errors.JaxRuntimeError:
        return None, temp / 2**30
    return times[REPEATS // 2] * 1e3, temp / 2**30


if __name__ == "__main__":
    print(f"{jax.devices()[0].device_kind}, d_model {D_MODEL}, {N_HEADS} heads, {jnp.dtype(DTYPE).name}, batch 1")
    results = {name: {} for name in MIXERS}
    for name, f in MIXERS.items():
        for T in LENGTHS:
            ms, gib = measure(f, T)
            results[name][T] = (ms, gib)
            mem = "-" if gib is None else f"{gib:.2f} GiB"
            print(f"{name:14s} T={T:6d}: {'OOM' if ms is None else f'{ms:9.2f} ms'}  {mem}", flush=True)
            if ms is None:
                break  # longer sequences won't fit either

    # Time table, then the growth per doubling of T: 2x = linear, 4x = quadratic
    print("\nforward ms (temp memory GiB)")
    print(f"{'T':>7}" + "".join(f"{name:>22}" for name in MIXERS))
    for T in LENGTHS:
        row = []
        for name in MIXERS:
            ms, gib = results[name].get(T, (None, None))
            row.append("OOM" if ms is None and T in results[name] else "" if ms is None else f"{ms:.1f} ({gib:.2f})")
        print(f"{T:>7}" + "".join(f"{c:>22}" for c in row))
    print("\ntime growth per doubling of T, over the last 3 doublings that ran (2x linear, 4x quadratic)")
    for name in MIXERS:
        ran = [T for T in LENGTHS if results[name].get(T, (None,))[0] is not None]
        if len(ran) >= 4:
            ratio = (results[name][ran[-1]][0] / results[name][ran[-4]][0]) ** (1 / 3)
            print(f"  {name:14s} {ratio:.2f}x  ({ran[-4]}-{ran[-1]})")

    os.makedirs("runs", exist_ok=True)
    with open("runs/bench/bench_attention.csv", "w") as log:
        log.write("mixer,seq_len,forward_ms,temp_gib\n")
        for name in MIXERS:
            for T, (ms, gib) in results[name].items():
                log.write(f"{name},{T},{'' if ms is None else f'{ms:.3f}'},{'' if gib is None else f'{gib:.3f}'}\n")
