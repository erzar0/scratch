"""Decode time of one attention layer vs context length (1k-128k tokens): the cost of producing one new token
from a cache of the T tokens before it. CSA (DeepSeek-V4) vs hierarchical chunk attention vs exact attention.

First checks every decode step against the prefill code (csa_attention, hier_attention, attention) at a small
size in fp32, then times the step at batch 1 and BATCH, same config as bench_attention.py (d_model 64, 1 head,
bf16). Per token, exact attention reads the whole cache; CSA reads top_k + window entries plus one indexer key
per block; hier reads one summary per chunk plus window + n_refined · chunk_size raw tokens. "hier" is hierarchical only
(2-chunk window, 64 tokens); "hier + window" adds a 512 or 1024-token sliding window. hier_cache(offload=True),
the raw K/V in host RAM, is not benchmarked here.
    .env/bin/python bench_decode.py      -> table on stdout, runs/bench/bench_decode.csv
"""
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from bench_attention import D_MODEL, DTYPE, LENGTHS, N_HEADS, key, p_att, p_csa4, p_csa32  # noqa: E402
from csa import csa_attention, csa_cache, csa_decode_step, init_csa  # noqa: E402
from hier import hier_attention, hier_cache, hier_decode_step  # noqa: E402
from transformer import attention, init_params  # noqa: E402

BATCH = 64  # sequences decoded together; at batch 1 a step reads ~MB, and launch overhead hides the reads
REPEATS = 3


def exact_cache(p, context, n_heads):
    T, D = context.shape
    _, k, v = (a.reshape(T, n_heads, D // n_heads) for a in jnp.split(context @ p["qkv"], 3, axis=-1))
    return {"key": k, "value": v}


def exact_decode_step(p, cache, x_t):
    """Row t of transformer.attention, one query. The new token's key is scored separately rather than
    concatenated onto the cache, which would copy the whole cache every step."""
    n_heads, head_dim = cache["key"].shape[1:]
    q, k_t, v_t = (a.reshape(n_heads, head_dim) for a in jnp.split(x_t @ p["qkv"], 3))
    logits = jnp.concatenate([jnp.einsum("hd,thd->ht", q, cache["key"]), (q * k_t).sum(-1, keepdims=True)], -1)
    attn = jax.nn.softmax(logits / jnp.sqrt(head_dim), -1)
    out = jnp.einsum("ht,thd->hd", attn[:, :-1], cache["value"]) + attn[:, -1:] * v_t
    return out.reshape(-1) @ p["proj"]


def check():
    """Decode step vs the prefill output at the same position, fp32, small sizes."""
    with jax.default_matmul_precision("highest"):
        x = jax.random.normal(key, (61, 32))
        p_csa = init_csa(key, 32, 4)
        full = csa_attention(p_csa, x, top_k=2, window=4)
        # 0: empty cache; 3, 6: fewer than top_k blocks; 13+: top_k drops blocks; 60: mid-block
        for t in [0, 3, 6, 13, 24, 40, 60]:
            y = csa_decode_step(p_csa, csa_cache(p_csa, x[:t], window=4), x[t], t, top_k=2)
            assert jnp.abs(y - full[t]).max() < 1e-4, f"[csa] decode != prefill at t={t}"

        p = init_params(key, 16, d_model=32, n_heads=4)["layers"][0]
        full = attention(p, x, 4)
        for t in [0, 17, 60]:
            assert jnp.abs(exact_decode_step(p, exact_cache(p, x[:t], 4), x[t]) - full[t]).max() < 1e-4

        # hier: decode a whole chunk carrying chunk_mass; only the chunk's last token must match prefill.
        # Windows of 2 and 3 chunks, small enough that summaries and refinement are used.
        chunk_size, early = 4, 0.0
        for window in [8, 12]:
            full = hier_attention(p, x, 4, chunk_size=chunk_size, n_refined=1, window=window)
            for chunk in range(14):
                mass = None
                for t in range(chunk * chunk_size, (chunk + 1) * chunk_size):
                    cache = hier_cache(p, x[:t], 4, chunk_size, window=window)
                    cache["chunk_mass"] = cache["chunk_mass"] if mass is None else mass
                    y, mass = hier_decode_step(p, cache, x[t], t, chunk_size, n_refined=1, window=window)
                    diff = jnp.abs(y - full[t]).max()
                    if t % chunk_size == chunk_size - 1:
                        assert diff < 1e-4, f"[hier] decode != prefill at chunk end t={t} (window {window})"
                    else:
                        early = max(early, float(diff))
    print(f"decode checks passed (hier before chunk end: max diff {early:.2f} vs prefill, expected: causal picks)")


def bench_one(step, cache, batch):
    """-> median µs per decode step for `batch` sequences, each with its own copy of the cache."""
    # real copies, tiled on the GPU, then put back where the original lives (GPU or pinned host)
    tile = lambda a: jnp.tile(jax.device_put(a, jax.memory.Space.Device)[None], (batch,) + (1,) * a.ndim)
    caches = jax.tree.map(lambda a: jax.device_put(tile(a), a.sharding), cache)
    x_t = jax.random.normal(key, (batch, D_MODEL), DTYPE)
    fn = jax.jit(jax.vmap(step)).lower(caches, x_t).compile()

    def per_call(n):  # back to back, one sync (see bench_attention.measure)
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn(caches, x_t)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / n

    per_call(1)
    n = min(2000, max(3, int(0.3 / per_call(3))))
    return sorted(per_call(n) for _ in range(REPEATS))[REPEATS // 2] * 1e6


# name -> (cache builder, decode step for the token at position T); all from a random context of T tokens
METHODS = {
    "csa block 4": (lambda c: csa_cache(p_csa4, c), lambda T: lambda cache, x: csa_decode_step(p_csa4, cache, x, T)),
    "csa block 32": (lambda c: csa_cache(p_csa32, c), lambda T: lambda cache, x: csa_decode_step(p_csa32, cache, x, T)),
    "hier": (lambda c: hier_cache(p_att, c, N_HEADS), lambda T: lambda cache, x: hier_decode_step(p_att, cache, x, T)[0]),
    **{f"hier + window {w}": (lambda c, w=w: hier_cache(p_att, c, N_HEADS, window=w),
                              lambda T, w=w: lambda cache, x: hier_decode_step(p_att, cache, x, T, window=w)[0])
       for w in [512, 1024]},
    "exact": (lambda c: exact_cache(p_att, c, N_HEADS), lambda T: lambda cache, x: exact_decode_step(p_att, cache, x)),
}

if __name__ == "__main__":
    check()
    print(f"{jax.devices()[0].device_kind}, d_model {D_MODEL}, {N_HEADS} heads, {jnp.dtype(DTYPE).name}")
    rows = []
    for T in LENGTHS:
        context = jax.random.normal(jax.random.fold_in(key, T), (T, D_MODEL), DTYPE)
        for name, (build, step) in METHODS.items():
            cache = jax.jit(build)(context)
            on_host = lambda a: a.sharding.memory_kind == "pinned_host"
            gpu_mb = sum(a.nbytes for a in jax.tree.leaves(cache) if not on_host(a)) / 2**20
            host_mb = sum(a.nbytes for a in jax.tree.leaves(cache) if on_host(a)) / 2**20
            us1, usb = bench_one(step(T), cache, 1), bench_one(step(T), cache, BATCH)
            rows.append((name, T, gpu_mb, host_mb, us1, usb))
            print(f"{name:17s} T={T:6d}: cache GPU {gpu_mb:7.2f} MB host {host_mb:6.2f} MB  batch 1 {us1:8.1f} µs  "
                  f"batch {BATCH} {usb:8.1f} µs", flush=True)

    print(f"\nµs per decode step, batch 1 / batch {BATCH}")
    print(f"{'T':>7}" + "".join(f"{name:>22}" for name in METHODS))
    for T in LENGTHS:
        cells = [f"{us1:.0f} / {usb:.0f}" for name, t, _, _, us1, usb in rows if t == T]
        print(f"{T:>7}" + "".join(f"{c:>22}" for c in cells))

    os.makedirs("runs", exist_ok=True)
    with open("runs/bench/bench_decode.csv", "w") as log:
        log.write(f"method,context,gpu_cache_mib_per_seq,host_cache_mib_per_seq,us_batch1,us_batch{BATCH}\n")
        for name, T, gpu_mb, host_mb, us1, usb in rows:
            log.write(f"{name},{T},{gpu_mb:.3f},{host_mb:.3f},{us1:.1f},{usb:.1f}\n")
