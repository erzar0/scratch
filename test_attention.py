"""Tests for csa.py, hier.py and the decode steps, against independent float64 NumPy references.
    .env/bin/python test_attention.py        (no pytest needed; exit code 1 if anything fails)
The references are written loop-style, one query and head at a time, straight from the definitions; they share
no code with the JAX versions, so a bug must be made twice the same way to slip through. Everything runs in
float64 (jax_enable_x64), so the tolerance vs a reference is 1e-9, not rounding noise."""
import os
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from bench_decode import exact_cache, exact_decode_step  # noqa: E402
from csa import csa_attention, csa_cache, csa_decode_step, init_csa  # noqa: E402
from hier import hier_attention, hier_cache, hier_decode_step  # noqa: E402
from transformer import attention, init_params  # noqa: E402

key = jax.random.PRNGKey(0)
TESTS = []


def test(f):
    TESTS.append(f)
    return f


def f64(tree):
    return jax.tree.map(lambda a: np.asarray(a, np.float64), tree)


def softmax(z):
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def max_diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    return float(np.abs(a - b).max()) if a.size else 0.0


def perturb(tree, k, scale=0.1):
    """Add noise to every leaf, so zero-initialized biases, unit norm gains and zero sinks get tested too."""
    leaves, treedef = jax.tree.flatten(tree)
    return treedef.unflatten([a + scale * jax.random.normal(kk, a.shape) for a, kk in
                              zip(leaves, jax.random.split(k, len(leaves)))])


def random_x(T, D, seed):
    return jax.random.normal(jax.random.fold_in(key, seed), (T, D))


def hier_params(D, seed=0):
    return init_params(jax.random.fold_in(key, seed), 16, d_model=D)["layers"][0]  # qkv, proj used


# ------------------------------------------------------------------ references

def naive_attention(p, x, n_heads):
    p, x = f64(p), np.asarray(x, np.float64)
    T, D = x.shape
    hd = D // n_heads
    q, k, v = (a.reshape(T, n_heads, hd) for a in np.split(x @ p["qkv"], 3, -1))
    out = np.zeros((T, n_heads, hd))
    for t in range(T):
        for h in range(n_heads):
            out[t, h] = softmax(k[: t + 1, h] @ q[t, h] / np.sqrt(hd)) @ v[: t + 1, h]
    return out.reshape(T, D) @ p["proj"]


def naive_hier(p, x, n_heads, chunk_size, n_refined, window, causal_picks=False):
    """Hier attention by definition: query t in chunk Q reads chunks Q-L+1 ... Q exactly (L = window // chunk_size),
    older chunks as summaries or, the n_refined with the most mass, token by token. As in the prefill, the sequence is zero-padded to whole chunks and the
    padded queries vote in the last chunk's picks. causal_picks: query t's picks use only the votes of the
    queries <= t of its chunk (what a decoder can do)."""
    p, x = f64(p), np.asarray(x, np.float64)
    T, D = x.shape
    C, hd, L = chunk_size, D // n_heads, window // chunk_size
    padded = -(-T // C) * C
    x = np.pad(x, ((0, padded - T), (0, 0)))
    q, k, v = (a.reshape(padded, n_heads, hd) for a in np.split(x @ p["qkv"], 3, -1))
    k_mean, v_mean = (a.reshape(padded // C, C, n_heads, hd).mean(1) for a in (k, v))
    scale = hd**-0.5

    def summary_probs(i, h):  # query i's cheap attention over [summaries ; local]: the summary part
        n_old = max(i // C - L + 1, 0)
        summary = k_mean[:n_old, h] @ q[i, h] * scale + np.log(C)
        local = k[n_old * C : i + 1, h] @ q[i, h] * scale
        return softmax(np.concatenate([summary, local]))[:n_old]

    out = np.zeros((padded, n_heads, hd))
    for t in range(padded):
        Q = t // C
        n_old = max(Q - L + 1, 0)
        voters = range(Q * C, t + 1) if causal_picks else range(Q * C, Q * C + C)
        for h in range(n_heads):
            mass = sum(summary_probs(i, h) for i in voters)
            refined = {int(c) for c in np.argsort(-mass, kind="stable")[:n_refined] if mass[c] > 0}
            logits, values = [], []
            for c in range(n_old):
                if c in refined:  # token by token
                    logits += list(k[c * C : (c + 1) * C, h] @ q[t, h] * scale)
                    values += list(v[c * C : (c + 1) * C, h])
                else:  # one summary token
                    logits.append(k_mean[c, h] @ q[t, h] * scale + np.log(C))
                    values.append(v_mean[c, h])
            for j in range(n_old * C, t + 1):  # local
                logits.append(k[j, h] @ q[t, h] * scale)
                values.append(v[j, h])
            out[t, h] = softmax(np.array(logits)) @ np.array(values)
    return out.reshape(padded, D)[:T] @ p["proj"]


def naive_csa(p, x, top_k, window):
    """CSA by definition -> (output (T, D), indexer loss). The loss matches csa.py's only for T % block_size == 0
    (csa.py averages it over the padded rows too)."""
    p, x = f64(p), np.asarray(x, np.float64)
    T, D = x.shape
    m = p["kv_compressor"]["bias_own"].shape[0]
    n_heads, c = p["sink"].shape[0], p["query_norm"].shape[0]

    def compress(cp, xs):  # Eqs. 9-12: entry b = per-channel softmax over block b (own) and block b-1 (prev)
        n, out_dim = len(xs) // m, cp["bias_own"].shape[1]
        val_own, val_prev, score_own, score_prev = np.split(xs[: n * m] @ cp["proj"], 4, -1)
        entries = []
        for b in range(n):
            own, prev = slice(b * m, (b + 1) * m), slice((b - 1) * m, b * m)
            scores, values = [score_own[own] + cp["bias_own"]], [val_own[own]]
            if b > 0:
                scores.append(score_prev[prev] + cp["bias_prev"])
                values.append(val_prev[prev])
            weight = softmax(np.concatenate(scores).T).T  # over tokens, separately per channel
            entries.append((weight * np.concatenate(values)).sum(0))
        return np.array(entries).reshape(n, out_dim)

    def rms(gain, a):
        return gain * a / np.sqrt((a * a).mean() + 1e-6)

    def rope(a, pos):  # rotate pairs (i, i + half) of the last quarter by pos · 10000^(-i/half)
        n_rot = len(a) // 4
        half = n_rot // 2
        angle = pos * 10000.0 ** (-np.arange(half) / half)
        x1, x2 = a[len(a) - n_rot : len(a) - half], a[len(a) - half :]
        return np.concatenate([a[: len(a) - n_rot], x1 * np.cos(angle) - x2 * np.sin(angle),
                               x1 * np.sin(angle) + x2 * np.cos(angle)])

    indexer = p["indexer"]
    n_indexer_heads = indexer["head_weight"].shape[1]
    latent = x @ p["query_down"]
    indexer_query = (latent @ indexer["query_up"]).reshape(T, n_indexer_heads, -1)
    head_weight = x @ indexer["head_weight"]
    indexer_keys = compress(indexer["key_compressor"], x)
    compressed_kv = compress(p["kv_compressor"], x)
    raw_kv = x @ p["window_kv"]
    n_groups = p["group_proj"].shape[0]
    y, loss = np.zeros((T, D)), 0.0
    for t in range(T):
        finished = range(t // m)  # blocks complete before t's block starts
        score = np.array([sum(head_weight[t, h] * max(indexer_query[t, h] @ indexer_keys[b], 0.0)
                              for h in range(n_indexer_heads)) for b in finished]) * indexer_query[t].size**-0.5
        picked = list(np.argsort(-score, kind="stable")[:top_k])
        recent = range(max(0, t - window + 1), t + 1)
        entries = [(compressed_kv[b], b * m + m // 2) for b in picked] + [(raw_kv[j], j) for j in recent]
        kv = np.array([rope(rms(p["kv_norm"], e), pos) for e, pos in entries])
        query = (latent[t] @ p["query_up"]).reshape(n_heads, c)
        heads, attn_on_picked = [], np.zeros(len(picked))
        for h in range(n_heads):
            qh = rope(rms(p["query_norm"], query[h]), t)
            weight = softmax(np.append(kv @ qh / np.sqrt(c), p["sink"][h]))[:-1]  # sink: weight on nothing
            attn_on_picked += weight[: len(picked)]
            heads.append(rope(weight @ kv, -t))
        per_group = np.concatenate(heads).reshape(n_groups, -1)
        y[t] = np.concatenate([per_group[g] @ p["group_proj"][g] for g in range(n_groups)]) @ p["out_proj"]
        if picked:  # cross-entropy from the attention on the picked blocks to the indexer's softmax over them
            s = score[picked]
            log_softmax = s - s.max() - np.log(np.exp(s - s.max()).sum())
            loss -= attn_on_picked / max(attn_on_picked.sum(), 1e-9) @ log_softmax
    return y, loss / T


# ------------------------------------------------------------------ hier prefill

HIER_CONFIGS = [  # (T, heads, chunk_size, n_refined, window, chunks_per_block), d_model 32
    (1, 2, 4, 2, 8, 64),  # one token
    (3, 2, 4, 2, 8, 64),  # shorter than a chunk
    (16, 2, 4, 1, 8, 64),  # whole chunks, one block
    (37, 4, 4, 1, 8, 3),  # partial last chunk, 3 blocks + remainder
    (37, 4, 4, 3, 12, 2),  # window of 3 chunks
    (50, 1, 8, 2, 8, 1),  # window of 1 chunk, one chunk per block
    (64, 2, 4, 20, 8, 64),  # n_refined >= old chunks: everything refined
    (40, 2, 1, 3, 5, 7),  # chunk_size 1
    (70, 2, 2, 4, 16, 4),  # window of 8 chunks
    (70, 2, 4, 2, 40, 3),  # window of 10 chunks: few summaries
]


@test
def hier_matches_reference():
    worst = 0.0
    for i, (T, H, C, S, W, cpb) in enumerate(HIER_CONFIGS):
        p, x = hier_params(32, i), random_x(T, 32, i)
        y = hier_attention(p, x, H, chunk_size=C, n_refined=S, window=W, chunks_per_block=cpb)
        d = max_diff(y, naive_hier(p, x, H, C, S, W))
        assert d < 1e-9, f"config {(T, H, C, S, W, cpb)}: max diff {d:.2e}"
        worst = max(worst, d)
    return f"{len(HIER_CONFIGS)} configs, max diff {worst:.1e}"


@test
def hier_refined_zero_is_summaries_only():
    p, x = hier_params(32), random_x(40, 32, 1)
    d = max_diff(hier_attention(p, x, 2, chunk_size=4, n_refined=0, window=8), naive_hier(p, x, 2, 4, 0, 8))
    assert d < 1e-9, f"max diff {d:.2e}"


@test
def hier_special_cases_equal_exact_attention():
    p, x = hier_params(32), random_x(45, 32, 2)
    exact = attention(p, x, 4)
    assert max_diff(exact, naive_attention(p, x, 4)) < 1e-9, "transformer.attention != reference"
    # every old chunk refined: nothing left as a summary
    assert max_diff(hier_attention(p, x, 4, chunk_size=4, n_refined=99, window=8, chunks_per_block=3), exact) < 1e-9
    # chunk_size 1: a summary is the token itself with logit + log 1 = + 0, whatever gets refined
    assert max_diff(hier_attention(p, x, 4, chunk_size=1, n_refined=2, window=3, chunks_per_block=5), exact) < 1e-9
    # window covering the whole sequence: no summaries at all
    assert max_diff(hier_attention(p, x, 4, chunk_size=4, n_refined=1, window=48), exact) < 1e-9
    # window=None is two chunks
    assert max_diff(hier_attention(p, x, 4, chunk_size=4, n_refined=1),
                    hier_attention(p, x, 4, chunk_size=4, n_refined=1, window=8)) < 1e-12


@test
def hier_block_size_invariant():
    p, x = hier_params(32), random_x(61, 32, 3)
    ys = [hier_attention(p, x, 4, chunk_size=4, n_refined=2, window=12, chunks_per_block=b)
          for b in [1, 2, 3, 5, 16, 64]]
    d = max(max_diff(y, ys[-1]) for y in ys)
    assert d < 1e-12, f"max diff {d:.2e}"


@test
def hier_causal_at_chunk_boundaries():
    # Picks depend on all queries of a chunk (documented leak), so only whole-chunk prefixes must match
    p, x = hier_params(32), random_x(48, 32, 4)
    for C, S, W, cpb in [(4, 1, 8, 3), (4, 3, 16, 64), (8, 2, 8, 1)]:
        f = lambda x: hier_attention(p, x, 4, chunk_size=C, n_refined=S, window=W, chunks_per_block=cpb)
        full = f(x)
        for n in range(C, 48, C):
            d = max_diff(f(x[:n]), full[:n])
            assert d < 1e-12, f"chunk {C}, prefix {n}: {d:.2e}"


@test
def hier_jit_and_vmap():
    p, xs = hier_params(32), jax.random.normal(key, (3, 37, 32))
    f = lambda x: hier_attention(p, x, 4, chunk_size=4, n_refined=2, window=8, chunks_per_block=3)
    eager = jnp.stack([f(x) for x in xs])
    assert max_diff(jax.jit(jax.vmap(f))(xs), eager) < 1e-12


@test
def hier_gradients():
    # finite differences vs reverse mode, through the block loop and its recomputation (checkpoint)
    p, x = hier_params(32), random_x(37, 32, 5)
    f = lambda x, qkv, proj: hier_attention({"qkv": qkv, "proj": proj}, x, 4, chunk_size=4, n_refined=2, window=8,
                                            chunks_per_block=3)
    check_grads(f, (x, p["qkv"], p["proj"]), order=1, modes=["rev"])


# ------------------------------------------------------------------ CSA prefill

CSA_CONFIGS = [  # (T, d_model, heads, block_size, groups, top_k, window)
    (1, 32, 4, 4, 2, 2, 4),
    (13, 32, 4, 4, 2, 2, 4),  # mid-block (padded)
    (40, 32, 4, 4, 2, 2, 4),  # top_k drops blocks
    (40, 32, 4, 4, 1, 8, 32),  # reference top_k / window
    (64, 64, 2, 8, 2, 3, 5),
    (48, 32, 1, 4, 1, 1, 1),  # one head, window of just the token itself
]


def csa_setup(i, T, D, H, m, G):
    p = perturb(init_csa(jax.random.fold_in(key, i), D, H, block_size=m, n_groups=G), jax.random.fold_in(key, 100 + i))
    return p, random_x(T, D, i)


@test
def csa_matches_reference():
    worst = 0.0
    for i, (T, D, H, m, G, k, w) in enumerate(CSA_CONFIGS):
        p, x = csa_setup(i, T, D, H, m, G)
        d = max_diff(csa_attention(p, x, top_k=k, window=w), naive_csa(p, x, k, w)[0])
        assert d < 1e-9, f"config {(T, D, H, m, G, k, w)}: max diff {d:.2e}"
        worst = max(worst, d)
    return f"{len(CSA_CONFIGS)} configs, max diff {worst:.1e}"


@test
def csa_causal_every_prefix():
    p, x = csa_setup(0, 37, 32, 4, 4, 2)
    full = csa_attention(p, x, top_k=2, window=4)
    d = max(max_diff(csa_attention(p, x[:n], top_k=2, window=4), full[:n]) for n in range(1, 37))
    assert d < 1e-12, f"max diff {d:.2e}"


@test
def csa_indexer_loss_gradient():
    # The layer output's gradient w.r.t. the indexer is the gradient of the indexer loss alone (with_aux_loss);
    # compare with central differences of the reference loss along a random direction.
    T, D, H, m, G, k, w = 40, 32, 4, 4, 2, 2, 4
    p, x = csa_setup(7, T, D, H, m, G)
    grad = jax.grad(lambda p: csa_attention(p, x, top_k=k, window=w).sum())(p)["indexer"]
    direction = perturb(jax.tree.map(jnp.zeros_like, p["indexer"]), jax.random.fold_in(key, 8), 1.0)
    analytic = sum(float((g * d).sum()) for g, d in zip(jax.tree.leaves(grad), jax.tree.leaves(direction)))
    eps = 1e-5
    shifted = lambda s: naive_csa({**p, "indexer": jax.tree.map(lambda a, d: a + s * d, p["indexer"], direction)},
                                  x, k, w)[1]
    numeric = (shifted(eps) - shifted(-eps)) / (2 * eps)
    assert abs(analytic - numeric) < 1e-6 * max(1.0, abs(numeric)), f"analytic {analytic:.8f} vs numeric {numeric:.8f}"
    return f"directional derivative {analytic:.6f}"


@test
def csa_main_gradients_untouched_by_indexer_loss():
    # Everything but the indexer gets the plain gradient of the output: the loss must not leak into the model
    T, D, H, m, G, k, w = 24, 32, 4, 4, 2, 2, 4
    p, x = csa_setup(9, T, D, H, m, G)
    main = {name: v for name, v in p.items() if name != "indexer"}
    f = lambda x, main: csa_attention({**main, "indexer": p["indexer"]}, x, top_k=k, window=w)
    check_grads(f, (x, main), order=1, modes=["rev"])


# ------------------------------------------------------------------ decode

@test
def csa_decode_every_position():
    for i, (T, D, H, m, G, k, w) in enumerate([CSA_CONFIGS[2], CSA_CONFIGS[4], CSA_CONFIGS[5]]):
        p, x = csa_setup(i, T, D, H, m, G)
        full = csa_attention(p, x, top_k=k, window=w)
        for t in range(T):
            y = csa_decode_step(p, csa_cache(p, x[:t], window=w), x[t], t, top_k=k)
            d = max_diff(y, full[t])
            assert d < 1e-10, f"config {(T, D, H, m, G, k, w)}, t={t}: {d:.2e}"


@test
def exact_decode_every_position():
    p, x = hier_params(32), random_x(30, 32, 10)
    full = naive_attention(p, x, 4)
    d = max(max_diff(exact_decode_step(p, exact_cache(p, x[:t], 4), x[t]), full[t]) for t in range(30))
    assert d < 1e-10, f"max diff {d:.2e}"


def hier_decode_all(p, x, H, C, S, W):
    """Decode every position of x, carrying chunk_mass through each chunk."""
    out, mass = [], None
    for t in range(len(x)):
        cache = hier_cache(p, x[:t], H, C, window=W)
        if t % C:
            cache["chunk_mass"] = mass
        y, mass = hier_decode_step(p, cache, x[t], t, C, S, window=W)
        out.append(y)
    return jnp.stack(out)


@test
def hier_decode_every_position():
    # Every position equals the reference with causal picks; chunk ends also equal the prefill
    for i, (T, H, C, S, W) in enumerate([(37, 4, 4, 1, 8), (40, 2, 4, 3, 12), (33, 1, 8, 2, 8), (20, 2, 2, 9, 4),
                                         (41, 2, 4, 2, 20)]):
        p, x = hier_params(32, i), random_x(T, 32, 20 + i)
        reference = naive_hier(p, x, H, C, S, W, causal_picks=True)
        prefill = hier_attention(p, x, H, chunk_size=C, n_refined=S, window=W)
        ends = np.arange(C - 1, T, C)
        y = hier_decode_all(p, x, H, C, S, W)
        d = max_diff(y, reference)
        assert d < 1e-10, f"config {(T, H, C, S, W)}: vs reference {d:.2e}"
        assert max_diff(y[ends], prefill[ends]) < 1e-10, f"config {(T, H, C, S, W)}: chunk ends != prefill"


@test
def hier_decode_batched_jit():
    # The benchmark's path: jit(vmap(step)) over a batch of caches
    B, T, H, C, S, W = 3, 29, 4, 4, 2, 8
    p, xs = hier_params(32), jax.random.normal(key, (B, T + 1, 32))
    caches = [hier_cache(p, x[:T], H, C, window=W) for x in xs]
    batched = jax.tree.map(lambda *a: jnp.stack(a), *caches)
    y, mass = jax.jit(jax.vmap(lambda c, x: hier_decode_step(p, c, x, T, C, S, window=W)))(batched, xs[:, T])
    for b in range(B):
        y_b, mass_b = hier_decode_step(p, caches[b], xs[b, T], T, C, S, window=W)
        assert max_diff(y[b], y_b) < 1e-12 and max_diff(mass[b], mass_b) < 1e-12, f"sequence {b} differs"


# ------------------------------------------------------------------ low precision and memory

@test
def bf16_close_to_float64():
    rel = lambda a, b: float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))
    bf16 = lambda tree: jax.tree.map(lambda a: a.astype(jnp.bfloat16), tree)
    p, x = hier_params(32), random_x(64, 32, 40)
    hier_err = rel(hier_attention(bf16(p), bf16(x), 4, chunk_size=4, n_refined=2, window=8),
                   naive_hier(p, x, 4, 4, 2, 8))
    pc, xc = csa_setup(41, 64, 32, 4, 4, 2)
    csa_err = rel(csa_attention(bf16(pc), bf16(xc), top_k=2, window=4), naive_csa(pc, xc, 2, 4)[0])
    assert hier_err < 0.03 and csa_err < 0.03, f"relative error hier {hier_err:.3f}, csa {csa_err:.3f}"
    return f"relative error hier {hier_err:.4f}, csa {csa_err:.4f}"


@test
def hier_memory_linear():
    # Compile only. Quadratic memory would grow 16x from 16k to 64k tokens; blocked hier should grow ~4x.
    p = jax.tree.map(lambda a: a.astype(jnp.bfloat16), init_params(key, 16, 64, 1, n_layers=1)["layers"][0])

    def temp(f, T):
        x = jax.ShapeDtypeStruct((T, 64), jnp.bfloat16)
        return jax.jit(f).lower(x).compile().memory_analysis().temp_size_in_bytes

    forward = lambda x: hier_attention(p, x, 1)
    backward = jax.grad(lambda x: hier_attention(p, x, 1).astype(jnp.float32).sum())
    ratios = {name: temp(f, 65536) / temp(f, 16384) for name, f in [("forward", forward), ("backward", backward)]}
    assert all(r < 6 for r in ratios.values()), f"memory growth 16k -> 64k: {ratios}"
    return "growth 16k -> 64k: " + ", ".join(f"{k} {v:.1f}x" for k, v in ratios.items())


if __name__ == "__main__":
    failed = []
    for f in TESTS:
        t0 = time.time()
        try:
            note = f() or ""
            print(f"PASS {f.__name__:45s} {time.time() - t0:6.1f}s  {note}", flush=True)
        except Exception:
            failed.append(f.__name__)
            print(f"FAIL {f.__name__:45s} {time.time() - t0:6.1f}s", flush=True)
            traceback.print_exc()
    print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    raise SystemExit(1 if failed else 0)
