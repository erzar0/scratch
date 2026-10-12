"""CSA decode with a bounded pool of compressed blocks: blocks are evicted by their attention statistic.

CSA's attention reads only top_k + window entries per token, but its pool of compressed blocks (one per
block_size tokens) grows with T, and the indexer scores every block for every token: O(T²/block_size) in
total. Here each layer keeps at most ~B(n) blocks, n = finished blocks so far:
    sqrt  B = c·√n         -> pool O(√T), indexer work O(T^1.5) in total
    log   B = c·log2(1+n)  -> pool O(log T), indexer work O(T log T)
After every token, per layer:
  stats     each head's attention weight on the blocks it read is added to the block's sum (blocks it didn't
            read get 0); avg = sum / tokens since the block left the raw window. Blocks still inside the
            window are rarely picked (the window has them at full resolution), so they are neither scored nor
            evictable yet.
  schedule  with over = pool size - B(n), one block is evicted with probability clip(over, 0, 1): the size
            tracks B(n).
  selection which block goes is sampled with p ∝ exp(-score / tau), score = max over heads of avg, divided
            by its mean over the evictable blocks. High scores go last. tau -> 0 is greedy (evict the
            minimum, as in H2O), tau -> inf is random; rule "oldest" evicts the oldest block instead
            (a sliding window over blocks).
Fine-tuning under eviction, exact: stream() runs the decode token by token and records when each block left
(evicted_at); the training step then runs the normal parallel forward with those blocks masked from the
indexer (csa_attention(evicted_at=...)), which gives every token exactly the pool it had in the decode.
    .env/bin/python evict.py check
    SEQ=2048 BATCH=2 STEPS=2000 PEAK_LR=1e-4 .env/bin/python evict.py tune sqrt 4   -> runs/train/tune_csa_sqrt4.*
    SEQ=2048 BATCH=2 .env/bin/python evict.py eval                                   -> runs/evict/evict.csv
"""
import math
import os
import pickle
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from csa import compress, csa_decode_step  # noqa: E402
from transformer import forward, init_params, layer_norm  # noqa: E402

KINDS = {"full": 0, "sqrt": 1, "log": 2}
RULES = {"stat": (1.0, False), "greedy": (1e-4, False), "random": (1e9, False), "oldest": (1.0, True)}  # tau, oldest
NEVER = 2**30  # evicted_at of a block that is never evicted
WINDOW = 32  # csa_attention's default, as used by transformer.mix


def budget(kind, c, n):
    """Pool size allowed with n finished blocks. kind: 0 full, 1 sqrt, 2 log (traced)."""
    return jnp.where(kind == 0, jnp.inf, jnp.where(kind == 1, c * jnp.sqrt(n), c * jnp.log2(1.0 + n)))


def stream(params, tokens, key, kind, c, tau, oldest):
    """Decode `tokens` (T,) one at a time with an evicting block pool in every layer.
    Returns hidden (T, D) (logits = hidden @ tok_emb.T), evicted_at (layers, ceil(T / block_size)) and the
    pool size after every token (layers, T)."""
    seq_len, d_model = len(tokens), params["tok_emb"].shape[1]
    first = params["layers"][0]["csa"]
    block_size, kv_dim = first["kv_compressor"]["bias_own"].shape
    indexer_dim = first["indexer"]["key_compressor"]["bias_own"].shape[1]
    n_heads = first["sink"].shape[0]
    n_blocks = -(-seq_len // block_size)
    cap = n_blocks + 1  # ponytail: room for every block, so one compile serves every budget; a real decoder
    # would size it to the budget (the pool size, not cap, is what the scaling is about)

    def empty_layer():
        return {
            "window_kv": jnp.zeros((WINDOW - 1, kv_dim)),  # raw entries of the last window - 1 tokens
            "recent": jnp.zeros((2 * block_size, d_model)),  # last 2 blocks of layer inputs, for compress
            "kv": jnp.zeros((cap, kv_dim)), "indexer_keys": jnp.zeros((cap, indexer_dim)),
            "block": jnp.full(cap, -1),  # block id per slot, -1 = empty
            "stat": jnp.zeros((cap, n_heads)), "seen": jnp.zeros(cap),  # attention sum, tokens counted
            "evicted_at": jnp.full(n_blocks, NEVER),
        }

    def evict(s, t, key):
        size = (s["block"] >= 0).sum()
        evictable = s["seen"] > 0  # has left the window (empty slots have seen 0)
        over = size - budget(kind, c, (t + 1) // block_size)
        key_do, key_pick = jax.random.split(key)
        do = (jax.random.uniform(key_do) < jnp.clip(over, 0, 1)) & evictable.any()
        score = (s["stat"] / jnp.maximum(s["seen"], 1)[:, None]).max(1)
        score = score / jnp.maximum(jnp.where(evictable, score, 0).sum() / jnp.maximum(evictable.sum(), 1), 1e-9)
        logits = jnp.where(oldest, -s["block"] * 1e4, -score / tau)
        slot = jax.random.categorical(key_pick, jnp.where(evictable, logits, -jnp.inf))
        gone = jnp.where(do, s["block"][slot], n_blocks)  # n_blocks: out of range, the scatter drops it
        s["evicted_at"] = s["evicted_at"].at[gone].set(t + 1, mode="drop")
        s["block"] = s["block"].at[slot].set(jnp.where(do, -1, s["block"][slot]))
        return s, size - do

    def layer_step(p, s, a, t, key):
        """One CSA layer for the token at t, a: its layer input. Returns the mixer output and the new state."""
        cache = {k: s[k] for k in ("window_kv", "indexer_keys", "block")} | {"compressed_kv": s["kv"]}
        y, picked, weight = csa_decode_step(p, cache, a, t, return_picks=True)
        s = dict(s)
        # stats, only for blocks that have left the window: the block's last token is before the window
        left = (s["block"] >= 0) & ((s["block"] + 1) * block_size <= t - WINDOW + 1)
        s["stat"] = jnp.where(left[:, None], s["stat"].at[picked].add(weight.T), 0.0)
        s["seen"] = s["seen"] + left
        # the window moves on; a block completed by this token joins the pool (usable from t + 1, as in prefill)
        s["window_kv"] = jnp.concatenate([s["window_kv"][1:], (a @ p["window_kv"])[None]])
        s["recent"] = jnp.concatenate([s["recent"][1:], a[None]])
        b = t // block_size
        complete = (t + 1) % block_size == 0
        slot = jnp.argmax(s["block"] < 0)

        def entry(compressor):  # block b: own tokens + previous block's; block 0 has no previous block
            first_block = compress(compressor, s["recent"][block_size:])[0]
            return jnp.where(b == 0, first_block, compress(compressor, s["recent"])[1])

        for name, compressor in (("kv", p["kv_compressor"]), ("indexer_keys", p["indexer"]["key_compressor"])):
            s[name] = s[name].at[slot].set(jnp.where(complete, entry(compressor), s[name][slot]))
        s["block"] = s["block"].at[slot].set(jnp.where(complete, b, s["block"][slot]))
        s["stat"] = s["stat"].at[slot].set(jnp.where(complete, 0.0, s["stat"][slot]))
        s["seen"] = s["seen"].at[slot].set(jnp.where(complete, 0.0, s["seen"][slot]))
        s, size = evict(s, t, key)
        return y, s, size

    def token_step(carry, inputs):
        states, key = carry
        token, t = inputs
        key, *layer_keys = jax.random.split(key, len(params["layers"]) + 1)
        x = params["tok_emb"][token] + params["pos_emb"][t]
        sizes = []
        for i, p in enumerate(params["layers"]):  # same layer as transformer.forward, one token
            y, states[i], size = layer_step(p["csa"], states[i], layer_norm(p["ln1"], x), t, layer_keys[i])
            x = x + y
            x = x + jax.nn.gelu(layer_norm(p["ln2"], x) @ p["fc1"]) @ p["fc2"]
            sizes.append(size)
        return (states, key), (layer_norm(params["ln_f"], x), jnp.stack(sizes))

    states = [empty_layer() for _ in params["layers"]]
    (states, _), (hidden, sizes) = jax.lax.scan(token_step, (states, key), (tokens, jnp.arange(seq_len)))
    return hidden, jnp.stack([s["evicted_at"] for s in states]), sizes.T


def run(params, tokens, key, kind, c, rule):
    """stream over a batch (B, T); each sequence gets its own random key."""
    tau, oldest = RULES[rule]
    keys = jax.random.split(key, len(tokens))
    return _run(params, tokens, keys, KINDS[kind], float(c), tau, oldest)


_run = jax.jit(jax.vmap(stream, in_axes=(None, 0, 0, None, None, None, None)))


def evict_loss(params, batch):
    """train_lm.loss_fn with the decode's evictions replayed: batch = (tokens (B, T+1), evicted_at)."""
    tokens, evicted_at = batch
    logits = jax.vmap(lambda x, e: forward(params, x, n_heads=params["layers"][0]["csa"]["sink"].shape[0],
                                           evicted_at=e))(tokens[:, :-1], evicted_at)
    target = jnp.take_along_axis(logits, tokens[:, 1:, None], axis=-1)[..., 0]
    return (jax.nn.logsumexp(logits, axis=-1) - target).mean()


def check():
    """stream vs the parallel forward on a small random CSA model, fp32."""
    key = jax.random.PRNGKey(0)
    params = init_params(key, 64, d_model=32, n_heads=4, n_layers=2, max_len=64, mixer="csa")
    tokens = jax.random.randint(key, (2, 61), 0, 64)
    logits = lambda hidden: hidden @ params["tok_emb"].T
    with jax.default_matmul_precision("highest"):
        full = jax.vmap(lambda x: forward(params, x))(tokens)
        hidden, evicted_at, sizes = run(params, tokens, key, "full", 0, "stat")
        assert (evicted_at == NEVER).all()
        assert jnp.abs(logits(hidden) - full).max() < 1e-4, "stream without eviction != forward"
        assert (sizes[:, :, -1] == 60 // 4).all()  # 61 tokens: 15 complete blocks
        for rule in RULES:
            hidden, evicted_at, sizes = run(params, tokens, key, "sqrt", 1, rule)
            replay = jax.vmap(lambda x, e: forward(params, x, evicted_at=e))(tokens, evicted_at)
            diff = jnp.abs(logits(hidden) - replay).max()
            assert diff < 1e-4, f"[{rule}] stream != forward with its evictions replayed: {diff:.1e}"
            assert (evicted_at < NEVER).sum() > 10, f"[{rule}] too few evictions"
            # pool <= budget + 1, except while the window holds more blocks than the budget allows
            n = (jnp.arange(61) + 1) // 4
            bound = jnp.maximum(jnp.ceil(jnp.sqrt(n)), WINDOW // 4 + 1) + 1
            assert (sizes <= bound).all(), f"[{rule}] pool over budget"
            if rule == "oldest":  # blocks leave in order
                for e in evicted_at.reshape(-1, evicted_at.shape[-1]):
                    gone = e[e < NEVER]
                    assert (jnp.diff(gone) >= 0).all()
        assert not jnp.allclose(full, replay, atol=1e-3), "evictions changed nothing"
    print("evict checks passed: decode == parallel forward, with and without evictions")


if __name__ == "__main__" and sys.argv[1] == "check":
    check()
    sys.exit()

import train_lm  # noqa: E402  (reads SEQ, BATCH, STEPS, PEAK_LR from the environment)


def load_params(name):
    with open(f"runs/train/{name}.pkl", "rb") as f:
        return jax.tree.map(jnp.asarray, pickle.load(f))


# The token-by-token stream costs ~2 s at T = 2048 whatever the batch (32 sequences: 2.2 s, step overhead
# dominates), the training step 0.07 s. So one stream call covers SIM_EVERY steps' batches.
# ponytail: evictions of steps 2..SIM_EVERY come from weights up to SIM_EVERY - 1 steps old (at LR <= 1e-4);
# SIM_EVERY = 1 is exact, ~6x slower.
SIM_EVERY = 8


def tune(kind, c):
    """Fine-tune the SEQ 2048 CSA model with its decode's evictions (kind "full": the no-eviction control)."""
    name = f"tune_csa_{kind}{c if kind != 'full' else ''}"
    params = load_params(f"csa_seq{train_lm.SEQ}")
    zeros = lambda: jax.tree.map(jnp.zeros_like, params)
    state = (params, zeros(), zeros(), jnp.zeros((), jnp.int32))
    train = train_lm.load("train")
    rng, key = np.random.default_rng(1), jax.random.PRNGKey(1)  # not the pretraining batches
    n_layers, n_blocks = len(params["layers"]), train_lm.SEQ // 4
    log = open(f"runs/train/{name}.csv", "w")
    log.write("step,tokens,train_loss,seconds\n")
    t0 = time.time()
    for step in range(1, train_lm.STEPS + 1):
        i = (step - 1) % SIM_EVERY
        if i == 0:  # the next SIM_EVERY batches, and their decode's evictions in one stream call
            starts = rng.integers(0, len(train) - train_lm.SEQ - 1, (SIM_EVERY, train_lm.BATCH))
            chunk = jnp.asarray(train[starts[..., None] + np.arange(train_lm.SEQ + 1)], jnp.int32)
            if kind == "full":
                evictions = jnp.full((SIM_EVERY, train_lm.BATCH, n_layers, n_blocks), NEVER)
            else:
                flat = chunk.reshape(-1, train_lm.SEQ + 1)[:, :-1]
                evictions = run(state[0], flat, jax.random.fold_in(key, step), kind, c, "stat")[1]
                evictions = evictions.reshape(SIM_EVERY, train_lm.BATCH, n_layers, n_blocks)
        tokens, evicted_at = chunk[i], evictions[i]
        state, loss, norm = train_lm.train_step(state, (tokens, evicted_at), loss=evict_loss)
        if step % 100 == 0:
            el = time.time() - t0
            print(f"[{name}] step {step}: loss {float(loss):.3f}  |g| {float(norm):.2f}  {el / step:.2f} s/step",
                  flush=True)
            log.write(f"{step},{step * train_lm.BATCH * train_lm.SEQ},{float(loss):.4f},{el:.1f}\n")
            log.flush()
    with open(f"runs/train/{name}.pkl", "wb") as f:
        pickle.dump(jax.device_get(state[0]), f)


@jax.jit
def seq_nll(tok_emb, hidden, targets):
    """Next-token loss per position from stream's hidden states, one sequence at a time (logits are big)."""
    one = lambda a: jax.nn.logsumexp(a[0] @ tok_emb.T, -1) - (a[0] * tok_emb[a[1]]).sum(-1)
    return jax.lax.map(one, (hidden, targets))


def evaluate(params, val, kind, c, rule, batch=32):
    """Decode every val window of SEQ tokens with eviction -> loss, mean pool at T, blocks scanned per
    sequence and layer, mean pool size after each token (T,)."""
    n_win = (len(val) - 1) // train_lm.SEQ
    losses, final, scanned = [], [], []
    for i in range(0, n_win, batch):
        idx = np.arange(i, min(i + batch, n_win))[:, None] * train_lm.SEQ + np.arange(train_lm.SEQ + 1)
        tokens = jnp.asarray(val[idx], jnp.int32)
        hidden, _, sizes = run(params, tokens[:, :-1], jax.random.PRNGKey(i), kind, c, rule)
        losses.append(np.asarray(seq_nll(params["tok_emb"], hidden, tokens[:, 1:])))
        final.append(np.asarray(sizes[:, :, -1]))
        scanned.append(np.asarray(sizes.sum(-1)))
    return (float(np.concatenate(losses).mean()), float(np.concatenate(final).mean()),
            float(np.concatenate(scanned).mean()), np.asarray(sizes).mean((0, 1)))


def main_eval():
    val = train_lm.load("validation")
    models = {"base": f"csa_seq{train_lm.SEQ}", "tuned full": "tune_csa_full", "tuned sqrt 4": "tune_csa_sqrt4",
              "tuned log 10": "tune_csa_log10"}
    configs = [("full", 0, "stat")] + [(kind, c, rule) for kind, cs in (("sqrt", (2, 4, 8)), ("log", (5, 10, 20)))
                                       for c in cs for rule in RULES]
    with open("runs/evict/evict.csv", "w") as log, open("runs/evict/evict_pool.csv", "w") as pool_log:
        log.write("model,budget,c,rule,pool_at_T,scanned_blocks_per_seq,val_loss,val_ppl\n")
        pool_log.write("budget,c,t,pool\n")
        for model, name in models.items():
            if not os.path.exists(f"runs/train/{name}.pkl"):
                continue
            params = load_params(name)
            for kind, c, rule in configs:
                if model != "base" and rule != "stat":  # tuned models: only the rule they were tuned with
                    continue
                loss, pool, scanned, pool_by_t = evaluate(params, val, kind, c, rule)
                print(f"{model:13s} {kind:4s} c={c:2d} {rule:6s}: pool at T {pool:6.1f}, scanned {scanned:9.0f}, "
                      f"val loss {loss:.4f} (ppl {math.exp(loss):.2f})", flush=True)
                log.write(f"{model},{kind},{c},{rule},{pool:.1f},{scanned:.0f},{loss:.4f},{math.exp(loss):.2f}\n")
                log.flush()
                if model == "base" and rule == "stat":  # pool size over time, once per budget
                    for t in range(15, train_lm.SEQ, 16):
                        pool_log.write(f"{kind},{c},{t + 1},{pool_by_t[t]:.2f}\n")


if __name__ == "__main__":
    {"tune": lambda: tune(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0), "eval": main_eval}[sys.argv[1]]()
