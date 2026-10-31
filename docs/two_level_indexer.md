# Double indexing: a two-level CSA2 indexer

Double indexing replaces how a DeepSeek-V4.1 CSA2 Full or Reindex layer chooses its top-k main KV entries. It first
scores groups of S consecutive entries (superblocks) and keeps the best s groups. It then runs the layer's own
indexer and top-k only over entries inside those groups and the query's own unfinished group. Compression,
cross-layer reuse, window, sink and attention are unchanged, so it drops into a trained CSA2 model.

Code: `src/deepseek/double_indexing.py` (`DoubleIndexing`, `Router`, `add_router`, `pick`, `summary_keys`,
`score_summaries`, `candidates`, `query_reads`). It plugs into `src/deepseek/csa2.py` through
`Csa2Attention.selector`. The JAX version and its results are frozen in `archive/jax/`.

## 1. Problem: pure CSA2's indexer cost

| Quantity | Pure CSA2, per Full or Reindex layer |
|---|---|
| Entries finished before token t (0-based) | n = (t + 1) // 4 |
| Indexer keys read per decoded token | n (all finished entries) |
| Prefill indexer work | Σ_t n ≈ T²/8 scores |
| At T = 131,072 | 32,768 keys per token |

- Reuse layers read nothing: they take the latest picks. Our 8 layers run Full, Reuse, Reindex, Reuse, twice
  (assumed), so layers 0, 2, 4, 6 select.
- The indexer is cheap per key but scans everything; it dominates at long context.
- HF-style prefill builds a (T, T/4) score matrix per indexer head; `bench/layer.py` measures where it OOMs.
- V4.1's Hierarchical Sparse Indexer (§2.3.2) narrows Reindex layers to a candidate pool, but its first Full layer
  still scans every entry. Double indexing removes that scan; it is not implemented here.

## 2. Design

```
layer input x (T, 512)
  │
  ├─ Full layer: compress 4 tokens → 1 main KV entry (64) ──► indexer key = projection (32)
  │                                        │                          │
  │                         ┌──────────────┘                          ▼
  │                         │             superblocks: S consecutive entries each
  │                         │                                         │
  │                         │      coarse: score n // S summaries, keep top s superblocks
  │                         │                                         │
  │                         │      fine: indexer scores entries of kept superblocks
  │                         │            + own unfinished superblock → top k = 8
  │                         ▼                                         │
  └─ attention over [8 picked entries ; 32-token window ; sink] ◄─────┘
     Reuse layers: same entries, same picks; Reindex layers: same entries, own picks
```

Path in words: the shared indexer keys feed the coarse step and then the fine step, which pick 8 entries for the
unchanged attention of the layer and its Reuse followers.

## 3. Algorithm

`DoubleIndexing(S, s, coarse)` is a `csa2.Selector`: `selector(attention, SelectorInputs) -> Picks`.

```
FOR each query t with n finished entries:
    done    = n // S                                     # complete superblocks
    coarse  = score_summaries(q_t, w_t, summary of g)  FOR g < done       (no grad)
    kept    = top_s(coarse)
    allowed = entries of kept superblocks  ∪  entries done·S … n − 1     # own superblock
    score   = indexer score of every allowed entry (Eq. 16 of V3.2: Σ_h w_h relu(q_h · k))
    RETURN top_k(score)  as Picks(entry, valid, score)                   # same as the indexer
```

- `DoubleIndexing.__call__` scores every entry and masks the disallowed ones: simple, used for every loss.
- `pick` scores only the candidates and returns the same picks; `bench/indexer.py` times it.
- Candidates per query: s·S kept entries + at most S own entries.

## 4. Reads per query (`query_reads`)

| Mode | Coarse keys | Fine keys | Total |
|---|---|---|---|
| pure CSA2 | n/a | n | n |
| mean | n // S | min(s, n // S)·S + n % S | ≈ n/S + s·S + S/2 |
| learned, r keys | r·(n // S) | same | ≈ r·n/S + s·S + S/2 |
| bound | 2·(n // S) | same | ≈ 2n/S + s·S + S/2 |
| oracle | n | same | > n (diagnostic) |

- The fine part is constant in T; only the coarse part grows.
- Example: T = 131,072, mean or learned, S = 16, s = 4 → ≈ 2,120 keys instead of 32,768 (15×).
- `reads` sums these over a sequence; `experiments/selector_forward.reads_per_seq` averages over selecting layers.

## 5. Coarse modes (`summary_keys`, `score_summaries`)

Notation: q_h indexer query of head h, w_h head weight (HF scorer's `weights_proj`, scaled), relu = max(·, 0),
scale = the scorer's `softmax_scale`.

| Mode | Stored per superblock | Coarse score | Training |
|---|---|---|---|
| mean | mean key k̄ (c) | Σ_h w_h · relu(q_h · k̄) · scale | none |
| bound | midpoint m, half-range r (2, c) | Σ_h relu(w_h) · relu(q_h · m + \|q_h\| · r) · scale | none |
| learned | n_keys pooled keys p_j | max_j Σ_h w'_h · relu(q'_h · p_j) · scale | router, §6 |
| oracle | all S entry keys | max over entries of the indexer score | none; reads all |

- bound (Quest-style): for heads with w_h > 0, an upper bound on the superblock's best entry score.
  - It catches single strong entries (needles) that a mean hides.
  - It is loose on superblocks with varied keys, so it needs a larger s.
- learned: q', w' come from the router's own indexer copy, not the layer's indexer.
- oracle: V4.1's blockwise candidate selection (§2.3.2) per layer; the ceiling of any summary key.

## 6. Learned router (`Router`, `add_router`)

A `Router` subclasses `csa2.Csa2Indexer` (query projection and head weights, copied from the layer's indexer at
init) and adds a pooling. `add_router(attention, S, n_keys)` sets `attention.router`.

| Parameter | Shape | Init |
|---|---|---|
| indexer weights | as `Csa2Indexer` | copy of the layer's indexer |
| score | (c, n_keys, c) | zeros |
| bias | (S, n_keys, c) | 0 on key j's own run of S / n_keys entries, −3 elsewhere |
| value | (c, c) | identity |
| temperature | () | 0 (log scale; loss only) |

Pooled key j, channel d, over the S entry keys b_i of a superblock:

```
weight[i, j, d] = softmax_i( (b_i · score[:, j, d]) + bias[i, j, d] )
p_j[d]          = Σ_i weight[i, j, d] · (b_i · value)[d]
```

- With n_keys = 1 at init, p equals the mean key, so learned starts as mean (`test_learned_router_starts_as_mean`).

### Training (`src/experiments/train_router.py`)

1. [frozen CSA2 model] → forward on one random train window, bf16 autocast → [per selecting layer: inputs, picks, attention on picks]
2. [attention on picks] → mean over heads and over the layer and its Reuse followers → [weight per pick]
3. [weight per pick] → drop picks in the query's unfinished superblock, sum the rest per superblock → [target (T, G)]
4. [router] → coarse scores × exp(temperature), log-softmax over complete superblocks → [log p (T, G)]
5. [target, log p] → loss = Σ_layers mean_t −Σ_g target · log p → Adam step on routers only → [router]

- Target rows are not normalized: a query that mostly attends to its window weighs little.
- Optimizer: Adam, lr 1e-3, cosine; `STEPS` default 600, `KEYS` default 1.
- Logs attention recall (share of the target inside the coarse top-s) on 4 val windows, for mean, bound, oracle and
  the trained router.
- Output: `runs/route/router_seq{T}_S{S}_r{n_keys}[_s{seed}].pt`, one state dict per selecting layer. A router is
  tied to one S (`coarse_scores` raises `ValueError` otherwise).

## 7. Per-layer routes and needle layers

`experiments/selector_forward.selecting(model, selector)` accepts:
- one selector for every Full and Reindex layer;
- a sequence with one entry per layer, indexed by layer (`None` = the layer's own indexer; entries of Reuse
  layers are ignored).

Rules (`src/experiments/eval_route.py`, mode `mixed`):
- Calibration routes one selecting layer at a time with learned (S, 4) on 2 train windows (every other layer flat),
  and measures the relative squared error of that layer's attention output.
- Every layer with error above the mean over selecting layers is a needle layer.
- Every needle layer uses `(S', s', "bound")`; every other selecting layer uses `(S, s, "learned")`.

## 8. Results

Full tables: `README.md` (Results) and `report/paper.pdf`. Cheapest setting of each mode that stays within 0.01 nats
of pure CSA2's validation loss, model 1, written as (superblock size S, superblocks kept s):

| Mode | 2048 tokens | 8192 tokens |
|---|---|---|
| mean | (8, 8), 2.7× fewer reads | (64, 2), 6.1× fewer reads |
| bound | (8, 8), 2.0× | (32, 4), 5.1× |
| learned | (16, 4), 3.1× | (16, 4), 7.7× |
| mixed | (8, 1), needle layers (8, 8), 3.0× | (16, 2), needle layers (16, 8), 7.2× |

- Needle layers: 2 and 4 on model 1 at 2048 tokens; layer 4 at 8192 tokens and on model 2.
- Model 2 at model 1's settings: mean, bound and learned still match; mixed does not.
- Copy test: bound has the lowest worst case, the learned router the highest.
- Indexer decode at 128k, 64 sequences: learned (16, 4) 109 µs vs pure CSA2 1518 µs.
- Prefill index is slower than pure CSA2 up to 16k tokens, faster from 32k.

## 9. Invariants (tested in `tests/test_double_indexing.py`)

- Every superblock kept (s ≥ G) → picks identical to the indexer's (`test_keeping_every_superblock_is_flat`).
- Picks match a per-query loop over the definition (`test_matches_per_query_loop`).
- Every candidate is a finished entry; superblock g is a coarse candidate only if complete
  (`test_routed_model_is_causal`).
- The own unfinished superblock is always a candidate, so recent entries are never routed away.
- `pick` (scores only candidates) picks the same entries as the masked `__call__` (`test_fast_pick_matches_mask`).
- Selectors run under `no_grad` and replace the indexer, so no indexer loss is attached while one is set.

## 10. API

| Name | Signature | Role |
|---|---|---|
| `DoubleIndexing` | `(superblock_size, kept_superblocks, coarse="mean")` | frozen dataclass and `csa2.Selector` |
| `DoubleIndexing.coarse_step` | `(score (B,S,G), finished (B,S,1))` | kept superblock ids (B, S, s) |
| `Router` | `(config, superblock_size, n_keys=1)` | learned summary keys and coarse query |
| `add_router` | `(attention, superblock_size, n_keys=1)` | fresh router on a selecting layer |
| `superblocks` | `(index_k (B,n,c), S)` | (B, G, S, c), complete superblocks |
| `summary_keys` | `(blocks, coarse, router=None)` | (B, G, r, c) keys the coarse step scores |
| `score_summaries` | `(q, weights, keys, coarse, scale)` | (B, S, G) coarse scores, float32 |
| `coarse_scores` | `(attention, inputs, q, route)` | the above for one layer, router query if learned |
| `candidates` | `(kept, finished, S, n)` | (B, S, n) allowed-entry mask |
| `query_reads`, `reads` | `(finished, route, n_keys)`, `(T, m, route, n_keys)` | §4 per query, per sequence |
| `pick` | `(q, weights, index_k, summaries, finished, top_k, route, scale)` | fast path → `Picks` |

Cache cost of routing: caching summary keys adds 1/S (mean), n_keys/S (learned) or 2/S (bound) of the indexer key
cache. ⚠ The CSA2 model has no KV cache; `bench/indexer.py` times `pick` on precomputed summary keys.

## 11. Reproduce

1. Train the base: `SEQ=8192 BATCH=1 STEPS=12000 uv run python -m model.train csa2`.
2. Pure CSA2 reference rows: `SEQ=8192 BATCH=1 uv run python -m experiments.eval_csa`.
3. Routers: `SEQ=8192 uv run python -m experiments.train_router 16 32`.
4. Each mode: `SEQ=8192 BATCH=1 uv run python -m experiments.eval_route <mode>` for mean, bound, oracle, learned,
   mixed.
5. Repeat 1–4 with `SEQ=2048 BATCH=2 STEPS=24000` and routers `8 16`.
6. Time: `uv run python -m bench.indexer` → `runs/bench/indexer.csv`.
7. Plot: `uv run python -m plots.route` → `runs/route/route.png`.
8. Test: `uv run pytest tests/test_double_indexing.py`.

⚠ Run GPU jobs one at a time (WSL: concurrent jobs slow each other and fail allocations).

## 12. Files

| Path | Content |
|---|---|
| `src/deepseek/double_indexing.py` | selection, coarse modes, router, reads, fast path |
| `src/deepseek/csa2.py` | CSA2 and the `selector` hook |
| `src/experiments/selector_forward.py` | LM forward and val loss with a selector per selecting layer |
| `src/experiments/train_router.py` | router training, recall diagnostics |
| `src/experiments/eval_route.py` | val loss vs reads; modes mean, bound, oracle, learned, mixed; needle calibration |
| `src/bench/indexer.py` | decode and prefill-index timing, 4k–128k: flat vs double indexing |
| `src/plots/route.py` | `runs/route/route.png`: loss fronts at 2048/8192, indexer bench |
| `runs/route/route_loss.csv` | T, seed, coarse, S, s, reads_per_seq, reads_ratio, val_loss, delta_vs_flat |
| `runs/bench/indexer.csv` | method, T, decode µs, prefill ms, prefill peak GiB, indexer KiB per token |
| `runs/route/router_seq*_S*_r*.pt` | trained routers (git-ignored) |
| `tests/test_double_indexing.py` | per-query loop reference, causality, fast path vs mask, router init, reads |

## 13. Limitations

- Two seeds at T = 2048, one at T = 8192; one small model (8 layers, d 512, 98M training tokens).
- Quality only up to the training length; 16k–128k is timing only.
- Plain PyTorch under `torch.compile` (decode in CUDA graphs); no fused gather / top-k kernel.
- The needle rule (error above the mean) and needle (16, 8) at T = 8192 were chosen after first results.
- Routing cuts indexer reads and compute, not cache memory.
- Routers train on pure-CSA2 layer inputs; under routing, later layers' inputs shift slightly.
- CSA2 layer modes, the index-key projection before RoPE and the V3.2 indexer loss are assumptions where V4.1 is
  silent.

## 14. Prior art

- MoBA: mean-pooled block keys for top-k block attention.
- Quest: per-page min/max key bounds for query-aware page selection.
- NSA: compressed coarse attention guiding fine block selection.
- HiP: hierarchical top-k pruning.
- HISA, LongCat: coarse-to-fine search for DeepSeek's token indexer.
- DeepSeek-V4.1 Hierarchical Sparse Indexer: a candidate pool from the first Full layer's exact scores, searched
  by later Reindex layers.
- Here: the two-level idea applied to every CSA2 selecting layer's full scan, plus a trained router and per-layer
  budgets, on a frozen model.
