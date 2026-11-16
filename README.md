# Sparse decode attention: synthetic benchmarks

Sparse attention for decoding: one new query attends over a long KV cache, reading only some keys.
- Every benchmarked version is listed in one registry: `src/attention/registry.py`.
- Three benchmarks run every version: latency, fidelity, tasks.
- Results go to local MLflow; there is one figure per benchmark, with every version on it.
- Synthetic data only, except one training test: modded-nanogpt record 91 with tournament attention (`src/nanogpt/`). The other natural-text work (Qwen, CSA2, WikiText, the paper) is removed; git tag `before-synthetic-reorg` recovers it.
- Hardware: one GPU (RTX 4070 Ti SUPER 16 GB, WSL).

## Layout

```
src/attention/            every attention version
  registry.py             VERSIONS: the only list of what is benchmarked
  harness.py              Chunk, Outcome, measures; hook into any HF model (AttentionInterface)
  dense.py exact_topk.py sparq.py loki.py bandit.py block_topk.py
  tournament/             general.py + one file per tournament variant
  triton/                 kernels: common.py, sparq.py, tournament_*.py
  fast_pytorch.py         plain PyTorch decode steps for versions without a kernel
src/benchmarks/
  synthetic.py            generated Q/K/V, task sequences
  latency.py fidelity.py tasks.py
  run.py                  suite runner (MLflow, resumable)
  plot.py                 figures + tables
  suites/standard.toml    the full suite; smoke.toml: minutes
src/nanogpt/              record 91 training (worktree ~/repos/modded-nanogpt-r91):
  tournament.py           tournament: shuffled blocks, growing budget, per query on its document prefix (Triton + flex)
  tournament_fast.py      the same at any block / winners / channels per round: autotuned kernels, document-window bits; tournament.py's winners at its settings
  plot.py                 every nanogpt figure -> runs/nanogpt/<topic>/ (training, checkpoints, selection_speed, long_context, settings; csa2: old)
src/common/               timing.py (do_bench_cudagraph), tracking.py (MLflow), style.py
tests/                    pytest, CPU (conftest hides the GPU; Triton tests skip); mirrors src/:
                          tests/<package>/test_<module>.py tests src/<package>/<module>.py
runs/benchmarks/<suite>/  figures (PDF + PNG), tables.md with captions
runs/mlflow.db, runs/mlartifacts/, runs/tasks/, runs/logs/   git-ignored
archive/bandits/          frozen, neither linted nor tested
docs/methods.md           every version and setting, explained simply
docs/papers/              reference papers (PDF)
```

## Naming rules

- Version id: `<family>/<variant>`, e.g. `tournament/global-halving`, `block-topk/double-indexing-bound`.
- No settings or codes in ids. Settings live in the registry entry and are logged as MLflow params.
- Each version has a short name; figures label rows by name and kernel (the implementation timed).
- [docs/methods.md](docs/methods.md) explains every version and setting in plain words.
- Every sparse version gives exact attention to a budget of keys: KEEP = 512 by default; the suites also sweep 128–2048.
  - Exceptions, checked in `tests/attention/test_correctness.py`: block top-k and double indexing up to 3 keys fewer (one chosen block can straddle the recent window).
  - Tournaments keep every key tied at a cutoff, so exact copies of a key (synthetic sinks, needles) can leave a few extra.
  - Block and bracket tournaments end below the budget (above half) unless the length is the budget times a power of two; the suites use only such lengths.
  - Shuffled blocks, growing budget: budget × 1024 ÷ 32768 winners per 1024-key block, so budget × n ÷ 32K keys; not dense on short contexts.
  - DeepSeek-V4 CSA: 128 raw recent keys plus top-budget compressed entries; latency only.
  - DeepSeek-V4.1 CSA2: 128 recent keys plus the top-budget keys by a random indexer; latency only.
  - DeepSeek-V4.1 CSA2 with our picks (exact, block-mean, block-mean wide pool): 128 recent keys plus budget picks before them, so budget + 128 keys.
- Figures are paper-style (7 pt serif, vector PDF + PNG), without titles; each caption (shape, dtype, distribution, repeats or seeds) is in `tables.md`.
- Every figure lists the versions in registry order, one row each.

## Versions

| Id | Name | Kernel |
|---|---|---|
| `dense` | Dense | PyTorch |
| `exact-topk` | Exact top-k | PyTorch |
| `exact-topk/summed-query` | Exact top-k, summed query | PyTorch |
| `sparq` | SparQ | Triton |
| `loki` | Loki | PyTorch |
| `deepseek-v4/csa` | DeepSeek-V4 CSA | transformers |
| `deepseek-v4/csa-flash` | DeepSeek-V4 CSA, V4-Flash size | transformers |
| `deepseek-v4.1/csa2` | DeepSeek-V4.1 CSA2 | PyTorch |
| `deepseek-v4.1/csa2-exact-picks` | DeepSeek-V4.1 CSA2, exact picks | PyTorch |
| `deepseek-v4.1/csa2-block-mean` | DeepSeek-V4.1 CSA2, block-mean picks | PyTorch |
| `deepseek-v4.1/csa2-block-mean-wide` | DeepSeek-V4.1 CSA2, block-mean picks, wide pool | PyTorch |
| `deepseek-v4.1/csa2-block-mean-half-wide` | DeepSeek-V4.1 CSA2, block-mean picks, half-wide pool | PyTorch |
| `deepseek-v4.1/csa2-block-mean-fp4-wide` | DeepSeek-V4.1 CSA2, FP4 block-mean picks, wide pool | PyTorch |
| `deepseek-v4.1/csa2-sign-sketch` | DeepSeek-V4.1 CSA2, sign-sketch picks | PyTorch |
| `deepseek-v4.1/csa2-pca-sketch` | DeepSeek-V4.1 CSA2, PCA-sketch picks | PyTorch |
| `deepseek-v4.1/csa2-random-sketch` | DeepSeek-V4.1 CSA2, random-sketch picks | PyTorch |
| `deepseek-v4.1/csa2-channel-sketch` | DeepSeek-V4.1 CSA2, top-channel picks | PyTorch |
| `deepseek-v4.1/csa2-head-max` | DeepSeek-V4.1 CSA2, max-over-heads picks | PyTorch |
| `block-topk/flat` | Block top-k | PyTorch |
| `block-topk/double-indexing-mean` | Double indexing: mean | PyTorch |
| `block-topk/double-indexing-bound` | Double indexing: bound | PyTorch |
| `tournament/global-halving` | Tournament: global halving | Triton |
| `tournament/blocks` | Tournament: blocks | PyTorch |
| `tournament/blocks-then-global` | Tournament: blocks, then global | Triton |
| `tournament/wide-first-round` | Tournament: wide first round | PyTorch |
| `tournament/sampled-cutoffs` | Tournament: sampled cutoffs | Triton |
| `tournament/sampled-cutoffs-approximate-losers` | Tournament: sampled cutoffs, approximate losers | Triton |
| `tournament/channel-groups` | Tournament: channel groups | PyTorch |
| `tournament/mean-value-losers` | Tournament: mean-value losers | PyTorch |
| `tournament/shuffled-blocks` | Tournament: shuffled blocks | Triton |
| `tournament/shuffled-blocks-growing` | Tournament: shuffled blocks, growing budget | Triton |
| `tournament/original` | Tournament: original | PyTorch |
| `bandit/exact-bound` | Bandit: exact bound | PyTorch |
| `bandit/gaussian-bound` | Bandit: Gaussian bound | PyTorch |

- Two implementations per version:
  - emulation: float32, exact selection and read counts; the quality benchmarks run it;
  - fast step: what the latency benchmark times.
- `tests/attention/test_registry.py` checks that both select the same keys.
- DeepSeek-V4 CSA is the transformers layer, unchanged, with random weights: fast step only, no emulation, so latency only. One row at the suite's shapes (projections sized down), one at the library's default V4-Flash sizes (64 heads × 512, batch 1).
- Bandit fast steps mask over all keys and do no compaction, so their latency is an upper bound.

## Benchmarks

| Benchmark | Input | Measures |
|---|---|---|
| latency | bf16 K/V from `synthetic.qkv`, 8K–128K keys, budgets 128–2048, batch 1, 8 KV heads × 2 query heads, head dim 128; iid, LLM-like | µs per decode step: median of 10 repeats, CUDA graphs |
| fidelity | float32 Q/K/V, 8K–128K keys, budgets 128–2048, 128 decode queries, 10 seeds; 7 distributions (docs/methods.md) | output error vs dense, attention mass kept, top-128 recall, K/V share read, K sectors |
| tasks (not usable yet) | one tiny Llama (4 layers, width 512, 4 query and 2 KV heads) trained on copy, MQAR and needle mixed, dense attention; tested at 4K–16K | answer accuracy with the version in every layer |

## Commands

```
uv sync
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests && MYPYPATH=src uv run mypy src tests

# full suite, detached; one GPU job at a time; a rerun skips finished runs
setsid nohup uv run python -m benchmarks.run src/benchmarks/suites/standard.toml > runs/logs/standard.log 2>&1 < /dev/null &
uv run python -m benchmarks.run src/benchmarks/suites/smoke.toml --only latency --versions dense sparq
uv run python -m benchmarks.plot                    # runs/benchmarks/standard/: latency, fidelity, fidelity-harder, budget (.png), tables.md
uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db

# record 91 with tournament attention, in the worktree; GPU check, then figures
PYTHONPATH=src:. ~/repos/modded-nanogpt-r91/.venv/bin/python -m tests.nanogpt.test_tournament
uv run python -m tests.nanogpt.test_tournament_fast     # GPU check of tournament_fast.py (about 30 min of first compiles)
cd ~/repos/modded-nanogpt-r91 && TRAIN_SEED=1 SWEEP_VAL_EVERY=500 setsid nohup ./run_tournament.sh > ~/repos/scratch/runs/logs/nanogpt_tournament.log 2>&1 < /dev/null &
uv run python -m nanogpt.plot                       # runs/nanogpt/<topic>/*.png
```

- ⚠ The tasks benchmark is not usable yet: dense attention scores at chance on MQAR and needle, and copy works only at some lengths, so the model has not learned content lookup. Run `--only latency fidelity` until that is fixed.
- A budget runs only where the context is at least 16 times it (`registry.MIN_RATIO`).
- Add `--force` to rerun finished runs.
- Kill a run by its PID; a killed run is not finished, so it reruns.

## Adding a version

1. Emulation: `Chunk -> Outcome` under `src/attention/`, in its own file.
2. Fast step: a Triton kernel in `src/attention/triton/`, or a function in `fast_pytorch.py`.
3. Add a `Version` to `registry.versions()`: id, name, implementation, emulation, fast step, all built at budget `keep`. Emulation `None` = latency only.
4. Explain it in `docs/methods.md` (`tests/test_docs.py` checks every id and name is there).
5. `uv run pytest tests/attention/test_registry.py tests/attention/test_correctness.py`; GPU cases: `uv run python -m tests.attention.test_registry`.
