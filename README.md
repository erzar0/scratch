# CSA2 and double indexing (PyTorch)

Small decoder LMs on WikiText-103 (GPT-2 BPE). A cheaper entry picker for DeepSeek-V4.1 Compressed Sparse Attention 2
(CSA2): double indexing, a two-level CSA2 indexer. The model is Hugging Face's DeepSeek-V4 (`transformers` 5.19.0), with
V4.1's CSA2 written as a diff on it. Single GPU (RTX 4070 Ti SUPER 16 GB, WSL).
Frozen: the JAX implementation and its results (`archive/jax/`), the bandit selectors (`archive/bandits/`).

## Layout

```
src/
  deepseek/     library, no side effects on import
    csa.py                 base: HF DeepseekV4ForCausalLM unchanged, every layer CSA, shrunk (SMALL config)
    csa2.py                V4.1 CSA2 as a diff on csa.py; each change marked "# CSA2 (V4.1 ...)"; modes
                           Full / Reuse / Reindex; Csa2Attention.selector hook replaces the indexer's top-k
    double_indexing.py     DoubleIndexing(S, s, coarse): coarse step on superblock summaries, Router
    reference.py           RandomEntries, Oracle
    v4_flash.py            DeepSeek-V4-Flash from its Hub config, no weights (shape checks)
  model/        train.py (csa / csa2, per-layer torch.compile, bf16 autocast)
  bench/        timing.py, layer.py (CSA2 vs SDPA flash layer), indexer.py (flat vs double indexing)
  experiments/  results.py, selector_forward.py (forward with any selector), eval_csa.py (E0), train_router.py (R),
                eval_route.py (E1), paired_se.py (E8: paired SE of each headline), pick_recall.py (E9: share of pure
                CSA2's picks found), copy_test.py (E10: repeated-text retrieval stress test)
  plots/        style.py + one module per figure; PNGs go next to their CSVs in runs/; paper.py and
                methods.py (method illustrations) -> report/figs
tests/          pytest, CPU (tests/conftest.py hides the GPU)
runs/           train/ csa/ route/ bench/ paper/ logs/  (*.pt git-ignored)
report/         paper.typ -> paper.pdf (figs/, numbers.json and the PDF from plots.paper pdf)
docs/           two_level_indexer.md
archive/        jax/: the JAX code and runs; bandits/: bandit selectors and dense control (neither linted nor tested)
```

## Setup

```
uv sync                       # .venv with torch 2.14.1 (CUDA 13), transformers 5.19.0; src/ installed editable
uv run pytest                 # CPU
uv run ruff check src tests && uv run ruff format --check src tests && MYPYPATH=src uv run mypy src tests
```

Run everything from the repo root as modules; `runs/` and `data/` paths are relative to it. Settings come from the
environment: `SEQ`, `BATCH`, `STEPS`, `PEAK_LR` (training and evals), `STEPS`, `LR`, `KEYS` (routers), `SEED` (model
seed; 0 = the first model, other seeds add `_s<n>` to weight, router and result file names).

```
SEQ=2048 BATCH=2 STEPS=24000 uv run python -m model.train csa2         # -> runs/train/csa2_seq2048.{csv,pt,log}
SEQ=2048 BATCH=2 uv run python -m experiments.eval_csa                 # E0 -> runs/csa/csa_loss.csv
SEQ=2048 uv run python -m experiments.train_router 8 16                # R  -> runs/route/router_seq2048_S*_r1.pt
SEQ=2048 BATCH=2 uv run python -m experiments.eval_route learned       # E1 (mean, bound, oracle, learned, mixed)
SEQ=2048 BATCH=2 uv run python -m experiments.paired_se               # E8 -> runs/paper/paired_se.csv
SEQ=2048 BATCH=2 uv run python -m experiments.pick_recall              # E9 -> runs/paper/pick_recall.csv
SEQ=2048 BATCH=2 uv run python -m experiments.copy_test                # E10 -> runs/paper/copy_test.csv
uv run python -m bench.indexer && uv run python -m bench.layer
uv run python -m plots.route                                           # likewise plots.{training,layer}
```

`runs/queue_eval.sh` runs E0 to E10 and the indexer bench for every model in order, one GPU job at a time.

## Results

Paper with every method, comparison and ranking: `report/paper.pdf` (NeurIPS template, `bloated-neurips`).
Rebuild figures, `numbers.json` and the PDF:

```
uv run --with typst python -m plots.paper pdf
```

All selectors run on frozen models; only the small routers are trained. "Extra loss" is a method's validation loss
minus pure CSA2's on the same text, in nats (lower is better). A method matches pure CSA2 if its extra loss is at most
0.01 nats (about 1% perplexity). "Times fewer reads" is how many times fewer index keys it reads than pure CSA2.

| Model | Context (tokens) | Validation loss | Test loss | Training time |
|---|---|---|---|---|
| model 1 | 2048 | 4.0462 | 4.0632 | 61 min |
| model 2 (another seed) | 2048 | 4.0336 | 4.0455 | 64 min |
| 8192-token model | 8192 | 3.9436 | 3.9602 | 43 min |

Cheapest matching setting of each selector on model 1 (`runs/paper/paired_se.csv`):

| Selector | Times fewer reads, 2048 tokens | Times fewer reads, 8192 tokens |
|---|---|---|
| double indexing, learned router | 3.1× | 7.7× |
| double indexing, mixed (more superblocks in needle layers) | 3.0× | 7.2× |
| double indexing, mean summary key (no training) | 2.7× | 6.1× |
| double indexing, bound summary key (no training) | 2.0× | 5.1× |

- Nothing beats pure CSA2's loss: an oracle picking the attention's own top 8 is slightly worse.
- Random entries cost about 0.16 nats, so selection matters.
- On model 2 (no re-tuning), learned, mean and bound still match; mixed does not.
- Copy test (second half repeats the first): every selector loses more than on text; bound loses least.
- Selection step at 128k tokens, 64 sequences: 109 µs per new token vs 1518 µs for pure CSA2 (`runs/bench/indexer.csv`).
- Whole layer at 128k: CSA2 decodes 5× faster than flash attention with a 43× smaller cache (`runs/bench/layer.csv`).
- CSA2 prefill is slower than flash at every length and runs out of 16 GB GPU memory from 64k tokens.

⚠ The needle-layer rule and one mixed setting at 8192 tokens were chosen after first results; see paper Limitations.

The JAX results (V4-style CSA, different models) stay in `archive/jax/runs/` and are not comparable.
