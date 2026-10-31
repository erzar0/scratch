# Plan: CSA2 attention in Qwen3-0.6B-Base (frozen rest) + benchmark suite vs original Qwen

## Context
- User request: "lets try changing attention in Qwen/Qwen3-0.6B-Base (while the rest of weights are frozen) and training it with new attention".
- Follow-up request: "prepare me a list of benchmarks we should do to test attention and overall model performance in comparison to the pretrained normal qwen, use newest research practices".
- Decisions already made (AskUserQuestion):
  - attention = faithful V4.1 CSA2 block, trained from scratch;
  - objective = imitate the teacher first, then fine-tune;
  - data = WikiText-103 for training, books (PG-19) for evaluation.
- Design part 1 (model) was accepted ("lets continue").
- Why: the existing CSA2 results come only from 49.6M self-trained LMs. A real pretrained model tests whether CSA2 and double indexing hold up at scale against a strong dense baseline.
- Outcome:
  - CSA2-Qwen checkpoint;
  - dense fine-tuned control;
  - one benchmark suite with plain-word tables and plots, comparing against the original Qwen.
- On approval:
  - save this file as `docs/superpowers/specs/2026-10-16-qwen-csa2-design.md`; not committed (no-commit rule);
  - implement in the order below.

## 1. Model (accepted part 1)
- Base: Qwen3-0.6B-Base. 28 layers, hidden 1024, 16 query heads, 8 KV heads × 128, 32k context. Download about 1.2 GB.
- Every non-attention weight is frozen: about 0.53B params (embeddings, MLPs, norms, LM head).
- Each `self_attn` is replaced by `QwenCsa2Attention(Csa2Attention)`, a subclass:
  - it adapts Qwen's call signature;
  - it builds its own `DeepseekV4RotaryEmbedding` cos/sin;
  - it reads a per-forward `SharedState`, reset by a forward pre-hook on `model.model`.
- Subclassing keeps `.indexer`, `.selector` and `.layer_idx`. So `experiments/selector_forward.py` (`selecting`, `reads_per_seq`) and the double-indexing selectors work unchanged.
- Layer modes: (FULL, REUSE, REINDEX, REUSE) × 7.
- Sizes, V4-Flash attention scaled to width 1024:
  - 16 heads × 128; RoPE on 16 channels;
  - q_lora 256; o_groups 2, o_lora 256;
  - indexer 16 heads × 32; top-64 entries; window 128; m = 4.
  - About 2.5M new params per layer, against Qwen's 6.3M.
- The original Qwen attention modules are kept beside the new ones (about 350 MB). A `teacher(model)` context manager swaps them back for teacher forwards under no_grad.
- `deepseek/csa2.py` and `double_indexing.py` stay unchanged.
- ⚠ Risk: top-64 of 1024 entries at 4k is very sparse, so loss may land well above Qwen's (unverified).

## 2. Training (part 2, new)
| Stage | What trains | Loss | Data | Time (estimate) |
|---|---|---|---|---|
| 1. attention transfer | CSA2 layers, run on the teacher's per-layer inputs | mean squared error to Qwen's attention output (after o_proj) + indexer loss | WikiText-103, 4096 tokens | about 20 min |
| 2. end to end | CSA2 layers | next-token cross-entropy + KL to Qwen's logits | WikiText-103, 4096 tokens, about 29M tokens | about 60 min |
| 2b. length extension (gated) | CSA2 layers | same as stage 2 | WikiText-103 windows of 16384 tokens | about 15 min |
| control | Qwen's own attention (rest frozen) | same as stage 2 | same | about 60 min |

- Stage 2b runs only if PG-19 loss rises with position past 4096 tokens after stage 2.
- Logits are 151936 wide, so CE and KL are computed in chunks of 512 positions; full logits never exist.
- Gradient checkpointing per decoder layer.
- Stage 1 hooks capture each layer's normed input and attention output from one teacher forward.
- The control separates domain adaptation to WikiText from learning the new attention.
- One seed per model (GPU budget); stated as a limitation.

## 3. Compared models (rows of every table)
| Row | What it is | Extra training |
|---|---|---|
| Qwen | original Qwen3-0.6B-Base | none |
| Qwen, fine-tuned | dense control from section 2 | stage-2 recipe |
| CSA2-Qwen | new attention, stages 1–2 (2b if gated in) | sections 1–2 |
| CSA2-Qwen, window only | CSA2-Qwen with every pick disabled: sliding window + sinks only | none (selector swap) |
| CSA2-Qwen + double indexing | mean, bound, learned router, mixed selectors | routers only (`experiments/train_router.py`) |
| CSA2-Qwen + random / oracle picks | reference selectors (`deepseek/reference.py`) | none |

- The window-only row shows whether the long-range branch is used at all. KL-guided distillation (ICLR 2026) shows short tasks saturate with tiny windows.

## 4. Benchmark suite
Tiers A–D test attention directly; E checks that overall ability is retained; F measures cost.

### A. Attention fidelity (cheap, every checkpoint)
| Benchmark | What it tests | Protocol |
|---|---|---|
| attention-output error per layer | how closely each CSA2 layer imitates Qwen's attention | relative squared error on teacher inputs, 64 WikiText val windows of 4096 tokens |
| KL to Qwen | how far the whole model's next-token distribution drifts | mean token KL, WikiText val and PG-19, 4096 tokens |
| pick recall | share of dense attention mass covered by the top-64 picks | reuse `experiments/pick_recall.py` logic |
| reads per sequence | indexer cost | `selector_forward.reads_per_seq` |

### B. Language modeling
| Benchmark | What it tests | Protocol |
|---|---|---|
| WikiText-103 val loss | in-domain quality | windows of 4096 tokens; extra loss vs Qwen ± paired standard error (`results.paired_delta`) |
| PG-19 test loss by position | out-of-domain books; whether distant context still helps; length beyond training | 100 test books, 32768-token windows; mean loss in position buckets 0–1k, 1–2k, 2–4k, …, 16–32k |

- Long-context validation loss is LongCat Sparse Attention's base-model metric (Aug 2026).

### C. Retrieval and long context (main attention test)
| Benchmark | What it tests | Protocol |
|---|---|---|
| RULER, 13 tasks | needle retrieval (single, multi-key, multi-value, multi-query), variable tracking, common/frequent word extraction, QA | lengths 4k, 8k, 16k, 32k; 50 samples per task and length; base-model template, one in-context demonstration for needle tasks |
| BABILong qa1–qa3 | facts scattered through long text, 1–3 reasoning steps | lengths 4k, 8k, 16k, 32k; 50 samples per task and length |
| HELMET many-shot in-context learning | label lookup across hundreds of demonstrations (TREC coarse, TREC fine, Banking77, Clinic150, NLU) | lengths 4k–32k; accuracy |
| recall-heavy short tasks: SWDE, FDA, SQuAD completion | exact copying of a value from 1–2k-token documents, out of the 128-token window's reach | lm-eval-harness tasks `swde`, `fda`, `squad_completion` |

- RULER + a realistic suite is the 2026 sparse-attention standard (MiniMax, LongCat, Opal). HELMET argues synthetic recall alone does not predict real tasks; many-shot ICL is HELMET's base-model task.
- Scoring without a KV cache: CSA2 runs uncached, so generating answers would re-run 32k-token prefills per token.
  - Answers are scored by teacher forcing instead: correct = every gold answer token is the model's argmax.
  - This equals greedy decoding exact match for that answer, in one forward per sample.
  - Multi-value tasks score each value in gold order (approximation; the same for every row).
  - Validated once on original Qwen against real lm-eval generation at 4k; the two should agree within a few points.
- RULER and HELMET prompts come from lm-eval-harness task definitions (`ruler`) and HELMET's released data (assumed available; checked at implementation).

### D. Double indexing on CSA2-Qwen
- Rows: pure CSA2-Qwen, mean, bound, learned router, mixed, random, oracle.
- Columns: times fewer reads; WikiText extra loss; PG-19 loss 16–32k; RULER needle average at 32k; MMLU.
- Same comparison as the paper, now on a pretrained 0.6B model.

### E. Overall ability retained (short context, lm-eval-harness)
| Benchmark | Shots | Metric |
|---|---|---|
| MMLU | 5 | accuracy |
| HellaSwag, ARC-Challenge | 0 | normalized accuracy |
| PIQA, ARC-Easy, WinoGrande | 0 | accuracy |
| LAMBADA (OpenAI) | 0 | accuracy, perplexity |
| GSM8K, 250-question subset (optional) | 4 | exact match, uncached generation |

- Reported as raw score and as % of Qwen's score; this is the retention metric of Opal and LoLCATs-style distillation papers.
- Run through `lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=...)`.
- ⚠ Short tasks mostly fit in the 128-token window. Good retention here does not show the long-range branch works; tiers B–C do.

### F. Efficiency
| Measure | Protocol |
|---|---|
| prefill time per layer and full model | 4k, 8k, 16k, 32k (layer up to 128k); CUDA-graphed (`bench/timing.py` `graphed`, `measure`) |
| decode time per token, per layer | 4k–128k; `bench/layer.py` `csa2_cache`, `csa2_decode_step` on a Qwen-sized layer vs Qwen3 attention with a static cache |
| cache memory per token | CSA2 entries vs Qwen's 8 KV heads × 128 × 2 × 28 layers |
| indexer-only time | flat vs double indexing, `bench/indexer.py` |

- ⚠ The CSA2 path is eager PyTorch (gather + einsum), while Qwen uses fused attention, so short contexts may favour Qwen. Reads and cache memory are the hardware-neutral numbers.

### Skipped, with reason
| Benchmark | Reason |
|---|---|
| LongBench v2 | four-choice; a 0.6B base model sits near the 25% chance level (assumed) |
| HELMET RAG, citation, summarization, re-ranking | need instruction following; built for chat models |
| GPQA, MATH, AIME, code | near the floor for a 0.6B base model (assumed) |
| plain needle-in-a-haystack heatmap | saturates; RULER's needle tasks cover it |

## 5. Code
- Placement follows the separate-variants rule: new package `src/qwen/`, plus `qwen` in `[tool.hatch.build.targets.wheel] packages`.

| File | Contents |
|---|---|
| `src/qwen/model.py` | `csa2_config(qwen_config)`, `QwenCsa2Attention`, `build()` (load, swap, freeze), `teacher()` context manager |
| `src/qwen/data.py` | WikiText-103 retokenized with Qwen's tokenizer → `data/qwen_wt103_{train,validation}.bin` (uint32), reusing `train.prep`'s parquet download; PG-19 test → `data/qwen_pg19_test.npz` |
| `src/qwen/train.py` | stage 1, stage 2 (and 2b), dense control; chunked CE + KL; reuses `train.windows`, `learning_rate`, `make_optimizer` |
| `src/qwen/evaluate.py` | tiers A, B, C-scoring, E (HFLM wrapper); window-only selector; teacher-forced answer scoring |
| `src/qwen/bench.py` | tier F on Qwen-sized layers, reusing `bench/timing.py` and `bench/layer.py` |
| `src/plots/qwen.py` | figures listed under Verification |
| `tests/test_qwen.py` | see below |

- Results go to `runs/qwen/*.csv` via `experiments/results.Results`. ⚠ Back up CSVs before reruns.
- Dependencies: optional group `eval = ["lm-eval", "datasets"]`, exact versions pinned by `uv lock`. HFLM compatibility with transformers 5.19 is unverified; fallback is our own loglikelihood loop over lm-eval's documents.
- Tests: one per non-trivial piece, Arrange-Act-Assert, tiny random Qwen3 config on CPU.
  1. Swapped model is causal: changing token t leaves logits before t unchanged.
  2. Teacher-forced answer scoring equals real greedy generation on a tiny model.
  3. Chunked CE + KL equals the unchunked value on a small vocab.

## 6. Order of work (one GPU job at a time)
1. Download Qwen3-0.6B-Base; tokenize data; run tests.
2. Baselines first: original Qwen on tiers B, C, E and F. Check the scoring paths here, including teacher-forced vs real generation.
3. Stage 1 → tier A check: per-layer error falls.
4. Stage 2 → tiers A–B; decide on stage 2b from the PG-19 position curve.
5. Dense control (stage-2 recipe).
6. Full suite (tiers A–C, E, F) on CSA2-Qwen and the control; window-only row on tiers B–C.
7. Train routers; tier D.
8. Plots, README section and paper section in plain words (plain-tables rule).

- GPU budget estimate: about 3 h of training and about 8–10 h of evaluation (unverified; RULER at 32k dominates).
- If the full budget is too long, tiers C and D drop to the needle tasks at 50 samples first.

## Verification
1. `uv run pytest` passes, with 3 new tests.
2. `uv run ruff check src tests && uv run ruff format --check src tests && MYPYPATH=src uv run mypy src tests` is clean.
3. Sanity of the eval code on original Qwen:
   - MMLU 5-shot lands near Qwen's published base score (checked against the Qwen3 report at run time);
   - teacher-forced RULER needle score matches real generation within a few points at 4k.
4. Stage 1: per-layer attention error falls in the live CSV log. Stage 2: KL to Qwen falls; the run has no NaNs.
5. Every result is plotted via `uv run python -m plots.qwen`, and each PNG is viewed:
   - training curves;
   - PG-19 loss by position (all rows);
   - RULER heatmap (task × length) per row;
   - retention bars;
   - efficiency vs context length.
6. Every table uses plain words: "extra loss vs Qwen (nats)", "% of Qwen's score", "times fewer reads".

## Sources (practice each grounds)
- LongCat Sparse Attention, https://arxiv.org/pdf/2608.01662: base-model long-context loss, n-shot retrieval, per-layer indexer latency, two-stage indexer close to double indexing.
- MiniMax Sparse Attention, https://arxiv.org/pdf/2606.13392: RULER + real long-context suite.
- Opal, https://arxiv.org/html/2609.31947: % retention on short tasks, RULER needle at 4k–32k, warning that likelihood suites hide long-context failures.
- KL-guided layer selection, ICLR 2026, https://proceedings.iclr.cc/paper_files/paper/2026/file/b08940ce0f8f2bd499f2e5440eb38ac5-Paper-Conference.pdf: recall-heavy SWDE/FDA/SQuAD; short tasks saturate with tiny windows.
- Jet-Nemotron PostNAS, https://arxiv.org/html/2508.15884v2: frozen MLPs, new attention trained on a pretrained model (our setup).
- Long-context attention benchmark, ICLR 2026, https://proceedings.iclr.cc/paper_files/paper/2026/file/7df69dbf39705c7a39b40f2d70e806c1-Paper-Conference.pdf.
- RULER through lm-eval-harness: https://pypi.org/project/flash-linear-attention/.

## Implementation notes (2026-10-16, after approval)
- Stage 1: 1500 steps, 2 × 4096 tokens, LR 1e-3; per-layer relative error 0.373 → 0.277 (validation).
- Stage 2: 2000 steps instead of about 3500 (time), LR 3e-4; dense control LR 3e-5 (1e-3 diverged in a smoke run).
- `torch.compile` not used: it breaks with gradient checkpointing (recompute metadata mismatch).
- HELMET ICL: 4 of 5 tasks (TREC coarse, TREC fine, Banking77 via `mteb/banking77`, CLINC150).
  - NLU skipped: `xingkunliuxtracta/nlu_evaluation_data` no longer on the Hub.
  - Demonstrations fill each length (HELMET fixes a count per length); shorter lengths reuse the first demonstrations of the longer ones.
  - Scored on label + separator, so a longer label is wrong.
- Tier D settings (from the paper's 8192-token results), superblocks of 16:
  - mean, bound, learned: keep 8;
  - mixed: learned keep 4 in most layers, bound keep 16 in needle layers.
- Every selector runs over 1024-query chunks (`evaluate.Chunked`); full scores at 32k would need 17 GB.
- Teacher-forced vs real generation check: `qwen.evaluate check qwen`, 25 samples at 4k, needles + variable tracking + BABILong qa1.
