# Bandit selectors and the dense control (frozen)

Archived 2026-10-15: the bandit selectors lost to double indexing, which is now the only line of work
(`src/attention/selectors/double_indexing.py`, `report/paper.typ`). Nothing here is linted, type-checked, tested
or run. Files keep their original repo paths below `archive/bandits/`, and their imports still name those paths
(`attention.dense`, `attention.selectors.superblock_bandit`, `model.heads`, ...).

```
src/attention/selectors/  superblock_bandit.py  double indexing with a bandit coarse step (UCB on sampled blocks)
                          channel_bandit.py     blocks are arms, pulls read indexer channels (successive elimination)
                          token_bandit.py       blocks are arms, pulls read raw tokens with the main query
src/attention/dense/      key_bandit.py         keys are arms, pulls read query/key channels; the "bandit" mixer
                          chunk_bandit.py       chunks are arms, sampled keys are pulls (+ oracle, random)
src/model/                heads.py              real softmax heads for the dense control
                          bandit_mixer.py       code cut from transformer.py / softmax.py (bandit mixer, approx hook)
src/experiments/          eval_csa_bandit.py (E2/E6/E7), eval_chunk_bandit.py (E4), sweep_key_bandit.py (E5)
src/bench/                chunk_bandit.py
src/plots/                csa_bandit.py, chunk_bandit.py, key_bandit.py,
                          methods_bandits.py, paper_bandits.py (cut from plots/methods.py and plots/paper.py)
tests/                    test_{superblock,channel,token}_bandit.py, test_methods.py,
                          reference_bandits.py, test_golden_bandits.py (cut from tests/; golden keys stay in
                          tests/data/golden.npz)
runs/                     csa_bandit/ chunk_bandit/ key_bandit/ logs/ train/bandit_seq1024.*
                          bench/{chunk_bandit,indexer}.csv, paper/*.csv: bandit rows split out of the live CSVs
```

## Revive

1. Restore a pre-archive tree: copy each file back to its path without the `archive/bandits/` prefix.
2. Re-add the cut code in `src/model/bandit_mixer.py`, `src/plots/*_bandits.py` and `tests/*_bandits.py` to the
   files their docstrings name.
3. Merge the archived CSV rows back: `pl.concat([live, archived])` per file (same headers).
4. Run `uv run pytest` and the ruff/mypy gate from the main README.

## Results (moved from the main README)

Δ = val loss − pure CSA at top_k 8, window 32; matches = Δ ≤ +0.01 nats; reads in indexer-key equivalents.

| Selector | Extra state | Fewer reads at Δ ≤ +0.01, T=2048 / 8192 | Notes |
|---|---|---|---|
| channel bandit (E6) | — | 1.87× / 1.51× (seed 0); 1.46× / 1.11× all 3 seeds | seed spread up to 0.029 |
| superblock bandit (E2) | — | none; best +0.031 @ 1.79× / +0.016 @ 1.97× | |
| token bandit (E7) | raw per-token KV | none; +0.100 / +0.165 at 8–9× more reads | ≈ random at T=8192 |

- Double indexing (learned 3.12× / 3.59×) dominates every bandit at every budget, both T.
- Channel bandit is the only bandit that matches pure CSA; its gain shrinks with T.
- Superblock bandit: max second_pulls vs none: −0.010 to −0.016; ucb_width 0–4 moves Δ ≤ 0.005.
- Token bandit: sampled raw tokens rank blocks worse than the trained indexer.

| Selector | Δ model 1 / 2 (T=2048) | Pick recall T=2048 / 8192 | Copy Δ T=2048 / 8192 / model 2 |
|---|---|---|---|
| channel bandit | +0.0093 / +0.0129 | 0.79 / 0.80 | +0.018 / +0.011 / +0.040 |
| superblock bandit | +0.0314 / +0.0328 | 0.63 / 0.77 | +0.139 / +0.047 / +0.138 |
| token bandit | +0.1000 / +0.1683 | 0.08 / 0.03 | +0.272 / +0.207 / +0.344 |

### Training

| Run | Steps × batch | Time | Val |
|---|---|---|---|
| bandit mixer T=1024 | 3000 × 4 | 15.3 min | 5.5051 (softmax T=1024: 5.5085) |

### E5 per-key bandit (`runs/key_bandit/key_bandit.png`, softmax T=1024 heads)

| ucb_width (tolerance 3, dims 8) | Reads vs exact | Rel. error | Δ val, all heads swapped |
|---|---|---|---|
| 1 | 0.36× | 0.098 | +0.0082 |
| 2 | 0.68× | 0.012 | +0.0001 |
| 4 | 0.97× | 0.00002 | 0.0000 |

### E4 chunk bandit (`runs/chunk_bandit/chunk_bandit.png`, softmax LMs, every head swapped)

| T | chunk, refined chunks | Tokens / query | KL bandit / oracle / random | Δ val bandit |
|---|---|---|---|---|
| 1024 | 16, 2 | 32 | 0.141 / 0.136 / 0.201 | +0.143 |
| 1024 | 32, 8 | 256 | 0.0139 / 0.0137 / 0.0241 | +0.0127 |
| 4096 | 32, 4 | 128 | 0.0550 / 0.0491 / 0.0735 | +0.0458 |
| 4096 | 64, 16 | 1024 | 0.0058 / 0.0030 / 0.0082 | +0.0016 |

- Bandit ≈ oracle at T=1024; at T=4096 the gap grows (c64 r16: 1.9× oracle KL).
- At equal q·k reads random refinement beats the bandit: T=4096 random c64 r16 KL 0.0082 @ 0.48× vs bandit
  c32 r16 0.0148 @ 0.49×; a 1024-token window gets 0.0120 @ 0.44×.
- Pull counts and ucb_width have no effect: KL 0.0142–0.0164 (T=4096, c64 r8).

### Benches (one layer, bf16, random weights, RTX 4070 Ti SUPER)

| Bench | Result |
|---|---|
| Superblock bandit vs flat indexer, 128k | decode 126–173 µs (2.6–3.6×); slower than double indexing (93 µs) |
| Chunk bandit vs flash, 32k | c64 r8 3.9 vs 14.9 ms (3.8×); no gain at 4k; 5.6× flash temp memory |

### Caveats

- Token bandit reads raw per-token KV: a decoder must keep 4× the compressed KV cache.
- Channel bandit reads strided channel slices; FLOP savings may not become wall-time savings (unverified).
