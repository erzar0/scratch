# Run 7: 1b0702e3 (B exact + CSA2's 1-head K=V window + picks every 4 layers)

Record 91 (modded-nanogpt, 1 GPU, proxy scale) with CSA2's KV layout, our
parameter-free exact top-64 entry picks made once per group of 4 layers, and CSA2's
window: one shared 128-channel K=V vector per token per layer instead of record 91's
own 6-head K and V. Trained 2026-11-12, seed 1. It is run 1
(`../best_win1_d3c62030/`) with picks every 4 layers instead of every 2
(`changes_vs_run1.diff`, 2 lines). Every training file here matches the code embedded
in the run's own log (`results/train_log.txt`).

## Results

| | run 7 | run 1 (picks every 2) | B exact (6-head window) | CSA2 |
|---|---|---|---|---|
| final val loss | 3.6004 | 3.5646 | 3.5839 | 3.6532 |
| per-position eval, all positions | 3.6149 | 3.5810 | 3.5987 | 3.669 |
| training time | 1.85 h | 1.94 h | 2.01 h | 3.01 h |
| peak memory | 7491 MiB | 7493 MiB | 7517 MiB | 10329 MiB |
| KV cache at 8K / 128K (synthetic) | 1.8 / 24.3 MiB | 1.8 / 24.3 MiB | 5.2 / 27.8 MiB | 2.2 / 30.3 MiB |

Decode, picks + attention, µs per token (random tensors, `results/decode_heat_rows.csv`):

| context | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| run 7, exact picks | 53 | 56 | 61 | 73 | 84 | 103 | 147 | 231 | 395 |
| run 7, prefilter keeps 512 (fails the loss gate) | 86 | 89 | 92 | 101 | 105 | 108 | 124 | 155 | 222 |
| run 7 at eval, picks at layers 0, 4 (up to 32K) / + block mean (from 64K) | 45 | 49 | 53 | 59 | 67 | 70 | 77 | 85 | 100 |
| CSA2, learned indexer | 54 | 57 | 67 | 78 | 83 | 103 | 125 | 165 | 275 |

- As trained (exact, 3 pick scans), run 7 is slower than CSA2 from 128K.
- At eval, picks at layers 0, 4 (layers 8-9 reuse layer 4's) plus the block mean
  prefilter from 64K beat CSA2 at every length (0.36x at 1M) and lower the loss
  (3.6140 vs 3.6149). Tests and limits: `ANALYSIS.md`; rows: `results/decode_block.csv`
  (its own CSA2 row: 53-282 µs).
- The old prefilter (keeps 512) is +0.016 at positions 4096-8192: it fails the gate.
- Per position (`results/per_position.csv`): flat past 2048, the training document
  length.
- One seed only. Seed noise is about 0.1: run 7 scored worse than both runs it combines
  (run 1 3.565, picks every 4 with the 6-head window 3.492).

## Model

- Every 4 attention layers share one K=V entry cache: m = 4 tokens per entry, gated
  pool (`csa2_flex.compress`), built by the group's first layer.
- The group's first layer picks the top 64 entries with its heads' summed q, exactly
  (`csa2_select_fast.exact`); its other 3 layers reuse those picks.
- Window: per token, e = rms_norm(x W_win); key = e rotated by the token's RoPE row,
  value = e; all 6 heads read it (`src/nanogpt/csa2_window.py`).
- Attention: one flex-attention softmax over the 128-token window plus the picked
  entries (`csa2_flex.attention`).

## Files

| file | what it is |
|---|---|
| `train_tournament_kv_win1_pick4_fast.py` | the training script (record 91 + all patches) |
| `triton_kernels.py`, `dc_triton_kernels.py` | record 91's kernels (read next to the script) |
| `src/nanogpt/csa2_window.py` | the shared K=V window |
| `src/nanogpt/csa2_flex.py` | entry compressor, entry RoPE, flex attention |
| `src/nanogpt/csa2_select_fast.py`, `csa2_select.py` | exact / tournament picks |
| `src/nanogpt/tournament*.py`, `block_topk_fast.py`, `src/attention/triton/common.py` | helpers the picks and flex attention import |
| `src/nanogpt/csa2_select_decode.py` | prefilter rule (eval option only) |
| `src/nanogpt/csa2_decode.py`, `csa2_attend_split.py`, `csa2_window_decode.py` | decode kernels: picks (`top64`, `prefilter64`), split attention, attention with the K=V window (`kv_attend`) |
| `bench/decode_heat.py`, `decode_csa2.py` | the decode benchmark (random tensors, 4K-1M) |
| `src/nanogpt/csa2_select_block.py` | block prefilter rule (eval option only; mean, max, min/max, mean + std) |
| `train_tournament_kv_win1_pick4_share04_fast.py` | run 7's script with picks at layers 0, 4 only (eval only; made by `make_pick_share_script.py`) |
| `bench/decode_block.py` | decode timing of shared picks and the block rule (random tensors) |
| `eval/` | per-position validation (`rule_eval.py` -> `packed_val.py`) |
| `run.sh` / `eval.sh` / `eval_rule.sh` | retrain / re-evaluate / re-evaluate under another pick rule |
| `results/` | the run's log (code + training output), per-position losses, eval rules (`eval_rules.csv`), decode rows |
| `ANALYSIS.md`, `RESEARCH.md` | where the time goes, tests of shared picks and block rules, recent papers |

## Run

Needs `~/repos/modded-nanogpt` (venv, fineweb data) and the `~/repos/modded-nanogpt-r91`
worktree. Logs and checkpoints are written to the worktree's `logs/`.

```
./run.sh                  # ~2 h on one GPU; SWEEP_ITERS=30 ./run.sh for a smoke test
./eval.sh [RUN_ID]        # default: run 7's checkpoint
./eval_rule.sh train_tournament_kv_win1_pick4_share04_fast.py block 8 64 mean  # 3.61402
```

The decode files are the current versions in `src/nanogpt/` (they are not part of
training); the benchmark runs from the repo root with the fork venv:
`PYTHONPATH=src:experiments/csa2/bench python experiments/csa2/bench/decode_heat.py`.
