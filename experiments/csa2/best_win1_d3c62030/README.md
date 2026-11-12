# Best run so far: d3c62030 (B exact + CSA2's 1-head K=V window)

Record 91 (modded-nanogpt, 1 GPU, proxy scale) with CSA2's KV layout and our
parameter-free exact top-64 entry picks, plus CSA2's window: one shared 128-channel
K=V vector per token per layer instead of record 91's own 6-head K and V.
Trained 2026-11-10 21:02-23:14, seed 1. Copied from `experiments/csa2/`; every
file here matches the code embedded in the run's own log (`results/train_log.txt`).

## Results

| | this run | B exact (6-head window) | CSA2 |
|---|---|---|---|
| final val loss | 3.5646 | 3.5839 | 3.6532 |
| per-position eval, all positions | 3.5810 | 3.5987 | 3.669 |
| training time | 1.94 h | 2.01 h | 3.01 h |
| peak memory | 7493 MiB | 7517 MiB | 10329 MiB |
| KV cache at 8K / 128K (synthetic) | 1.8 / 24.3 MiB | 5.2 / 27.8 MiB | 2.2 / 30.3 MiB |

- Per position (`results/per_position.csv`): below B exact in every bin. Loss stays flat
  past 2048, the training document length.
- One seed only. CSA2's two seeds differ by 0.034.

## What changed vs B exact (`changes_vs_b_exact.diff`, 82 lines)

- Window K, V: `src/nanogpt/csa2_window.py` `shared_window`. Per token,
  e = rms_norm(x W_win). Key = e rotated by the token's RoPE row; value = e.
  All 6 heads read the same vector.
- W_win: 1 new `kv_bank` slice per attention layer (same NorMuon setup as the
  entry compressor).
- The heads' own v is no longer computed. Their k is still computed but unused
  (ponytail: wasted projection).

## Model, unchanged from B exact

- Every 4 attention layers share one K=V entry cache: m = 4 tokens per entry, gated
  pool (`csa2_flex.compress`).
- Even layers pick the top 64 entries with their own q, exactly
  (`csa2_select_fast.exact`); odd layers reuse those picks.
- Attention: one flex-attention softmax over the 128-token window plus the picked
  entries (`csa2_flex.attention`).

## Files

| file | what it is |
|---|---|
| `train_tournament_kv_win1_fast.py` | the training script (record 91 + all patches) |
| `triton_kernels.py`, `dc_triton_kernels.py` | record 91's kernels (read next to the script) |
| `src/nanogpt/csa2_window.py` | the new window |
| `src/nanogpt/csa2_flex.py` | entry compressor, entry RoPE, flex attention |
| `src/nanogpt/csa2_select_fast.py`, `csa2_select.py` | exact / tournament picks |
| `src/nanogpt/tournament*.py`, `block_topk_fast.py`, `src/attention/triton/common.py` | helpers the picks and flex attention import |
| `src/nanogpt/csa2_select_decode.py` | prefilter rule (eval option only) |
| `eval/` | per-position validation (`rule_eval.py` -> `packed_val.py`) |
| `run.sh` / `eval.sh` | retrain / re-evaluate from this folder |
| `results/` | the run's log (code + training output) and per-position losses |

## Run

Needs `~/repos/modded-nanogpt` (venv, fineweb data) and the `~/repos/modded-nanogpt-r91`
worktree. Logs and checkpoints are written to the worktree's `logs/`.

```
./run.sh                  # ~2 h on one GPU; SWEEP_ITERS=30 ./run.sh for a smoke test
./eval.sh [RUN_ID]        # default: d3c62030's checkpoint
```
