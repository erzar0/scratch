# Run 7: where the cost goes, what to change

Analysis 2026-11-12. All tests below are on run 7's checkpoint, no retraining
(record 91 validation rows of 8192 tokens, 128 rows; decode on random tensors).

## Data flow (one attention layer; 10 layers, groups of 4: layers 0-3, 4-7, 8-9)

```
x (768) ─┬─ x·qk_w ──> q (6 heads × 96) ......... k (6 × 96): computed, never read
         ├─ x·W_win ─> rms_norm ─> e (128) ─> key = RoPE(e), value = e
         │                                    window cache: last 128 tokens, 1 vector each
         └─ group's first layer only (0, 4, 8):
              4 tokens ─> softmax-gated sum of x·w_kv ─> rms_norm ─> entry (128)
                                                    entry cache: N/4 per group
              picks = top 64 entries by (q summed over heads)·RoPE(entry), exact scan
attention, each head: one softmax over 128 window tokens + the group's 64 picked entries
              ─> y (6 × 128) ─> o_w
```

Path in words: the window and the entries are both one shared vector per token (or per
4 tokens); only the picks depend on the query, and they are made 3 times per token.

## Cost per stat

| stat | run 7 | CSA2 | where run 7's cost goes |
|---|---|---|---|
| decode at 512K, µs per token | 231 | 165 | picks 203 (3 exact scans of 131072 entries, 96 MiB read), attention 34 |
| decode at 1M, µs per token | 395 | 275 | picks 364, attention 34 |
| KV cache at 128K | 24.3 MiB | 30.3 MiB | entries 24.0 (3 groups × 32768 × 256 bytes), window 0.3 |
| training time | 1.85 h | 3.01 h | MLP 43%, other matrix multiplies 31%, flex attention 10%, picks under 1% of GPU time (profile of B exact, stage 3; run 7 not profiled) |
| final validation loss | 3.600 | 3.653 | seed noise about 0.1 (1 seed) |

## Findings in the code

1. Picks dominate decode from 64K: 53 of 84 µs at 64K, 203 of 231 at 512K. Attention
   is flat (34 µs). Every decode win has to come from the picks -> tests 1 and 2.
2. The k half of the QK projection (576 of 1152 columns per layer) is computed, normed
   and rotated in training and never read: the shared window replaced it. Deleting it
   saves a matrix multiply per layer (about 2% of a step, estimate, unverified); needs a
   retrain to confirm the loss is unchanged (it should be: no gradient reaches it).
3. The shared window is expanded to 6 heads by `repeat_interleave`
   (`csa2_window.shared_window`): two (T, 6, 128) copies per layer in training, where
   flex attention could read one head with a GQA mapping instead.
4. Record 91's partial key offset (long-window layers) is applied to the unused k only;
   the shared window keys never get it. Possible loss cost, untested.
5. B is stripped of record 91's attention output gates, XSA, value embeddings and DC
   (`attn_gate_w=None`, `xsa_alpha=None`, `dc_w=None`). Output gates are the cheapest to
   put back (gated attention, NeurIPS 2025: better loss and length extrapolation).
6. Entries are cached in bf16 and the picks read all 128 channels. DeepSeek-V4 keeps its
   indexer keys in FP4; FP8 pick keys would halve the bytes the picks read (eval-only
   test possible; not run).

## Tests

Gate for "no loss cost": within 0.005 of exact over all positions and within 0.01 in
every position bin past 2048. Differences are paired per token against exact picks on
the same tokens; ± is one standard error over documents (`eval/paired.py`).

### 1. Picks shared across groups (IndexCache)

Entry j covers tokens 4j..4j+3 in every group, so a group can reuse another group's
picks unchanged. Scripts: `../b/make_pick_share_script.py`.

| picks made at | pick scans per token | all positions | positions 4096-8192 | gate |
|---|---|---|---|---|
| layers 0, 4, 8 (as trained) | 3 | 3.6149 | 3.6165 | |
| layers 0, 4 (layers 8-9 reuse layer 4's) | 2 | -0.0010 ± 0.0002 | -0.0080 ± 0.0009 | passes |
| layers 0, 8 (layers 4-7 reuse layer 0's) | 2 | +0.0099 ± 0.0008 | +0.0272 ± 0.0078 | fails |
| layer 0 only | 1 | +0.0099 ± 0.0008 | +0.0212 ± 0.0084 | fails |

- Layers 4-7 need their own picks; layers 8-9 do better with layer 4's than their own.
- Free win: one pick scan fewer per token and lower loss at every position past 2048.

### 2. Block prefilter (QSA mean, max, Quest min/max bound, UNIQUE mean + std)

Score aligned blocks of entries by a summary with the summed q, keep the best blocks plus
every partly readable block (document start, current tail), exact top 64 inside
(`src/nanogpt/csa2_select_block.py`). Rows of 8192 tokens hold at most 2048 entries, so
keeping 256 entries is 12.5% or more of the context; at decode lengths the kept share is
far lower.

Change in loss vs exact picks; the last column is positions 4096-8192, at 512 entries.

| summary | 128 entries (16 blocks of 8) | 256 (16 of 16) | 256 (32 of 8) | 512 (64 of 8) | 512, positions 4096-8192 |
|---|---|---|---|---|---|
| mean key | +0.0058 | +0.0018 | +0.0012 | +0.0002 | +0.0037 ± 0.0007 |
| max key | +0.0073 | +0.0025 | +0.0017 | | |
| min/max bound | +0.0050 | +0.0017 | +0.0010 | +0.0002 | +0.0069 ± 0.0022 |
| mean + std | +0.0043 | +0.0014 | +0.0009 | +0.0001 | +0.0031 ± 0.0014 |
| prefilter keeps 512 (existing) | | | | +0.0008 | +0.0161 ± 0.0024 |

- At 256 entries every summary passes over all positions and fails past 4096
  (+0.015 to +0.025).
- At 512 entries mean, min/max and mean + std pass; max is the worst summary at every size.
- The existing prefilter fails at the same 512 entries (+0.016 past 4096): it ranks
  every entry on one aligned block of 16 channels (largest sum of |q|); the block rules
  rank blocks on all channels of a summary. The scoreboard no longer counts it for run 7 or
  run 1 (+0.032).
- Mean is as good as mean + std and the min/max bound at 512 and reads half the bytes
  (128 channels per block instead of 256).

### 3. Both: picks at layers 0, 4 and the block rule

| rule | all positions | positions 4096-8192 | gate |
|---|---|---|---|
| block mean + std, 256 entries, picks at 0, 4 | -0.0001 ± 0.0002 | +0.0067 ± 0.0030 | passes |
| block mean + std, 512 entries, picks at 0, 4 | -0.0009 ± 0.0001 | -0.0049 ± 0.0014 | passes |
| block mean, 512 entries, picks at 0, 4 | -0.0008 ± 0.0001 | -0.0045 ± 0.0013 | passes |

The sharing gain outweighs the block rule's cost: both together still beat exact picks.

### 4. Decode time (picks + attention, µs per token, random tensors)

`bench/decode_block.py`: stage 1 `top64` over block summaries (64 blocks of 8 = 512
entries), stage 2 `top64` over the gathered entries; checked equal to PyTorch.

| context | 4K | 8K | 32K | 128K | 512K | 1M |
|---|---|---|---|---|---|---|
| CSA2, learned indexer | 53 | 57 | 78 | 103 | 166 | 282 |
| run 7, exact picks every 4 layers (as trained) | 51 | 57 | 73 | 108 | 240 | 400 |
| exact picks at layers 0, 4 | 45 | 49 | 59 | 82 | 171 | 295 |
| block mean, 512 entries, every 4 layers | 68 | 70 | 79 | 89 | 111 | 133 |
| block mean, 512 entries, picks at 0, 4 | 56 | 58 | 63 | 70 | 85 | 100 |
| block mean + std, 512 entries, picks at 0, 4 | 57 | 59 | 66 | 75 | 98 | 126 |
| prefilter keeps 512 (fails the gate) | 85 | 89 | 104 | 109 | 157 | 223 |

- Best rule per length: exact picks at layers 0, 4 up to 32K; block mean with picks at
  0, 4 from 64K (tie at 64K: 68 vs 67).
- Against CSA2: 0.86x at 8K, 0.76x at 32K, 0.68x at 128K, 0.51x at 512K, 0.36x at 1M.
- The block rule costs about 25 µs fixed (two launches, a gather) and wins from 64K.
- Full grid, all parts: `results/decode_block.csv`, plot
  `runs/nanogpt/long_context/decode_block.png`.

### Limits

- Eval rows are 8192 tokens (at most 2048 entries), so 512 kept entries is at least 25%
  of the context. At 128K it is 1.6%, at 1M 0.2%; the loss there is untested (training
  documents end by 2048, so this data has nothing longer to test on).
- One seed, one checkpoint. Picks at layers 0, 4 beating the trained layout is an
  eval-time result; training with it is the next test.

## Next (need the user's OK: training)

1. Train run 7 with picks at layers 0, 4 (2 scans) and seed 2 of runs 7 and 4.
2. Delete the unused k projection; read the window with a GQA mapping (findings 2, 3).
3. Put the output gates back (finding 5).
4. Train with the block rule in the loop (mean + std), then re-test past 4096.
5. FP8 pick keys at eval (finding 6; no training).
