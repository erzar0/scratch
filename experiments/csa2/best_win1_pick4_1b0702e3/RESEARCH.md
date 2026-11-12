# Recent attention papers vs run 7

Searched 2026-11-12. Most entries are read from abstracts only; details marked
(abstract) were not checked against the full paper.

## Where run 7 loses time

- Decode, picks: 24 µs at 4K -> 203 µs at 512K -> 364 µs at 1M per token (3 exact scans of
  all N/4 entries x 96 channels, bf16). Attention is flat at 34 µs.
- Training: 1.85 h (CSA2 3.01 h, dense stripped about 1.66 h).
- Loss: 3.600 (CSA2 3.653); seed noise about 0.1.

## Papers

| paper | date | idea | helps | our test |
|---|---|---|---|---|
| [DeepSeek-V4](https://arxiv.org/abs/2606.19348) (CSA, HCA; the "CSA2" we compare to) | 2026-06 | KV compressed 4x by gated pooling; FP4 lightning indexer picks top-k compressed blocks; HCA layers compress 128x and attend densely | decode, cache | our base. Its indexer keys are FP4; ours are bf16 -> FP8 pick keys (not tested yet) |
| [IndexCache](https://arxiv.org/abs/2603.12201) | 2026-03 | adjacent layers' top-k overlap 70-100%; only "full" layers pick, others reuse; 1.48x decode on a 30B DSA model | decode picks | C1: picks shared across our 3 groups |
| [Qwen3.8-Next QSA](https://arxiv.org/abs/2608.30320) | 2026-08 | indexer scores micro-blocks of 4 tokens by their mean key; up to 512 blocks + current tail | decode picks | C2: block mean summary |
| [UNIQUE](https://arxiv.org/abs/2605.27740) | 2026-05 | page score = mean key + std offset; training-free or sparsity-aware fine-tuning (sigmoid soft mask at the top-k boundary) | decode picks | C2: block meanstd summary |
| Quest (2024; baseline in UNIQUE) | 2024 | per-channel min / max per page: an upper bound on any key's score | decode picks | C2: block minmax summary |
| [HySparse](https://arxiv.org/abs/2602.03560), [HySparse2](https://arxiv.org/abs/2609.26368) | 2026-02, 2026-09 | one full-attention layer's token picks and KV reused by the following sparse layers; KV about 10x smaller | decode, cache | same idea as our groups of 4 and C1 |
| [Latent Sparse Attention](https://arxiv.org/abs/2610.42147) | 2026-10 | keys clustered hierarchically in a learned latent space; queries descend the hierarchy instead of scoring every key; native or distilled (abstract) | decode, prefill at 1M+ | not tested; hierarchical version of C2 |
| [HyScale](https://arxiv.org/abs/2610.32057) | 2026-10 | training: exact top-k on the fly without a score table (FlashTopK); replay forward picks in recomputation; 2x on DeepSeek-V4-Flash CSA layers at 1M | training time | our exact picks already run without grad, once per group; no recomputation in our setup (unverified) |
| [Sample-guided exact top-k](https://arxiv.org/abs/2609.08450) | 2026-09 | strided sample proposes the top-tail cutoff, one full pass certifies it; 1.55x over the fastest exact selector | decode picks, training picks | not tested; could replace the radix steps in csa2_decode.top64 |
| [Gated attention](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html) (Qwen, NeurIPS 2025) | 2025 | per-head sigmoid gate on the attention output; better loss, no attention sink, better length extrapolation | loss | not tested (needs training). B is stripped of record 91's two gated layers |
| [HSA length generalization](https://arxiv.org/abs/2510.17196) (ICLR 2026) | 2025-10 | chunk sparse attention extrapolates 4K -> 32M with a non-linear chunk encoder, a bypass residual path and sparsity enforced in training; top-k 8 beat 64 at 32M | loss past 2048 | not tested; relevant to the breaks in runs 2, 3, 6, 8 |
| [Inference-time context sparsity](https://arxiv.org/abs/2605.24168) | 2026-05 | models robust to 50x decode sparsity; top-k failures blamed on deterministic, locality-biased selection (vAttention: stochastic) (abstract) | decode | not tested |

## Ranked ideas for run 7

1. Picks shared across groups (IndexCache): no training to test, cuts pick scans 3 -> 2 or 1.
2. Block prefilter with a parameter-free summary (QSA, UNIQUE, Quest): scan N/64 summaries
   instead of N/4 entries.
3. FP8 pick keys (DeepSeek-V4 uses FP4): halves pick bytes; eval-only test possible.
4. Faster exact top-k selection (sample-guided): kernel-only.
5. Output gates (gated attention): needs training; loss and extrapolation.

Results of 1 and 2: `ANALYSIS.md`.
