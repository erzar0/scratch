## Qwen, CSA2-Qwen and controls

| model | WikiText loss (nats) | extra loss vs Qwen (nats) | KL to Qwen (nats) | PG-19 loss, tokens 16k-32k | RULER, 4k (%) | RULER, 32k (%) | BABILong, 4k (%) | BABILong, 32k (%) | many-shot ICL, 4k (%) | many-shot ICL, 32k (%) | short tasks, % of Qwen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen (original) | 2.488 | — | — | 2.763 | 63.2 | 49.9 | 34.0 | 10.7 | 49.5 | 60.0 | 100.0 |
| Qwen, attention fine-tuned | 2.500 | +0.013 ± 0.001 | 0.018 | 2.762 | 65.0 | 52.8 |  |  |  |  |  |
| CSA2-Qwen | 4.497 | +2.009 ± 0.028 | 2.006 | 3.675 | 0.2 | 0.2 | 0.0 | 0.0 | 0.5 | 0.0 | 46.6 |
| CSA2-Qwen, window only | 4.837 | +2.349 ± 0.029 | 2.345 | 4.092 |  |  |  |  |  |  |  |
| CSA2-Qwen, random picks | 4.522 | +2.035 ± 0.028 | 2.030 | 3.683 |  |  |  |  |  |  |  |
| CSA2-Qwen, oracle picks | 4.497 | +2.010 ± 0.028 | 2.006 | 3.690 |  |  |  |  |  |  |  |

## Double indexing on CSA2-Qwen

| model | WikiText loss (nats) | extra loss vs Qwen (nats) | KL to Qwen (nats) | PG-19 loss, tokens 16k-32k | RULER, 4k (%) | RULER, 32k (%) | BABILong, 4k (%) | BABILong, 32k (%) | many-shot ICL, 4k (%) | many-shot ICL, 32k (%) | short tasks, % of Qwen | times fewer indexer reads, 4k | times fewer indexer reads, 32k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pure CSA2 | 4.497 | +2.009 ± 0.028 | 2.006 | 3.675 | 0.2 | 0.2 | 0.0 | 0.0 | 0.5 | 0.0 | 46.6 | 1.00× | 1.00× |
| mean summary | 4.498 | +2.011 ± 0.028 | 2.007 | 3.669 |  |  |  |  |  |  |  | 3.24× | 10.50× |
| bound summary | 4.497 | +2.010 ± 0.028 | 2.006 | 3.670 |  |  |  |  |  |  |  | 2.70× | 6.35× |

## Qwen's own attention, masked like CSA2

| tokens each query may read | WikiText loss (nats) | extra loss vs all tokens |
|---|---|---|
| all earlier tokens (Qwen) | 2.35 | +0.000 |
| last 128 tokens | 5.06 | +2.707 |
| last 128 + first 4 tokens | 2.68 | +0.328 |
| last 128 + first 4 + best 64 blocks of 4 | 2.36 | +0.016 |
| last 128; first 4 take weight but give zeros | 4.95 | +2.604 |
| the same + best 64 blocks of 4 | 3.90 | +1.546 |

## One CSA2 layer trained alone (error on Qwen's inputs)

| sizes | parameters (M) | layer 0 error, web + books | layer 0 error, WikiText | layer 4 error, web + books | layer 4 error, WikiText |
|---|---|---|---|---|---|
| current sizes | 2.4 | 0.26 | 0.28 | 0.56 | 0.58 |
| key/value 512 wide | 7.2 | 0.21 | 0.22 | 0.52 | 0.54 |
| full-rank query and output | 8.3 | 0.23 | 0.24 | 0.56 | 0.57 |
| every channel rotated | 2.8 | 0.20 | 0.21 | 0.50 | 0.51 |
| no pooling (blocks of 1) | 2.4 | 0.26 | 0.28 | 0.55 | 0.57 |
| all four | 30.2 | 0.10 | 0.11 | 0.43 | 0.45 |
| current sizes + first 4 tokens | 2.4 | 0.26 | 0.28 | 0.55 | 0.57 |
| all four + first 4 tokens | 30.2 | 0.11 | 0.12 | 0.43 | 0.45 |
