| Method | Loss above dense (nats) | Minus SparQ, 32 dims, top-k = keep (nats) | Minus first round on 8 dims, losers dropped, same keep (nats) | K and V read (share of dense) | K sectors, best layout (share of dense K) |
|---|---|---|---|---|---|
| tournament: |q| order, halving, losers dropped, keep 128 | +0.329 ± 0.012 | +0.114 ± 0.008 | — | 0.066 | 0.226 |
| tournament: |q| order, halving, losers keep approximate scores, keep 128 | +0.193 ± 0.009 | -0.021 ± 0.008 | -0.135 ± 0.008 | 0.564 | 0.237 |
| tournament: |q| order, first round on 16 dims, halving, losers dropped, keep 128 | +0.331 ± 0.013 | +0.117 ± 0.009 | +0.002 ± 0.010 | 0.097 | 0.290 |
| tournament: |q| order, first round on 16 dims, halving, losers keep approximate scores, keep 128 | +0.137 ± 0.009 | -0.078 ± 0.008 | -0.192 ± 0.009 | 0.595 | 0.290 |
| tournament: |q| order, first round on 16 dims, halving, losers' weight on their mean value, keep 128 | +0.231 ± 0.010 | +0.016 ± 0.008 | -0.098 ± 0.009 | 0.097 | 0.290 |
| tournament: |q| order, halving, losers dropped, keep 256 | +0.184 ± 0.009 | +0.062 ± 0.004 | — | 0.069 | 0.228 |
| tournament: |q| order, halving, losers keep approximate scores, keep 256 | +0.140 ± 0.006 | +0.018 ± 0.005 | -0.045 ± 0.005 | 0.565 | 0.228 |
| tournament: |q| order, first round on 16 dims, halving, losers dropped, keep 256 | +0.158 ± 0.008 | +0.036 ± 0.004 | -0.026 ± 0.003 | 0.100 | 0.292 |
| tournament: |q| order, first round on 16 dims, halving, losers keep approximate scores, keep 256 | +0.095 ± 0.006 | -0.027 ± 0.005 | -0.090 ± 0.006 | 0.596 | 0.292 |
| tournament: |q| order, halving, losers dropped, keep 512 | +0.113 ± 0.006 | +0.045 ± 0.004 | — | 0.075 | 0.231 |
| tournament: |q| order, halving, losers keep approximate scores, keep 512 | +0.096 ± 0.005 | +0.028 ± 0.004 | -0.016 ± 0.003 | 0.567 | 0.231 |
| tournament: |q| order, first round on 16 dims, halving, losers dropped, keep 512 | +0.084 ± 0.006 | +0.016 ± 0.004 | -0.029 ± 0.002 | 0.106 | 0.295 |
| tournament: |q| order, first round on 16 dims, halving, losers keep approximate scores, keep 512 | +0.064 ± 0.004 | -0.004 ± 0.004 | -0.049 ± 0.003 | 0.598 | 0.295 |
| tournament: |q| order, first round on 8 dims, halving, losers' weight on their mean value, keep 512 | +0.113 ± 0.005 | +0.045 ± 0.004 | +0.001 ± 0.003 | 0.075 | 0.231 |
| tournament: |q| order, first round on 16 dims, halving, losers' weight on their mean value, keep 512 | +0.086 ± 0.005 | +0.018 ± 0.004 | -0.027 ± 0.003 | 0.106 | 0.295 |
| tournament: fixed groups of 8 dims (rotary pairs) by |q|, first round on 8 dims, halving, losers dropped, keep 512 | +1.699 ± 0.055 | +1.632 ± 0.056 | +1.587 ± 0.056 | 0.075 | 0.160 |
| tournament: fixed groups of 8 dims (as stored) by |q|, first round on 8 dims, halving, losers dropped, keep 512 | +2.738 ± 0.054 | +2.671 ± 0.055 | +2.626 ± 0.054 | 0.075 | 0.161 |
| SparQ, 32 dims, top-128 | +0.214 ± 0.011 | — | — | 0.129 | 0.254 |
| SparQ, 32 dims, top-256 | +0.122 ± 0.008 | — | — | 0.133 | 0.258 |
| SparQ, 32 dims, top-512 | +0.068 ± 0.005 | — | — | 0.141 | 0.266 |
| SparQ, 64 dims, top-128 | +0.163 ± 0.009 | -0.052 ± 0.004 | — | 0.254 | 0.504 |
| SparQ, 64 dims, top-256 | +0.086 ± 0.006 | -0.036 ± 0.004 | — | 0.258 | 0.508 |
| SparQ, 64 dims, top-512 | +0.046 ± 0.003 | -0.022 ± 0.002 | — | 0.266 | 0.516 |

| Method | Books | Sampled minus exact cutoffs, paired over books (nats) | Within 2 standard errors or under 0.01 nats |
|---|---|---|---|
| tournament: |q| order, first round on 8 dims, halving with sampled-median cutoff, losers dropped, keep 512 | 16 | +0.0023 ± 0.0014 | yes |
| tournament: |q| order, first round on 16 dims, halving with sampled-median cutoff, losers dropped, keep 512 | 16 | -0.0002 ± 0.0006 | yes |
| tournament: |q| order, first round on 16 dims, halving with sampled-median cutoff, losers keep approximate scores, keep 512 | 16 | +0.0013 ± 0.0007 | yes |

| Context length | tournament: |q| order, halving, losers keep approximate scores, keep 512 (%) | tournament: |q| order, halving, losers dropped, keep 512 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 70.4 | 69.3 | +1.1 ± 0.8 |
| 8192 | 52.5 | 48.6 | +3.9 ± 1.2 |
| 16384 | 29.2 | 20.1 | +9.1 ± 2.0 |
| 32768 | 47.2 | 38.3 | +8.9 ± 2.1 |
| all | 49.8 | 44.1 | +5.7 ± 0.8 |

| Context length | tournament: |q| order, first round on 16 dims, halving, losers dropped, keep 512 (%) | tournament: |q| order, halving, losers dropped, keep 512 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 68.5 | 69.3 | -0.8 ± 0.9 |
| 8192 | 48.7 | 48.6 | +0.1 ± 0.5 |
| 16384 | 46.9 | 20.1 | +26.8 ± 2.6 |
| 32768 | 42.4 | 38.3 | +4.1 ± 2.1 |
| all | 51.6 | 44.1 | +7.5 ± 1.0 |

| Context length | tournament: |q| order, halving, losers keep approximate scores, keep 512 (%) | SparQ, 32 dims, top-128 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 70.4 | 52.5 | +17.9 ± 2.5 |
| 8192 | 52.5 | 41.6 | +10.8 ± 2.2 |
| 16384 | 29.2 | 29.1 | +0.1 ± 2.7 |
| 32768 | 47.2 | 28.6 | +18.5 ± 3.0 |
| all | 49.8 | 38.0 | +11.8 ± 1.3 |

| Context length | tournament: |q| order, first round on 16 dims, halving, losers dropped, keep 512 (%) | SparQ, 32 dims, top-128 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 68.5 | 52.5 | +15.9 ± 2.4 |
| 8192 | 48.7 | 41.6 | +7.1 ± 1.9 |
| 16384 | 46.9 | 29.1 | +17.8 ± 3.0 |
| 32768 | 42.4 | 28.6 | +13.7 ± 2.6 |
| all | 51.6 | 38.0 | +13.6 ± 1.3 |

| Context length | tournament: |q| order, first round on 16 dims, halving, losers keep approximate scores, keep 512 (%) | tournament: |q| order, first round on 16 dims, halving, losers dropped, keep 512 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 70.8 | 68.5 | +2.3 ± 0.8 |
| 8192 | 52.6 | 48.7 | +3.9 ± 1.3 |
| 16384 | 47.4 | 46.9 | +0.6 ± 1.7 |
| 32768 | 51.7 | 42.4 | +9.3 ± 2.2 |
| all | 55.6 | 51.6 | +4.0 ± 0.8 |

| Context length | tournament: |q| order, first round on 16 dims, halving, losers' weight on their mean value, keep 512 (%) | tournament: |q| order, first round on 16 dims, halving, losers dropped, keep 512 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 70.8 | 68.5 | +2.3 ± 0.8 |
| 8192 | 51.8 | 48.7 | +3.1 ± 1.1 |
| 16384 | 47.6 | 46.9 | +0.8 ± 1.4 |
| 32768 | 46.9 | 42.4 | +4.5 ± 2.4 |
| all | 54.3 | 51.6 | +2.7 ± 0.8 |

| Context length | tournament: |q| order, halving, losers dropped, keep 512 (%) | SparQ, 32 dims, top-512 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 69.3 | 70.8 | -1.4 ± 0.8 |
| 8192 | 48.6 | 48.8 | -0.2 ± 0.7 |
| 16384 | 20.1 | 52.6 | -32.5 ± 2.7 |
| 32768 | 38.3 | 47.8 | -9.5 ± 2.6 |
| all | 44.1 | 55.0 | -10.9 ± 1.1 |

| Context length | tournament: |q| order, first round on 16 dims, halving, losers dropped, keep 512 (%) | SparQ, 32 dims, top-512 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 68.5 | 70.8 | -2.3 ± 0.9 |
| 8192 | 48.7 | 48.8 | -0.1 ± 0.7 |
| 16384 | 46.9 | 52.6 | -5.8 ± 2.1 |
| 32768 | 42.4 | 47.8 | -5.4 ± 2.3 |
| all | 51.6 | 55.0 | -3.4 ± 0.8 |

| Context length | tournament: |q| order, first round on 16 dims, halving, losers keep approximate scores, keep 512 (%) | SparQ, 32 dims, top-512 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 70.8 | 70.8 | +0.0 ± 0.6 |
| 8192 | 52.6 | 48.8 | +3.8 ± 1.4 |
| 16384 | 47.4 | 52.6 | -5.2 ± 2.2 |
| 32768 | 51.7 | 47.8 | +3.9 ± 2.5 |
| all | 55.6 | 55.0 | +0.6 ± 0.9 |

| Context length | dense (µs) | SparQ, 64 dims, top-128 (µs, speedup) | SparQ, 32 dims, top-128 (µs, speedup) | tournament, 3 rounds in blocks of 1024 then global, keep 512 (µs, speedup) | tournament, global halving, exact cutoffs, keep 512 (µs, speedup) | global halving, sampled cutoffs, 8 first dims, keep 512 (µs, speedup) | global halving, sampled cutoffs, 16 first dims, keep 512 (µs, speedup) | the same, losers keep approximate scores (reads all V) (µs, speedup) | tournament, every round in 32 shuffled blocks per KV head, keep 512 (µs, speedup) |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 35.9 | 86.6, 0.4× | 84.8, 0.4× | 73.7, 0.5× | 83.1, 0.4× | 92.1, 0.4× | 93.1, 0.4× | 135.8, 0.3× | 61.0, 0.6× |
| 16384 | 116.5 | 113.2, 1.0× | 107.9, 1.1× | 85.4, 1.4× | 106.7, 1.1× | 113.1, 1.0× | 114.8, 1.0× | 202.5, 0.6× | 68.3, 1.7× |
| 32768 | 227.9 | 116.8, 2.0× | 90.9, 2.5× | 101.2, 2.3× | 138.2, 1.6× | 146.3, 1.6× | 152.9, 1.5× | 344.0, 0.7× | 79.0, 2.9× |
| 65536 | 443.8 | 208.0, 2.1× | 106.0, 4.2× | 176.7, 2.5× | 266.4, 1.7× | 256.7, 1.7× | 286.7, 1.5× | 566.1, 0.8× | 165.6, 2.7× |
| 131072 | 876.5 | 331.3, 2.6× | 215.3, 4.1× | 288.5, 3.0× | 426.9, 2.1× | 427.2, 2.1× | 460.5, 1.9× | 985.0, 0.9× | 235.0, 3.7× |

| Context length | hybrid, bisection (µs) | SparQ 32, same run (µs) | hybrid, radix select (µs) | SparQ 32, same run (µs) | hybrid ÷ SparQ 32, bisection | hybrid ÷ SparQ 32, radix select |
|---|---|---|---|---|---|---|
| 8192 | 98.8 | 96.1 | 73.7 | 84.8 | 1.03 | 0.87 |
| 16384 | 115.9 | 120.7 | 85.4 | 107.9 | 0.96 | 0.79 |
| 32768 | 141.2 | 107.3 | 101.2 | 90.9 | 1.32 | 1.11 |
| 65536 | 227.9 | 130.5 | 176.7 | 106.0 | 1.75 | 1.67 |
| 131072 | 350.1 | 247.2 | 288.5 | 215.3 | 1.42 | 1.34 |
