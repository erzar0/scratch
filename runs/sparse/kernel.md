| Method | Loss above dense (nats) | Minus SparQ, 32 dims (nats) | K and V read (share of dense) | K sectors, best layout (share of dense K) |
|---|---|---|---|---|
| SparQ, 32 dims, top-128 | +0.214 ± 0.011 | — | 0.129 | 0.254 |
| tournament: |q| order, halving, losers dropped, keep 256 | +0.184 ± 0.009 | -0.030 ± 0.006 | 0.069 | 0.228 |
| tournament: |q| order, halving in blocks of 1024, losers dropped, keep 256 | +0.726 ± 0.020 | +0.511 ± 0.024 | 0.069 | 0.292 |
| tournament: |q| order, halving in blocks of 4096, losers dropped, keep 256 | +0.436 ± 0.021 | +0.222 ± 0.021 | 0.069 | 0.272 |
| tournament: |q| order, halving in blocks of 1024, losers dropped, keep 512 | +0.405 ± 0.014 | +0.191 ± 0.016 | 0.075 | 0.288 |
| tournament: |q| order, halving in blocks of 4096, losers dropped, keep 512 | +0.295 ± 0.016 | +0.080 ± 0.017 | 0.075 | 0.271 |
| tournament: |q| order, halving, losers dropped, keep 512 | +0.113 ± 0.006 | -0.102 ± 0.008 | 0.075 | 0.231 |
| tournament: |q| order, first round in blocks of 1024, then global, losers dropped, keep 512 | +0.121 ± 0.006 | -0.094 ± 0.008 | 0.075 | 0.249 |
| tournament: |q| order, first 2 rounds in blocks of 1024, then global, losers dropped, keep 512 | +0.137 ± 0.007 | -0.078 ± 0.007 | 0.075 | 0.266 |
| tournament: |q| order, first 3 rounds in blocks of 1024, then global, losers dropped, keep 512 | +0.162 ± 0.007 | -0.052 ± 0.008 | 0.075 | 0.278 |
| tournament: |q| order, first 4 rounds in blocks of 1024, then global, losers dropped, keep 512 | +0.202 ± 0.007 | -0.012 ± 0.008 | 0.075 | 0.285 |

| Context length | tournament: |q| order, halving, losers dropped, keep 256 (%) | SparQ, 32 dims, top-128 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 58.6 | 52.5 | +6.0 ± 1.5 |
| 8192 | 42.9 | 41.6 | +1.2 ± 1.5 |
| 16384 | 17.3 | 29.1 | -11.7 ± 2.1 |
| 32768 | 28.2 | 28.6 | -0.4 ± 2.2 |
| all | 36.8 | 38.0 | -1.2 ± 1.0 |

| Context length | tournament: |q| order, halving, losers dropped, keep 512 (%) | SparQ, 32 dims, top-128 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 69.3 | 52.5 | +16.8 ± 2.4 |
| 8192 | 48.6 | 41.6 | +7.0 ± 1.8 |
| 16384 | 20.1 | 29.1 | -8.9 ± 2.2 |
| 32768 | 38.3 | 28.6 | +9.6 ± 2.7 |
| all | 44.1 | 38.0 | +6.1 ± 1.2 |

| Context length | tournament: |q| order, first 3 rounds in blocks of 1024, then global, losers dropped, keep 512 (%) | SparQ, 32 dims, top-128 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 69.7 | 52.5 | +17.1 ± 2.4 |
| 8192 | 46.9 | 41.6 | +5.2 ± 1.8 |
| 16384 | 31.8 | 29.1 | +2.7 ± 3.0 |
| 32768 | 32.1 | 28.6 | +3.5 ± 2.3 |
| all | 45.1 | 38.0 | +7.1 ± 1.2 |

| Context length | tournament: |q| order, first 3 rounds in blocks of 1024, then global, losers dropped, keep 512 (%) | tournament: |q| order, halving, losers dropped, keep 512 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 69.7 | 69.3 | +0.4 ± 0.7 |
| 8192 | 46.9 | 48.6 | -1.7 ± 1.1 |
| 16384 | 31.8 | 20.1 | +11.7 ± 2.5 |
| 32768 | 32.1 | 38.3 | -6.2 ± 2.3 |
| all | 45.1 | 44.1 | +1.0 ± 0.9 |

| Context length | dense (µs) | SparQ, 64 dims, top-128 (PG-19 loss of the keep-512 hybrid) (µs, speedup) | SparQ, 32 dims, top-128 (µs, speedup) | SparQ, 16 dims, top-128 (µs, speedup) | tournament hybrid, last 3 rounds global, keep 256 (µs, speedup) | tournament hybrid, last 3 rounds global, keep 512 (µs, speedup) |
|---|---|---|---|---|---|---|
| 8192 | 38.4 | 92.6, 0.4× | 90.9, 0.4× | 84.4, 0.5× | 77.2, 0.5× | 80.8, 0.5× |
| 16384 | 123.7 | 124.1, 1.0× | 115.5, 1.1× | 108.9, 1.1× | 86.2, 1.4× | 91.7, 1.3× |
| 32768 | 242.0 | 104.9, 2.3× | 95.4, 2.5× | 89.1, 2.7× | 95.1, 2.5× | 103.5, 2.3× |
| 65536 | 481.8 | 219.4, 2.2× | 114.0, 4.2× | 103.1, 4.7× | 187.0, 2.6× | 188.8, 2.6× |
| 131072 | 933.1 | 345.1, 2.7× | 225.4, 4.1× | 121.7, 7.7× | 370.9, 2.5× | 293.7, 3.2× |
