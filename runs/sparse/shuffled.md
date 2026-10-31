| Selection, keep 512 | Keys in position order: loss above dense (nats) | Keys in random slots: loss above dense (nats) | Random slots minus position order (nats) | Random slots minus tournament: |q| order, halving, losers dropped, keep 512 (nats) | Recall of the exact top-128, position order / random slots |
|---|---|---|---|---|---|
| halving in blocks of 1024 | +0.405 ± 0.014 | +0.113 ± 0.007 | -0.293 ± 0.014 | -0.000 ± 0.002 | 0.346 / 0.878 |
| halving in blocks of 4096 | +0.295 ± 0.016 | +0.114 ± 0.007 | -0.180 ± 0.015 | +0.002 ± 0.002 | 0.540 / 0.882 |
| bracket | +3.619 ± 0.077 | +0.260 ± 0.008 | -3.359 ± 0.077 | +0.147 ± 0.008 | 0.115 / 0.697 |

| Reference | Loss above dense (nats) | Recall |
|---|---|---|
| tournament: |q| order, halving, losers dropped, keep 512 | +0.113 ± 0.006 | 0.883 |
| SparQ, 32 dims, top-512 | +0.068 ± 0.005 | 0.876 |
| SparQ, 32 dims, top-256 | +0.122 ± 0.008 | 0.766 |

| Context length | tournament, keys in random slots: |q| order, halving in blocks of 1024, losers dropped, keep 512 (%) | tournament: |q| order, halving, losers dropped, keep 512 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 69.3 | 69.3 | -0.1 ± 0.6 |
| 8192 | 49.0 | 48.6 | +0.4 ± 0.3 |
| 16384 | 21.5 | 20.1 | +1.4 ± 1.3 |
| 32768 | 39.0 | 38.3 | +0.7 ± 1.8 |
| all | 44.7 | 44.1 | +0.6 ± 0.6 |

| Context length | tournament, keys in random slots: |q| order, halving in blocks of 1024, losers dropped, keep 512 (%) | SparQ, 32 dims, top-512 (%) | Difference (points, paired by sample) |
|---|---|---|---|
| 4096 | 69.3 | 70.8 | -1.5 ± 1.0 |
| 8192 | 49.0 | 48.8 | +0.2 ± 0.7 |
| 16384 | 21.5 | 52.6 | -31.1 ± 2.7 |
| 32768 | 39.0 | 47.8 | -8.8 ± 2.5 |
| all | 44.7 | 55.0 | -10.3 ± 1.1 |
