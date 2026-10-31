| Method | PG-19 loss above dense at 32768 tokens (nats) | µs, 32768 tokens | µs, 131072 tokens |
|---|---|---|---|
| SparQ, 64 dims, top-512 | +0.046 ± 0.003 | 108.7 | 340.4 |
| tournament: |q| order, first round on 16 dims, halving with sampled-median cutoff, losers keep approximate scores, keep 512 | +0.065 ± 0.005 | 344.0 | 985.0 |
| SparQ, 32 dims, top-512 | +0.068 ± 0.005 | 100.4 | 226.4 |
| tournament: |q| order, first round on 16 dims, halving with sampled-median cutoff, losers dropped, keep 512 | +0.084 ± 0.006 | 152.9 | 460.5 |
| SparQ, 64 dims, top-256 | +0.086 ± 0.006 | 108.2 | 338.6 |
| tournament, keys in random slots: |q| order, halving in blocks of 1024, losers dropped, keep 512 | +0.113 ± 0.007 | 79.0 | 235.0 |
| tournament: |q| order, halving, losers dropped, keep 512 | +0.113 ± 0.006 | 138.2 | 426.9 |
| tournament: |q| order, first round on 8 dims, halving with sampled-median cutoff, losers dropped, keep 512 | +0.115 ± 0.006 | 146.3 | 427.2 |
| SparQ, 32 dims, top-256 | +0.122 ± 0.008 | 98.8 | 223.8 |
| tournament: |q| order, first 3 rounds in blocks of 1024, then global, losers dropped, keep 512 | +0.162 ± 0.007 | 101.2 | — |
| SparQ, 64 dims, top-128 | +0.163 ± 0.009 | 116.8 | 331.3 |
| SparQ, 32 dims, top-128 | +0.214 ± 0.011 | 90.9 | 215.3 |
| SparQ, 16 dims, top-128 | +0.377 ± 0.018 | 85.5 | 113.3 |
