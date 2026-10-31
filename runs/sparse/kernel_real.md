| Method | Random keys (µs) | Real keys, mean over layers (µs) | Real keys, fastest and slowest layer (µs) | Real ÷ random |
|---|---|---|---|---|
| dense attention (best of SDPA and bmm-softmax-bmm) | 225.8 | 220.9 | 219.5–224.8 | 0.98 |
| SparQ, 64 dims, top-128 | 101.7 | 103.1 | 100.3–123.1 | 1.01 |
| SparQ, 32 dims, top-128 | 90.4 | 91.3 | 88.9–93.3 | 1.01 |
| tournament, 3 rounds in blocks of 1024 then global, keep 512 | 100.9 | 99.0 | 95.8–100.7 | 0.98 |
| tournament, global halving, exact cutoffs, keep 512 | 139.3 | 132.6 | 125.1–135.9 | 0.95 |
| global halving, sampled cutoffs, 8 first dims, keep 512 | 146.3 | 139.7 | 131.6–142.1 | 0.95 |
| global halving, sampled cutoffs, 16 first dims, keep 512 | 152.6 | 145.0 | 136.9–147.7 | 0.95 |
