| Bandit, keep 512 | Loss above dense (nats) | Minus tournament: |q| order, halving, losers dropped, keep 512 (nats) | Minus SparQ, 32 dims, top-512 (nats) | Share of K and V read | Recall of the exact top-128 |
|---|---|---|---|---|---|
| bandit: |q| order, exact bound, radius × 0.125, keep 512 | +0.112 ± 0.005 | -0.001 ± 0.004 | +0.044 ± 0.004 | 0.098 | 0.951 |
| bandit: |q| order, Gaussian bound, 1 standard deviation, keep 512 | +0.115 ± 0.006 | +0.003 ± 0.004 | +0.048 ± 0.004 | 0.085 | 0.951 |
| bandit: |q| order, Gaussian bound, 0.5 standard deviations, keep 512 | +0.218 ± 0.009 | +0.106 ± 0.008 | +0.151 ± 0.007 | 0.059 | 0.844 |
| bandit: |q| order, exact bound, radius × 0.25, keep 512 | +0.075 ± 0.005 | -0.038 ± 0.003 | +0.007 ± 0.003 | 0.160 | 0.967 |
| bandit: |q| order, Gaussian bound, 2 standard deviations, keep 512 | +0.076 ± 0.005 | -0.037 ± 0.003 | +0.008 ± 0.003 | 0.144 | 0.968 |
| bandit: |q| order, exact bound, radius × 0.0625, keep 512 | +0.284 ± 0.023 | +0.172 ± 0.024 | +0.217 ± 0.022 | 0.065 | 0.874 |

| Reference | Loss above dense (nats) | Share of K and V read | Recall |
|---|---|---|---|
| tournament: |q| order, halving, losers dropped, keep 64 | +0.634 ± 0.021 | 0.064 | 0.452 |
| tournament: |q| order, halving, losers dropped, keep 128 | +0.329 ± 0.012 | 0.066 | 0.703 |
| tournament: |q| order, halving, losers dropped, keep 256 | +0.184 ± 0.009 | 0.069 | 0.832 |
| tournament: |q| order, halving, losers dropped, keep 512 | +0.113 ± 0.006 | 0.075 | 0.883 |
| SparQ, 16 dims, top-128 | +0.377 ± 0.018 | 0.066 | 0.415 |
| SparQ, 16 dims, top-256 | +0.237 ± 0.011 | 0.070 | 0.567 |
| SparQ, 32 dims, top-128 | +0.214 ± 0.011 | 0.129 | 0.583 |
| SparQ, 32 dims, top-64 | +0.394 ± 0.015 | 0.127 | 0.361 |
| SparQ, 32 dims, top-256 | +0.122 ± 0.008 | 0.133 | 0.766 |
| SparQ, 32 dims, top-512 | +0.068 ± 0.005 | 0.141 | 0.876 |
| exact top-128 | +0.139 ± 0.008 | 0.502 | 1.000 |
| exact top-64 | +0.282 ± 0.014 | 0.501 | 0.500 |
| exact top-256 | +0.074 ± 0.005 | 0.504 | 1.000 |
