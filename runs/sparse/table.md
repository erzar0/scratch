| Method | Loss above dense (nats) | K and V read (share of dense) | K sectors, rows / 8-dim chunks / transpose / two copies | Exact top-128 kept | Exact weight on kept keys | Output error |
|---|---|---|---|---|---|---|
| dense attention | +0.000 ± 0.000 | — | — | — | — | — |
| exact top-256 | +0.074 ± 0.005 | 0.504 | — | 1.00 | 0.82 | 0.181 |
| tournament: |q| order, halving, losers dropped, keep 512 | +0.113 ± 0.006 | 0.075 | 0.763 / 0.617 / 0.280 / 0.231 | 0.88 | 0.83 | 0.201 |
| tournament: |q| order, first round in blocks of 1024, then global, losers dropped, keep 512 | +0.121 ± 0.006 | 0.075 | 0.762 / 0.638 / 0.300 / 0.249 | 0.86 | 0.82 | 0.211 |
| tournament: |q| order, first 2 rounds in blocks of 1024, then global, losers dropped, keep 512 | +0.137 ± 0.007 | 0.075 | 0.762 / 0.642 / 0.319 / 0.266 | 0.84 | 0.81 | 0.216 |
| exact top-128 | +0.139 ± 0.008 | 0.502 | — | 1.00 | 0.77 | 0.248 |
| tournament: |q| order, first 3 rounds in blocks of 1024, then global, losers dropped, keep 512 | +0.162 ± 0.007 | 0.075 | 0.762 / 0.644 / 0.339 / 0.278 | 0.78 | 0.80 | 0.229 |
| SparQ, 64 dims, top-128 | +0.163 ± 0.009 | 0.254 | 0.999 / 0.987 / 0.514 / 0.504 | 0.78 | 0.75 | 0.267 |
| tournament: |q| order, halving, losers dropped, keep 256 | +0.184 ± 0.009 | 0.069 | 0.766 / 0.618 / 0.252 / 0.228 | 0.83 | 0.78 | 0.258 |
| tournament: |q| order, halving, losers keep approximate scores, keep 128 | +0.193 ± 0.009 | 0.564 | 0.762 / 0.615 / 0.237 / — | 0.71 | 0.79 | 0.216 |
| tournament: |q| order, first 4 rounds in blocks of 1024, then global, losers dropped, keep 512 | +0.202 ± 0.007 | 0.075 | 0.763 / 0.645 / 0.356 / 0.285 | 0.66 | 0.77 | 0.261 |
| SparQ, 32 dims, top-128 | +0.214 ± 0.011 | 0.129 | 0.974 / 0.872 / 0.272 / 0.254 | 0.58 | 0.72 | 0.318 |
| SparQ, 16 dims, top-256 | +0.237 ± 0.011 | 0.070 | 0.849 / 0.641 / 0.175 / 0.133 | 0.57 | 0.72 | 0.372 |
| exact top-64 | +0.282 ± 0.014 | 0.501 | — | 0.50 | 0.70 | 0.324 |
| tournament: |q| order, halving in blocks of 4096, losers dropped, keep 512 | +0.295 ± 0.016 | 0.075 | 0.766 / 0.640 / 0.354 / 0.271 | 0.54 | 0.74 | 0.312 |
| tournament: |q| order, halving, losers dropped, keep 128 | +0.329 ± 0.012 | 0.066 | 0.770 / 0.620 / 0.237 / 0.226 | 0.70 | 0.73 | 0.325 |
| tournament: |q| order, halving, losers dropped, keep 128, last 16 kept | +0.334 ± 0.013 | 0.066 | 0.770 / 0.620 / 0.236 / — | 0.70 | 0.73 | 0.314 |
| Loki post-rotary, 64 axes, top-128 | +0.356 ± 0.016 | 0.254 | — | 0.69 | 0.70 | 0.425 |
| tournament: |q| order, halving, losers dropped, keep 128, 16 dims per round | +0.360 ± 0.016 | 0.127 | 0.922 / 0.819 / 0.448 / — | 0.83 | 0.76 | 0.268 |
| SparQ, 16 dims, top-128 | +0.377 ± 0.018 | 0.066 | 0.853 / 0.642 / 0.152 / 0.129 | 0.41 | 0.65 | 0.475 |
| SparQ, 32 dims, top-64 | +0.394 ± 0.015 | 0.127 | 0.976 / 0.873 / 0.261 / 0.252 | 0.36 | 0.64 | 0.420 |
| tournament: |q| order, halving in blocks of 1024, losers dropped, keep 512 | +0.405 ± 0.014 | 0.075 | 0.764 / 0.647 / 0.385 / 0.288 | 0.35 | 0.66 | 0.403 |
| tournament: |q| order, halving in blocks of 4096, losers dropped, keep 256 | +0.436 ± 0.021 | 0.069 | 0.768 / 0.641 / 0.315 / 0.272 | 0.35 | 0.66 | 0.405 |
| tournament: |q| order, halving, losers dropped, keep 64 | +0.634 ± 0.021 | 0.064 | 0.773 / 0.621 / 0.228 / 0.223 | 0.45 | 0.66 | 0.403 |
| tournament: PCA order, halving, losers dropped, keep 128 | +0.660 ± 0.019 | 0.066 | 0.169 / 0.148 / 0.237 / — | 0.64 | 0.68 | 0.637 |
| tournament: |q| order, halving in blocks of 1024, losers dropped, keep 256 | +0.726 ± 0.020 | 0.069 | 0.766 / 0.648 / 0.341 / 0.292 | 0.22 | 0.57 | 0.531 |
| Loki post-rotary, 32 axes, top-128 | +0.763 ± 0.027 | 0.129 | — | 0.51 | 0.65 | 0.739 |
| tournament: PCA order, halving, losers to mean value, keep 128 | +1.019 ± 0.037 | 0.066 | 0.169 / 0.146 / 0.224 / — | 0.68 | 0.76 | 0.505 |
| Loki post-rotary, 16 axes, top-128 | +2.469 ± 0.064 | 0.066 | — | 0.34 | 0.57 | 1.202 |
| tournament: |q| order, halving, losers to mean value, keep 128 | +3.335 ± 0.059 | 0.066 | 0.757 / 0.611 / 0.228 / — | 0.75 | 0.81 | 0.283 |
| tournament: random order, halving, losers to mean value, keep 128 | +4.039 ± 0.090 | 0.066 | 0.169 / 0.151 / 0.258 / — | 0.19 | 0.27 | 2.262 |
| tournament: |q| order, bracket, losers dropped, keep 128 | +4.284 ± 0.086 | 0.065 | 0.770 / 0.750 / 0.399 / — | 0.06 | 0.26 | 0.953 |
| tournament: random order, halving, losers dropped, keep 128 | +4.329 ± 0.083 | 0.066 | 0.169 / 0.151 / 0.257 / — | 0.20 | 0.28 | 2.819 |
| Loki pre-rotary, 32 axes, top-128 | +4.807 ± 0.083 | 0.129 | — | 0.21 | 0.40 | 0.752 |
| tournament: random order, bracket, losers to mean value, keep 128 | +4.992 ± 0.132 | 0.065 | 0.168 / 0.190 / 0.399 / — | 0.04 | 0.22 | 2.084 |
| tournament: Hadamard order, halving, losers to mean value, keep 128 | +5.350 ± 0.129 | 0.066 | 0.169 / 0.151 / 0.258 / — | 0.20 | 0.41 | 1.444 |
| tournament: Hadamard order, halving, losers dropped, keep 128 | +5.392 ± 0.133 | 0.066 | 0.169 / 0.151 / 0.257 / — | 0.18 | 0.38 | 1.740 |
| tournament: random order, bracket, losers keep approximate scores, keep 128 | +6.790 ± 0.119 | 0.563 | 0.168 / 0.190 / 0.399 / — | 0.03 | 0.17 | 2.681 |
| tournament: random order, bracket, losers dropped, keep 128 | +6.801 ± 0.101 | 0.065 | 0.168 / 0.190 / 0.399 / — | 0.03 | 0.17 | 2.760 |
