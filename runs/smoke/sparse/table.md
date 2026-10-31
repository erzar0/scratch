| Method | Loss above dense (nats) | K and V read (share of dense) | K sectors, rows / 8-dim chunks / transpose | Exact top-128 kept | Exact weight on kept keys | Output error |
|---|---|---|---|---|---|---|
| exact top-128 | -0.233 ± nan | 0.516 | — | 1.00 | 0.85 | 0.175 |
| Loki post-rotary, 32 axes, top-128 | -0.054 ± nan | 0.157 | — | 0.63 | 0.76 | 0.482 |
| dense attention | +0.000 ± 0.000 | — | — | — | — | — |
| SparQ, 32 dims, top-128 | +0.179 ± nan | 0.129 | — | 0.58 | 0.72 | 0.326 |
| tournament: |q| order, halving, losers dropped, keep 128 | +0.273 ± nan | 0.065 | 0.767 / 0.617 / 0.237 | 0.70 | 0.73 | 0.329 |
| Loki pre-rotary, 32 axes, top-128 | +3.292 ± nan | 0.157 | — | 0.37 | 0.67 | 0.406 |
| tournament: random order, halving, losers dropped, keep 128 | +3.449 ± nan | 0.087 | 0.184 / 0.179 / 0.381 | 0.26 | 0.38 | 2.719 |
| tournament: random order, bracket, losers dropped, keep 1024 | +3.497 ± nan | 0.281 | 0.344 / 0.563 / 1.002 | 0.34 | 0.47 | 1.347 |
| tournament: random order, bracket, losers to mean value, keep 128 | +4.280 ± nan | 0.087 | 0.184 / 0.223 / 0.658 | 0.10 | 0.28 | 2.009 |
| tournament: PCA order, bracket, losers dropped, keep 128 | +4.879 ± nan | 0.087 | 0.184 / 0.223 / 0.658 | 0.14 | 0.35 | 0.912 |
| tournament: random order, bracket, losers dropped, keep 128 | +4.910 ± nan | 0.087 | 0.184 / 0.223 / 0.658 | 0.09 | 0.26 | 2.246 |
| tournament: random order, bracket, losers keep approximate scores, keep 128 | +5.398 ± nan | 0.571 | 0.184 / 0.223 / 0.658 | 0.09 | 0.26 | 2.198 |
| tournament: Hadamard order, halving, losers to mean value, keep 128 | +5.900 ± nan | 0.087 | 0.184 / 0.181 / 0.395 | 0.24 | 0.48 | 1.453 |
