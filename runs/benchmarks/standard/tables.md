# Benchmarks: suite standard

### Latency, µs per decode step, budget 512: median (min–max)

latency.png: Decode-step latency on NVIDIA GeForce RTX 4070 Ti SUPER: median of 10 repeats of CUDA-graph replays; whiskers min to max. Batch 1, 8 KV heads × 2 query heads, head dim 128, bf16 K and V; every sparse version gives exact attention to about 512 keys (exceptions: docs/methods.md). Panels: iid: iid Gaussian q, k, v; LLM-like: outlier channels, 4 sink keys and 4 needle keys per query. docs/methods.md explains each row's version and its settings; n/a: a version this benchmark does not run.

#### iid

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 36 (36–36) | 116 (116–116) | 227 (226–227) | 443 (443–443) | 873 (872–873) |
| Exact top-k | PyTorch | 109 (109–109) | 154 (154–155) | 225 (225–225) | 365 (365–365) | 631 (631–632) |
| SparQ | Triton | 97 (97–97) | 126 (126–126) | 137 (113–155) | 145 (145–145) | 288 (288–288) |
| Loki | PyTorch | 105 (105–105) | 147 (147–147) | 153 (152–154) | 223 (223–224) | 387 (387–388) |
| DeepSeek-V4 CSA | transformers | 334 (330–336) | 500 (499–503) | 809 (808–810) | 1642 (1640–1646) | 3085 (3080–3087) |
| DeepSeek-V4 CSA, V4-Flash size | transformers | 747 (746–1014) | 893 (892–1312) | 1164 (1162–1558) | 1732 (1731–1734) | 2893 (2892–3037) |
| Block top-k | PyTorch | 93 (92–95) | 103 (102–104) | 129 (127–132) | 176 (171–177) | 255 (252–265) |
| Double indexing: mean | PyTorch | 150 (134–157) | 139 (120–148) | 131 (129–152) | 136 (124–141) | 128 (128–133) |
| Double indexing: bound | PyTorch | 148 (130–174) | 144 (138–158) | 157 (135–159) | 135 (131–141) | 138 (137–144) |
| Tournament: global halving | Triton | 83 (83–84) | 106 (106–111) | 139 (138–140) | 264 (264–264) | 427 (427–428) |
| Tournament: blocks | PyTorch | 348 (335–379) | 463 (463–470) | 703 (703–712) | 1316 (1316–1316) | 2457 (2454–2461) |
| Tournament: blocks, then global | Triton | 74 (73–74) | 85 (85–86) | 101 (100–101) | 176 (176–177) | 290 (290–290) |
| Tournament: wide first round | PyTorch | 421 (420–422) | 712 (712–714) | 1196 (1196–1197) | 2036 (2032–2043) | 3816 (3816–3822) |
| Tournament: sampled cutoffs | Triton | 92 (92–94) | 113 (113–114) | 147 (146–148) | 255 (255–255) | 425 (425–425) |
| Tournament: sampled cutoffs, approximate losers | Triton | 133 (133–134) | 193 (192–194) | 332 (332–334) | 542 (542–543) | 943 (941–944) |
| Tournament: channel groups | PyTorch | 426 (424–426) | 717 (717–717) | 1211 (1210–1212) | 2043 (2041–2045) | 3824 (3823–3833) |
| Tournament: mean-value losers | PyTorch | 474 (474–479) | 777 (777–781) | 1279 (1278–1279) | 2152 (2151–2156) | 4021 (4018–4028) |
| Tournament: shuffled blocks | Triton | 66 (66–66) | 73 (73–74) | 84 (83–86) | 170 (170–171) | 240 (240–240) |
| Tournament: shuffled blocks, growing budget | Triton | 69 (68–71) | 72 (72–74) | 84 (84–85) | 134 (134–134) | 234 (234–235) |
| Tournament: original | PyTorch | 468 (467–474) | 797 (795–802) | 1390 (1389–1392) | 2693 (2691–2694) | 5257 (5255–5259) |
| Bandit: exact bound | PyTorch | 1793 (1792–1808) | 2591 (2588–2599) | 2884 (2883–2887) | 4853 (4840–4863) | 9275 (9271–9286) |
| Bandit: Gaussian bound | PyTorch | 1706 (1692–1715) | 2423 (2416–2437) | 2564 (2553–2901) | 3961 (3931–3963) | 6496 (6494–6505) |

#### LLM-like

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 36 (36–36) | 116 (116–116) | 226 (226–226) | 443 (443–443) | 873 (873–874) |
| Exact top-k | PyTorch | 111 (110–111) | 156 (156–156) | 225 (225–226) | 365 (364–365) | 632 (631–633) |
| SparQ | Triton | 95 (95–95) | 121 (121–121) | 114 (112–115) | 143 (142–145) | 289 (288–289) |
| Loki | PyTorch | 108 (108–108) | 150 (150–150) | 154 (152–155) | 223 (223–224) | 388 (388–389) |
| DeepSeek-V4 CSA | transformers | 335 (332–336) | 501 (500–505) | 825 (820–827) | 1645 (1643–1648) | 3085 (3082–3087) |
| DeepSeek-V4 CSA, V4-Flash size | transformers | 748 (747–1060) | 894 (893–1011) | 1165 (1161–1167) | 1731 (1729–1852) | 2897 (2891–2904) |
| Block top-k | PyTorch | 95 (93–113) | 102 (101–104) | 127 (126–128) | 186 (185–194) | 263 (261–265) |
| Double indexing: mean | PyTorch | 119 (117–133) | 119 (118–134) | 120 (120–133) | 125 (123–138) | 194 (148–203) |
| Double indexing: bound | PyTorch | 143 (124–172) | 166 (148–189) | 160 (149–182) | 142 (137–155) | 142 (140–171) |
| Tournament: global halving | Triton | 84 (83–86) | 107 (106–108) | 139 (138–140) | 264 (264–265) | 428 (428–430) |
| Tournament: blocks | PyTorch | 341 (334–364) | 464 (464–469) | 707 (705–708) | 1320 (1319–1327) | 2453 (2453–2454) |
| Tournament: blocks, then global | Triton | 74 (73–75) | 85 (85–86) | 101 (100–101) | 177 (177–177) | 289 (289–289) |
| Tournament: wide first round | PyTorch | 422 (416–422) | 700 (693–700) | 1197 (1196–1202) | 2025 (2025–2028) | 3825 (3823–3833) |
| Tournament: sampled cutoffs | Triton | 93 (92–97) | 114 (113–115) | 146 (146–147) | 256 (256–257) | 430 (430–431) |
| Tournament: sampled cutoffs, approximate losers | Triton | 134 (133–135) | 190 (189–191) | 333 (332–334) | 543 (542–544) | 944 (943–944) |
| Tournament: channel groups | PyTorch | 423 (420–423) | 711 (711–712) | 1210 (1207–1213) | 2032 (2032–2033) | 3826 (3824–3834) |
| Tournament: mean-value losers | PyTorch | 477 (470–481) | 766 (760–766) | 1279 (1278–1280) | 2146 (2146–2148) | 4029 (4028–4038) |
| Tournament: shuffled blocks | Triton | 66 (66–73) | 73 (73–74) | 84 (83–84) | 169 (169–170) | 241 (241–241) |
| Tournament: shuffled blocks, growing budget | Triton | 70 (69–70) | 72 (72–73) | 84 (83–85) | 135 (135–135) | 234 (234–234) |
| Tournament: original | PyTorch | 449 (449–449) | 797 (796–799) | 1389 (1388–1390) | 2691 (2690–2694) | 5264 (5256–5265) |
| Bandit: exact bound | PyTorch | 1803 (1802–1825) | 2578 (2573–2593) | 2888 (2885–2900) | 4849 (4847–4860) | 9284 (9282–9294) |
| Bandit: Gaussian bound | PyTorch | 1703 (1692–1712) | 2398 (2397–2403) | 2560 (2556–2586) | 3943 (3913–3953) | 6487 (6486–6504) |

### fidelity: output error relative to dense, budget 512, mean ± SE

fidelity.png: Output error against dense attention, |o − o_dense| / |o_dense| per head: mean over 128 decode queries, ± SE over 10 seeds. 8 KV heads × 2 query heads, head dim 128; every sparse version gives exact attention to about 512 keys and runs as its float32 PyTorch emulation, which selects the same keys as its kernel (tests/test_registry.py). On iid data attention is near uniform, so no 512 keys come close to dense. Panels: iid: iid Gaussian q, k, v; Outlier channels: 4 outlier channels in q and k; LLM-like: outlier channels, 4 sink keys and 4 needle keys per query. docs/methods.md explains each row's version and its settings; n/a: a version this benchmark does not run.

#### iid

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| Exact top-k | PyTorch | 2.65 ± 0.006 | 4.12 ± 0.012 | 6.18 ± 0.017 | 9.13 ± 0.028 | 13.3 ± 0.035 |
| SparQ | Triton | 3.18 ± 0.0073 | 4.87 ± 0.013 | 7.22 ± 0.02 | 10.6 ± 0.033 | 15.4 ± 0.038 |
| Loki | PyTorch | 3.43 ± 0.0079 | 5.16 ± 0.016 | 7.53 ± 0.019 | 10.9 ± 0.034 | 15.6 ± 0.042 |
| Block top-k | PyTorch | 3.52 ± 0.0083 | 5.26 ± 0.016 | 7.65 ± 0.02 | 11.1 ± 0.036 | 15.9 ± 0.042 |
| Double indexing: mean | PyTorch | 3.56 ± 0.0085 | 5.26 ± 0.016 | 7.56 ± 0.018 | 10.8 ± 0.031 | 15.4 ± 0.047 |
| Double indexing: bound | PyTorch | 3.6 ± 0.0088 | 5.33 ± 0.014 | 7.65 ± 0.023 | 11 ± 0.034 | 15.5 ± 0.043 |
| Tournament: global halving | Triton | 3.17 ± 0.0075 | 4.72 ± 0.013 | 6.86 ± 0.018 | 9.92 ± 0.032 | 14.2 ± 0.037 |
| Tournament: blocks | PyTorch | 3.17 ± 0.0078 | 4.73 ± 0.013 | 6.87 ± 0.02 | 9.96 ± 0.033 | 14.3 ± 0.038 |
| Tournament: blocks, then global | Triton | 3.17 ± 0.0076 | 4.72 ± 0.013 | 6.86 ± 0.018 | 9.92 ± 0.032 | 14.2 ± 0.038 |
| Tournament: wide first round | PyTorch | 3.06 ± 0.0066 | 4.6 ± 0.013 | 6.72 ± 0.018 | 9.77 ± 0.033 | 14.1 ± 0.04 |
| Tournament: sampled cutoffs | Triton | 3.17 ± 0.0076 | 4.72 ± 0.013 | 6.86 ± 0.018 | 9.92 ± 0.032 | 14.2 ± 0.037 |
| Tournament: sampled cutoffs, approximate losers | Triton | 2.39 ± 0.0052 | 3.14 ± 0.011 | 3.88 ± 0.013 | 4.54 ± 0.021 | 5.02 ± 0.02 |
| Tournament: channel groups | PyTorch | 3.34 ± 0.0069 | 4.96 ± 0.014 | 7.2 ± 0.02 | 10.4 ± 0.031 | 14.9 ± 0.044 |
| Tournament: mean-value losers | PyTorch | 2.35 ± 0.0051 | 3.14 ± 0.011 | 3.94 ± 0.013 | 4.65 ± 0.02 | 5.2 ± 0.023 |
| Tournament: shuffled blocks | Triton | 3.18 ± 0.0072 | 4.73 ± 0.012 | 6.87 ± 0.018 | 9.94 ± 0.031 | 14.3 ± 0.039 |
| Tournament: shuffled blocks, growing budget | Triton | 6.8 ± 0.014 | 6.87 ± 0.019 | 6.87 ± 0.018 | 6.9 ± 0.021 | 6.9 ± 0.017 |
| Tournament: original | PyTorch | 2.54 ± 0.0068 | 3.02 ± 0.01 | 3.33 ± 0.013 | 3.48 ± 0.015 | 3.46 ± 0.01 |
| Bandit: exact bound | PyTorch | 2.66 ± 0.0057 | 4.15 ± 0.012 | 6.23 ± 0.017 | 9.24 ± 0.029 | 13.5 ± 0.037 |
| Bandit: Gaussian bound | PyTorch | 2.66 ± 0.0057 | 4.15 ± 0.012 | 6.23 ± 0.017 | 9.24 ± 0.029 | 13.5 ± 0.038 |

#### Outlier channels

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| Exact top-k | PyTorch | 1.06 ± 0.0072 | 1.75 ± 0.008 | 2.82 ± 0.019 | 4.48 ± 0.022 | 7.12 ± 0.053 |
| SparQ | Triton | 1.14 ± 0.0079 | 1.82 ± 0.0077 | 2.79 ± 0.018 | 4.21 ± 0.021 | 6.36 ± 0.035 |
| Loki | PyTorch | 1.22 ± 0.0084 | 2.02 ± 0.011 | 3.25 ± 0.024 | 5.12 ± 0.028 | 7.95 ± 0.048 |
| Block top-k | PyTorch | 2.59 ± 0.0077 | 3.82 ± 0.0078 | 5.52 ± 0.014 | 7.86 ± 0.037 | 11.2 ± 0.045 |
| Double indexing: mean | PyTorch | 2.72 ± 0.0072 | 3.95 ± 0.01 | 5.57 ± 0.015 | 7.74 ± 0.031 | 10.8 ± 0.031 |
| Double indexing: bound | PyTorch | 2.82 ± 0.0077 | 4.27 ± 0.0089 | 6.31 ± 0.022 | 9.16 ± 0.049 | 13.3 ± 0.049 |
| Tournament: global halving | Triton | 1.61 ± 0.01 | 2.64 ± 0.016 | 4.2 ± 0.024 | 6.27 ± 0.051 | 9.2 ± 0.075 |
| Tournament: blocks | PyTorch | 1.61 ± 0.011 | 2.65 ± 0.014 | 4.2 ± 0.027 | 6.34 ± 0.059 | 9.31 ± 0.071 |
| Tournament: blocks, then global | Triton | 1.61 ± 0.01 | 2.64 ± 0.016 | 4.2 ± 0.024 | 6.27 ± 0.051 | 9.2 ± 0.074 |
| Tournament: wide first round | PyTorch | 1.59 ± 0.013 | 2.63 ± 0.013 | 4.18 ± 0.028 | 6.26 ± 0.052 | 9.19 ± 0.076 |
| Tournament: sampled cutoffs | Triton | 1.61 ± 0.01 | 2.64 ± 0.016 | 4.2 ± 0.024 | 6.27 ± 0.051 | 9.2 ± 0.075 |
| Tournament: sampled cutoffs, approximate losers | Triton | 1.42 ± 0.0098 | 2.18 ± 0.014 | 3.23 ± 0.025 | 4.39 ± 0.044 | 5.84 ± 0.059 |
| Tournament: channel groups | PyTorch | 1.76 ± 0.0092 | 2.77 ± 0.013 | 4.33 ± 0.036 | 6.35 ± 0.038 | 9.21 ± 0.082 |
| Tournament: mean-value losers | PyTorch | 1.4 ± 0.011 | 2.17 ± 0.012 | 3.22 ± 0.027 | 4.37 ± 0.044 | 5.82 ± 0.057 |
| Tournament: shuffled blocks | Triton | 1.61 ± 0.011 | 2.66 ± 0.013 | 4.23 ± 0.031 | 6.3 ± 0.056 | 9.25 ± 0.08 |
| Tournament: shuffled blocks, growing budget | Triton | 3.72 ± 0.024 | 3.94 ± 0.011 | 4.23 ± 0.031 | 4.34 ± 0.034 | 4.53 ± 0.032 |
| Tournament: original | PyTorch | 2.61 ± 0.02 | 3.39 ± 0.033 | 4.35 ± 0.029 | 5.4 ± 0.062 | 6.61 ± 0.074 |
| Bandit: exact bound | PyTorch | 1.53 ± 0.0084 | 2.57 ± 0.014 | 4.14 ± 0.031 | 6.23 ± 0.054 | 9.12 ± 0.068 |
| Bandit: Gaussian bound | PyTorch | 1.53 ± 0.011 | 2.54 ± 0.014 | 4.09 ± 0.025 | 6.16 ± 0.057 | 9.02 ± 0.083 |

#### LLM-like

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| Exact top-k | PyTorch | 0.0173 ± 0.00056 | 0.0378 ± 0.0012 | 0.079 ± 0.0024 | 0.154 ± 0.0036 | 0.328 ± 0.015 |
| SparQ | Triton | 0.0217 ± 0.00067 | 0.0453 ± 0.0013 | 0.0913 ± 0.0027 | 0.174 ± 0.0041 | 0.364 ± 0.017 |
| Loki | PyTorch | 0.0191 ± 0.00062 | 0.0416 ± 0.0013 | 0.087 ± 0.0027 | 0.171 ± 0.0045 | 0.371 ± 0.018 |
| Block top-k | PyTorch | 0.173 ± 0.0048 | 0.243 ± 0.0052 | 0.364 ± 0.0059 | 0.582 ± 0.012 | 1.02 ± 0.034 |
| Double indexing: mean | PyTorch | 0.8 ± 0.0057 | 1.08 ± 0.0081 | 1.34 ± 0.0093 | 1.58 ± 0.012 | 1.88 ± 0.038 |
| Double indexing: bound | PyTorch | 0.838 ± 0.012 | 1.06 ± 0.0056 | 1.28 ± 0.013 | 1.49 ± 0.016 | 1.85 ± 0.029 |
| Tournament: global halving | Triton | 0.0414 ± 0.0011 | 0.0759 ± 0.0019 | 0.145 ± 0.0047 | 0.258 ± 0.0087 | 0.532 ± 0.028 |
| Tournament: blocks | PyTorch | 0.0417 ± 0.0011 | 0.0766 ± 0.0019 | 0.146 ± 0.0046 | 0.26 ± 0.0087 | 0.54 ± 0.028 |
| Tournament: blocks, then global | Triton | 0.0413 ± 0.0011 | 0.0759 ± 0.0019 | 0.145 ± 0.0047 | 0.258 ± 0.0087 | 0.532 ± 0.028 |
| Tournament: wide first round | PyTorch | 0.0383 ± 0.0011 | 0.0729 ± 0.0019 | 0.142 ± 0.0045 | 0.255 ± 0.0086 | 0.529 ± 0.028 |
| Tournament: sampled cutoffs | Triton | 0.0413 ± 0.0011 | 0.0759 ± 0.0019 | 0.145 ± 0.0047 | 0.258 ± 0.0087 | 0.532 ± 0.028 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.0407 ± 0.0011 | 0.0743 ± 0.0018 | 0.141 ± 0.0044 | 0.249 ± 0.0083 | 0.504 ± 0.027 |
| Tournament: channel groups | PyTorch | 0.0592 ± 0.0017 | 0.0979 ± 0.0029 | 0.168 ± 0.0052 | 0.284 ± 0.0094 | 0.564 ± 0.029 |
| Tournament: mean-value losers | PyTorch | 0.0376 ± 0.0011 | 0.0712 ± 0.0019 | 0.137 ± 0.0042 | 0.245 ± 0.0082 | 0.499 ± 0.027 |
| Tournament: shuffled blocks | Triton | 0.0426 ± 0.0011 | 0.0772 ± 0.0019 | 0.147 ± 0.0047 | 0.26 ± 0.0088 | 0.534 ± 0.028 |
| Tournament: shuffled blocks, growing budget | Triton | 0.093 ± 0.0017 | 0.103 ± 0.0021 | 0.147 ± 0.0047 | 0.222 ± 0.0074 | 0.403 ± 0.021 |
| Tournament: original | PyTorch | 0.481 ± 0.027 | 0.497 ± 0.018 | 0.54 ± 0.011 | 0.696 ± 0.014 | 0.958 ± 0.026 |
| Bandit: exact bound | PyTorch | 0.0303 ± 0.00088 | 0.064 ± 0.0017 | 0.132 ± 0.0044 | 0.246 ± 0.0086 | 0.52 ± 0.028 |
| Bandit: Gaussian bound | PyTorch | 0.0419 ± 0.0014 | 0.0878 ± 0.003 | 0.17 ± 0.0056 | 0.295 ± 0.0096 | 0.592 ± 0.029 |

### fidelity-harder: output error relative to dense, budget 512, mean ± SE

fidelity-harder.png: Output error against dense attention, |o − o_dense| / |o_dense| per head: mean over 128 decode queries, ± SE over 10 seeds. 8 KV heads × 2 query heads, head dim 128; every sparse version gives exact attention to about 512 keys and runs as its float32 PyTorch emulation, which selects the same keys as its kernel (tests/test_registry.py). On Flat data attention is near uniform, so no 512 keys come close to dense. Panels: Flat: LLM-like with q × 0.5 (softer attention); Sharp: LLM-like with q × 2 (sharper attention); Local: LLM-like with neighbouring keys alike: noise averaged over 16 positions, needles as runs of 16 keys; Many needles: LLM-like with 64 needle keys per query. docs/methods.md explains each row's version and its settings; n/a: a version this benchmark does not run.

#### Flat

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| Exact top-k | PyTorch | 0.851 ± 0.026 | 1.77 ± 0.042 | 3.49 ± 0.076 | 6.51 ± 0.086 | 12.4 ± 0.39 |
| SparQ | Triton | 0.914 ± 0.029 | 1.9 ± 0.048 | 3.76 ± 0.085 | 7.08 ± 0.1 | 13.6 ± 0.45 |
| Loki | PyTorch | 0.872 ± 0.027 | 1.82 ± 0.044 | 3.58 ± 0.08 | 6.71 ± 0.091 | 12.8 ± 0.42 |
| Block top-k | PyTorch | 1.88 ± 0.029 | 3.51 ± 0.062 | 6.57 ± 0.11 | 12 ± 0.13 | 22.3 ± 0.58 |
| Double indexing: mean | PyTorch | 2.27 ± 0.031 | 4.07 ± 0.063 | 7.1 ± 0.092 | 11.9 ± 0.11 | 20 ± 0.34 |
| Double indexing: bound | PyTorch | 2.32 ± 0.034 | 4.08 ± 0.056 | 7.16 ± 0.097 | 12.2 ± 0.13 | 20.8 ± 0.45 |
| Tournament: global halving | Triton | 1.04 ± 0.032 | 2.13 ± 0.054 | 4.14 ± 0.09 | 7.58 ± 0.11 | 14.2 ± 0.47 |
| Tournament: blocks | PyTorch | 1.05 ± 0.032 | 2.14 ± 0.054 | 4.17 ± 0.091 | 7.66 ± 0.11 | 14.4 ± 0.47 |
| Tournament: blocks, then global | Triton | 1.04 ± 0.032 | 2.13 ± 0.054 | 4.14 ± 0.09 | 7.58 ± 0.11 | 14.2 ± 0.47 |
| Tournament: wide first round | PyTorch | 1.01 ± 0.031 | 2.08 ± 0.052 | 4.06 ± 0.087 | 7.46 ± 0.11 | 14 ± 0.46 |
| Tournament: sampled cutoffs | Triton | 1.04 ± 0.032 | 2.13 ± 0.054 | 4.14 ± 0.09 | 7.58 ± 0.11 | 14.2 ± 0.47 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.72 ± 0.023 | 1.26 ± 0.03 | 2 ± 0.039 | 2.8 ± 0.046 | 3.79 ± 0.13 |
| Tournament: channel groups | PyTorch | 1.2 ± 0.038 | 2.39 ± 0.062 | 4.58 ± 0.1 | 8.27 ± 0.12 | 15.3 ± 0.51 |
| Tournament: mean-value losers | PyTorch | 0.688 ± 0.022 | 1.22 ± 0.029 | 1.93 ± 0.038 | 2.73 ± 0.046 | 3.7 ± 0.13 |
| Tournament: shuffled blocks | Triton | 1.06 ± 0.032 | 2.15 ± 0.054 | 4.17 ± 0.09 | 7.62 ± 0.11 | 14.3 ± 0.47 |
| Tournament: shuffled blocks, growing budget | Triton | 2.24 ± 0.043 | 2.98 ± 0.069 | 4.17 ± 0.09 | 5.58 ± 0.08 | 7.24 ± 0.21 |
| Tournament: original | PyTorch | 1.59 ± 0.015 | 2.33 ± 0.034 | 3.29 ± 0.03 | 4.48 ± 0.076 | 5.66 ± 0.14 |
| Bandit: exact bound | PyTorch | 0.933 ± 0.027 | 1.94 ± 0.047 | 3.84 ± 0.082 | 7.14 ± 0.1 | 13.5 ± 0.44 |
| Bandit: Gaussian bound | PyTorch | 0.993 ± 0.03 | 2.11 ± 0.054 | 4.24 ± 0.092 | 7.89 ± 0.13 | 15 ± 0.51 |

#### Sharp

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| Exact top-k | PyTorch | 1.5e-05 ± 1.3e-06 | 4.65e-05 ± 4.7e-06 | 0.000135 ± 1.6e-05 | 0.000267 ± 2.7e-05 | 0.000711 ± 7.3e-05 |
| SparQ | Triton | 6e-05 ± 8.6e-06 | 0.000267 ± 0.00012 | 0.000504 ± 9.4e-05 | 0.000814 ± 0.00013 | 0.00244 ± 0.00016 |
| Loki | PyTorch | 2.82e-05 ± 1.6e-06 | 0.000108 ± 1.3e-05 | 0.000584 ± 0.00021 | 0.00206 ± 0.00027 | 0.00707 ± 0.00095 |
| Block top-k | PyTorch | 0.0978 ± 0.0051 | 0.135 ± 0.0055 | 0.189 ± 0.0029 | 0.29 ± 0.0049 | 0.4 ± 0.0048 |
| Double indexing: mean | PyTorch | 0.768 ± 0.0066 | 1.03 ± 0.0086 | 1.21 ± 0.0089 | 1.32 ± 0.011 | 1.35 ± 0.019 |
| Double indexing: bound | PyTorch | 0.796 ± 0.012 | 0.994 ± 0.0062 | 1.15 ± 0.011 | 1.24 ± 0.011 | 1.32 ± 0.012 |
| Tournament: global halving | Triton | 0.00163 ± 0.00022 | 0.00284 ± 0.00034 | 0.00397 ± 0.00042 | 0.0049 ± 0.00052 | 0.0086 ± 0.00077 |
| Tournament: blocks | PyTorch | 0.00186 ± 0.00021 | 0.0029 ± 0.00034 | 0.00416 ± 0.00043 | 0.00514 ± 0.00048 | 0.0114 ± 0.00092 |
| Tournament: blocks, then global | Triton | 0.00169 ± 0.0002 | 0.00285 ± 0.00034 | 0.00402 ± 0.00043 | 0.00493 ± 0.00051 | 0.00859 ± 0.00077 |
| Tournament: wide first round | PyTorch | 0.00135 ± 0.00022 | 0.00242 ± 0.0003 | 0.00354 ± 0.00039 | 0.0046 ± 0.00046 | 0.00843 ± 0.00082 |
| Tournament: sampled cutoffs | Triton | 0.00163 ± 0.00022 | 0.00284 ± 0.00034 | 0.00398 ± 0.00042 | 0.0049 ± 0.00052 | 0.00859 ± 0.00077 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.00163 ± 0.00022 | 0.00285 ± 0.00034 | 0.00398 ± 0.00042 | 0.0049 ± 0.00052 | 0.00859 ± 0.00077 |
| Tournament: channel groups | PyTorch | 0.00675 ± 0.00072 | 0.00888 ± 0.00093 | 0.0106 ± 0.0014 | 0.0103 ± 0.00084 | 0.0139 ± 0.0014 |
| Tournament: mean-value losers | PyTorch | 0.00135 ± 0.00022 | 0.00241 ± 0.0003 | 0.00354 ± 0.00039 | 0.0046 ± 0.00046 | 0.00843 ± 0.00082 |
| Tournament: shuffled blocks | Triton | 0.00207 ± 0.00027 | 0.00304 ± 0.00035 | 0.00418 ± 0.00043 | 0.0052 ± 0.00056 | 0.00878 ± 0.0008 |
| Tournament: shuffled blocks, growing budget | Triton | 0.00574 ± 0.00052 | 0.00492 ± 0.00044 | 0.00417 ± 0.00043 | 0.00361 ± 0.00039 | 0.00483 ± 0.00062 |
| Tournament: original | PyTorch | 0.41 ± 0.029 | 0.383 ± 0.02 | 0.357 ± 0.012 | 0.396 ± 0.015 | 0.399 ± 0.021 |
| Bandit: exact bound | PyTorch | 0.000723 ± 0.00014 | 0.00165 ± 0.00025 | 0.00298 ± 0.00038 | 0.0044 ± 0.00048 | 0.00798 ± 0.0009 |
| Bandit: Gaussian bound | PyTorch | 0.00414 ± 0.00047 | 0.0128 ± 0.0015 | 0.0217 ± 0.0018 | 0.0345 ± 0.0025 | 0.0582 ± 0.003 |

#### Local

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| Exact top-k | PyTorch | 0.000162 ± 7.6e-06 | 0.00103 ± 5.7e-05 | 0.00341 ± 0.00022 | 0.00913 ± 0.00028 | 0.0216 ± 0.0011 |
| SparQ | Triton | 0.000961 ± 8.2e-05 | 0.00324 ± 0.00019 | 0.00779 ± 0.00056 | 0.0162 ± 0.00053 | 0.0329 ± 0.0016 |
| Loki | PyTorch | 0.00028 ± 1.3e-05 | 0.00185 ± 0.00011 | 0.00562 ± 0.00039 | 0.0138 ± 0.00042 | 0.0325 ± 0.0016 |
| Block top-k | PyTorch | 0.0364 ± 0.0016 | 0.0551 ± 0.0024 | 0.0878 ± 0.0055 | 0.127 ± 0.0025 | 0.187 ± 0.0049 |
| Double indexing: mean | PyTorch | 0.132 ± 0.0076 | 0.204 ± 0.017 | 0.284 ± 0.014 | 0.336 ± 0.012 | 0.431 ± 0.013 |
| Double indexing: bound | PyTorch | 0.0573 ± 0.0024 | 0.101 ± 0.0026 | 0.151 ± 0.0061 | 0.228 ± 0.004 | 0.276 ± 0.0072 |
| Tournament: global halving | Triton | 0.029 ± 0.00095 | 0.031 ± 0.0011 | 0.0414 ± 0.0018 | 0.056 ± 0.0018 | 0.078 ± 0.003 |
| Tournament: blocks | PyTorch | 0.0345 ± 0.0013 | 0.045 ± 0.002 | 0.0863 ± 0.0039 | 0.279 ± 0.0045 | 0.735 ± 0.014 |
| Tournament: blocks, then global | Triton | 0.029 ± 0.00095 | 0.0311 ± 0.0011 | 0.0415 ± 0.0018 | 0.0561 ± 0.0018 | 0.0792 ± 0.003 |
| Tournament: wide first round | PyTorch | 0.0232 ± 0.00086 | 0.0262 ± 0.0009 | 0.0367 ± 0.0018 | 0.0518 ± 0.0014 | 0.0739 ± 0.0029 |
| Tournament: sampled cutoffs | Triton | 0.029 ± 0.00096 | 0.031 ± 0.0011 | 0.0414 ± 0.0018 | 0.056 ± 0.0018 | 0.078 ± 0.003 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.029 ± 0.00096 | 0.0309 ± 0.0011 | 0.0413 ± 0.0018 | 0.0556 ± 0.0018 | 0.0764 ± 0.0029 |
| Tournament: channel groups | PyTorch | 0.0642 ± 0.0021 | 0.0665 ± 0.0017 | 0.0771 ± 0.0032 | 0.0935 ± 0.0021 | 0.114 ± 0.0037 |
| Tournament: mean-value losers | PyTorch | 0.0232 ± 0.00086 | 0.0262 ± 0.0009 | 0.0365 ± 0.0018 | 0.0514 ± 0.0014 | 0.0723 ± 0.0029 |
| Tournament: shuffled blocks | Triton | 0.0325 ± 0.0011 | 0.0337 ± 0.0012 | 0.0454 ± 0.0018 | 0.0599 ± 0.0019 | 0.0816 ± 0.0031 |
| Tournament: shuffled blocks, growing budget | Triton | 0.113 ± 0.0023 | 0.0688 ± 0.0023 | 0.0454 ± 0.0018 | 0.033 ± 0.0013 | 0.0293 ± 0.0016 |
| Tournament: original | PyTorch | 1.35 ± 0.034 | 1.68 ± 0.024 | 1.91 ± 0.043 | 2.17 ± 0.023 | 2.41 ± 0.026 |
| Bandit: exact bound | PyTorch | 0.0135 ± 0.00063 | 0.0177 ± 0.00079 | 0.0266 ± 0.0016 | 0.0419 ± 0.0015 | 0.0658 ± 0.0026 |
| Bandit: Gaussian bound | PyTorch | 0.0398 ± 0.0012 | 0.0617 ± 0.0019 | 0.0956 ± 0.003 | 0.133 ± 0.0035 | 0.198 ± 0.0052 |

#### Many needles

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| Exact top-k | PyTorch | 0.0702 ± 0.0022 | 0.0777 ± 0.0015 | 0.0861 ± 0.0014 | 0.0983 ± 0.0015 | 0.12 ± 0.0031 |
| SparQ | Triton | 0.0845 ± 0.0024 | 0.0927 ± 0.0016 | 0.103 ± 0.0016 | 0.116 ± 0.0018 | 0.142 ± 0.0035 |
| Loki | PyTorch | 0.0821 ± 0.0023 | 0.0897 ± 0.0016 | 0.0984 ± 0.0015 | 0.111 ± 0.0017 | 0.136 ± 0.0036 |
| Block top-k | PyTorch | 0.995 ± 0.022 | 1.2 ± 0.02 | 1.32 ± 0.022 | 1.4 ± 0.018 | 1.41 ± 0.044 |
| Double indexing: mean | PyTorch | 1.54 ± 0.022 | 2.04 ± 0.015 | 2.55 ± 0.021 | 3.1 ± 0.02 | 3.71 ± 0.018 |
| Double indexing: bound | PyTorch | 1.78 ± 0.016 | 2.39 ± 0.016 | 3.09 ± 0.019 | 3.79 ± 0.019 | 4.45 ± 0.018 |
| Tournament: global halving | Triton | 0.127 ± 0.004 | 0.127 ± 0.0018 | 0.135 ± 0.0023 | 0.144 ± 0.0026 | 0.171 ± 0.0035 |
| Tournament: blocks | PyTorch | 0.128 ± 0.0043 | 0.127 ± 0.0018 | 0.135 ± 0.0022 | 0.146 ± 0.0027 | 0.179 ± 0.0042 |
| Tournament: blocks, then global | Triton | 0.127 ± 0.0037 | 0.127 ± 0.0017 | 0.134 ± 0.0023 | 0.144 ± 0.0026 | 0.171 ± 0.0036 |
| Tournament: wide first round | PyTorch | 0.122 ± 0.003 | 0.125 ± 0.0017 | 0.133 ± 0.002 | 0.143 ± 0.0026 | 0.171 ± 0.0038 |
| Tournament: sampled cutoffs | Triton | 0.127 ± 0.0037 | 0.127 ± 0.0019 | 0.135 ± 0.0023 | 0.144 ± 0.0026 | 0.171 ± 0.0036 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.12 ± 0.0033 | 0.121 ± 0.0021 | 0.126 ± 0.0021 | 0.135 ± 0.0026 | 0.159 ± 0.0029 |
| Tournament: channel groups | PyTorch | 0.19 ± 0.0068 | 0.167 ± 0.0031 | 0.158 ± 0.003 | 0.159 ± 0.0039 | 0.18 ± 0.0043 |
| Tournament: mean-value losers | PyTorch | 0.115 ± 0.0027 | 0.118 ± 0.0014 | 0.124 ± 0.002 | 0.133 ± 0.0025 | 0.157 ± 0.0032 |
| Tournament: shuffled blocks | Triton | 0.128 ± 0.0039 | 0.128 ± 0.0017 | 0.134 ± 0.0021 | 0.145 ± 0.0026 | 0.172 ± 0.0038 |
| Tournament: shuffled blocks, growing budget | Triton | 0.18 ± 0.0061 | 0.153 ± 0.0023 | 0.134 ± 0.0021 | 0.118 ± 0.002 | 0.113 ± 0.0024 |
| Tournament: original | PyTorch | 0.861 ± 0.041 | 0.802 ± 0.033 | 0.72 ± 0.023 | 0.73 ± 0.028 | 0.684 ± 0.018 |
| Bandit: exact bound | PyTorch | 0.116 ± 0.0028 | 0.121 ± 0.0017 | 0.129 ± 0.002 | 0.141 ± 0.0026 | 0.168 ± 0.0035 |
| Bandit: Gaussian bound | PyTorch | 0.142 ± 0.0055 | 0.162 ± 0.0029 | 0.184 ± 0.0052 | 0.22 ± 0.0046 | 0.286 ± 0.0091 |

### Fidelity: Dense attention weight on the picked keys, budget 512, mean ± SE

#### iid

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 4.4e-10 | 1 ± 3.3e-10 | 1 ± 3.6e-10 | 1 ± 2.8e-10 | 1 ± 3.2e-10 |
| Exact top-k | PyTorch | 0.228 ± 0.00015 | 0.142 ± 8.7e-05 | 0.0879 ± 7.4e-05 | 0.0537 ± 5e-05 | 0.0325 ± 2.9e-05 |
| SparQ | Triton | 0.168 ± 0.00015 | 0.0992 ± 7.7e-05 | 0.0579 ± 6.6e-05 | 0.0335 ± 4.1e-05 | 0.0193 ± 2.5e-05 |
| Loki | PyTorch | 0.129 ± 0.00011 | 0.0711 ± 6.5e-05 | 0.0388 ± 4.5e-05 | 0.0211 ± 2.3e-05 | 0.0115 ± 1e-05 |
| Block top-k | PyTorch | 0.12 ± 6.7e-05 | 0.0659 ± 4.1e-05 | 0.0361 ± 1.7e-05 | 0.0197 ± 1.4e-05 | 0.0107 ± 8e-06 |
| Double indexing: mean | PyTorch | 0.104 ± 4.5e-05 | 0.0534 ± 2.7e-05 | 0.0274 ± 9.3e-06 | 0.014 ± 7.5e-06 | 0.00717 ± 4.2e-06 |
| Double indexing: bound | PyTorch | 0.0975 ± 5.3e-05 | 0.0491 ± 1.6e-05 | 0.0247 ± 7.6e-06 | 0.0124 ± 6.2e-06 | 0.00625 ± 2e-06 |
| Tournament: global halving | Triton | 0.166 ± 0.00015 | 0.105 ± 7.9e-05 | 0.0666 ± 7e-05 | 0.042 ± 4.7e-05 | 0.0263 ± 2.8e-05 |
| Tournament: blocks | PyTorch | 0.166 ± 0.00015 | 0.105 ± 8.2e-05 | 0.0661 ± 6.8e-05 | 0.0414 ± 4.7e-05 | 0.0256 ± 2.8e-05 |
| Tournament: blocks, then global | Triton | 0.166 ± 0.00015 | 0.105 ± 8e-05 | 0.0666 ± 6.9e-05 | 0.042 ± 4.8e-05 | 0.0263 ± 2.8e-05 |
| Tournament: wide first round | PyTorch | 0.18 ± 0.00015 | 0.114 ± 7.8e-05 | 0.0719 ± 6.6e-05 | 0.0451 ± 4.6e-05 | 0.0281 ± 2.6e-05 |
| Tournament: sampled cutoffs | Triton | 0.166 ± 0.00015 | 0.105 ± 8e-05 | 0.0666 ± 7.1e-05 | 0.042 ± 4.8e-05 | 0.0263 ± 2.8e-05 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.166 ± 0.00015 | 0.105 ± 8e-05 | 0.0666 ± 7.1e-05 | 0.042 ± 4.8e-05 | 0.0263 ± 2.8e-05 |
| Tournament: channel groups | PyTorch | 0.145 ± 0.00014 | 0.0896 ± 6.6e-05 | 0.0555 ± 6.1e-05 | 0.0344 ± 3.8e-05 | 0.0214 ± 2.7e-05 |
| Tournament: mean-value losers | PyTorch | 0.18 ± 0.00015 | 0.114 ± 7.8e-05 | 0.0719 ± 6.6e-05 | 0.0451 ± 4.6e-05 | 0.0281 ± 2.6e-05 |
| Tournament: shuffled blocks | Triton | 0.165 ± 0.00015 | 0.104 ± 7.8e-05 | 0.0661 ± 7e-05 | 0.0417 ± 4.8e-05 | 0.0261 ± 2.8e-05 |
| Tournament: shuffled blocks, growing budget | Triton | 0.0663 ± 7.3e-05 | 0.0662 ± 4.9e-05 | 0.0661 ± 7e-05 | 0.0661 ± 6.7e-05 | 0.0661 ± 6.1e-05 |
| Tournament: original | PyTorch | 0.103 ± 6.9e-05 | 0.0608 ± 4.9e-05 | 0.0361 ± 3.2e-05 | 0.0217 ± 2.5e-05 | 0.0131 ± 1.4e-05 |
| Bandit: exact bound | PyTorch | 0.227 ± 0.00015 | 0.142 ± 8.7e-05 | 0.0878 ± 7.4e-05 | 0.0536 ± 5e-05 | 0.0324 ± 2.9e-05 |
| Bandit: Gaussian bound | PyTorch | 0.227 ± 0.00015 | 0.142 ± 8.7e-05 | 0.0875 ± 7.3e-05 | 0.0534 ± 5e-05 | 0.0322 ± 2.9e-05 |

#### Outlier channels

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 2.4e-10 | 1 ± 3.2e-10 | 1 ± 3.7e-10 | 1 ± 3.5e-10 | 1 ± 3.6e-10 |
| Exact top-k | PyTorch | 0.548 ± 0.0015 | 0.443 ± 0.0011 | 0.353 ± 0.0016 | 0.28 ± 0.0012 | 0.219 ± 0.0011 |
| SparQ | Triton | 0.518 ± 0.0015 | 0.414 ± 0.0011 | 0.327 ± 0.0016 | 0.258 ± 0.0011 | 0.201 ± 0.001 |
| Loki | PyTorch | 0.518 ± 0.0015 | 0.415 ± 0.0011 | 0.328 ± 0.0016 | 0.259 ± 0.0012 | 0.201 ± 0.0011 |
| Block top-k | PyTorch | 0.222 ± 0.00076 | 0.144 ± 0.00049 | 0.0929 ± 0.00045 | 0.0602 ± 0.0004 | 0.038 ± 0.00029 |
| Double indexing: mean | PyTorch | 0.161 ± 0.00035 | 0.0877 ± 0.00025 | 0.0471 ± 0.00028 | 0.0251 ± 0.00018 | 0.0132 ± 0.00013 |
| Double indexing: bound | PyTorch | 0.191 ± 0.00064 | 0.118 ± 0.00089 | 0.074 ± 0.00067 | 0.0476 ± 0.00056 | 0.0307 ± 0.00047 |
| Tournament: global halving | Triton | 0.503 ± 0.0013 | 0.407 ± 0.00096 | 0.327 ± 0.0014 | 0.262 ± 0.001 | 0.207 ± 0.001 |
| Tournament: blocks | PyTorch | 0.502 ± 0.0013 | 0.406 ± 0.00096 | 0.325 ± 0.0014 | 0.258 ± 0.001 | 0.201 ± 0.00098 |
| Tournament: blocks, then global | Triton | 0.503 ± 0.0013 | 0.407 ± 0.00096 | 0.327 ± 0.0014 | 0.262 ± 0.001 | 0.207 ± 0.001 |
| Tournament: wide first round | PyTorch | 0.506 ± 0.0013 | 0.41 ± 0.00096 | 0.329 ± 0.0014 | 0.263 ± 0.001 | 0.208 ± 0.001 |
| Tournament: sampled cutoffs | Triton | 0.503 ± 0.0013 | 0.407 ± 0.00096 | 0.327 ± 0.0014 | 0.262 ± 0.001 | 0.207 ± 0.001 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.503 ± 0.0013 | 0.407 ± 0.00096 | 0.327 ± 0.0014 | 0.262 ± 0.001 | 0.207 ± 0.001 |
| Tournament: channel groups | PyTorch | 0.464 ± 0.0016 | 0.38 ± 0.0013 | 0.307 ± 0.0019 | 0.247 ± 0.0011 | 0.197 ± 0.001 |
| Tournament: mean-value losers | PyTorch | 0.506 ± 0.0013 | 0.41 ± 0.00096 | 0.329 ± 0.0014 | 0.263 ± 0.001 | 0.208 ± 0.001 |
| Tournament: shuffled blocks | Triton | 0.499 ± 0.0013 | 0.404 ± 0.00096 | 0.325 ± 0.0014 | 0.26 ± 0.001 | 0.206 ± 0.00099 |
| Tournament: shuffled blocks, growing budget | Triton | 0.321 ± 0.0012 | 0.324 ± 0.0011 | 0.325 ± 0.0014 | 0.326 ± 0.0011 | 0.325 ± 0.0012 |
| Tournament: original | PyTorch | 0.146 ± 0.0048 | 0.107 ± 0.0058 | 0.0756 ± 0.0046 | 0.0603 ± 0.0054 | 0.0433 ± 0.0038 |
| Bandit: exact bound | PyTorch | 0.52 ± 0.0012 | 0.42 ± 0.00097 | 0.335 ± 0.0014 | 0.268 ± 0.001 | 0.211 ± 0.001 |
| Bandit: Gaussian bound | PyTorch | 0.519 ± 0.0012 | 0.419 ± 0.00097 | 0.335 ± 0.0014 | 0.267 ± 0.001 | 0.21 ± 0.001 |

#### LLM-like

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 9.5e-10 | 1 ± 7.2e-10 | 1 ± 1.1e-09 | 1 ± 2.1e-09 | 1 ± 5.3e-09 |
| Exact top-k | PyTorch | 0.984 ± 0.00051 | 0.968 ± 0.00093 | 0.94 ± 0.0016 | 0.899 ± 0.0017 | 0.829 ± 0.0062 |
| SparQ | Triton | 0.981 ± 0.00059 | 0.962 ± 0.001 | 0.933 ± 0.0017 | 0.89 ± 0.0018 | 0.82 ± 0.0064 |
| Loki | PyTorch | 0.983 ± 0.00056 | 0.965 ± 0.001 | 0.936 ± 0.0017 | 0.894 ± 0.0018 | 0.823 ± 0.0064 |
| Block top-k | PyTorch | 0.898 ± 0.0019 | 0.866 ± 0.0021 | 0.821 ± 0.0015 | 0.749 ± 0.0019 | 0.657 ± 0.0051 |
| Double indexing: mean | PyTorch | 0.607 ± 0.0027 | 0.47 ± 0.0034 | 0.354 ± 0.0038 | 0.269 ± 0.0044 | 0.216 ± 0.01 |
| Double indexing: bound | PyTorch | 0.585 ± 0.0061 | 0.48 ± 0.0031 | 0.385 ± 0.0052 | 0.313 ± 0.0042 | 0.244 ± 0.006 |
| Tournament: global halving | Triton | 0.97 ± 0.00079 | 0.951 ± 0.0012 | 0.921 ± 0.0018 | 0.88 ± 0.0018 | 0.811 ± 0.0063 |
| Tournament: blocks | PyTorch | 0.97 ± 0.00079 | 0.951 ± 0.0012 | 0.92 ± 0.0018 | 0.878 ± 0.0018 | 0.808 ± 0.0062 |
| Tournament: blocks, then global | Triton | 0.97 ± 0.00079 | 0.951 ± 0.0012 | 0.921 ± 0.0018 | 0.88 ± 0.0018 | 0.811 ± 0.0063 |
| Tournament: wide first round | PyTorch | 0.972 ± 0.00074 | 0.953 ± 0.0012 | 0.922 ± 0.0018 | 0.881 ± 0.0018 | 0.812 ± 0.0063 |
| Tournament: sampled cutoffs | Triton | 0.97 ± 0.00079 | 0.951 ± 0.0012 | 0.921 ± 0.0018 | 0.88 ± 0.0018 | 0.811 ± 0.0063 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.97 ± 0.00079 | 0.951 ± 0.0012 | 0.921 ± 0.0018 | 0.88 ± 0.0018 | 0.811 ± 0.0063 |
| Tournament: channel groups | PyTorch | 0.96 ± 0.0012 | 0.939 ± 0.0016 | 0.91 ± 0.0021 | 0.871 ± 0.0018 | 0.804 ± 0.0065 |
| Tournament: mean-value losers | PyTorch | 0.972 ± 0.00074 | 0.953 ± 0.0012 | 0.922 ± 0.0018 | 0.881 ± 0.0018 | 0.812 ± 0.0063 |
| Tournament: shuffled blocks | Triton | 0.969 ± 0.00078 | 0.95 ± 0.0012 | 0.92 ± 0.0018 | 0.879 ± 0.0018 | 0.81 ± 0.0063 |
| Tournament: shuffled blocks, growing budget | Triton | 0.938 ± 0.00066 | 0.936 ± 0.0012 | 0.92 ± 0.0018 | 0.892 ± 0.0017 | 0.84 ± 0.0057 |
| Tournament: original | PyTorch | 0.749 ± 0.012 | 0.74 ± 0.0084 | 0.726 ± 0.0036 | 0.673 ± 0.0048 | 0.615 ± 0.0057 |
| Bandit: exact bound | PyTorch | 0.977 ± 0.00062 | 0.957 ± 0.0011 | 0.926 ± 0.0017 | 0.884 ± 0.0018 | 0.814 ± 0.0063 |
| Bandit: Gaussian bound | PyTorch | 0.971 ± 0.00093 | 0.946 ± 0.0015 | 0.911 ± 0.0025 | 0.866 ± 0.0022 | 0.792 ± 0.0069 |

#### Flat

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 4.5e-10 | 1 ± 3.6e-10 | 1 ± 4.8e-10 | 1 ± 5.3e-10 | 1 ± 2.7e-10 |
| Exact top-k | PyTorch | 0.602 ± 0.0091 | 0.44 ± 0.0073 | 0.3 ± 0.006 | 0.203 ± 0.0032 | 0.127 ± 0.0068 |
| SparQ | Triton | 0.586 ± 0.0093 | 0.425 ± 0.0075 | 0.288 ± 0.006 | 0.194 ± 0.0033 | 0.121 ± 0.0068 |
| Loki | PyTorch | 0.599 ± 0.0092 | 0.437 ± 0.0074 | 0.297 ± 0.006 | 0.2 ± 0.0032 | 0.125 ± 0.0069 |
| Block top-k | PyTorch | 0.396 ± 0.0043 | 0.274 ± 0.0041 | 0.176 ± 0.0034 | 0.109 ± 0.0017 | 0.0631 ± 0.0035 |
| Double indexing: mean | PyTorch | 0.291 ± 0.0043 | 0.17 ± 0.0034 | 0.0918 ± 0.0022 | 0.0506 ± 0.0013 | 0.0289 ± 0.0022 |
| Double indexing: bound | PyTorch | 0.281 ± 0.0051 | 0.171 ± 0.0029 | 0.095 ± 0.0026 | 0.0548 ± 0.0014 | 0.0294 ± 0.0021 |
| Tournament: global halving | Triton | 0.58 ± 0.0094 | 0.426 ± 0.0075 | 0.292 ± 0.006 | 0.199 ± 0.0032 | 0.125 ± 0.0068 |
| Tournament: blocks | PyTorch | 0.579 ± 0.0094 | 0.424 ± 0.0074 | 0.29 ± 0.0059 | 0.196 ± 0.0032 | 0.122 ± 0.0066 |
| Tournament: blocks, then global | Triton | 0.58 ± 0.0094 | 0.426 ± 0.0075 | 0.292 ± 0.006 | 0.199 ± 0.0032 | 0.125 ± 0.0068 |
| Tournament: wide first round | PyTorch | 0.585 ± 0.0093 | 0.429 ± 0.0074 | 0.294 ± 0.006 | 0.2 ± 0.0032 | 0.126 ± 0.0068 |
| Tournament: sampled cutoffs | Triton | 0.58 ± 0.0094 | 0.426 ± 0.0075 | 0.292 ± 0.006 | 0.199 ± 0.0032 | 0.125 ± 0.0068 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.58 ± 0.0094 | 0.426 ± 0.0075 | 0.292 ± 0.006 | 0.199 ± 0.0032 | 0.125 ± 0.0068 |
| Tournament: channel groups | PyTorch | 0.554 ± 0.01 | 0.406 ± 0.0079 | 0.28 ± 0.006 | 0.192 ± 0.0031 | 0.122 ± 0.0068 |
| Tournament: mean-value losers | PyTorch | 0.585 ± 0.0093 | 0.429 ± 0.0074 | 0.294 ± 0.006 | 0.2 ± 0.0032 | 0.126 ± 0.0068 |
| Tournament: shuffled blocks | Triton | 0.576 ± 0.0093 | 0.423 ± 0.0074 | 0.29 ± 0.0059 | 0.198 ± 0.0032 | 0.125 ± 0.0067 |
| Tournament: shuffled blocks, growing budget | Triton | 0.414 ± 0.0061 | 0.36 ± 0.0065 | 0.29 ± 0.0059 | 0.238 ± 0.0035 | 0.193 ± 0.0077 |
| Tournament: original | PyTorch | 0.356 ± 0.0027 | 0.252 ± 0.0034 | 0.17 ± 0.0028 | 0.108 ± 0.0023 | 0.065 ± 0.0031 |
| Bandit: exact bound | PyTorch | 0.596 ± 0.009 | 0.436 ± 0.0073 | 0.298 ± 0.006 | 0.202 ± 0.0032 | 0.127 ± 0.0068 |
| Bandit: Gaussian bound | PyTorch | 0.588 ± 0.0094 | 0.427 ± 0.0077 | 0.29 ± 0.006 | 0.196 ± 0.0033 | 0.123 ± 0.0068 |

#### Sharp

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 6.6e-10 | 1 ± 5.4e-10 | 1 ± 5.4e-10 | 1 ± 5.5e-10 | 1 ± 1.2e-09 |
| Exact top-k | PyTorch | 1 ± 1.2e-06 | 1 ± 4.4e-06 | 1 ± 1.5e-05 | 1 ± 2.4e-05 | 0.999 ± 6.3e-05 |
| SparQ | Triton | 1 ± 5.1e-06 | 1 ± 6.4e-05 | 1 ± 4.7e-05 | 0.999 ± 8.5e-05 | 0.998 ± 9.3e-05 |
| Loki | PyTorch | 1 ± 1.6e-06 | 1 ± 8.1e-06 | 1 ± 0.00015 | 0.999 ± 0.00016 | 0.995 ± 0.0007 |
| Block top-k | PyTorch | 0.956 ± 0.0021 | 0.939 ± 0.0024 | 0.915 ± 0.0011 | 0.868 ± 0.0022 | 0.816 ± 0.002 |
| Double indexing: mean | PyTorch | 0.632 ± 0.0032 | 0.49 ± 0.0037 | 0.372 ± 0.0041 | 0.287 ± 0.0046 | 0.239 ± 0.011 |
| Double indexing: bound | PyTorch | 0.613 ± 0.0062 | 0.507 ± 0.0035 | 0.415 ± 0.0049 | 0.348 ± 0.0042 | 0.29 ± 0.0053 |
| Tournament: global halving | Triton | 0.999 ± 0.00011 | 0.998 ± 0.00016 | 0.998 ± 0.00019 | 0.997 ± 0.00024 | 0.995 ± 0.00029 |
| Tournament: blocks | PyTorch | 0.999 ± 0.0001 | 0.998 ± 0.00016 | 0.998 ± 0.0002 | 0.997 ± 0.00022 | 0.994 ± 0.00031 |
| Tournament: blocks, then global | Triton | 0.999 ± 0.0001 | 0.998 ± 0.00016 | 0.998 ± 0.0002 | 0.997 ± 0.00024 | 0.995 ± 0.00029 |
| Tournament: wide first round | PyTorch | 0.999 ± 0.00011 | 0.999 ± 0.00013 | 0.998 ± 0.00017 | 0.997 ± 0.00021 | 0.996 ± 0.0003 |
| Tournament: sampled cutoffs | Triton | 0.999 ± 0.00011 | 0.998 ± 0.00016 | 0.998 ± 0.0002 | 0.997 ± 0.00024 | 0.995 ± 0.00029 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.999 ± 0.00011 | 0.998 ± 0.00016 | 0.998 ± 0.0002 | 0.997 ± 0.00024 | 0.995 ± 0.00029 |
| Tournament: channel groups | PyTorch | 0.996 ± 0.00047 | 0.995 ± 0.00054 | 0.994 ± 0.00083 | 0.994 ± 0.0004 | 0.992 ± 0.0006 |
| Tournament: mean-value losers | PyTorch | 0.999 ± 0.00011 | 0.999 ± 0.00013 | 0.998 ± 0.00017 | 0.997 ± 0.00021 | 0.996 ± 0.0003 |
| Tournament: shuffled blocks | Triton | 0.999 ± 0.00013 | 0.998 ± 0.00017 | 0.998 ± 0.0002 | 0.997 ± 0.00022 | 0.995 ± 0.00029 |
| Tournament: shuffled blocks, growing budget | Triton | 0.997 ± 0.00026 | 0.997 ± 0.00019 | 0.998 ± 0.0002 | 0.998 ± 0.00017 | 0.997 ± 0.00022 |
| Tournament: original | PyTorch | 0.803 ± 0.014 | 0.814 ± 0.0097 | 0.826 ± 0.0052 | 0.806 ± 0.0068 | 0.804 ± 0.01 |
| Bandit: exact bound | PyTorch | 0.999 ± 8.5e-05 | 0.999 ± 0.00011 | 0.998 ± 0.00015 | 0.997 ± 0.00021 | 0.996 ± 0.00032 |
| Bandit: Gaussian bound | PyTorch | 0.997 ± 0.00029 | 0.992 ± 0.00083 | 0.986 ± 0.0011 | 0.977 ± 0.0016 | 0.961 ± 0.0021 |

#### Local

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 6e-10 | 1 ± 4e-10 | 1 ± 9.5e-10 | 1 ± 1.5e-09 | 1 ± 3.8e-09 |
| Exact top-k | PyTorch | 1 ± 7.4e-06 | 0.999 ± 5.5e-05 | 0.997 ± 0.0002 | 0.992 ± 0.00024 | 0.981 ± 0.00087 |
| SparQ | Triton | 0.999 ± 4.9e-05 | 0.997 ± 0.00014 | 0.994 ± 0.00042 | 0.986 ± 0.00039 | 0.973 ± 0.0012 |
| Loki | PyTorch | 1 ± 1.2e-05 | 0.998 ± 9.6e-05 | 0.995 ± 0.00032 | 0.988 ± 0.00033 | 0.974 ± 0.0012 |
| Block top-k | PyTorch | 0.982 ± 0.00074 | 0.974 ± 0.001 | 0.958 ± 0.0024 | 0.94 ± 0.0011 | 0.912 ± 0.002 |
| Double indexing: mean | PyTorch | 0.924 ± 0.0049 | 0.879 ± 0.012 | 0.838 ± 0.0085 | 0.81 ± 0.0074 | 0.762 ± 0.0078 |
| Double indexing: bound | PyTorch | 0.968 ± 0.0013 | 0.943 ± 0.0016 | 0.915 ± 0.0034 | 0.875 ± 0.002 | 0.851 ± 0.0035 |
| Tournament: global halving | Triton | 0.984 ± 0.00052 | 0.982 ± 0.00059 | 0.976 ± 0.00089 | 0.967 ± 0.00077 | 0.953 ± 0.0014 |
| Tournament: blocks | PyTorch | 0.981 ± 0.00071 | 0.975 ± 0.001 | 0.952 ± 0.0021 | 0.851 ± 0.0019 | 0.633 ± 0.0061 |
| Tournament: blocks, then global | Triton | 0.984 ± 0.00052 | 0.982 ± 0.00059 | 0.976 ± 0.00089 | 0.967 ± 0.00077 | 0.953 ± 0.0014 |
| Tournament: wide first round | PyTorch | 0.987 ± 0.00047 | 0.985 ± 0.0005 | 0.979 ± 0.00079 | 0.969 ± 0.00066 | 0.955 ± 0.0014 |
| Tournament: sampled cutoffs | Triton | 0.984 ± 0.00052 | 0.982 ± 0.0006 | 0.976 ± 0.00089 | 0.967 ± 0.00077 | 0.953 ± 0.0014 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.984 ± 0.00052 | 0.982 ± 0.0006 | 0.976 ± 0.00089 | 0.967 ± 0.00077 | 0.953 ± 0.0014 |
| Tournament: channel groups | PyTorch | 0.964 ± 0.0011 | 0.964 ± 0.00095 | 0.958 ± 0.0016 | 0.948 ± 0.00077 | 0.937 ± 0.0017 |
| Tournament: mean-value losers | PyTorch | 0.987 ± 0.00047 | 0.985 ± 0.0005 | 0.979 ± 0.00079 | 0.969 ± 0.00066 | 0.955 ± 0.0014 |
| Tournament: shuffled blocks | Triton | 0.983 ± 0.00057 | 0.981 ± 0.00066 | 0.974 ± 0.00086 | 0.965 ± 0.00077 | 0.952 ± 0.0014 |
| Tournament: shuffled blocks, growing budget | Triton | 0.94 ± 0.00099 | 0.963 ± 0.0011 | 0.974 ± 0.00086 | 0.98 ± 0.00056 | 0.98 ± 0.00083 |
| Tournament: original | PyTorch | 0.351 ± 0.014 | 0.261 ± 0.0054 | 0.224 ± 0.011 | 0.182 ± 0.0066 | 0.149 ± 0.0032 |
| Bandit: exact bound | PyTorch | 0.992 ± 0.00034 | 0.989 ± 0.00044 | 0.984 ± 0.00068 | 0.974 ± 0.0006 | 0.959 ± 0.0013 |
| Bandit: Gaussian bound | PyTorch | 0.978 ± 0.00065 | 0.967 ± 0.001 | 0.948 ± 0.0018 | 0.928 ± 0.0017 | 0.899 ± 0.0022 |

#### Many needles

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 5.3e-10 | 1 ± 5.2e-10 | 1 ± 4.4e-10 | 1 ± 1e-09 | 1 ± 2.8e-09 |
| Exact top-k | PyTorch | 0.941 ± 0.0017 | 0.936 ± 0.0011 | 0.93 ± 0.001 | 0.922 ± 0.00095 | 0.909 ± 0.002 |
| SparQ | Triton | 0.932 ± 0.0019 | 0.927 ± 0.0011 | 0.921 ± 0.0011 | 0.913 ± 0.001 | 0.899 ± 0.0021 |
| Loki | PyTorch | 0.934 ± 0.0017 | 0.93 ± 0.0012 | 0.924 ± 0.0011 | 0.916 ± 0.001 | 0.903 ± 0.0021 |
| Block top-k | PyTorch | 0.537 ± 0.0094 | 0.46 ± 0.0078 | 0.426 ± 0.0074 | 0.412 ± 0.0062 | 0.428 ± 0.013 |
| Double indexing: mean | PyTorch | 0.331 ± 0.0063 | 0.225 ± 0.0023 | 0.16 ± 0.0019 | 0.119 ± 0.0009 | 0.0944 ± 0.0013 |
| Double indexing: bound | PyTorch | 0.274 ± 0.0039 | 0.181 ± 0.0029 | 0.123 ± 0.0015 | 0.0915 ± 0.0011 | 0.0688 ± 0.00083 |
| Tournament: global halving | Triton | 0.917 ± 0.0021 | 0.915 ± 0.0011 | 0.912 ± 0.0012 | 0.906 ± 0.0011 | 0.894 ± 0.0019 |
| Tournament: blocks | PyTorch | 0.917 ± 0.0021 | 0.915 ± 0.0011 | 0.912 ± 0.0012 | 0.905 ± 0.0011 | 0.891 ± 0.0021 |
| Tournament: blocks, then global | Triton | 0.917 ± 0.002 | 0.916 ± 0.0011 | 0.912 ± 0.0012 | 0.906 ± 0.0011 | 0.894 ± 0.0019 |
| Tournament: wide first round | PyTorch | 0.919 ± 0.0019 | 0.917 ± 0.0011 | 0.913 ± 0.0012 | 0.907 ± 0.0011 | 0.895 ± 0.0019 |
| Tournament: sampled cutoffs | Triton | 0.917 ± 0.0021 | 0.916 ± 0.0011 | 0.912 ± 0.0012 | 0.906 ± 0.0011 | 0.894 ± 0.0019 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.917 ± 0.0021 | 0.916 ± 0.0011 | 0.912 ± 0.0012 | 0.906 ± 0.0011 | 0.894 ± 0.0019 |
| Tournament: channel groups | PyTorch | 0.889 ± 0.0033 | 0.899 ± 0.0016 | 0.903 ± 0.0012 | 0.9 ± 0.0015 | 0.89 ± 0.002 |
| Tournament: mean-value losers | PyTorch | 0.919 ± 0.0019 | 0.917 ± 0.0011 | 0.913 ± 0.0012 | 0.907 ± 0.0011 | 0.895 ± 0.0019 |
| Tournament: shuffled blocks | Triton | 0.916 ± 0.0021 | 0.915 ± 0.0011 | 0.912 ± 0.0012 | 0.905 ± 0.0011 | 0.893 ± 0.002 |
| Tournament: shuffled blocks, growing budget | Triton | 0.888 ± 0.0027 | 0.901 ± 0.0014 | 0.912 ± 0.0012 | 0.921 ± 0.00087 | 0.928 ± 0.001 |
| Tournament: original | PyTorch | 0.638 ± 0.013 | 0.66 ± 0.012 | 0.692 ± 0.0068 | 0.69 ± 0.0094 | 0.703 ± 0.0068 |
| Bandit: exact bound | PyTorch | 0.922 ± 0.0019 | 0.919 ± 0.0011 | 0.914 ± 0.0012 | 0.908 ± 0.0011 | 0.896 ± 0.0019 |
| Bandit: Gaussian bound | PyTorch | 0.913 ± 0.0028 | 0.904 ± 0.0015 | 0.894 ± 0.0017 | 0.879 ± 0.0019 | 0.853 ± 0.0041 |

### Fidelity: Recall of the dense top-128 keys, budget 512, mean ± SE

#### iid

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| SparQ | Triton | 0.665 ± 0.00075 | 0.587 ± 0.0004 | 0.521 ± 0.00067 | 0.464 ± 0.00059 | 0.415 ± 0.0006 |
| Loki | PyTorch | 0.38 ± 0.00072 | 0.283 ± 0.00065 | 0.21 ± 0.00041 | 0.158 ± 0.00045 | 0.119 ± 0.00034 |
| Block top-k | PyTorch | 0.329 ± 0.00062 | 0.244 ± 0.00036 | 0.183 ± 0.00018 | 0.138 ± 0.00036 | 0.104 ± 0.00031 |
| Double indexing: mean | PyTorch | 0.228 ± 0.00039 | 0.135 ± 0.00028 | 0.0793 ± 0.00019 | 0.0463 ± 0.00015 | 0.0268 ± 0.00019 |
| Double indexing: bound | PyTorch | 0.195 ± 0.00042 | 0.108 ± 0.00028 | 0.0602 ± 0.0002 | 0.0331 ± 0.00014 | 0.0185 ± 0.00011 |
| Tournament: global halving | Triton | 0.628 ± 0.00076 | 0.634 ± 0.00063 | 0.655 ± 0.0006 | 0.688 ± 0.00062 | 0.728 ± 0.00063 |
| Tournament: blocks | PyTorch | 0.626 ± 0.0008 | 0.63 ± 0.00066 | 0.646 ± 0.00063 | 0.67 ± 0.00067 | 0.692 ± 0.00059 |
| Tournament: blocks, then global | Triton | 0.628 ± 0.00076 | 0.634 ± 0.00065 | 0.654 ± 0.00063 | 0.687 ± 0.0006 | 0.727 ± 0.00066 |
| Tournament: wide first round | PyTorch | 0.736 ± 0.00074 | 0.744 ± 0.00046 | 0.767 ± 0.00053 | 0.799 ± 0.00047 | 0.839 ± 0.00048 |
| Tournament: sampled cutoffs | Triton | 0.628 ± 0.00077 | 0.634 ± 0.00063 | 0.655 ± 0.00063 | 0.688 ± 0.00067 | 0.728 ± 0.00064 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.628 ± 0.00077 | 0.634 ± 0.00063 | 0.655 ± 0.00063 | 0.688 ± 0.00067 | 0.728 ± 0.00064 |
| Tournament: channel groups | PyTorch | 0.487 ± 0.00065 | 0.464 ± 0.00042 | 0.457 ± 0.00057 | 0.464 ± 0.00059 | 0.481 ± 0.0007 |
| Tournament: mean-value losers | PyTorch | 0.736 ± 0.00074 | 0.744 ± 0.00046 | 0.767 ± 0.00053 | 0.799 ± 0.00047 | 0.839 ± 0.00048 |
| Tournament: shuffled blocks | Triton | 0.62 ± 0.00064 | 0.626 ± 0.0006 | 0.646 ± 0.00068 | 0.679 ± 0.0007 | 0.72 ± 0.00065 |
| Tournament: shuffled blocks, growing budget | Triton | 0.427 ± 0.00047 | 0.544 ± 0.00049 | 0.646 ± 0.00068 | 0.731 ± 0.00062 | 0.797 ± 0.00056 |
| Tournament: original | PyTorch | 0.236 ± 0.0004 | 0.199 ± 0.00048 | 0.177 ± 0.0003 | 0.164 ± 0.00051 | 0.159 ± 0.0005 |
| Bandit: exact bound | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 7.6e-07 | 1 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.999 ± 4e-05 | 0.997 ± 3.7e-05 | 0.995 ± 5.8e-05 | 0.992 ± 9.6e-05 | 0.987 ± 0.00012 |

#### Outlier channels

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| SparQ | Triton | 0.975 ± 0.00036 | 0.967 ± 0.00064 | 0.961 ± 0.00053 | 0.957 ± 0.0004 | 0.952 ± 0.00032 |
| Loki | PyTorch | 0.973 ± 0.00045 | 0.962 ± 0.00046 | 0.951 ± 0.00059 | 0.941 ± 0.0005 | 0.929 ± 0.00046 |
| Block top-k | PyTorch | 0.316 ± 0.00068 | 0.244 ± 0.0008 | 0.19 ± 0.00045 | 0.148 ± 0.00031 | 0.115 ± 0.00047 |
| Double indexing: mean | PyTorch | 0.22 ± 0.00059 | 0.133 ± 0.00047 | 0.0803 ± 0.00041 | 0.0479 ± 0.0005 | 0.0285 ± 0.00015 |
| Double indexing: bound | PyTorch | 0.239 ± 0.0005 | 0.164 ± 0.0005 | 0.118 ± 0.00042 | 0.0903 ± 0.00048 | 0.0716 ± 0.00052 |
| Tournament: global halving | Triton | 0.935 ± 0.00097 | 0.938 ± 0.00083 | 0.946 ± 0.00079 | 0.955 ± 0.00056 | 0.966 ± 0.00046 |
| Tournament: blocks | PyTorch | 0.934 ± 0.00098 | 0.936 ± 0.00083 | 0.942 ± 0.0007 | 0.947 ± 0.00054 | 0.947 ± 0.00042 |
| Tournament: blocks, then global | Triton | 0.935 ± 0.00097 | 0.938 ± 0.00083 | 0.946 ± 0.00079 | 0.955 ± 0.00056 | 0.966 ± 0.00045 |
| Tournament: wide first round | PyTorch | 0.943 ± 0.00099 | 0.945 ± 0.00079 | 0.952 ± 0.00076 | 0.96 ± 0.00056 | 0.97 ± 0.00051 |
| Tournament: sampled cutoffs | Triton | 0.935 ± 0.00097 | 0.938 ± 0.00083 | 0.946 ± 0.00079 | 0.955 ± 0.00055 | 0.966 ± 0.00046 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.935 ± 0.00097 | 0.938 ± 0.00083 | 0.946 ± 0.00079 | 0.955 ± 0.00055 | 0.966 ± 0.00046 |
| Tournament: channel groups | PyTorch | 0.822 ± 0.0029 | 0.836 ± 0.0028 | 0.85 ± 0.003 | 0.864 ± 0.0019 | 0.881 ± 0.0017 |
| Tournament: mean-value losers | PyTorch | 0.943 ± 0.00099 | 0.945 ± 0.00079 | 0.952 ± 0.00076 | 0.96 ± 0.00056 | 0.97 ± 0.00051 |
| Tournament: shuffled blocks | Triton | 0.93 ± 0.00095 | 0.933 ± 0.00085 | 0.942 ± 0.0008 | 0.952 ± 0.00051 | 0.963 ± 0.00043 |
| Tournament: shuffled blocks, growing budget | Triton | 0.696 ± 0.00093 | 0.86 ± 0.0011 | 0.942 ± 0.0008 | 0.978 ± 0.00036 | 0.993 ± 0.00017 |
| Tournament: original | PyTorch | 0.2 ± 0.008 | 0.177 ± 0.012 | 0.155 ± 0.012 | 0.156 ± 0.017 | 0.141 ± 0.014 |
| Bandit: exact bound | PyTorch | 0.964 ± 0.00096 | 0.962 ± 0.0008 | 0.964 ± 0.00086 | 0.969 ± 0.00067 | 0.976 ± 0.00046 |
| Bandit: Gaussian bound | PyTorch | 0.96 ± 0.001 | 0.957 ± 0.00074 | 0.959 ± 0.00088 | 0.964 ± 0.00061 | 0.97 ± 0.00053 |

#### LLM-like

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| SparQ | Triton | 0.988 ± 0.00067 | 0.975 ± 0.0015 | 0.96 ± 0.0015 | 0.945 ± 0.0021 | 0.92 ± 0.0046 |
| Loki | PyTorch | 0.999 ± 9.9e-05 | 0.997 ± 0.00027 | 0.993 ± 0.00024 | 0.985 ± 0.00045 | 0.972 ± 0.0018 |
| Block top-k | PyTorch | 0.418 ± 0.00096 | 0.35 ± 0.0022 | 0.29 ± 0.0019 | 0.238 ± 0.0013 | 0.193 ± 0.0016 |
| Double indexing: mean | PyTorch | 0.313 ± 0.00084 | 0.212 ± 0.0011 | 0.141 ± 0.00067 | 0.0926 ± 0.0007 | 0.0661 ± 0.00066 |
| Double indexing: bound | PyTorch | 0.301 ± 0.0016 | 0.212 ± 0.0011 | 0.149 ± 0.0012 | 0.11 ± 0.00089 | 0.0833 ± 0.0012 |
| Tournament: global halving | Triton | 0.909 ± 0.0036 | 0.879 ± 0.0039 | 0.853 ± 0.0044 | 0.839 ± 0.0026 | 0.821 ± 0.0047 |
| Tournament: blocks | PyTorch | 0.907 ± 0.0037 | 0.875 ± 0.004 | 0.846 ± 0.0045 | 0.825 ± 0.0024 | 0.793 ± 0.0044 |
| Tournament: blocks, then global | Triton | 0.909 ± 0.0036 | 0.879 ± 0.0039 | 0.853 ± 0.0044 | 0.838 ± 0.0026 | 0.821 ± 0.0047 |
| Tournament: wide first round | PyTorch | 0.927 ± 0.0026 | 0.898 ± 0.0033 | 0.869 ± 0.0039 | 0.852 ± 0.0022 | 0.832 ± 0.0043 |
| Tournament: sampled cutoffs | Triton | 0.909 ± 0.0036 | 0.879 ± 0.0039 | 0.853 ± 0.0044 | 0.839 ± 0.0026 | 0.821 ± 0.0047 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.909 ± 0.0036 | 0.88 ± 0.0039 | 0.853 ± 0.0044 | 0.839 ± 0.0026 | 0.821 ± 0.0047 |
| Tournament: channel groups | PyTorch | 0.819 ± 0.0076 | 0.779 ± 0.0075 | 0.755 ± 0.0073 | 0.749 ± 0.0027 | 0.742 ± 0.0076 |
| Tournament: mean-value losers | PyTorch | 0.927 ± 0.0026 | 0.898 ± 0.0033 | 0.869 ± 0.0039 | 0.852 ± 0.0022 | 0.832 ± 0.0043 |
| Tournament: shuffled blocks | Triton | 0.901 ± 0.0037 | 0.872 ± 0.0039 | 0.846 ± 0.0044 | 0.833 ± 0.0025 | 0.816 ± 0.0047 |
| Tournament: shuffled blocks, growing budget | Triton | 0.611 ± 0.0027 | 0.767 ± 0.0036 | 0.846 ± 0.0044 | 0.892 ± 0.0024 | 0.917 ± 0.0037 |
| Tournament: original | PyTorch | 0.451 ± 0.0089 | 0.408 ± 0.0073 | 0.388 ± 0.0045 | 0.347 ± 0.0075 | 0.324 ± 0.0059 |
| Bandit: exact bound | PyTorch | 0.966 ± 0.0011 | 0.941 ± 0.0022 | 0.908 ± 0.0028 | 0.882 ± 0.0017 | 0.855 ± 0.0038 |
| Bandit: Gaussian bound | PyTorch | 0.92 ± 0.0037 | 0.863 ± 0.0054 | 0.809 ± 0.0064 | 0.779 ± 0.0044 | 0.746 ± 0.0099 |

#### Flat

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| SparQ | Triton | 0.995 ± 0.00045 | 0.986 ± 0.0012 | 0.976 ± 0.0011 | 0.965 ± 0.0018 | 0.946 ± 0.0047 |
| Loki | PyTorch | 1 ± 6.8e-06 | 1 ± 4.2e-05 | 0.999 ± 6e-05 | 0.998 ± 0.00013 | 0.996 ± 0.00043 |
| Block top-k | PyTorch | 0.465 ± 0.0013 | 0.402 ± 0.002 | 0.345 ± 0.0018 | 0.286 ± 0.0016 | 0.235 ± 0.0018 |
| Double indexing: mean | PyTorch | 0.336 ± 0.00093 | 0.228 ± 0.00099 | 0.152 ± 0.0007 | 0.0998 ± 0.00073 | 0.0706 ± 0.00072 |
| Double indexing: bound | PyTorch | 0.324 ± 0.0015 | 0.232 ± 0.0013 | 0.166 ± 0.0013 | 0.124 ± 0.001 | 0.0959 ± 0.0013 |
| Tournament: global halving | Triton | 0.979 ± 0.0013 | 0.98 ± 0.0013 | 0.984 ± 0.00083 | 0.99 ± 0.00049 | 0.992 ± 0.00057 |
| Tournament: blocks | PyTorch | 0.978 ± 0.0013 | 0.978 ± 0.0014 | 0.981 ± 0.00093 | 0.983 ± 0.0006 | 0.971 ± 0.001 |
| Tournament: blocks, then global | Triton | 0.979 ± 0.0013 | 0.98 ± 0.0013 | 0.984 ± 0.00082 | 0.99 ± 0.00049 | 0.992 ± 0.00057 |
| Tournament: wide first round | PyTorch | 0.988 ± 0.00072 | 0.988 ± 0.00076 | 0.991 ± 0.00042 | 0.995 ± 0.00028 | 0.996 ± 0.00025 |
| Tournament: sampled cutoffs | Triton | 0.979 ± 0.0013 | 0.98 ± 0.0013 | 0.984 ± 0.00082 | 0.99 ± 0.0005 | 0.992 ± 0.00057 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.979 ± 0.0013 | 0.98 ± 0.0013 | 0.984 ± 0.00082 | 0.99 ± 0.0005 | 0.992 ± 0.00057 |
| Tournament: channel groups | PyTorch | 0.911 ± 0.0055 | 0.903 ± 0.0059 | 0.91 ± 0.0049 | 0.921 ± 0.0026 | 0.93 ± 0.005 |
| Tournament: mean-value losers | PyTorch | 0.988 ± 0.00072 | 0.988 ± 0.00075 | 0.991 ± 0.00042 | 0.995 ± 0.00028 | 0.996 ± 0.00025 |
| Tournament: shuffled blocks | Triton | 0.975 ± 0.0014 | 0.976 ± 0.0015 | 0.981 ± 0.00089 | 0.987 ± 0.00058 | 0.99 ± 0.00065 |
| Tournament: shuffled blocks, growing budget | Triton | 0.76 ± 0.0018 | 0.931 ± 0.0024 | 0.981 ± 0.00089 | 0.995 ± 0.00034 | 0.998 ± 0.00024 |
| Tournament: original | PyTorch | 0.469 ± 0.0095 | 0.43 ± 0.0079 | 0.412 ± 0.0049 | 0.373 ± 0.0093 | 0.35 ± 0.008 |
| Bandit: exact bound | PyTorch | 0.998 ± 0.0002 | 0.997 ± 0.00016 | 0.998 ± 0.00013 | 0.999 ± 9.2e-05 | 0.999 ± 5.2e-05 |
| Bandit: Gaussian bound | PyTorch | 0.982 ± 0.0014 | 0.967 ± 0.0027 | 0.953 ± 0.0025 | 0.945 ± 0.0027 | 0.931 ± 0.0067 |

#### Sharp

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| SparQ | Triton | 0.985 ± 0.00066 | 0.97 ± 0.0016 | 0.952 ± 0.0016 | 0.932 ± 0.002 | 0.901 ± 0.0043 |
| Loki | PyTorch | 0.991 ± 0.00062 | 0.979 ± 0.0011 | 0.96 ± 0.0014 | 0.93 ± 0.0015 | 0.9 ± 0.0027 |
| Block top-k | PyTorch | 0.402 ± 0.001 | 0.33 ± 0.0024 | 0.269 ± 0.002 | 0.217 ± 0.0011 | 0.171 ± 0.0018 |
| Double indexing: mean | PyTorch | 0.304 ± 0.00093 | 0.204 ± 0.0012 | 0.134 ± 0.00074 | 0.0875 ± 0.00076 | 0.0613 ± 0.00073 |
| Double indexing: bound | PyTorch | 0.292 ± 0.0017 | 0.205 ± 0.0011 | 0.143 ± 0.0012 | 0.105 ± 0.00094 | 0.077 ± 0.0011 |
| Tournament: global halving | Triton | 0.896 ± 0.0042 | 0.86 ± 0.0044 | 0.824 ± 0.0049 | 0.799 ± 0.0027 | 0.757 ± 0.0064 |
| Tournament: blocks | PyTorch | 0.893 ± 0.0042 | 0.856 ± 0.0044 | 0.817 ± 0.005 | 0.785 ± 0.0026 | 0.728 ± 0.0059 |
| Tournament: blocks, then global | Triton | 0.896 ± 0.0042 | 0.86 ± 0.0044 | 0.824 ± 0.0049 | 0.799 ± 0.0027 | 0.757 ± 0.0064 |
| Tournament: wide first round | PyTorch | 0.915 ± 0.0031 | 0.88 ± 0.0038 | 0.842 ± 0.0043 | 0.814 ± 0.0025 | 0.769 ± 0.0061 |
| Tournament: sampled cutoffs | Triton | 0.896 ± 0.0042 | 0.86 ± 0.0044 | 0.824 ± 0.005 | 0.799 ± 0.0027 | 0.757 ± 0.0064 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.896 ± 0.0042 | 0.86 ± 0.0044 | 0.824 ± 0.005 | 0.799 ± 0.0027 | 0.757 ± 0.0064 |
| Tournament: channel groups | PyTorch | 0.802 ± 0.0081 | 0.755 ± 0.0082 | 0.722 ± 0.0077 | 0.707 ± 0.0028 | 0.676 ± 0.0089 |
| Tournament: mean-value losers | PyTorch | 0.915 ± 0.0031 | 0.88 ± 0.0038 | 0.842 ± 0.0043 | 0.814 ± 0.0025 | 0.769 ± 0.0061 |
| Tournament: shuffled blocks | Triton | 0.887 ± 0.0042 | 0.851 ± 0.0044 | 0.817 ± 0.0049 | 0.793 ± 0.0027 | 0.751 ± 0.0063 |
| Tournament: shuffled blocks, growing budget | Triton | 0.586 ± 0.003 | 0.737 ± 0.0041 | 0.817 ± 0.0049 | 0.862 ± 0.0027 | 0.879 ± 0.0054 |
| Tournament: original | PyTorch | 0.444 ± 0.0088 | 0.398 ± 0.0073 | 0.375 ± 0.0045 | 0.332 ± 0.0074 | 0.304 ± 0.0058 |
| Bandit: exact bound | PyTorch | 0.96 ± 0.0012 | 0.928 ± 0.0025 | 0.887 ± 0.0032 | 0.847 ± 0.0019 | 0.794 ± 0.0056 |
| Bandit: Gaussian bound | PyTorch | 0.905 ± 0.0043 | 0.838 ± 0.0061 | 0.775 ± 0.0068 | 0.733 ± 0.0045 | 0.679 ± 0.011 |

#### Local

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| SparQ | Triton | 0.993 ± 0.00025 | 0.99 ± 0.00047 | 0.987 ± 0.00059 | 0.984 ± 0.00073 | 0.975 ± 0.001 |
| Loki | PyTorch | 0.999 ± 5.3e-05 | 0.996 ± 0.00028 | 0.992 ± 0.00042 | 0.986 ± 0.0004 | 0.97 ± 0.0012 |
| Block top-k | PyTorch | 0.888 ± 0.0024 | 0.874 ± 0.0023 | 0.862 ± 0.0024 | 0.855 ± 0.0016 | 0.842 ± 0.0014 |
| Double indexing: mean | PyTorch | 0.759 ± 0.0035 | 0.687 ± 0.0063 | 0.639 ± 0.0066 | 0.617 ± 0.0049 | 0.592 ± 0.0068 |
| Double indexing: bound | PyTorch | 0.827 ± 0.0037 | 0.761 ± 0.0047 | 0.703 ± 0.004 | 0.685 ± 0.0027 | 0.692 ± 0.0044 |
| Tournament: global halving | Triton | 0.903 ± 0.0018 | 0.904 ± 0.002 | 0.905 ± 0.0019 | 0.912 ± 0.0013 | 0.911 ± 0.0018 |
| Tournament: blocks | PyTorch | 0.885 ± 0.0026 | 0.862 ± 0.0025 | 0.798 ± 0.0034 | 0.624 ± 0.0021 | 0.403 ± 0.0029 |
| Tournament: blocks, then global | Triton | 0.903 ± 0.0018 | 0.904 ± 0.002 | 0.905 ± 0.0019 | 0.912 ± 0.0013 | 0.909 ± 0.0018 |
| Tournament: wide first round | PyTorch | 0.915 ± 0.0018 | 0.914 ± 0.002 | 0.914 ± 0.0019 | 0.919 ± 0.0013 | 0.917 ± 0.0018 |
| Tournament: sampled cutoffs | Triton | 0.903 ± 0.0018 | 0.904 ± 0.002 | 0.905 ± 0.0019 | 0.912 ± 0.0013 | 0.911 ± 0.0018 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.903 ± 0.0018 | 0.904 ± 0.002 | 0.905 ± 0.0019 | 0.912 ± 0.0013 | 0.911 ± 0.0018 |
| Tournament: channel groups | PyTorch | 0.848 ± 0.0037 | 0.846 ± 0.002 | 0.85 ± 0.0021 | 0.863 ± 0.0014 | 0.866 ± 0.0019 |
| Tournament: mean-value losers | PyTorch | 0.915 ± 0.0018 | 0.914 ± 0.002 | 0.914 ± 0.0019 | 0.919 ± 0.0013 | 0.917 ± 0.0018 |
| Tournament: shuffled blocks | Triton | 0.896 ± 0.0019 | 0.897 ± 0.002 | 0.898 ± 0.002 | 0.907 ± 0.0014 | 0.906 ± 0.0017 |
| Tournament: shuffled blocks, growing budget | Triton | 0.622 ± 0.0036 | 0.794 ± 0.0027 | 0.898 ± 0.002 | 0.954 ± 0.00067 | 0.978 ± 0.001 |
| Tournament: original | PyTorch | 0.171 ± 0.0016 | 0.136 ± 0.0011 | 0.117 ± 0.00092 | 0.102 ± 0.0013 | 0.0934 ± 0.0014 |
| Bandit: exact bound | PyTorch | 0.939 ± 0.002 | 0.936 ± 0.0023 | 0.934 ± 0.0018 | 0.935 ± 0.0011 | 0.928 ± 0.0017 |
| Bandit: Gaussian bound | PyTorch | 0.903 ± 0.0019 | 0.879 ± 0.0016 | 0.853 ± 0.0028 | 0.834 ± 0.0015 | 0.8 ± 0.0028 |

#### Many needles

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| SparQ | Triton | 0.946 ± 0.0011 | 0.94 ± 0.0013 | 0.936 ± 0.00081 | 0.936 ± 0.00081 | 0.933 ± 0.002 |
| Loki | PyTorch | 0.968 ± 0.00096 | 0.967 ± 0.00078 | 0.967 ± 0.00092 | 0.966 ± 0.0011 | 0.963 ± 0.0018 |
| Block top-k | PyTorch | 0.324 ± 0.0043 | 0.292 ± 0.0037 | 0.28 ± 0.0041 | 0.275 ± 0.0031 | 0.284 ± 0.0083 |
| Double indexing: mean | PyTorch | 0.225 ± 0.0031 | 0.161 ± 0.002 | 0.123 ± 0.0016 | 0.0947 ± 0.0013 | 0.0761 ± 0.00096 |
| Double indexing: bound | PyTorch | 0.187 ± 0.0022 | 0.128 ± 0.0014 | 0.0918 ± 0.0011 | 0.0688 ± 0.00055 | 0.0534 ± 0.00085 |
| Tournament: global halving | Triton | 0.792 ± 0.0025 | 0.811 ± 0.0021 | 0.828 ± 0.003 | 0.838 ± 0.0017 | 0.845 ± 0.0017 |
| Tournament: blocks | PyTorch | 0.791 ± 0.0025 | 0.809 ± 0.0019 | 0.826 ± 0.0029 | 0.831 ± 0.0016 | 0.83 ± 0.0016 |
| Tournament: blocks, then global | Triton | 0.793 ± 0.0022 | 0.812 ± 0.0022 | 0.828 ± 0.003 | 0.838 ± 0.0016 | 0.845 ± 0.0018 |
| Tournament: wide first round | PyTorch | 0.806 ± 0.0027 | 0.821 ± 0.0019 | 0.834 ± 0.003 | 0.845 ± 0.0018 | 0.849 ± 0.0011 |
| Tournament: sampled cutoffs | Triton | 0.793 ± 0.0025 | 0.812 ± 0.0021 | 0.828 ± 0.0031 | 0.838 ± 0.0016 | 0.845 ± 0.0017 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.793 ± 0.0024 | 0.812 ± 0.0022 | 0.828 ± 0.003 | 0.839 ± 0.0017 | 0.844 ± 0.0018 |
| Tournament: channel groups | PyTorch | 0.712 ± 0.0025 | 0.748 ± 0.003 | 0.779 ± 0.0033 | 0.797 ± 0.0016 | 0.81 ± 0.002 |
| Tournament: mean-value losers | PyTorch | 0.807 ± 0.0027 | 0.821 ± 0.0019 | 0.834 ± 0.003 | 0.845 ± 0.0018 | 0.849 ± 0.0012 |
| Tournament: shuffled blocks | Triton | 0.787 ± 0.0021 | 0.807 ± 0.002 | 0.826 ± 0.003 | 0.835 ± 0.0016 | 0.842 ± 0.0017 |
| Tournament: shuffled blocks, growing budget | Triton | 0.61 ± 0.0035 | 0.739 ± 0.002 | 0.826 ± 0.003 | 0.883 ± 0.0014 | 0.928 ± 0.0012 |
| Tournament: original | PyTorch | 0.346 ± 0.0043 | 0.395 ± 0.0045 | 0.442 ± 0.0032 | 0.463 ± 0.0035 | 0.491 ± 0.003 |
| Bandit: exact bound | PyTorch | 0.841 ± 0.0019 | 0.849 ± 0.0019 | 0.855 ± 0.0026 | 0.859 ± 0.0022 | 0.861 ± 0.0015 |
| Bandit: Gaussian bound | PyTorch | 0.828 ± 0.0026 | 0.825 ± 0.0019 | 0.822 ± 0.003 | 0.814 ± 0.0016 | 0.801 ± 0.0016 |

### Fidelity: Share of K and V elements read, budget 512, mean ± SE

#### iid

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 0.531 ± 0 | 0.516 ± 0 | 0.508 ± 0 | 0.504 ± 0 | 0.502 ± 0 |
| SparQ | Triton | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Loki | PyTorch | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Block top-k | PyTorch | 0.188 ± 6.6e-07 | 0.156 ± 2e-07 | 0.141 ± 6.8e-08 | 0.133 ± 3e-08 | 0.129 ± 8.3e-09 |
| Double indexing: mean | PyTorch | 0.103 ± 6.3e-07 | 0.0551 ± 2.3e-07 | 0.0314 ± 1.2e-07 | 0.0196 ± 6e-08 | 0.0137 ± 2.4e-08 |
| Double indexing: bound | PyTorch | 0.11 ± 7.2e-07 | 0.0629 ± 2e-07 | 0.0392 ± 1.8e-07 | 0.0274 ± 7e-08 | 0.0215 ± 5.4e-08 |
| Tournament: global halving | Triton | 0.114 ± 8.3e-10 | 0.0871 ± 7e-09 | 0.0743 ± 5.2e-09 | 0.0682 ± 1.1e-09 | 0.0652 ± 1.3e-09 |
| Tournament: blocks | PyTorch | 0.114 ± 6.2e-10 | 0.0871 ± 7.7e-09 | 0.0743 ± 5.1e-10 | 0.0682 ± 1.5e-09 | 0.0652 ± 6.4e-10 |
| Tournament: blocks, then global | Triton | 0.114 ± 5.8e-10 | 0.0871 ± 1e-08 | 0.0743 ± 3.5e-10 | 0.0682 ± 1.1e-09 | 0.0652 ± 1e-09 |
| Tournament: wide first round | PyTorch | 0.143 ± 9.9e-09 | 0.117 ± 1e-08 | 0.105 ± 2.3e-09 | 0.0992 ± 1.2e-09 | 0.0963 ± 8.3e-10 |
| Tournament: sampled cutoffs | Triton | 0.114 ± 1.8e-06 | 0.0871 ± 1.7e-06 | 0.0743 ± 2e-06 | 0.0682 ± 3e-06 | 0.0652 ± 3e-06 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.582 ± 1.8e-06 | 0.571 ± 1.7e-06 | 0.566 ± 2e-06 | 0.564 ± 3e-06 | 0.563 ± 3e-06 |
| Tournament: channel groups | PyTorch | 0.143 ± 8e-10 | 0.117 ± 6.6e-09 | 0.105 ± 2.8e-10 | 0.0992 ± 1.6e-09 | 0.0963 ± 8.8e-10 |
| Tournament: mean-value losers | PyTorch | 0.143 ± 9.9e-09 | 0.117 ± 1e-08 | 0.105 ± 2.3e-09 | 0.0992 ± 1.2e-09 | 0.0963 ± 8.3e-10 |
| Tournament: shuffled blocks | Triton | 0.114 ± 2.2e-08 | 0.0871 ± 6.6e-09 | 0.0743 ± 4.4e-10 | 0.0682 ± 1.5e-09 | 0.0652 ± 5.3e-10 |
| Tournament: shuffled blocks, growing budget | Triton | 0.0746 ± 1.1e-09 | 0.0744 ± 3.8e-10 | 0.0743 ± 4.4e-10 | 0.0743 ± 1.3e-09 | 0.0742 ± 5.1e-10 |
| Tournament: original | PyTorch | 0.582 ± 0 | 0.571 ± 0 | 0.566 ± 0 | 0.564 ± 0 | 0.563 ± 0 |
| Bandit: exact bound | PyTorch | 0.36 ± 0.00011 | 0.318 ± 0.00012 | 0.289 ± 0.00013 | 0.267 ± 0.00013 | 0.249 ± 0.00014 |
| Bandit: Gaussian bound | PyTorch | 0.268 ± 0.00017 | 0.214 ± 0.00016 | 0.177 ± 0.00016 | 0.15 ± 0.00016 | 0.129 ± 0.00015 |

#### Outlier channels

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 0.531 ± 0 | 0.516 ± 0 | 0.508 ± 0 | 0.504 ± 0 | 0.502 ± 0 |
| SparQ | Triton | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Loki | PyTorch | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Block top-k | PyTorch | 0.188 ± 6.3e-07 | 0.156 ± 2.3e-07 | 0.141 ± 4e-08 | 0.133 ± 1.7e-08 | 0.129 ± 7.6e-09 |
| Double indexing: mean | PyTorch | 0.103 ± 5.7e-07 | 0.0551 ± 3e-07 | 0.0314 ± 1.1e-07 | 0.0196 ± 4.9e-08 | 0.0137 ± 2.5e-08 |
| Double indexing: bound | PyTorch | 0.11 ± 7e-07 | 0.0629 ± 3e-07 | 0.0392 ± 1.4e-07 | 0.0274 ± 7.5e-08 | 0.0215 ± 3.2e-08 |
| Tournament: global halving | Triton | 0.114 ± 1e-08 | 0.0871 ± 6.6e-09 | 0.0743 ± 5.9e-10 | 0.0682 ± 2e-10 | 0.0652 ± 7.4e-10 |
| Tournament: blocks | PyTorch | 0.114 ± 1.4e-08 | 0.0871 ± 5.2e-09 | 0.0743 ± 5e-10 | 0.0682 ± 1.6e-09 | 0.0652 ± 5.6e-10 |
| Tournament: blocks, then global | Triton | 0.114 ± 5.1e-10 | 0.0871 ± 7e-09 | 0.0743 ± 4.6e-10 | 0.0682 ± 3.3e-10 | 0.0652 ± 6.3e-10 |
| Tournament: wide first round | PyTorch | 0.143 ± 1e-08 | 0.117 ± 4.8e-09 | 0.105 ± 2.2e-09 | 0.0992 ± 2.5e-10 | 0.0963 ± 7.1e-10 |
| Tournament: sampled cutoffs | Triton | 0.114 ± 5.1e-06 | 0.0871 ± 9.8e-06 | 0.0743 ± 5.6e-06 | 0.0681 ± 7e-06 | 0.0652 ± 1.3e-05 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.582 ± 5.1e-06 | 0.571 ± 9.8e-06 | 0.566 ± 5.6e-06 | 0.564 ± 7e-06 | 0.563 ± 1.3e-05 |
| Tournament: channel groups | PyTorch | 0.143 ± 5.7e-10 | 0.117 ± 4.7e-09 | 0.105 ± 3.4e-10 | 0.0992 ± 9.9e-10 | 0.0963 ± 5.5e-10 |
| Tournament: mean-value losers | PyTorch | 0.143 ± 1e-08 | 0.117 ± 4.8e-09 | 0.105 ± 2.2e-09 | 0.0992 ± 2.5e-10 | 0.0963 ± 7.1e-10 |
| Tournament: shuffled blocks | Triton | 0.114 ± 6.2e-10 | 0.0871 ± 7.1e-09 | 0.0743 ± 5.4e-09 | 0.0682 ± 1.6e-09 | 0.0652 ± 8.7e-10 |
| Tournament: shuffled blocks, growing budget | Triton | 0.0746 ± 0 | 0.0744 ± 4.9e-10 | 0.0743 ± 5.4e-09 | 0.0743 ± 2.7e-09 | 0.0742 ± 6.2e-10 |
| Tournament: original | PyTorch | 0.582 ± 0 | 0.571 ± 0 | 0.566 ± 0 | 0.564 ± 0 | 0.563 ± 0 |
| Bandit: exact bound | PyTorch | 0.197 ± 0.00053 | 0.14 ± 0.00028 | 0.103 ± 0.00038 | 0.079 ± 0.00023 | 0.0636 ± 0.00014 |
| Bandit: Gaussian bound | PyTorch | 0.121 ± 0.00026 | 0.0792 ± 0.00011 | 0.0568 ± 8e-05 | 0.0449 ± 5.1e-05 | 0.0385 ± 4.1e-05 |

#### LLM-like

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 0.531 ± 0 | 0.516 ± 0 | 0.508 ± 0 | 0.504 ± 0 | 0.502 ± 0 |
| SparQ | Triton | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Loki | PyTorch | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Block top-k | PyTorch | 0.188 ± 7e-07 | 0.156 ± 1.5e-07 | 0.141 ± 6.5e-08 | 0.133 ± 1.4e-08 | 0.129 ± 4.9e-09 |
| Double indexing: mean | PyTorch | 0.103 ± 6.8e-07 | 0.0551 ± 3.3e-07 | 0.0314 ± 1.5e-07 | 0.0196 ± 4.8e-08 | 0.0137 ± 2.9e-08 |
| Double indexing: bound | PyTorch | 0.11 ± 5.6e-07 | 0.0629 ± 2.8e-07 | 0.0392 ± 1.7e-07 | 0.0274 ± 8e-08 | 0.0215 ± 3.4e-08 |
| Tournament: global halving | Triton | 0.114 ± 9.2e-07 | 0.0871 ± 5.3e-07 | 0.0743 ± 2.2e-07 | 0.0682 ± 1.1e-07 | 0.0652 ± 6.4e-08 |
| Tournament: blocks | PyTorch | 0.114 ± 8.2e-07 | 0.0871 ± 6.6e-07 | 0.0743 ± 3.3e-07 | 0.0682 ± 9.8e-08 | 0.0652 ± 5.4e-08 |
| Tournament: blocks, then global | Triton | 0.114 ± 7.9e-07 | 0.0871 ± 4.6e-07 | 0.0743 ± 2.4e-07 | 0.0682 ± 1.1e-07 | 0.0652 ± 6.3e-08 |
| Tournament: wide first round | PyTorch | 0.143 ± 1e-06 | 0.117 ± 3.1e-07 | 0.105 ± 1.7e-07 | 0.0992 ± 1.2e-07 | 0.0963 ± 4.6e-08 |
| Tournament: sampled cutoffs | Triton | 0.114 ± 6.2e-06 | 0.0871 ± 1.4e-05 | 0.0743 ± 1.5e-05 | 0.0681 ± 1.5e-05 | 0.0652 ± 1.4e-05 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.582 ± 5.6e-06 | 0.571 ± 1.4e-05 | 0.566 ± 1.5e-05 | 0.564 ± 1.5e-05 | 0.563 ± 1.4e-05 |
| Tournament: channel groups | PyTorch | 0.143 ± 5.5e-07 | 0.117 ± 4.6e-07 | 0.105 ± 2e-07 | 0.0992 ± 8.5e-08 | 0.0963 ± 5.6e-08 |
| Tournament: mean-value losers | PyTorch | 0.143 ± 8.8e-07 | 0.117 ± 3.8e-07 | 0.105 ± 1.8e-07 | 0.0992 ± 1.3e-07 | 0.0963 ± 4.5e-08 |
| Tournament: shuffled blocks | Triton | 0.114 ± 2.1e-06 | 0.0872 ± 5.2e-07 | 0.0743 ± 3.3e-07 | 0.0682 ± 1.5e-07 | 0.0652 ± 4.2e-08 |
| Tournament: shuffled blocks, growing budget | Triton | 0.0747 ± 2.2e-06 | 0.0744 ± 9.5e-07 | 0.0743 ± 3.3e-07 | 0.0743 ± 8.8e-08 | 0.0742 ± 2.2e-08 |
| Tournament: original | PyTorch | 0.582 ± 0 | 0.571 ± 0 | 0.566 ± 0 | 0.564 ± 0 | 0.563 ± 0 |
| Bandit: exact bound | PyTorch | 0.17 ± 0.0023 | 0.126 ± 0.0016 | 0.0949 ± 0.00087 | 0.0729 ± 0.0007 | 0.0597 ± 0.00098 |
| Bandit: Gaussian bound | PyTorch | 0.111 ± 0.00068 | 0.0743 ± 0.0004 | 0.0544 ± 0.00018 | 0.0435 ± 0.0001 | 0.0379 ± 0.00011 |

#### Flat

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 0.531 ± 0 | 0.516 ± 0 | 0.508 ± 0 | 0.504 ± 0 | 0.502 ± 0 |
| SparQ | Triton | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Loki | PyTorch | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Block top-k | PyTorch | 0.188 ± 6.8e-07 | 0.156 ± 1.2e-07 | 0.141 ± 7.9e-08 | 0.133 ± 1.6e-08 | 0.129 ± 4.6e-09 |
| Double indexing: mean | PyTorch | 0.103 ± 7.6e-07 | 0.0551 ± 2.9e-07 | 0.0314 ± 1.5e-07 | 0.0196 ± 4.3e-08 | 0.0137 ± 3.3e-08 |
| Double indexing: bound | PyTorch | 0.11 ± 6.1e-07 | 0.0629 ± 2.8e-07 | 0.0392 ± 2e-07 | 0.0274 ± 8.5e-08 | 0.0215 ± 3.9e-08 |
| Tournament: global halving | Triton | 0.114 ± 1.7e-06 | 0.0871 ± 7.5e-07 | 0.0743 ± 2.1e-07 | 0.0682 ± 1.2e-07 | 0.0652 ± 8.6e-08 |
| Tournament: blocks | PyTorch | 0.114 ± 1.1e-06 | 0.0871 ± 6.2e-07 | 0.0743 ± 2.5e-07 | 0.0682 ± 1.2e-07 | 0.0652 ± 6.5e-08 |
| Tournament: blocks, then global | Triton | 0.114 ± 1.4e-06 | 0.0871 ± 8e-07 | 0.0743 ± 2.2e-07 | 0.0682 ± 1.1e-07 | 0.0652 ± 9e-08 |
| Tournament: wide first round | PyTorch | 0.143 ± 8.6e-07 | 0.117 ± 5.6e-07 | 0.105 ± 1.6e-07 | 0.0992 ± 8.1e-08 | 0.0963 ± 7.8e-08 |
| Tournament: sampled cutoffs | Triton | 0.114 ± 6.7e-06 | 0.0871 ± 1.4e-05 | 0.0743 ± 1.6e-05 | 0.0681 ± 1.6e-05 | 0.0652 ± 1.4e-05 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.582 ± 6e-06 | 0.571 ± 1.4e-05 | 0.566 ± 1.6e-05 | 0.564 ± 1.6e-05 | 0.563 ± 1.4e-05 |
| Tournament: channel groups | PyTorch | 0.143 ± 1.2e-06 | 0.117 ± 6e-07 | 0.105 ± 1.6e-07 | 0.0992 ± 8.3e-08 | 0.0963 ± 4.9e-08 |
| Tournament: mean-value losers | PyTorch | 0.143 ± 5.5e-07 | 0.117 ± 5.3e-07 | 0.105 ± 1.7e-07 | 0.0992 ± 8.4e-08 | 0.0963 ± 7.7e-08 |
| Tournament: shuffled blocks | Triton | 0.114 ± 2e-06 | 0.0872 ± 6.9e-07 | 0.0743 ± 4.2e-07 | 0.0682 ± 1.6e-07 | 0.0652 ± 3.5e-08 |
| Tournament: shuffled blocks, growing budget | Triton | 0.0747 ± 1.9e-06 | 0.0744 ± 9.8e-07 | 0.0743 ± 4.1e-07 | 0.0743 ± 6.5e-08 | 0.0742 ± 1.1e-08 |
| Tournament: original | PyTorch | 0.582 ± 0 | 0.571 ± 0 | 0.566 ± 0 | 0.564 ± 0 | 0.563 ± 0 |
| Bandit: exact bound | PyTorch | 0.176 ± 0.0025 | 0.132 ± 0.0018 | 0.101 ± 0.001 | 0.0779 ± 0.0008 | 0.0641 ± 0.0012 |
| Bandit: Gaussian bound | PyTorch | 0.113 ± 0.00079 | 0.076 ± 0.00049 | 0.0555 ± 0.00023 | 0.0442 ± 0.00012 | 0.0383 ± 0.00015 |

#### Sharp

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 0.531 ± 0 | 0.516 ± 0 | 0.508 ± 0 | 0.504 ± 0 | 0.502 ± 0 |
| SparQ | Triton | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Loki | PyTorch | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Block top-k | PyTorch | 0.188 ± 6.5e-07 | 0.156 ± 1.5e-07 | 0.141 ± 5.5e-08 | 0.133 ± 1.4e-08 | 0.129 ± 5.1e-09 |
| Double indexing: mean | PyTorch | 0.103 ± 6.2e-07 | 0.0551 ± 3.1e-07 | 0.0314 ± 1.5e-07 | 0.0196 ± 5.1e-08 | 0.0137 ± 2.9e-08 |
| Double indexing: bound | PyTorch | 0.11 ± 4.2e-07 | 0.0629 ± 1.9e-07 | 0.0392 ± 1.6e-07 | 0.0274 ± 7.6e-08 | 0.0215 ± 3.5e-08 |
| Tournament: global halving | Triton | 0.114 ± 8e-07 | 0.0871 ± 4.9e-07 | 0.0743 ± 1.9e-07 | 0.0682 ± 1.1e-07 | 0.0652 ± 7.1e-08 |
| Tournament: blocks | PyTorch | 0.114 ± 1.2e-06 | 0.0871 ± 5.3e-07 | 0.0743 ± 3e-07 | 0.0682 ± 1.1e-07 | 0.0652 ± 5.9e-08 |
| Tournament: blocks, then global | Triton | 0.114 ± 6.8e-07 | 0.0871 ± 5e-07 | 0.0743 ± 1.4e-07 | 0.0682 ± 1.2e-07 | 0.0652 ± 7.4e-08 |
| Tournament: wide first round | PyTorch | 0.143 ± 1.1e-06 | 0.117 ± 4.6e-07 | 0.105 ± 2.6e-07 | 0.0992 ± 1.6e-07 | 0.0963 ± 8.9e-08 |
| Tournament: sampled cutoffs | Triton | 0.114 ± 5.9e-06 | 0.0871 ± 1.4e-05 | 0.0743 ± 1.5e-05 | 0.0681 ± 1.5e-05 | 0.0652 ± 1.4e-05 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.582 ± 5.8e-06 | 0.571 ± 1.4e-05 | 0.566 ± 1.5e-05 | 0.564 ± 1.5e-05 | 0.563 ± 1.4e-05 |
| Tournament: channel groups | PyTorch | 0.143 ± 8.2e-07 | 0.117 ± 5.8e-07 | 0.105 ± 1.6e-07 | 0.0992 ± 9.1e-08 | 0.0963 ± 4.1e-08 |
| Tournament: mean-value losers | PyTorch | 0.143 ± 1.1e-06 | 0.117 ± 4.8e-07 | 0.105 ± 2.5e-07 | 0.0992 ± 1.6e-07 | 0.0963 ± 8.8e-08 |
| Tournament: shuffled blocks | Triton | 0.114 ± 2.1e-06 | 0.0872 ± 5.8e-07 | 0.0743 ± 2.8e-07 | 0.0682 ± 1.4e-07 | 0.0652 ± 5.3e-08 |
| Tournament: shuffled blocks, growing budget | Triton | 0.0747 ± 2.4e-06 | 0.0744 ± 1e-06 | 0.0743 ± 2.8e-07 | 0.0743 ± 7.7e-08 | 0.0742 ± 1.8e-08 |
| Tournament: original | PyTorch | 0.582 ± 0 | 0.571 ± 0 | 0.566 ± 0 | 0.564 ± 0 | 0.563 ± 0 |
| Bandit: exact bound | PyTorch | 0.167 ± 0.0022 | 0.123 ± 0.0015 | 0.0922 ± 0.0008 | 0.0707 ± 0.00065 | 0.0579 ± 0.00089 |
| Bandit: Gaussian bound | PyTorch | 0.11 ± 0.00062 | 0.0737 ± 0.00036 | 0.0539 ± 0.00016 | 0.0433 ± 9.2e-05 | 0.0377 ± 0.0001 |

#### Local

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 0.531 ± 0 | 0.516 ± 0 | 0.508 ± 0 | 0.504 ± 0 | 0.502 ± 0 |
| SparQ | Triton | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Loki | PyTorch | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Block top-k | PyTorch | 0.188 ± 2.7e-07 | 0.156 ± 1.6e-07 | 0.141 ± 5e-08 | 0.133 ± 1.8e-08 | 0.129 ± 4.9e-09 |
| Double indexing: mean | PyTorch | 0.103 ± 3.8e-07 | 0.0551 ± 1.2e-07 | 0.0314 ± 8.8e-08 | 0.0196 ± 1.7e-08 | 0.0137 ± 3e-09 |
| Double indexing: bound | PyTorch | 0.111 ± 3.7e-07 | 0.0629 ± 1.3e-07 | 0.0392 ± 8e-08 | 0.0274 ± 2.4e-08 | 0.0215 ± 8.2e-09 |
| Tournament: global halving | Triton | 0.114 ± 3.8e-09 | 0.0871 ± 2.1e-08 | 0.0743 ± 3.9e-08 | 0.0682 ± 1.1e-08 | 0.0652 ± 7.1e-09 |
| Tournament: blocks | PyTorch | 0.114 ± 1.4e-07 | 0.0871 ± 4.6e-07 | 0.0743 ± 2.9e-07 | 0.0682 ± 1.1e-07 | 0.0652 ± 1.3e-07 |
| Tournament: blocks, then global | Triton | 0.114 ± 3.3e-09 | 0.0871 ± 1.8e-08 | 0.0743 ± 4e-08 | 0.0682 ± 1e-08 | 0.0652 ± 8e-09 |
| Tournament: wide first round | PyTorch | 0.143 ± 1.4e-08 | 0.117 ± 1.3e-08 | 0.105 ± 4.1e-08 | 0.0992 ± 1.6e-08 | 0.0963 ± 6.8e-09 |
| Tournament: sampled cutoffs | Triton | 0.114 ± 2.4e-05 | 0.0871 ± 1.4e-05 | 0.0743 ± 1.3e-05 | 0.0681 ± 1.5e-05 | 0.0652 ± 7.4e-06 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.582 ± 2.4e-05 | 0.571 ± 1.4e-05 | 0.566 ± 1.3e-05 | 0.564 ± 1.5e-05 | 0.563 ± 7.4e-06 |
| Tournament: channel groups | PyTorch | 0.143 ± 1.3e-08 | 0.117 ± 4.1e-08 | 0.105 ± 3.5e-08 | 0.0992 ± 1.3e-08 | 0.0963 ± 7.3e-09 |
| Tournament: mean-value losers | PyTorch | 0.143 ± 1.4e-08 | 0.117 ± 1.3e-08 | 0.105 ± 4.1e-08 | 0.0992 ± 1.6e-08 | 0.0963 ± 6.8e-09 |
| Tournament: shuffled blocks | Triton | 0.114 ± 1.1e-08 | 0.0871 ± 4.2e-10 | 0.0743 ± 3.6e-10 | 0.0682 ± 1.8e-09 | 0.0652 ± 6.1e-10 |
| Tournament: shuffled blocks, growing budget | Triton | 0.0746 ± 8.3e-10 | 0.0744 ± 4.7e-09 | 0.0743 ± 3.6e-10 | 0.0743 ± 1.7e-09 | 0.0742 ± 8.7e-10 |
| Tournament: original | PyTorch | 0.582 ± 0 | 0.571 ± 0 | 0.566 ± 0 | 0.564 ± 0 | 0.563 ± 0 |
| Bandit: exact bound | PyTorch | 0.145 ± 0.002 | 0.104 ± 0.0014 | 0.0767 ± 0.00076 | 0.0594 ± 0.00057 | 0.0501 ± 0.00083 |
| Bandit: Gaussian bound | PyTorch | 0.103 ± 0.00039 | 0.0678 ± 0.0002 | 0.0497 ± 7.6e-05 | 0.0406 ± 3.5e-05 | 0.036 ± 3.2e-05 |

#### Many needles

| Name | Kernel | 8K | 16K | 32K | 64K | 128K |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 1 ± 0 |
| Exact top-k | PyTorch | 0.531 ± 0 | 0.516 ± 0 | 0.508 ± 0 | 0.504 ± 0 | 0.502 ± 0 |
| SparQ | Triton | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Loki | PyTorch | 0.188 ± 0 | 0.156 ± 0 | 0.141 ± 0 | 0.133 ± 0 | 0.129 ± 0 |
| Block top-k | PyTorch | 0.188 ± 2.3e-07 | 0.156 ± 1.1e-07 | 0.141 ± 4.2e-08 | 0.133 ± 9.9e-09 | 0.129 ± 4.2e-09 |
| Double indexing: mean | PyTorch | 0.103 ± 3.8e-07 | 0.0551 ± 2e-07 | 0.0314 ± 5.6e-08 | 0.0196 ± 2.4e-08 | 0.0137 ± 1.6e-08 |
| Double indexing: bound | PyTorch | 0.111 ± 4.8e-07 | 0.0629 ± 1.7e-07 | 0.0392 ± 4.1e-08 | 0.0274 ± 1.5e-08 | 0.0215 ± 2.9e-08 |
| Tournament: global halving | Triton | 0.116 ± 2.4e-05 | 0.0884 ± 9.7e-06 | 0.0749 ± 5.7e-06 | 0.0686 ± 3.2e-06 | 0.0654 ± 5.6e-06 |
| Tournament: blocks | PyTorch | 0.116 ± 8.4e-06 | 0.0884 ± 4.5e-06 | 0.075 ± 1.9e-06 | 0.0685 ± 1.4e-06 | 0.0654 ± 1.2e-06 |
| Tournament: blocks, then global | Triton | 0.116 ± 1.9e-05 | 0.0884 ± 9.4e-06 | 0.0749 ± 6.2e-06 | 0.0686 ± 4.6e-06 | 0.0654 ± 5.4e-06 |
| Tournament: wide first round | PyTorch | 0.145 ± 9.3e-06 | 0.119 ± 9.8e-06 | 0.106 ± 6.6e-06 | 0.0996 ± 3.6e-06 | 0.0965 ± 5.1e-06 |
| Tournament: sampled cutoffs | Triton | 0.116 ± 2.5e-05 | 0.0884 ± 2.1e-05 | 0.0749 ± 1.4e-05 | 0.0686 ± 2e-05 | 0.0654 ± 1.7e-05 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.583 ± 1.5e-05 | 0.572 ± 1.9e-05 | 0.567 ± 1.3e-05 | 0.564 ± 2e-05 | 0.563 ± 1.6e-05 |
| Tournament: channel groups | PyTorch | 0.145 ± 1.6e-05 | 0.119 ± 1.1e-05 | 0.106 ± 4.7e-06 | 0.0996 ± 3.8e-06 | 0.0965 ± 5.1e-06 |
| Tournament: mean-value losers | PyTorch | 0.145 ± 1.1e-05 | 0.119 ± 6.9e-06 | 0.106 ± 5.2e-06 | 0.0996 ± 4e-06 | 0.0965 ± 5.1e-06 |
| Tournament: shuffled blocks | Triton | 0.116 ± 1.1e-05 | 0.0884 ± 7.1e-06 | 0.075 ± 2.8e-06 | 0.0685 ± 1.5e-06 | 0.0654 ± 1.4e-06 |
| Tournament: shuffled blocks, growing budget | Triton | 0.0767 ± 1.1e-05 | 0.0757 ± 5e-06 | 0.075 ± 2.3e-06 | 0.0746 ± 2.1e-06 | 0.0744 ± 3.1e-06 |
| Tournament: original | PyTorch | 0.582 ± 0 | 0.571 ± 0 | 0.566 ± 0 | 0.564 ± 0 | 0.563 ± 0 |
| Bandit: exact bound | PyTorch | 0.187 ± 0.0011 | 0.128 ± 0.0012 | 0.0914 ± 0.00064 | 0.0683 ± 0.00061 | 0.0556 ± 0.00088 |
| Bandit: Gaussian bound | PyTorch | 0.125 ± 0.00021 | 0.0792 ± 0.00015 | 0.0553 ± 6.8e-05 | 0.0435 ± 4.6e-05 | 0.0375 ± 3.2e-05 |

### Budget: latency and output error per budget

budget.png: Every version at each budget, the number of keys attended exactly: latency at 32K keys, LLM-like; output error at 32K keys, LLM-like. A budget runs where the context is at least 16 times it. Whiskers as in the per-benchmark figures. docs/methods.md explains each row's version and its settings; n/a: a version this benchmark does not run.

#### Latency

| Name | Kernel | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 226 (226–226) | 226 (226–226) | 226 (226–226) | 226 (226–226) | 226 (226–226) |
| Exact top-k | PyTorch | 200 (200–201) | 213 (212–213) | 225 (225–226) | 254 (254–255) | 318 (318–319) |
| SparQ | Triton | 107 (104–110) | 115 (111–125) | 114 (112–115) | 117 (115–118) | 134 (119–140) |
| Loki | PyTorch | 129 (127–130) | 142 (140–142) | 154 (152–155) | 181 (180–181) | 235 (235–236) |
| DeepSeek-V4 CSA | transformers | 812 (809–815) | 821 (819–823) | 825 (820–827) | 823 (821–829) | 826 (823–829) |
| DeepSeek-V4 CSA, V4-Flash size | transformers | 1151 (1148–1153) | 1159 (1157–1293) | 1165 (1161–1167) | 1163 (1161–1252) | 1159 (1158–1470) |
| Block top-k | PyTorch | 104 (104–105) | 112 (111–115) | 127 (126–128) | 164 (162–169) | 221 (213–223) |
| Double indexing: mean | PyTorch | 97 (95–100) | 103 (103–104) | 120 (120–133) | 161 (160–162) | 225 (222–238) |
| Double indexing: bound | PyTorch | 114 (112–149) | 135 (125–159) | 160 (149–182) | 195 (166–205) | 231 (229–236) |
| Tournament: global halving | Triton | 149 (148–150) | 140 (140–141) | 139 (138–140) | 157 (157–158) | 420 (419–420) |
| Tournament: blocks | PyTorch | 800 (798–804) | 750 (749–753) | 707 (705–708) | 678 (676–689) | 667 (666–668) |
| Tournament: blocks, then global | Triton | 89 (89–89) | 92 (91–93) | 101 (100–101) | 127 (127–128) | 392 (391–392) |
| Tournament: wide first round | PyTorch | 1297 (1297–1297) | 1249 (1249–1251) | 1197 (1196–1202) | 1160 (1160–1164) | 1139 (1139–1145) |
| Tournament: sampled cutoffs | Triton | 170 (169–170) | 156 (155–156) | 146 (146–147) | 152 (151–152) | 178 (177–178) |
| Tournament: sampled cutoffs, approximate losers | Triton | 348 (347–349) | 333 (332–333) | 333 (332–334) | 371 (371–371) | 515 (514–515) |
| Tournament: channel groups | PyTorch | 1304 (1304–1305) | 1258 (1258–1259) | 1210 (1207–1213) | 1170 (1170–1174) | 1150 (1150–1151) |
| Tournament: mean-value losers | PyTorch | 1390 (1389–1391) | 1336 (1335–1338) | 1279 (1278–1280) | 1232 (1231–1233) | 1195 (1184–1200) |
| Tournament: shuffled blocks | Triton | 94 (93–94) | 88 (88–89) | 84 (83–84) | 80 (79–81) | 80 (79–82) |
| Tournament: shuffled blocks, growing budget | Triton | 94 (94–95) | 89 (88–90) | 84 (83–85) | 81 (80–81) | 80 (79–80) |
| Tournament: original | PyTorch | 1489 (1487–1491) | 1437 (1436–1438) | 1389 (1388–1390) | 1347 (1347–1350) | 1310 (1309–1313) |
| Bandit: exact bound | PyTorch | 2763 (2754–2801) | 2875 (2871–2879) | 2888 (2885–2900) | 2921 (2919–2961) | 2993 (2993–2998) |
| Bandit: Gaussian bound | PyTorch | 2426 (2417–2546) | 2543 (2539–2560) | 2560 (2556–2586) | 2587 (2586–2598) | 2659 (2658–2665) |

#### Output error

| Name | Kernel | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---:|---:|---:|---:|---:|
| Dense | PyTorch | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| Exact top-k | PyTorch | 0.128 ± 0.003 | 0.101 ± 0.0027 | 0.079 ± 0.0024 | 0.0613 ± 0.002 | 0.0452 ± 0.0015 |
| SparQ | Triton | 0.153 ± 0.0033 | 0.119 ± 0.003 | 0.0913 ± 0.0027 | 0.0698 ± 0.0022 | 0.0512 ± 0.0017 |
| Loki | PyTorch | 0.146 ± 0.0035 | 0.112 ± 0.0031 | 0.087 ± 0.0027 | 0.0682 ± 0.0023 | 0.0512 ± 0.0018 |
| DeepSeek-V4 CSA | transformers | not run | not run | not run | not run | not run |
| DeepSeek-V4 CSA, V4-Flash size | transformers | not run | not run | not run | not run | not run |
| Block top-k | PyTorch | 0.757 ± 0.0076 | 0.526 ± 0.0061 | 0.364 ± 0.0059 | 0.255 ± 0.0039 | 0.177 ± 0.004 |
| Double indexing: mean | PyTorch | 1.59 ± 0.016 | 1.49 ± 0.013 | 1.34 ± 0.0093 | 1.11 ± 0.007 | 0.766 ± 0.0045 |
| Double indexing: bound | PyTorch | 1.7 ± 0.015 | 1.52 ± 0.015 | 1.28 ± 0.013 | 0.957 ± 0.013 | 0.582 ± 0.0065 |
| Tournament: global halving | Triton | 0.206 ± 0.0054 | 0.175 ± 0.005 | 0.145 ± 0.0047 | 0.117 ± 0.004 | 0.0882 ± 0.0034 |
| Tournament: blocks | PyTorch | 0.215 ± 0.0054 | 0.177 ± 0.005 | 0.146 ± 0.0046 | 0.117 ± 0.004 | 0.0884 ± 0.0034 |
| Tournament: blocks, then global | Triton | 0.206 ± 0.0054 | 0.176 ± 0.005 | 0.145 ± 0.0047 | 0.117 ± 0.004 | 0.0883 ± 0.0034 |
| Tournament: wide first round | PyTorch | 0.204 ± 0.0053 | 0.173 ± 0.005 | 0.142 ± 0.0045 | 0.113 ± 0.004 | 0.0841 ± 0.0032 |
| Tournament: sampled cutoffs | Triton | 0.206 ± 0.0054 | 0.175 ± 0.0051 | 0.145 ± 0.0047 | 0.117 ± 0.004 | 0.0882 ± 0.0034 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.192 ± 0.005 | 0.168 ± 0.0047 | 0.141 ± 0.0044 | 0.114 ± 0.0039 | 0.0868 ± 0.0033 |
| Tournament: channel groups | PyTorch | 0.223 ± 0.0056 | 0.196 ± 0.0054 | 0.168 ± 0.0052 | 0.141 ± 0.0045 | 0.11 ± 0.0037 |
| Tournament: mean-value losers | PyTorch | 0.188 ± 0.0049 | 0.165 ± 0.0047 | 0.137 ± 0.0042 | 0.111 ± 0.0038 | 0.0827 ± 0.0031 |
| Tournament: shuffled blocks | Triton | 0.209 ± 0.0055 | 0.177 ± 0.0051 | 0.147 ± 0.0047 | 0.117 ± 0.0041 | 0.0883 ± 0.0034 |
| Tournament: shuffled blocks, growing budget | Triton | 0.209 ± 0.0055 | 0.178 ± 0.0051 | 0.147 ± 0.0047 | 0.117 ± 0.0041 | 0.0884 ± 0.0034 |
| Tournament: original | PyTorch | 0.618 ± 0.01 | 0.58 ± 0.011 | 0.54 ± 0.011 | 0.502 ± 0.011 | 0.462 ± 0.011 |
| Bandit: exact bound | PyTorch | 0.2 ± 0.0052 | 0.166 ± 0.0049 | 0.132 ± 0.0044 | 0.102 ± 0.0038 | 0.0753 ± 0.003 |
| Bandit: Gaussian bound | PyTorch | 0.264 ± 0.0076 | 0.214 ± 0.0064 | 0.17 ± 0.0056 | 0.129 ± 0.0045 | 0.0911 ± 0.0034 |
