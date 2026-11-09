# Benchmarks: suite smoke

### Latency, µs per decode step, budget 512: median (min–max)

latency.png: Decode-step latency on NVIDIA GeForce RTX 4070 Ti SUPER: median of 3 repeats of CUDA-graph replays; whiskers min to max. Batch 1, 8 KV heads × 2 query heads, head dim 128, bf16 K and V; every sparse version gives exact attention to about 512 keys (exceptions: docs/methods.md). Panels: iid: iid Gaussian q, k, v. docs/methods.md explains each row's version and its settings.

#### iid

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 227 (227–227) |
| Exact top-k | PyTorch | 225 (225–226) |
| SparQ | Triton | 139 (116–150) |
| Loki | PyTorch | 154 (154–154) |
| Block top-k | PyTorch | 125 (125–126) |
| Double indexing: mean | PyTorch | 117 (116–118) |
| Double indexing: bound | PyTorch | 125 (125–133) |
| Tournament: global halving | Triton | 139 (139–139) |
| Tournament: blocks | PyTorch | 701 (701–704) |
| Tournament: blocks, then global | Triton | 100 (100–101) |
| Tournament: wide first round | PyTorch | 1200 (1199–1200) |
| Tournament: sampled cutoffs | Triton | 146 (145–146) |
| Tournament: sampled cutoffs, approximate losers | Triton | 332 (332–333) |
| Tournament: channel groups | PyTorch | 1215 (1215–1216) |
| Tournament: mean-value losers | PyTorch | 1285 (1285–1285) |
| Tournament: shuffled blocks | Triton | 84 (84–84) |
| Tournament: original | PyTorch | 1395 (1395–1397) |
| Bandit: exact bound | PyTorch | 2882 (2882–2886) |
| Bandit: Gaussian bound | PyTorch | 2572 (2563–2592) |

### fidelity: output error relative to dense, budget 512, mean ± SE

fidelity.png: Output error against dense attention, |o − o_dense| / |o_dense| per head: mean over 32 decode queries, ± SE over 1 seeds. 8 KV heads × 2 query heads, head dim 128; every sparse version gives exact attention to about 512 keys and runs as its float32 PyTorch emulation, which selects the same keys as its kernel (tests/test_registry.py). On iid data attention is near uniform, so no 512 keys come close to dense. Panels: iid: iid Gaussian q, k, v; Outlier channels: 4 outlier channels in q and k; LLM-like: outlier channels, 4 sink keys and 4 needle keys per query. docs/methods.md explains each row's version and its settings.

#### iid

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 0 ± 0 |
| Exact top-k | PyTorch | 6.14 ± 0 |
| SparQ | Triton | 7.19 ± 0 |
| Loki | PyTorch | 7.49 ± 0 |
| Block top-k | PyTorch | 7.62 ± 0 |
| Double indexing: mean | PyTorch | 7.59 ± 0 |
| Double indexing: bound | PyTorch | 7.6 ± 0 |
| Tournament: global halving | Triton | 6.79 ± 0 |
| Tournament: blocks | PyTorch | 6.82 ± 0 |
| Tournament: blocks, then global | Triton | 6.79 ± 0 |
| Tournament: wide first round | PyTorch | 6.67 ± 0 |
| Tournament: sampled cutoffs | Triton | 6.79 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 3.79 ± 0 |
| Tournament: channel groups | PyTorch | 7.17 ± 0 |
| Tournament: mean-value losers | PyTorch | 3.8 ± 0 |
| Tournament: shuffled blocks | Triton | 6.81 ± 0 |
| Tournament: original | PyTorch | 3.24 ± 0 |
| Bandit: exact bound | PyTorch | 6.2 ± 0 |
| Bandit: Gaussian bound | PyTorch | 6.19 ± 0 |

#### Outlier channels

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 0 ± 0 |
| Exact top-k | PyTorch | 2.96 ± 0 |
| SparQ | Triton | 2.9 ± 0 |
| Loki | PyTorch | 3.41 ± 0 |
| Block top-k | PyTorch | 5.48 ± 0 |
| Double indexing: mean | PyTorch | 5.56 ± 0 |
| Double indexing: bound | PyTorch | 6.27 ± 0 |
| Tournament: global halving | Triton | 4.57 ± 0 |
| Tournament: blocks | PyTorch | 4.52 ± 0 |
| Tournament: blocks, then global | Triton | 4.57 ± 0 |
| Tournament: wide first round | PyTorch | 4.52 ± 0 |
| Tournament: sampled cutoffs | Triton | 4.57 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 3.45 ± 0 |
| Tournament: channel groups | PyTorch | 4.54 ± 0 |
| Tournament: mean-value losers | PyTorch | 3.39 ± 0 |
| Tournament: shuffled blocks | Triton | 4.4 ± 0 |
| Tournament: original | PyTorch | 4.38 ± 0 |
| Bandit: exact bound | PyTorch | 4.41 ± 0 |
| Bandit: Gaussian bound | PyTorch | 4.38 ± 0 |

#### LLM-like

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 0 ± 0 |
| Exact top-k | PyTorch | 0.0792 ± 0 |
| SparQ | Triton | 0.0869 ± 0 |
| Loki | PyTorch | 0.0904 ± 0 |
| Block top-k | PyTorch | 0.286 ± 0 |
| Double indexing: mean | PyTorch | 1.27 ± 0 |
| Double indexing: bound | PyTorch | 0.957 ± 0 |
| Tournament: global halving | Triton | 0.144 ± 0 |
| Tournament: blocks | PyTorch | 0.144 ± 0 |
| Tournament: blocks, then global | Triton | 0.144 ± 0 |
| Tournament: wide first round | PyTorch | 0.142 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.144 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.138 ± 0 |
| Tournament: channel groups | PyTorch | 0.151 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.136 ± 0 |
| Tournament: shuffled blocks | Triton | 0.144 ± 0 |
| Tournament: original | PyTorch | 0.486 ± 0 |
| Bandit: exact bound | PyTorch | 0.136 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.156 ± 0 |

### fidelity-harder: output error relative to dense, budget 512, mean ± SE

fidelity-harder.png: Output error against dense attention, |o − o_dense| / |o_dense| per head: mean over 32 decode queries, ± SE over 1 seeds. 8 KV heads × 2 query heads, head dim 128; every sparse version gives exact attention to about 512 keys and runs as its float32 PyTorch emulation, which selects the same keys as its kernel (tests/test_registry.py). On Flat data attention is near uniform, so no 512 keys come close to dense. Panels: Flat: LLM-like with q × 0.5 (softer attention); Sharp: LLM-like with q × 2 (sharper attention); Local: LLM-like, each key averaging the noise of 16 neighbouring positions; Many needles: LLM-like with 64 needle keys per query. docs/methods.md explains each row's version and its settings.

#### Flat

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 0 ± 0 |
| Exact top-k | PyTorch | 4.48 ± 0 |
| SparQ | Triton | 4.67 ± 0 |
| Loki | PyTorch | 4.66 ± 0 |
| Block top-k | PyTorch | 6.87 ± 0 |
| Double indexing: mean | PyTorch | 7.37 ± 0 |
| Double indexing: bound | PyTorch | 7.44 ± 0 |
| Tournament: global halving | Triton | 5.17 ± 0 |
| Tournament: blocks | PyTorch | 5.19 ± 0 |
| Tournament: blocks, then global | Triton | 5.17 ± 0 |
| Tournament: wide first round | PyTorch | 5.11 ± 0 |
| Tournament: sampled cutoffs | Triton | 5.17 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 2.26 ± 0 |
| Tournament: channel groups | PyTorch | 5.44 ± 0 |
| Tournament: mean-value losers | PyTorch | 2.23 ± 0 |
| Tournament: shuffled blocks | Triton | 5.19 ± 0 |
| Tournament: original | PyTorch | 3.56 ± 0 |
| Bandit: exact bound | PyTorch | 4.93 ± 0 |
| Bandit: Gaussian bound | PyTorch | 5.25 ± 0 |

#### Sharp

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 0 ± 0 |
| Exact top-k | PyTorch | 0.00016 ± 0 |
| SparQ | Triton | 0.000216 ± 0 |
| Loki | PyTorch | 0.000214 ± 0 |
| Block top-k | PyTorch | 0.152 ± 0 |
| Double indexing: mean | PyTorch | 1.2 ± 0 |
| Double indexing: bound | PyTorch | 0.875 ± 0 |
| Tournament: global halving | Triton | 0.0043 ± 0 |
| Tournament: blocks | PyTorch | 0.00431 ± 0 |
| Tournament: blocks, then global | Triton | 0.0043 ± 0 |
| Tournament: wide first round | PyTorch | 0.00394 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.0043 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.0043 ± 0 |
| Tournament: channel groups | PyTorch | 0.00616 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.00394 ± 0 |
| Tournament: shuffled blocks | Triton | 0.00433 ± 0 |
| Tournament: original | PyTorch | 0.315 ± 0 |
| Bandit: exact bound | PyTorch | 0.00383 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.0115 ± 0 |

#### Local

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 0 ± 0 |
| Exact top-k | PyTorch | 0.0796 ± 0 |
| SparQ | Triton | 0.0878 ± 0 |
| Loki | PyTorch | 0.094 ± 0 |
| Block top-k | PyTorch | 1.12 ± 0 |
| Double indexing: mean | PyTorch | 1.41 ± 0 |
| Double indexing: bound | PyTorch | 0.721 ± 0 |
| Tournament: global halving | Triton | 0.146 ± 0 |
| Tournament: blocks | PyTorch | 0.145 ± 0 |
| Tournament: blocks, then global | Triton | 0.146 ± 0 |
| Tournament: wide first round | PyTorch | 0.143 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.146 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.14 ± 0 |
| Tournament: channel groups | PyTorch | 0.152 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.138 ± 0 |
| Tournament: shuffled blocks | Triton | 0.146 ± 0 |
| Tournament: original | PyTorch | 0.45 ± 0 |
| Bandit: exact bound | PyTorch | 0.139 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.159 ± 0 |

#### Many needles

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 0 ± 0 |
| Exact top-k | PyTorch | 0.0196 ± 0 |
| SparQ | Triton | 0.0238 ± 0 |
| Loki | PyTorch | 0.02 ± 0 |
| Block top-k | PyTorch | 0.622 ± 0 |
| Double indexing: mean | PyTorch | 1.89 ± 0 |
| Double indexing: bound | PyTorch | 2.22 ± 0 |
| Tournament: global halving | Triton | 0.038 ± 0 |
| Tournament: blocks | PyTorch | 0.0379 ± 0 |
| Tournament: blocks, then global | Triton | 0.038 ± 0 |
| Tournament: wide first round | PyTorch | 0.0375 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.0379 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.0368 ± 0 |
| Tournament: channel groups | PyTorch | 0.0417 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.0361 ± 0 |
| Tournament: shuffled blocks | Triton | 0.0383 ± 0 |
| Tournament: original | PyTorch | 0.437 ± 0 |
| Bandit: exact bound | PyTorch | 0.0354 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.0394 ± 0 |

### Fidelity: Dense attention weight on the picked keys, budget 512, mean ± SE

#### iid

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.0878 ± 0 |
| SparQ | Triton | 0.058 ± 0 |
| Loki | PyTorch | 0.0386 ± 0 |
| Block top-k | PyTorch | 0.0362 ± 0 |
| Double indexing: mean | PyTorch | 0.0274 ± 0 |
| Double indexing: bound | PyTorch | 0.0244 ± 0 |
| Tournament: global halving | Triton | 0.0666 ± 0 |
| Tournament: blocks | PyTorch | 0.0662 ± 0 |
| Tournament: blocks, then global | Triton | 0.0666 ± 0 |
| Tournament: wide first round | PyTorch | 0.0719 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.0666 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.0666 ± 0 |
| Tournament: channel groups | PyTorch | 0.0554 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.0719 ± 0 |
| Tournament: shuffled blocks | Triton | 0.0662 ± 0 |
| Tournament: original | PyTorch | 0.0359 ± 0 |
| Bandit: exact bound | PyTorch | 0.0877 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.0874 ± 0 |

#### Outlier channels

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.345 ± 0 |
| SparQ | Triton | 0.318 ± 0 |
| Loki | PyTorch | 0.32 ± 0 |
| Block top-k | PyTorch | 0.0928 ± 0 |
| Double indexing: mean | PyTorch | 0.0448 ± 0 |
| Double indexing: bound | PyTorch | 0.0705 ± 0 |
| Tournament: global halving | Triton | 0.319 ± 0 |
| Tournament: blocks | PyTorch | 0.316 ± 0 |
| Tournament: blocks, then global | Triton | 0.319 ± 0 |
| Tournament: wide first round | PyTorch | 0.321 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.319 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.319 ± 0 |
| Tournament: channel groups | PyTorch | 0.301 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.321 ± 0 |
| Tournament: shuffled blocks | Triton | 0.317 ± 0 |
| Tournament: original | PyTorch | 0.0722 ± 0 |
| Bandit: exact bound | PyTorch | 0.327 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.326 ± 0 |

#### LLM-like

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.944 ± 0 |
| SparQ | Triton | 0.94 ± 0 |
| Loki | PyTorch | 0.94 ± 0 |
| Block top-k | PyTorch | 0.869 ± 0 |
| Double indexing: mean | PyTorch | 0.39 ± 0 |
| Double indexing: bound | PyTorch | 0.532 ± 0 |
| Tournament: global halving | Triton | 0.931 ± 0 |
| Tournament: blocks | PyTorch | 0.931 ± 0 |
| Tournament: blocks, then global | Triton | 0.931 ± 0 |
| Tournament: wide first round | PyTorch | 0.932 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.931 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.931 ± 0 |
| Tournament: channel groups | PyTorch | 0.927 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.932 ± 0 |
| Tournament: shuffled blocks | Triton | 0.931 ± 0 |
| Tournament: original | PyTorch | 0.76 ± 0 |
| Bandit: exact bound | PyTorch | 0.934 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.927 ± 0 |

#### Flat

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.266 ± 0 |
| SparQ | Triton | 0.258 ± 0 |
| Loki | PyTorch | 0.262 ± 0 |
| Block top-k | PyTorch | 0.187 ± 0 |
| Double indexing: mean | PyTorch | 0.0994 ± 0 |
| Double indexing: bound | PyTorch | 0.123 ± 0 |
| Tournament: global halving | Triton | 0.26 ± 0 |
| Tournament: blocks | PyTorch | 0.26 ± 0 |
| Tournament: blocks, then global | Triton | 0.26 ± 0 |
| Tournament: wide first round | PyTorch | 0.261 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.26 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.26 ± 0 |
| Tournament: channel groups | PyTorch | 0.253 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.261 ± 0 |
| Tournament: shuffled blocks | Triton | 0.26 ± 0 |
| Tournament: original | PyTorch | 0.157 ± 0 |
| Bandit: exact bound | PyTorch | 0.264 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.261 ± 0 |

#### Sharp

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 |
| SparQ | Triton | 1 ± 0 |
| Loki | PyTorch | 1 ± 0 |
| Block top-k | PyTorch | 0.934 ± 0 |
| Double indexing: mean | PyTorch | 0.39 ± 0 |
| Double indexing: bound | PyTorch | 0.559 ± 0 |
| Tournament: global halving | Triton | 0.998 ± 0 |
| Tournament: blocks | PyTorch | 0.998 ± 0 |
| Tournament: blocks, then global | Triton | 0.998 ± 0 |
| Tournament: wide first round | PyTorch | 0.998 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.998 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.998 ± 0 |
| Tournament: channel groups | PyTorch | 0.997 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.998 ± 0 |
| Tournament: shuffled blocks | Triton | 0.998 ± 0 |
| Tournament: original | PyTorch | 0.847 ± 0 |
| Bandit: exact bound | PyTorch | 0.998 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.994 ± 0 |

#### Local

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.944 ± 0 |
| SparQ | Triton | 0.94 ± 0 |
| Loki | PyTorch | 0.94 ± 0 |
| Block top-k | PyTorch | 0.423 ± 0 |
| Double indexing: mean | PyTorch | 0.0873 ± 0 |
| Double indexing: bound | PyTorch | 0.687 ± 0 |
| Tournament: global halving | Triton | 0.931 ± 0 |
| Tournament: blocks | PyTorch | 0.93 ± 0 |
| Tournament: blocks, then global | Triton | 0.931 ± 0 |
| Tournament: wide first round | PyTorch | 0.932 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.932 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.932 ± 0 |
| Tournament: channel groups | PyTorch | 0.927 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.932 ± 0 |
| Tournament: shuffled blocks | Triton | 0.931 ± 0 |
| Tournament: original | PyTorch | 0.775 ± 0 |
| Bandit: exact bound | PyTorch | 0.934 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.927 ± 0 |

#### Many needles

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.983 ± 0 |
| SparQ | Triton | 0.98 ± 0 |
| Loki | PyTorch | 0.982 ± 0 |
| Block top-k | PyTorch | 0.715 ± 0 |
| Double indexing: mean | PyTorch | 0.236 ± 0 |
| Double indexing: bound | PyTorch | 0.198 ± 0 |
| Tournament: global halving | Triton | 0.973 ± 0 |
| Tournament: blocks | PyTorch | 0.973 ± 0 |
| Tournament: blocks, then global | Triton | 0.973 ± 0 |
| Tournament: wide first round | PyTorch | 0.973 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.973 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.973 ± 0 |
| Tournament: channel groups | PyTorch | 0.97 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.973 ± 0 |
| Tournament: shuffled blocks | Triton | 0.973 ± 0 |
| Tournament: original | PyTorch | 0.827 ± 0 |
| Bandit: exact bound | PyTorch | 0.974 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.972 ± 0 |

### Fidelity: Recall of the dense top-128 keys, budget 512, mean ± SE

#### iid

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 |
| SparQ | Triton | 0.519 ± 0 |
| Loki | PyTorch | 0.21 ± 0 |
| Block top-k | PyTorch | 0.185 ± 0 |
| Double indexing: mean | PyTorch | 0.0806 ± 0 |
| Double indexing: bound | PyTorch | 0.0589 ± 0 |
| Tournament: global halving | Triton | 0.656 ± 0 |
| Tournament: blocks | PyTorch | 0.646 ± 0 |
| Tournament: blocks, then global | Triton | 0.655 ± 0 |
| Tournament: wide first round | PyTorch | 0.766 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.656 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.656 ± 0 |
| Tournament: channel groups | PyTorch | 0.454 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.766 ± 0 |
| Tournament: shuffled blocks | Triton | 0.648 ± 0 |
| Tournament: original | PyTorch | 0.175 ± 0 |
| Bandit: exact bound | PyTorch | 1 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.995 ± 0 |

#### Outlier channels

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 |
| SparQ | Triton | 0.956 ± 0 |
| Loki | PyTorch | 0.95 ± 0 |
| Block top-k | PyTorch | 0.19 ± 0 |
| Double indexing: mean | PyTorch | 0.0777 ± 0 |
| Double indexing: bound | PyTorch | 0.115 ± 0 |
| Tournament: global halving | Triton | 0.944 ± 0 |
| Tournament: blocks | PyTorch | 0.938 ± 0 |
| Tournament: blocks, then global | Triton | 0.944 ± 0 |
| Tournament: wide first round | PyTorch | 0.949 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.944 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.944 ± 0 |
| Tournament: channel groups | PyTorch | 0.851 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.949 ± 0 |
| Tournament: shuffled blocks | Triton | 0.94 ± 0 |
| Tournament: original | PyTorch | 0.145 ± 0 |
| Bandit: exact bound | PyTorch | 0.96 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.957 ± 0 |

#### LLM-like

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 |
| SparQ | Triton | 0.931 ± 0 |
| Loki | PyTorch | 0.963 ± 0 |
| Block top-k | PyTorch | 0.261 ± 0 |
| Double indexing: mean | PyTorch | 0.123 ± 0 |
| Double indexing: bound | PyTorch | 0.172 ± 0 |
| Tournament: global halving | Triton | 0.778 ± 0 |
| Tournament: blocks | PyTorch | 0.774 ± 0 |
| Tournament: blocks, then global | Triton | 0.778 ± 0 |
| Tournament: wide first round | PyTorch | 0.797 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.778 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.778 ± 0 |
| Tournament: channel groups | PyTorch | 0.685 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.797 ± 0 |
| Tournament: shuffled blocks | Triton | 0.774 ± 0 |
| Tournament: original | PyTorch | 0.292 ± 0 |
| Bandit: exact bound | PyTorch | 0.836 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.749 ± 0 |

#### Flat

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 |
| SparQ | Triton | 0.951 ± 0 |
| Loki | PyTorch | 0.998 ± 0 |
| Block top-k | PyTorch | 0.321 ± 0 |
| Double indexing: mean | PyTorch | 0.135 ± 0 |
| Double indexing: bound | PyTorch | 0.192 ± 0 |
| Tournament: global halving | Triton | 0.969 ± 0 |
| Tournament: blocks | PyTorch | 0.964 ± 0 |
| Tournament: blocks, then global | Triton | 0.969 ± 0 |
| Tournament: wide first round | PyTorch | 0.983 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.969 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.969 ± 0 |
| Tournament: channel groups | PyTorch | 0.865 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.983 ± 0 |
| Tournament: shuffled blocks | Triton | 0.964 ± 0 |
| Tournament: original | PyTorch | 0.31 ± 0 |
| Bandit: exact bound | PyTorch | 0.997 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.931 ± 0 |

#### Sharp

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 |
| SparQ | Triton | 0.922 ± 0 |
| Loki | PyTorch | 0.916 ± 0 |
| Block top-k | PyTorch | 0.241 ± 0 |
| Double indexing: mean | PyTorch | 0.116 ± 0 |
| Double indexing: bound | PyTorch | 0.164 ± 0 |
| Tournament: global halving | Triton | 0.751 ± 0 |
| Tournament: blocks | PyTorch | 0.744 ± 0 |
| Tournament: blocks, then global | Triton | 0.751 ± 0 |
| Tournament: wide first round | PyTorch | 0.767 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.751 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.751 ± 0 |
| Tournament: channel groups | PyTorch | 0.656 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.767 ± 0 |
| Tournament: shuffled blocks | Triton | 0.745 ± 0 |
| Tournament: original | PyTorch | 0.28 ± 0 |
| Bandit: exact bound | PyTorch | 0.809 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.715 ± 0 |

#### Local

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 |
| SparQ | Triton | 0.931 ± 0 |
| Loki | PyTorch | 0.954 ± 0 |
| Block top-k | PyTorch | 0.388 ± 0 |
| Double indexing: mean | PyTorch | 0.184 ± 0 |
| Double indexing: bound | PyTorch | 0.234 ± 0 |
| Tournament: global halving | Triton | 0.78 ± 0 |
| Tournament: blocks | PyTorch | 0.759 ± 0 |
| Tournament: blocks, then global | Triton | 0.779 ± 0 |
| Tournament: wide first round | PyTorch | 0.797 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.78 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.78 ± 0 |
| Tournament: channel groups | PyTorch | 0.683 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.797 ± 0 |
| Tournament: shuffled blocks | Triton | 0.775 ± 0 |
| Tournament: original | PyTorch | 0.319 ± 0 |
| Bandit: exact bound | PyTorch | 0.837 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.748 ± 0 |

#### Many needles

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 1 ± 0 |
| SparQ | Triton | 0.975 ± 0 |
| Loki | PyTorch | 1 ± 0 |
| Block top-k | PyTorch | 0.45 ± 0 |
| Double indexing: mean | PyTorch | 0.176 ± 0 |
| Double indexing: bound | PyTorch | 0.146 ± 0 |
| Tournament: global halving | Triton | 0.885 ± 0 |
| Tournament: blocks | PyTorch | 0.878 ± 0 |
| Tournament: blocks, then global | Triton | 0.884 ± 0 |
| Tournament: wide first round | PyTorch | 0.885 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.884 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.885 ± 0 |
| Tournament: channel groups | PyTorch | 0.84 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.885 ± 0 |
| Tournament: shuffled blocks | Triton | 0.878 ± 0 |
| Tournament: original | PyTorch | 0.553 ± 0 |
| Bandit: exact bound | PyTorch | 0.907 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.877 ± 0 |

### Fidelity: Share of K and V elements read, budget 512, mean ± SE

#### iid

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.508 ± 0 |
| SparQ | Triton | 0.141 ± 0 |
| Loki | PyTorch | 0.141 ± 0 |
| Block top-k | PyTorch | 0.141 ± 0 |
| Double indexing: mean | PyTorch | 0.0313 ± 0 |
| Double indexing: bound | PyTorch | 0.039 ± 0 |
| Tournament: global halving | Triton | 0.0742 ± 0 |
| Tournament: blocks | PyTorch | 0.0742 ± 0 |
| Tournament: blocks, then global | Triton | 0.0742 ± 0 |
| Tournament: wide first round | PyTorch | 0.105 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.0742 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.566 ± 0 |
| Tournament: channel groups | PyTorch | 0.105 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.105 ± 0 |
| Tournament: shuffled blocks | Triton | 0.0742 ± 0 |
| Tournament: original | PyTorch | 0.566 ± 0 |
| Bandit: exact bound | PyTorch | 0.288 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.176 ± 0 |

#### Outlier channels

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.508 ± 0 |
| SparQ | Triton | 0.141 ± 0 |
| Loki | PyTorch | 0.141 ± 0 |
| Block top-k | PyTorch | 0.141 ± 0 |
| Double indexing: mean | PyTorch | 0.0313 ± 0 |
| Double indexing: bound | PyTorch | 0.039 ± 0 |
| Tournament: global halving | Triton | 0.0742 ± 0 |
| Tournament: blocks | PyTorch | 0.0742 ± 0 |
| Tournament: blocks, then global | Triton | 0.0742 ± 0 |
| Tournament: wide first round | PyTorch | 0.105 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.0742 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.566 ± 0 |
| Tournament: channel groups | PyTorch | 0.105 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.105 ± 0 |
| Tournament: shuffled blocks | Triton | 0.0742 ± 0 |
| Tournament: original | PyTorch | 0.566 ± 0 |
| Bandit: exact bound | PyTorch | 0.102 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.0565 ± 0 |

#### LLM-like

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.508 ± 0 |
| SparQ | Triton | 0.141 ± 0 |
| Loki | PyTorch | 0.141 ± 0 |
| Block top-k | PyTorch | 0.141 ± 0 |
| Double indexing: mean | PyTorch | 0.0313 ± 0 |
| Double indexing: bound | PyTorch | 0.0391 ± 0 |
| Tournament: global halving | Triton | 0.0742 ± 0 |
| Tournament: blocks | PyTorch | 0.0742 ± 0 |
| Tournament: blocks, then global | Triton | 0.0742 ± 0 |
| Tournament: wide first round | PyTorch | 0.105 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.0741 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.566 ± 0 |
| Tournament: channel groups | PyTorch | 0.105 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.105 ± 0 |
| Tournament: shuffled blocks | Triton | 0.0742 ± 0 |
| Tournament: original | PyTorch | 0.566 ± 0 |
| Bandit: exact bound | PyTorch | 0.101 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.0563 ± 0 |

#### Flat

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.508 ± 0 |
| SparQ | Triton | 0.141 ± 0 |
| Loki | PyTorch | 0.141 ± 0 |
| Block top-k | PyTorch | 0.141 ± 0 |
| Double indexing: mean | PyTorch | 0.0313 ± 0 |
| Double indexing: bound | PyTorch | 0.0391 ± 0 |
| Tournament: global halving | Triton | 0.0742 ± 0 |
| Tournament: blocks | PyTorch | 0.0742 ± 0 |
| Tournament: blocks, then global | Triton | 0.0742 ± 0 |
| Tournament: wide first round | PyTorch | 0.105 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.0741 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.566 ± 0 |
| Tournament: channel groups | PyTorch | 0.105 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.105 ± 0 |
| Tournament: shuffled blocks | Triton | 0.0742 ± 0 |
| Tournament: original | PyTorch | 0.566 ± 0 |
| Bandit: exact bound | PyTorch | 0.109 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.0579 ± 0 |

#### Sharp

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.508 ± 0 |
| SparQ | Triton | 0.141 ± 0 |
| Loki | PyTorch | 0.141 ± 0 |
| Block top-k | PyTorch | 0.141 ± 0 |
| Double indexing: mean | PyTorch | 0.0313 ± 0 |
| Double indexing: bound | PyTorch | 0.0391 ± 0 |
| Tournament: global halving | Triton | 0.0742 ± 0 |
| Tournament: blocks | PyTorch | 0.0742 ± 0 |
| Tournament: blocks, then global | Triton | 0.0742 ± 0 |
| Tournament: wide first round | PyTorch | 0.105 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.0741 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.566 ± 0 |
| Tournament: channel groups | PyTorch | 0.105 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.105 ± 0 |
| Tournament: shuffled blocks | Triton | 0.0742 ± 0 |
| Tournament: original | PyTorch | 0.566 ± 0 |
| Bandit: exact bound | PyTorch | 0.098 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.0557 ± 0 |

#### Local

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.508 ± 0 |
| SparQ | Triton | 0.141 ± 0 |
| Loki | PyTorch | 0.141 ± 0 |
| Block top-k | PyTorch | 0.141 ± 0 |
| Double indexing: mean | PyTorch | 0.0314 ± 0 |
| Double indexing: bound | PyTorch | 0.0391 ± 0 |
| Tournament: global halving | Triton | 0.0742 ± 0 |
| Tournament: blocks | PyTorch | 0.0742 ± 0 |
| Tournament: blocks, then global | Triton | 0.0742 ± 0 |
| Tournament: wide first round | PyTorch | 0.105 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.0742 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.566 ± 0 |
| Tournament: channel groups | PyTorch | 0.105 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.105 ± 0 |
| Tournament: shuffled blocks | Triton | 0.0742 ± 0 |
| Tournament: original | PyTorch | 0.566 ± 0 |
| Bandit: exact bound | PyTorch | 0.101 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.0564 ± 0 |

#### Many needles

| Name | Kernel | 32K |
|---|---|---:|
| Dense | PyTorch | 1 ± 0 |
| Exact top-k | PyTorch | 0.508 ± 0 |
| SparQ | Triton | 0.141 ± 0 |
| Loki | PyTorch | 0.141 ± 0 |
| Block top-k | PyTorch | 0.141 ± 0 |
| Double indexing: mean | PyTorch | 0.0314 ± 0 |
| Double indexing: bound | PyTorch | 0.0391 ± 0 |
| Tournament: global halving | Triton | 0.0749 ± 0 |
| Tournament: blocks | PyTorch | 0.0748 ± 0 |
| Tournament: blocks, then global | Triton | 0.0749 ± 0 |
| Tournament: wide first round | PyTorch | 0.106 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.0748 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.567 ± 0 |
| Tournament: channel groups | PyTorch | 0.105 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.106 ± 0 |
| Tournament: shuffled blocks | Triton | 0.0748 ± 0 |
| Tournament: original | PyTorch | 0.566 ± 0 |
| Bandit: exact bound | PyTorch | 0.0874 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.0528 ± 0 |

### Budget: latency and output error per budget

budget.png: Every version at each budget, the number of keys attended exactly: latency at 32K keys, iid; output error at 32K keys, LLM-like. A budget runs where the context is at least 16 times it. Whiskers as in the per-benchmark figures. docs/methods.md explains each row's version and its settings.

#### Latency

| Name | Kernel | 512 | 2048 |
|---|---|---:|---:|
| Dense | PyTorch | 227 (227–227) | 227 (227–227) |
| Exact top-k | PyTorch | 225 (225–226) | 318 (317–318) |
| SparQ | Triton | 139 (116–150) | 122 (121–122) |
| Loki | PyTorch | 154 (154–154) | 235 (235–235) |
| Block top-k | PyTorch | 125 (125–126) | 217 (217–217) |
| Double indexing: mean | PyTorch | 117 (116–118) | 224 (222–224) |
| Double indexing: bound | PyTorch | 125 (125–133) | 228 (226–228) |
| Tournament: global halving | Triton | 139 (139–139) | 414 (413–415) |
| Tournament: blocks | PyTorch | 701 (701–704) | 665 (665–667) |
| Tournament: blocks, then global | Triton | 100 (100–101) | 388 (388–388) |
| Tournament: wide first round | PyTorch | 1200 (1199–1200) | 1140 (1139–1144) |
| Tournament: sampled cutoffs | Triton | 146 (145–146) | 189 (183–191) |
| Tournament: sampled cutoffs, approximate losers | Triton | 332 (332–333) | 514 (513–514) |
| Tournament: channel groups | PyTorch | 1215 (1215–1216) | 1159 (1159–1161) |
| Tournament: mean-value losers | PyTorch | 1285 (1285–1285) | 1200 (1199–1200) |
| Tournament: shuffled blocks | Triton | 84 (84–84) | 80 (79–80) |
| Tournament: original | PyTorch | 1395 (1395–1397) | 1316 (1316–1317) |
| Bandit: exact bound | PyTorch | 2882 (2882–2886) | 2983 (2981–2996) |
| Bandit: Gaussian bound | PyTorch | 2572 (2563–2592) | 2676 (2670–2691) |

#### Output error

| Name | Kernel | 512 | 2048 |
|---|---|---:|---:|
| Dense | PyTorch | 0 ± 0 | 0 ± 0 |
| Exact top-k | PyTorch | 0.0792 ± 0 | 0.0482 ± 0 |
| SparQ | Triton | 0.0869 ± 0 | 0.0533 ± 0 |
| Loki | PyTorch | 0.0904 ± 0 | 0.056 ± 0 |
| Block top-k | PyTorch | 0.286 ± 0 | 0.14 ± 0 |
| Double indexing: mean | PyTorch | 1.27 ± 0 | 0.703 ± 0 |
| Double indexing: bound | PyTorch | 0.957 ± 0 | 0.411 ± 0 |
| Tournament: global halving | Triton | 0.144 ± 0 | 0.097 ± 0 |
| Tournament: blocks | PyTorch | 0.144 ± 0 | 0.0976 ± 0 |
| Tournament: blocks, then global | Triton | 0.144 ± 0 | 0.097 ± 0 |
| Tournament: wide first round | PyTorch | 0.142 ± 0 | 0.0955 ± 0 |
| Tournament: sampled cutoffs | Triton | 0.144 ± 0 | 0.097 ± 0 |
| Tournament: sampled cutoffs, approximate losers | Triton | 0.138 ± 0 | 0.0945 ± 0 |
| Tournament: channel groups | PyTorch | 0.151 ± 0 | 0.108 ± 0 |
| Tournament: mean-value losers | PyTorch | 0.136 ± 0 | 0.093 ± 0 |
| Tournament: shuffled blocks | Triton | 0.144 ± 0 | 0.0967 ± 0 |
| Tournament: original | PyTorch | 0.486 ± 0 | 0.429 ± 0 |
| Bandit: exact bound | PyTorch | 0.136 ± 0 | 0.091 ± 0 |
| Bandit: Gaussian bound | PyTorch | 0.156 ± 0 | 0.0986 ± 0 |
