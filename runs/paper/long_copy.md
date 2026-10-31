### 8192 tokens

| Method | Copy loss above pure CSA2 (nats) | Text loss above pure CSA2 (nats) | Copy hit rate | Index reads saved vs pure CSA2 |
|---|---|---|---|---|
| pure CSA2 | 3.852 (loss) | 3.871 (loss) | 0.014 | 1.00× |
| random entries, 8 entries | +0.172 ± 0.008 | +0.132 ± 0.007 | 0.005 | — |
| oracle, 8 entries | +0.001 ± 0.001 | +0.002 ± 0.001 | 0.020 | — |
| double indexing: mean, superblocks of 64, keep 2 | +0.033 ± 0.003 | +0.006 ± 0.001 | 0.003 | 6.06× |
| double indexing: bound, superblocks of 32, keep 4 | +0.030 ± 0.002 | +0.006 ± 0.001 | 0.003 | 5.08× |
| double indexing: learned, superblocks of 16, keep 4 | +0.033 ± 0.002 | +0.007 ± 0.001 | 0.001 | 7.65× |
| double indexing: mixed, superblocks of 16, keep 2 (needle layers: keep 8) | +0.022 ± 0.002 | +0.004 ± 0.001 | 0.004 | 7.24× |

### 16384 tokens

| Method | Copy loss above pure CSA2 (nats) | Text loss above pure CSA2 (nats) | Copy hit rate | Index reads saved vs pure CSA2 |
|---|---|---|---|---|
| pure CSA2 | 3.981 (loss) | 3.944 (loss) | 0.024 | 1.00× |
| random entries, 8 entries | +0.127 ± 0.009 | +0.151 ± 0.007 | 0.003 | — |
| oracle, 8 entries | -0.000 ± 0.001 | +0.003 ± 0.001 | 0.031 | — |
| double indexing: mean, superblocks of 64, keep 2 | +0.047 ± 0.003 | +0.009 ± 0.001 | 0.006 | 10.89× |
| double indexing: bound, superblocks of 32, keep 4 | +0.037 ± 0.003 | +0.009 ± 0.001 | 0.009 | 7.64× |
| double indexing: learned, superblocks of 16, keep 4 | +0.052 ± 0.003 | +0.010 ± 0.001 | 0.004 | 10.32× |
| double indexing: mixed, superblocks of 16, keep 2 (needle layers: keep 8) | +0.026 ± 0.003 | +0.006 ± 0.001 | 0.010 | 9.22× |

### 32768 tokens

| Method | Copy loss above pure CSA2 (nats) | Text loss above pure CSA2 (nats) | Copy hit rate | Index reads saved vs pure CSA2 |
|---|---|---|---|---|
| pure CSA2 | 4.146 (loss) | 4.041 (loss) | 0.004 | 1.00× |
| random entries, 8 entries | -0.006 ± 0.010 | +0.085 ± 0.008 | 0.001 | — |
| oracle, 8 entries | +0.015 ± 0.003 | +0.004 ± 0.001 | 0.008 | — |
| double indexing: mean, superblocks of 64, keep 2 | +0.006 ± 0.004 | +0.008 ± 0.003 | 0.000 | 18.49× |
| double indexing: bound, superblocks of 32, keep 4 | +0.024 ± 0.004 | +0.017 ± 0.003 | 0.000 | 10.31× |
| double indexing: learned, superblocks of 16, keep 4 | +0.027 ± 0.003 | +0.013 ± 0.003 | 0.000 | 12.54× |
| double indexing: mixed, superblocks of 16, keep 2 (needle layers: keep 8) | +0.024 ± 0.003 | +0.016 ± 0.002 | 0.001 | 10.71× |

### 65536 tokens

| Method | Copy loss above pure CSA2 (nats) | Text loss above pure CSA2 (nats) | Copy hit rate | Index reads saved vs pure CSA2 |
|---|---|---|---|---|
| pure CSA2 | 4.102 (loss) | 4.100 (loss) | 0.039 | 1.00× |
| random entries, 8 entries | +0.034 ± 0.011 | +0.030 ± 0.012 | 0.001 | — |
| oracle, 8 entries | +0.006 ± 0.004 | +0.010 ± 0.002 | 0.061 | — |
| double indexing: mean, superblocks of 64, keep 2 | +0.003 ± 0.006 | +0.001 ± 0.005 | 0.034 | 28.62× |
| double indexing: bound, superblocks of 32, keep 4 | +0.046 ± 0.006 | +0.015 ± 0.004 | 0.019 | 12.53× |
| double indexing: learned, superblocks of 16, keep 4 | +0.031 ± 0.005 | +0.014 ± 0.004 | 0.033 | 14.05× |
| double indexing: mixed, superblocks of 16, keep 2 (needle layers: keep 8) | +0.037 ± 0.005 | +0.017 ± 0.005 | 0.023 | 11.66× |

### 131072 tokens

| Method | Copy loss above pure CSA2 (nats) | Text loss above pure CSA2 (nats) | Copy hit rate | Index reads saved vs pure CSA2 |
|---|---|---|---|---|
| pure CSA2 | 4.246 (loss) | 4.171 (loss) | 0.013 | 1.00× |
| random entries, 8 entries | -0.105 ± 0.009 | -0.034 ± 0.008 | 0.000 | — |
| oracle, 8 entries | +0.047 ± 0.014 | +0.023 ± 0.003 | 0.024 | — |
| double indexing: mean, superblocks of 64, keep 2 | -0.029 ± 0.006 | -0.015 ± 0.005 | 0.004 | 39.51× |
| double indexing: bound, superblocks of 32, keep 4 | +0.005 ± 0.005 | +0.005 ± 0.003 | 0.003 | 14.05× |
| double indexing: learned, superblocks of 16, keep 4 | +0.034 ± 0.005 | +0.014 ± 0.002 | 0.003 | 14.96× |
| double indexing: mixed, superblocks of 16, keep 2 (needle layers: keep 8) | +0.011 ± 0.004 | +0.014 ± 0.003 | 0.004 | 12.20× |
