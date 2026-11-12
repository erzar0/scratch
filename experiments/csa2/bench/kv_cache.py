"""KV cache each attention method must keep for decode, batch 1, record 91 stripped
model (10 attention layers, 6 heads, K padded to 128, V 128, bf16): allocate it and
read torch's allocated bytes. Windows from train_dense_stripped.py bm_sizes at eval
(short 768, long 2560 on layers 3 and 10; layer 6 has no attention).
    uv run python kv_cache.py OUT.csv"""

import csv
import sys

import torch

H, D = 6, 128
WINDOWS = [768, 768, 768, 2560, 768, 768, 768, 768, 768, 2560]


def bf16(*shape: int) -> torch.Tensor:
    return torch.empty(*shape, device="cuda", dtype=torch.bfloat16)


def dense(n: int, w: int) -> list[torch.Tensor]:
    return [bf16(min(n, w), H, D), bf16(min(n, w), H, D)]


def full(n: int, w: int) -> list[torch.Tensor]:
    return [bf16(n, H, D), bf16(n, H, D)]


def transposed(n: int, w: int) -> list[torch.Tensor]:
    return [bf16(H, D, n), bf16(n, H, D)]


def both(n: int, w: int) -> list[torch.Tensor]:
    return [*full(n, w), bf16(H, D, n)]


def tiles(n: int, w: int) -> list[torch.Tensor]:
    means = torch.empty(n // 32, H, D, device="cuda", dtype=torch.float32)
    return [*full(n, w), means]


GROUP = 4  # CSA2: one compressed K=V head (m = 4) per group of 4 attention layers


def csa2(n: int, w: int, layer: int, index_dim: int = 32) -> list[torch.Tensor]:
    """CSA2: K=V window of 128 per layer; per group main KV + indexer K (n / 4)."""
    shared = [bf16(n // 4, D), bf16(n // 4, index_dim)] if layer % GROUP == 0 else []
    return [bf16(min(n, 128), D), *shared]


def csa2_picks(n: int, w: int, layer: int) -> list[torch.Tensor]:
    """A: CSA2 without indexer K (picks from main KV)."""
    return [x for x in csa2(n, w, layer, 0) if x.numel()]


def r91_csa2(n: int, w: int, layer: int) -> list[torch.Tensor]:
    """B: record 91's own K, V in a 128 window per layer + main KV per group."""
    shared = [bf16(n // 4, D)] if layer % GROUP == 0 else []
    return [*dense(n, 128), *shared]


def r91_kv_window(groups: int):
    """B with CSA2's window (csa2_window): G shared K=V vectors per window token."""

    def state(n: int, w: int, layer: int) -> list[torch.Tensor]:
        shared = [bf16(n // 4, D)] if layer % GROUP == 0 else []
        return [bf16(min(n, 128), groups, D), *shared]

    return state


kv_win1, kv_win2, kv_win6 = r91_kv_window(1), r91_kv_window(2), r91_kv_window(6)

METHODS = {
    "Dense stripped, FlashAttention 2 sliding windows 768 / 2560": dense,
    "Run 4: exact top 64 per block, two Triton kernels": full,
    "Runs 2 and 3: tournament, K stored transposed only": transposed,
    "Runs 2 and 3: tournament, K in both layouts": both,
    "Run 5: tile selection, 32-key blocks (+ block mean keys)": tiles,
    "CSA2 stripped, learned indexer": csa2,
    "CSA2 stripped, exact or tournament picks instead of the indexer": csa2_picks,
    "CSA2 KV layout in stripped record 91, exact or tournament top 64 entries": (
        r91_csa2
    ),
    "CSA2 KV layout in stripped record 91, 1 shared K=V window head": kv_win1,
    "CSA2 KV layout in stripped record 91, 2 shared K=V window heads": kv_win2,
    "CSA2 KV layout in stripped record 91, per-head K=V window": kv_win6,
}
LAYERED = {csa2, csa2_picks, r91_csa2, kv_win1, kv_win2, kv_win6}

rows = []
for name, state in METHODS.items():
    for n in (8192, 32768, 131072):
        before = torch.cuda.memory_allocated()
        cache = [
            state(n, w, i) if state in LAYERED else state(n, w)  # type: ignore[call-arg]
            for i, w in enumerate(WINDOWS)
        ]
        mib = (torch.cuda.memory_allocated() - before) / 2**20
        rows.append((name, n, round(mib, 1)))
        print(rows[-1], flush=True)
        del cache
with open(sys.argv[1], "w", newline="") as f:
    csv.writer(f).writerows([("version", "n", "cache_mib"), *rows])
