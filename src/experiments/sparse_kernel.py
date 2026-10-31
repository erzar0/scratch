"""Decode attention timing for one layer: dense, SparQ, the tournament kernels.

Batch 1, one new token, Qwen3-0.6B's layer shape (8 KV heads, 2 query heads each,
head dim 128), bf16 K and V with random values, K also stored transposed for the
partial scores (src/sparse/kernel.py). The tournament runs its last 3 rounds
globally and the rest in blocks of 1024 keys, so the rounds in blocks grow with n
(3 of 6 at 32768 tokens and keep 512, the setting measured on PG-19). The global
kernel (src/sparse/kernel_global.py) runs every round over all survivors; the
sampled one (src/sparse/kernel_sampled.py) takes early cutoffs from a sample.
The parallel one (src/sparse/kernel_parallel.py) runs every round inside 32 blocks
per KV head, for keys stored in a random order.
Best of 3 medians of CUDA-graph replays -> runs/sparse/kernel.csv.

uv run python -m experiments.sparse_kernel
"""

import math
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import triton.testing

from experiments.results import Results
from sparse import kernel, kernel_global, kernel_parallel, kernel_sampled

OUT = Path("runs/sparse/kernel.csv")
LENGTHS = (8192, 16384, 32768, 65536, 131072)
GROUPS, DIM = 8, 128


def methods(
    q: torch.Tensor, k: torch.Tensor, k_t: torch.Tensor, v: torch.Tensor
) -> dict[str, Callable[[], torch.Tensor]]:
    """Method name -> one decode step of attention for this layer."""
    q4, k4, v4 = q.reshape(1, 2 * GROUPS, 1, DIM), k[None], v[None]
    scaling = DIM**-0.5
    return {
        "dense": lambda: F.scaled_dot_product_attention(q4, k4, v4, enable_gqa=True),
        "dense-bmm": lambda: ((q @ k.mT) * scaling).softmax(-1) @ v,
        "sparq-64-128": lambda: kernel.sparq(q, k, k_t, v, 64, 128),
        "sparq-32-128": lambda: kernel.sparq(q, k, k_t, v, 32, 128),
        "sparq-16-128": lambda: kernel.sparq(q, k, k_t, v, 16, 128),
        "sparq-64-256": lambda: kernel.sparq(q, k, k_t, v, 64, 256),
        "sparq-64-512": lambda: kernel.sparq(q, k, k_t, v, 64, 512),
        "sparq-32-256": lambda: kernel.sparq(q, k, k_t, v, 32, 256),
        "sparq-32-512": lambda: kernel.sparq(q, k, k_t, v, 32, 512),
        "hybrid-256": lambda: kernel.tournament(q, k, k_t, v, 256),
        "hybrid-512": lambda: kernel.tournament(q, k, k_t, v, 512),
        "global-128": lambda: kernel_global.tournament(q, k, k_t, v, 128),
        "global-256": lambda: kernel_global.tournament(q, k, k_t, v, 256),
        "global-512": lambda: kernel_global.tournament(q, k, k_t, v, 512),
        "sampled-8-drop-512": lambda: kernel_sampled.tournament(q, k, k_t, v, 512),
        "sampled-16-drop-512": lambda: kernel_sampled.tournament(q, k, k_t, v, 512, 16),
        "sampled-16-approx-512": lambda: kernel_sampled.tournament_approx(
            q, k, k_t, v, 512, 16
        ),
        "parallel-32-512": lambda: kernel_parallel.tournament(q, k, k_t, v, 512),
    }


@torch.no_grad()
def main() -> None:
    """Time every method at every length; see the module docstring."""
    results = Results(OUT, {"study": "decode"})
    gpu = torch.cuda.get_device_name()
    for n in LENGTHS:
        torch.manual_seed(0)
        q = torch.randn(GROUPS, 2, DIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(GROUPS, n, DIM, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        k_t = k.transpose(1, 2).contiguous()
        for name, step in methods(q, k, k_t, v).items():
            step()  # compile before capture
            ms = min(  # the GPU clock under WSL wanders: best of 3 medians
                triton.testing.do_bench_cudagraph(step, rep=100, return_mode="median")
                for _ in range(3)
            )
            keep = int(name.split("-")[1]) if name.startswith("hybrid") else 0
            total = math.ceil(math.log2(n / keep)) if keep else 0
            rounds = total - kernel.GLOBAL_ROUNDS if keep else None
            results.add(
                {
                    "study": "decode",
                    "gpu": gpu,
                    "method": name,
                    "n": n,
                    "us": 1000 * ms,
                    "block_rounds": rounds,
                }
            )
            print(f"{n:>6} {name:<13} {1000 * ms:8.1f} us", flush=True)


if __name__ == "__main__":
    main()
