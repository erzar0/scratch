"""Decode attention timing on real Qwen3-0.6B keys at 32768 tokens.

experiments/sparse_kernel.py times random keys, whose round survivors spread
evenly: the worst case for the tournament kernels' scattered reads, where one
32-byte sector of the transposed K is one dim of 16 neighbouring keys. Real
survivors may cluster (recent keys, sinks), touching fewer sectors. Here each
layer's q, K and V are those of the last position of the first PG-19 test book
longer than 32768 tokens (dense attention in every layer); each method of
sparse_kernel.methods is timed on each layer, best of 3 medians of CUDA-graph
replays -> runs/sparse/kernel_real.csv (per layer and method).

uv run python -m experiments.sparse_kernel_real
"""

from pathlib import Path

import torch
import triton.testing

from experiments.results import Results
from experiments.sparse_kernel import methods
from model import train
from qwen import data
from qwen import model as qm
from sparse import attend

OUT = Path("runs/sparse/kernel_real.csv")
LONG = 32768
TIMED = (
    "dense",
    "dense-bmm",
    "sparq-64-128",
    "sparq-32-128",
    "hybrid-512",
    "global-512",
    "sampled-8-drop-512",
    "sampled-16-drop-512",
)


@torch.no_grad()
def capture() -> list[tuple[torch.Tensor, ...]]:
    """Per layer: (G, 2, D) q, (G, N, D) K and V of the last position, bf16, CPU."""
    layers = []

    def keep(chunk: attend.Chunk) -> attend.Outcome:
        last = (chunk.query[:, :, -1], chunk.key, chunk.value)
        layers.append(tuple(t.to("cpu", torch.bfloat16) for t in last))
        return attend.dense(chunk)

    model = qm.load()
    attend.attach(model, keep, 1)
    book = next(b for b in data.books() if len(b) > LONG)
    model.model(input_ids=train.to_device(book[None, :LONG]))
    return layers


@torch.no_grad()
def main() -> None:
    """Time each method on each layer; see the module docstring."""
    layers = capture()
    torch.cuda.empty_cache()
    results = Results(OUT, {"study": "real"})
    gpu = torch.cuda.get_device_name()
    for layer, tensors in enumerate(layers):
        q, k, v = (t.cuda() for t in tensors)
        steps = methods(q, k, k.transpose(1, 2).contiguous(), v)
        for name in TIMED:
            steps[name]()  # compile before capture
            ms = min(  # the GPU clock under WSL wanders: best of 3 medians
                triton.testing.do_bench_cudagraph(
                    steps[name], rep=100, return_mode="median"
                )
                for _ in range(3)
            )
            row = {"layer": layer, "method": name, "n": k.shape[1], "us": 1000 * ms}
            results.add({"study": "real", "gpu": gpu} | row)
            print(f"layer {layer:2d} {name:<20} {1000 * ms:8.1f} us", flush=True)


if __name__ == "__main__":
    main()
