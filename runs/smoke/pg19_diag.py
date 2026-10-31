"""Throwaway: is CSA2-Qwen's PG-19 loss a length bug, a dtype issue or the domain?"""

import torch

from model import train
from qwen import data, evaluate
from qwen import train as qt

torch.set_float32_matmul_precision("high")
books = [b for b in data.books() if len(b) > 32768][:2]
val = data.load("validation")


def buckets(losses: torch.Tensor) -> list[str]:
    edges = [0, 1024, 4096, 8192, 16384, 32768]
    return [f"{losses[a:b].mean():.2f}" for a, b in zip(edges, edges[1:]) if a < len(losses)]


for row in ("qwen", "csa2"):
    model, selector = evaluate.load_row(row)
    with evaluate.scoped(model, selector):
        for i, book in enumerate(books):
            long = evaluate.token_losses(model, train.to_device(book[None, :32769]))
            short = evaluate.token_losses(model, train.to_device(book[None, :4097]))
            gap = (long[:4096] - short).abs().max()
            print(row, "book", i, "32k:", buckets(long), "4k:", buckets(short), f"max diff {gap:.3f}")
        wiki = evaluate.token_losses(model, train.to_device(val[None, :32769]))
        print(row, "wikitext 32k:", buckets(wiki))
    del model
    torch.cuda.empty_cache()

model = qt.build("csa2", "e2e").eval()  # float32 students, autocast as in training
with torch.no_grad(), qt.autocast(), evaluate.scoped(model, evaluate.own_indexer):
    short = evaluate.token_losses(model, train.to_device(books[0][None, :4097]))
print("csa2 fp32+autocast book 0 4k:", buckets(short))
