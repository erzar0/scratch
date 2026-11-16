"""Decode time of one token (batch 1) at DeepSeek V4.1-Flash-1017's attention size
(src/deepseek4.1-1017/inference/config.json): 40 layers, 64 query heads of 512
channels on one K=V head, 128-token window, 512 picked entries per layer. Entries of
2 tokens from layers 2, 8, 14 (each read by its next 6 layers), of 1 token from layer
20 (read by layers 20-39); layers 0-1 window only; picks at layers 2, 8, 14, 20, 24,
28, 32, 36, every other layer reusing the last pick layer's.
    reference      its indexer (32 heads x 128 channels, per-head weights; index keys
                   cached at layers 2, 8, 14, 20): layers 2, 8, 14 score every index
                   key of their source; layer 20 every key, keeps the 2048 blocks of
                   8 entries with the largest score (newest pinned) and takes its top
                   512 inside; layers 24-36 score only inside those blocks (model.py
                   Indexer, select_candidate_blocks; csa2_ref_decode.block_max).
    parameter-free src/nanogpt/csa2_pf.py at this size: no index keys, no sink;
                   score = (sum of the 64 heads' queries) . entry, among entries that
                   end before the window:
        exact          every entry (csa2_decode.top64; up to 128K, see LENGTHS).
        blocks         a mean per 8 entries kept at cache write; the best 2048 blocks
                       by score of their mean (Triton GEMV + torch.topk), then the top
                       512 inside (= nanogpt's passing best 256 blocks of 8 at top 64:
                       32 candidates per pick).
        blocks, reuse  as blocks, but layers 24-36 score inside layer 20's blocks (as
                       the reference).
        Flash-1017 structure  exact at layers 2, 8, 14 (as the reference scores
                       every index key there), blocks at 20, reuse at 24-36.
        FP4 means      blocks with the means stored as V4.1's FP4 main-cache row
                       (csa2_sketch_decode.fp4_scores; 288 instead of 512 bytes).
        1024 blocks    blocks keeping 1024 blocks of 8 (16 candidates per pick).
Attention is the same kernel for every version: csa2_attend_mqa (window + 512 picks,
sink, inverse RoPE on 64 channels; tl.dot over 16 heads per program, every row read
once per head group; csa2_attend_split, one head per program, is
decode_flash_split.csv). FP8 rows: every cache float8_e4m3fn (TensorRT-LLM stores FP4
entries and index keys; this GPU's Triton reads no FP4 natively).
Cache writes (compressor, index keys, means) are not timed. Picks checked against
PyTorch at 128K. Also writes the KV cache of both at this size (40 layers, bf16 /
FP8 / TensorRT-LLM FP4).
-> runs/nanogpt/long_context/decode_flash.csv (version, part, n, us_per_token),
   runs/nanogpt/long_context/flash_cache.csv (version, format, n, mib),
   runs/nanogpt/long_context/flash_bytes.csv (version, n, picks_mb, attention_mb):
   bytes one decode token reads with every cache FP8 (H200's model: time ~ bytes).
Fork venv, from the repo root:
    PYTHONPATH=src:experiments/csa2/bench python experiments/csa2/bench/decode_flash.py
"""

import csv
from collections.abc import Callable
from pathlib import Path

import torch
import triton
import triton.language as tl
from decode_csa2 import graphed_ms
from torch import Tensor

from nanogpt import csa2_attend_mqa as ma
from nanogpt import csa2_decode as cd
from nanogpt import csa2_ref_decode as rd
from nanogpt.csa2_sketch_decode import fp4_scores, pack_fp4, unpack_fp4

# ruff: noqa: N803  (Triton's convention: upper-case constexpr arguments)

OUT = Path("runs/nanogpt/long_context")
LAYERS, HEADS, DIM, ROPE, WINDOW, TOP = 40, 64, 512, 64, 128, 512
INDEX_HEADS, INDEX_DIM, BLOCKS, SIZE = 32, 128, 2048, 8
SOURCES = {2: 2, 8: 2, 14: 2, 20: 1}  # kv source layer: tokens per entry
PICKS = (2, 8, 14, 20, 24, 28, 32, 36)
# ponytail: top64 at top 512 runs at most 32 programs; past 2^18 keys per call it
# needs 128 KiB shared memory (this GPU: 99 KiB): no 1M, exact picks up to 128K
LENGTHS = (8192, 32768, 131072, 524288)
EXACT_MAX = 131072
REF = "V4.1-Flash-1017 attention, its indexer, 2048 candidate blocks"
PF = "parameter-free, summed query"
BEST = f"{PF}, best 2048 blocks of 8"
REUSE = f"{BEST}, layers 24-36 inside layer 20's blocks"
STRUCTURE = f"{PF}, Flash-1017 structure: exact at 2-14, best 2048 blocks of 8 at 20-36"
FP4 = f"{BEST}, FP4 means"
FP4_REUSE = f"{REUSE}, FP4 means"
HALF = f"{PF}, best 1024 blocks of 8"
Pick = Callable[[dict, int], Tensor]


def source(i: int) -> int:
    """The kv source layer whose entries layer i reads."""
    return max(s for s in SOURCES if s <= i)


@triton.jit
def _gemv(k_ptr, q_ptr, out_ptr, n, D: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    c = tl.arange(0, D)
    k = tl.load(k_ptr + rows[:, None] * D + c[None, :], mask=rows[:, None] < n)
    q = tl.load(q_ptr + c).to(tl.float32)
    tl.store(out_ptr + rows, tl.sum(k.to(tl.float32) * q[None, :], 1), mask=rows < n)


def gemv(keys: Tensor, q: Tensor) -> Tensor:
    """keys (N, D) of any dtype . q (D,) -> (N,) fp32."""
    n, dim = keys.shape
    out = torch.empty(n, dtype=torch.float32, device=keys.device)
    _gemv[(triton.cdiv(n, 32),)](keys, q, out, n, D=dim, R=32, num_warps=4)
    return out


def rand(*shape: int) -> Tensor:
    return torch.randn(*shape, device="cuda").bfloat16()


def state(n: int) -> dict:
    """Random caches and one token's queries at length n (bf16)."""
    s: dict = {"n": n}
    s["entries"] = {i: rand(n // m, DIM) for i, m in SOURCES.items()}
    s["index"] = {i: rand(n // m, INDEX_DIM) for i, m in SOURCES.items()}
    # parameter-free: entries that end before the window, a mean per full block
    s["pf"] = {i: k[: (n - WINDOW) // m] for (i, m), k in
               zip(SOURCES.items(), s["entries"].values(), strict=True)}  # fmt: skip
    s["means"] = {i: k[: len(k) // SIZE * SIZE].view(-1, SIZE, DIM).float().mean(1)
                  .to(k.dtype) for i, k in s["pf"].items()}  # fmt: skip
    s["fp4"] = {i: pack_fp4(m) for i, m in s["means"].items()}  # stays FP4 in fp8()
    s["window"] = [rand(WINDOW, DIM) for _ in range(LAYERS)]
    s["q"] = [rand(HEADS, DIM) for _ in range(LAYERS)]
    s["qa"] = {i: s["q"][i].float().sum(0).bfloat16() for i in PICKS}
    s["qi"] = {i: rand(INDEX_HEADS, INDEX_DIM) for i in PICKS}
    s["w"] = {i: torch.rand(INDEX_HEADS, device="cuda").bfloat16() for i in PICKS}
    s["sinks"] = torch.zeros(HEADS, device="cuda")
    s["cos"], s["sin"] = rand(ROPE), rand(ROPE)
    # attention-only part: valid in every source of both versions
    s["picks"] = torch.randperm((n - WINDOW) // 2, device="cuda")[:TOP]
    return s


def fp8(s: dict) -> dict:
    """s with every cache float8_e4m3fn (queries stay bf16)."""
    out = dict(s)
    for key in ("entries", "index", "pf", "means"):
        out[key] = {i: x.to(torch.float8_e4m3fn) for i, x in s[key].items()}
    out["window"] = [x.to(torch.float8_e4m3fn) for x in s["window"]]
    return out


def reference(s: dict, i: int) -> Tensor:
    keys, q, w = s["index"][source(i)], s["qi"][i], s["w"][i]
    if i < 20 or len(keys) <= BLOCKS * SIZE:  # every entry (every block a candidate)
        return cd.top64(q, keys, w, top=TOP)
    if i == 20:
        best = rd.block_max(q, w, keys, SIZE)
        best[-1:].fill_(float("inf"))  # newest block pinned
        s["ref_blocks"] = best.topk(BLOCKS).indices
    return cd.top64(q, keys, w, top=TOP, within=s["ref_blocks"], size=SIZE)


def exact(s: dict, i: int) -> Tensor:
    return cd.top64(s["qa"][i], s["pf"][source(i)], top=TOP)


def blocks(reuse: bool, count: int = BLOCKS, fp4: bool = False) -> Pick:
    def pick(s: dict, i: int) -> Tensor:
        q, keys, means = s["qa"][i], s["pf"][source(i)], s["means"][source(i)]
        if len(means) <= count:  # every block a candidate
            return cd.top64(q, keys[: len(means) * SIZE], top=TOP)
        if not (reuse and i > 20):
            score = fp4_scores(*s["fp4"][source(i)], q) if fp4 else gemv(means, q)
            s["pf_blocks"] = score.topk(count, sorted=False).indices
        return cd.top64(q, keys, top=TOP, within=s["pf_blocks"], size=SIZE)

    return pick


def structure(s: dict, i: int) -> Tensor:
    return exact(s, i) if i < 20 else blocks(True)(s, i)


VERSIONS: dict[str, tuple[Pick, str]] = {  # pick rule, its entries for attention
    REF: (reference, "entries"),
    f"{PF}, exact picks": (exact, "pf"),
    BEST: (blocks(False), "pf"),
    REUSE: (blocks(True), "pf"),
    STRUCTURE: (structure, "pf"),
    FP4: (blocks(False, fp4=True), "pf"),
    FP4_REUSE: (blocks(True, fp4=True), "pf"),
    HALF: (blocks(False, BLOCKS // 2), "pf"),
}


def parts(s: dict, pick: Pick, kv: str) -> dict[str, Callable[[], None]]:
    """Picks (8 pick layers), attention (40 layers, fixed picks), both."""
    sinks = s["sinks"] if kv == "entries" else torch.full_like(s["sinks"], -1e30)

    def attend(i: int, picks: Tensor) -> None:
        main = s[kv][source(max(i, 2))]
        picks = picks[:0] if i < 2 else picks  # layers 0-1: window only
        ma.csa2_attend(s["q"][i], s["window"][i], main, picks, sinks, s["cos"],
                       s["sin"])  # fmt: skip

    def picks() -> None:
        for i in PICKS:
            pick(s, i)

    def attention() -> None:
        for i in range(LAYERS):
            attend(i, s["picks"])

    def both() -> None:
        chosen = s["picks"]
        for i in range(LAYERS):
            if i in PICKS:
                chosen = pick(s, i)
            attend(i, chosen)

    return {"Picks": picks, "Attention": attention, "Picks + attention": both}


def check(s: dict) -> None:
    """Every rule's picks at every pick layer = PyTorch's (fp32 scores)."""

    def top(score: Tensor, among: Tensor | None = None) -> set[int]:
        if among is None:
            return set(score.topk(TOP).indices.tolist())
        return set(among[score[among].topk(TOP).indices].tolist())

    def cand(block_score: Tensor, count: int = BLOCKS) -> Tensor:
        best = block_score.topk(min(count, len(block_score))).indices
        return (best[:, None] * SIZE + torch.arange(SIZE, device="cuda")).flatten()

    held: dict[str, Tensor] = {}
    for i in PICKS:
        keys = s["index"][source(i)].float()
        score = ((keys @ s["qi"][i].float().T).relu() * s["w"][i].float()).sum(1)
        if i == 20:
            block = score.view(-1, SIZE).amax(1)
            block[-1] = float("inf")
            held["ref"] = cand(block)
        want = top(score, held["ref"] if i >= 20 else None)
        assert set(reference(s, i).tolist()) == want, ("reference", i)
        q, keys = s["qa"][i].float(), s["pf"][source(i)].float()
        score = keys @ q
        assert set(exact(s, i).tolist()) == top(score), ("exact", i)
        own = cand(s["means"][source(i)].float() @ q)
        if i == 20:
            held["pf"] = own
        assert set(blocks(False)(s, i).tolist()) == top(score, own), ("blocks", i)
        own = cand(unpack_fp4(*s["fp4"][source(i)]) @ q)
        got = set(blocks(False, fp4=True)(s, i).tolist())
        assert got == top(score, own), ("FP4 means", i)
        own = cand(s["means"][source(i)].float() @ q, BLOCKS // 2)
        got = set(blocks(False, BLOCKS // 2)(s, i).tolist())
        assert got == top(score, own), ("1024 blocks", i)
    for i in PICKS[3:]:  # 20, then 24-36 inside layer 20's blocks
        score = s["pf"][source(i)].float() @ s["qa"][i].float()
        got = set(blocks(True)(s, i).tolist())
        assert got == top(score, held["pf"]), ("reuse", i)
    print("picks = PyTorch at every pick layer (reference, exact, blocks, reuse, FP4"
          " means, 1024 blocks)", flush=True)  # fmt: skip


def cache_mib(n: int) -> list[tuple[str, str, int, float]]:
    """KV cache of one sequence: 40 layers' windows + entries (+ index keys / means).
    TensorRT-LLM FP4: entry 4 bits + E4M3 scale per 16 values, index key 4 bits +
    UE8M0 per 32, window FP8 + UE8M0 per 32."""
    per = {"bf16": (2, 2, 2), "FP8": (1, 1, 1), "FP4": (9 / 16, 17 / 32, 33 / 32)}
    entries = sum(n // m for m in SOURCES.values())
    rows = []
    for fmt, (entry, index, window) in per.items():
        win = LAYERS * min(n, WINDOW) * DIM * window
        ref = win + entries * (DIM * entry + INDEX_DIM * index)
        pf = win + entries * DIM * entry
        means = entries // SIZE * DIM
        for name, b in ((REF, ref), (f"{PF}, exact picks", pf),
                        (BEST, pf + means * entry),
                        (FP4, pf + means * 9 / 16)):  # fmt: skip
            rows.append((name, fmt, n, round(b / 2**20, 2)))
    return rows


def bytes_mb(n: int) -> list[tuple[str, int, float, float]]:
    """MB one decode token reads, every cache FP8 (1 byte per value; FP4 means 9/16):
    the pick layers' scoring passes, and the 40 layers' attention (window + picks;
    layers 0-1 window only). Score writes and top-k passes are not counted."""
    half = (n - WINDOW) // 2  # entries of layers 2, 8, 14 before the window
    full = n - WINDOW  # layer 20's
    cand = BLOCKS * SIZE * DIM  # bytes rescored inside 2048 blocks of 8
    ref_cand = BLOCKS * SIZE * INDEX_DIM

    def means(*sizes: int, fp4: bool = False) -> float:
        return sum(m // SIZE for m in sizes) * DIM * (9 / 16 if fp4 else 1)

    picks = {
        REF: 3 * half * INDEX_DIM + full * INDEX_DIM + 5 * ref_cand,
        f"{PF}, exact picks": 3 * half * DIM + 5 * full * DIM,
        BEST: means(half, half, half, *[full] * 5) + 8 * cand,
        REUSE: means(half, half, half, full) + 8 * cand,
        STRUCTURE: 3 * half * DIM + means(full) + 5 * cand,
        FP4: means(half, half, half, *[full] * 5, fp4=True) + 8 * cand,
        FP4_REUSE: means(half, half, half, full, fp4=True) + 8 * cand,
        HALF: means(half, half, half, *[full] * 5) + 4 * cand,
    }
    attention = (2 * WINDOW + (LAYERS - 2) * (WINDOW + TOP)) * DIM
    return [(v, n, round(b / 1e6, 2), round(attention / 1e6, 2))
            for v, b in picks.items()]  # fmt: skip


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "flash_cache.csv").open("w", newline="") as f:
        cache = [r for n in LENGTHS for r in cache_mib(n)]
        csv.writer(f).writerows([("version", "format", "n", "mib"), *cache])
    for r in cache:
        print(*r, sep=" | ", flush=True)
    with (OUT / "flash_bytes.csv").open("w", newline="") as f:
        read = [r for n in LENGTHS for r in bytes_mb(n)]
        header = ("version", "n", "picks_mb", "attention_mb")
        csv.writer(f).writerows([header, *read])
    for b in read:
        print(*b, sep=" | ", flush=True)
    rows = []
    for n in LENGTHS:
        torch.manual_seed(0)
        s = state(n)
        s8 = fp8(s)
        if n == 131072:
            check(s)
            check(s8)
        for fmt, st in (("", s), (", FP8 caches", s8)):
            for name, (pick, kv) in VERSIONS.items():
                if pick in (exact, structure) and n > EXACT_MAX:
                    continue
                for part, fn in parts(st, pick, kv).items():
                    us = 1000 * graphed_ms(fn)
                    rows.append((name + fmt, part, n, round(us, 1)))
                    print(*rows[-1], sep=" | ", flush=True)
        del s, s8
        torch.cuda.empty_cache()
    with (OUT / "decode_flash.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("version", "part", "n", "us_per_token"), *rows])
    print("wrote", OUT / "decode_flash.csv", flush=True)
