"""Every stat of each model under each decode rule, one row per (model, picks at
decode): loss (final validation loss as logged; eval with bf16, FP8 and FP4 caches; eval
by position), the per-token change vs the run's own (trained) rule in the same cache
format (eval/paired.py; gate: all positions <= +0.005, every range past 2048 <= +0.01),
training hours and peak memory, KV cache at 8K-512K tokens in each format (block-mean
rows: plus one mean per 8 entries, in the entries' format), and decode time per token
with bf16 and FP8 caches (bench/decode_ref_picks.py, bench/decode_pf_share.py). Every
eval clips the lm_head weights to the training head's FP8 range (eval/variant_eval.py
HEAD_CLIP, queue_head_clip*.sh); the logged validation loss leaves them unclipped.
Tournament picks are the training rule: no decode time. Writes
runs/nanogpt/csa2_rule_table.csv (empty value: not applicable) and prints one
markdown table per panel. From the repo root:
    uv run python experiments/csa2/rule_table.py
"""

import csv
import dataclasses
import sys
from pathlib import Path

import scoreboard as sb

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))
import paired  # noqa: E402

OUT = sb.RUNS / "csa2_rule_table.csv"
BINS = ("0-128", "128-512", "512-1024", "1024-2048", "2048-2560", "2560-4096",
        "4096-8192")  # fmt: skip
FORMATS = {"bf16": "", "FP8": "fp8_", "FP4": "fp4_"}  # eval csv tag prefix
DECODE_LENGTHS = [4096 << i for i in range(9)]  # 4K .. 1M
LAYOUT = "V4.1 reference layout, summed query"
R4 = "V4.1 reference, entries of 4 tokens at layers 1-9, summed query"
PF = "parameter-free CSA2, V4.1 layout, summed query"
BLOCKS = ", block-mean picks, best {} blocks of 8"
SHARE = ", layer 9 reuses layer 5's picks"  # eval/variant_eval.py SHARE9
LOGGED = "Final validation loss (logged, lm_head unclipped)"
TOUR: tuple[str, str, str | None] = ("tournament (as trained)", "tournament", None)


def rules(layout: str, pools: tuple[int, ...]) -> list[tuple[str, str, str | None]]:
    """(picks label, eval tag, decode row) for tournament, exact and block mean."""
    return [
        TOUR,
        ("exact", "exact", f"{layout}, exact picks"),
        *((f"best {p} blocks of 8", f"block_mean_8x{p}", layout + BLOCKS.format(p))
          for p in pools),
    ]  # fmt: skip


Rule = tuple[str, str, str | None]  # picks label, eval tag, decode row
# scoreboard's C3S versions include block means: entries alone here (with_means)
C3S, C3S_FP4 = (dataclasses.replace(v, entries=v.entries[:2])
                for v in (sb.C3S, sb.C3S_FP4))  # fmt: skip
MODELS: list[tuple[str, sb.Version, sb.Version, str, list[Rule]]] = [
    # name, bf16 version, FP4 version, own tag, rules
    ("V4.1 reference", sb.REF, sb.REF_FP4, "indexer",
     [("its learned indexer", "indexer", sb.REF_DECODE)]),
    ("V4.1 reference trained with our tournament picks", sb.C2, sb.C2_FP4,
     "tournament", rules(LAYOUT, (256, 64))),
    ("V4.1 reference trained with our tournament picks, 4-token entries, smooth stage"
     " switches", C3S, C3S_FP4, "tournament", rules(R4, (64,))),
    ("Parameter-free V4.1 block, trained with tournament picks", sb.C5, sb.C5_FP4,
     "tournament", [*rules(PF, (256, 64)),
                    ("best 256 blocks of 8" + SHARE, "share9_block_mean_8x256",
                     PF + BLOCKS.format(256) + SHARE)]),
    ("Parameter-free V4.1 block with a fixed sink (ln 128), trained with tournament"
     " picks", sb.C6, sb.C6_FP4, "tournament", rules(PF, (256, 64))),
]  # fmt: skip


def eval_rows(run: str, tag: str) -> dict[str, float]:
    """Eval loss by position range ("0-8192": all positions)."""
    with (sb.EVAL / f"rule_{run[:8]}_{tag}.csv").open() as f:
        return {f"{r['start']}-{r['end']}": float(r["loss"]) for r in csv.DictReader(f)}


def fp8(v: sb.Version) -> sb.Version:
    """v's caches as plain E4M3: 1 byte per bf16 value."""
    entries = tuple((m, size // 2) for m, size in v.entries)
    return dataclasses.replace(v, window_bytes=v.window_bytes // 2, entries=entries)


def with_means(v: sb.Version) -> sb.Version:
    """v plus the block-mean rules' cache: one mean per 8 entries, same format."""
    means = tuple((m * 8, size) for m, size in v.entries)
    return dataclasses.replace(v, entries=v.entries + means)


def decode() -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for name in ("decode_pf_share", "decode_ref_picks"):  # same rows: ref_picks'
        with (sb.RUNS / f"long_context/{name}.csv").open() as f:
            out |= {
                (r["version"], int(r["n"])): float(r["us_per_token"])
                for r in csv.DictReader(f)
                if r["part"] == "Picks + attention"
            }
    return out


def row(
    v: sb.Version, v4: sb.Version, own: str, tag: str, dec: str | None, us: dict
) -> list[tuple[str, str, float | None]]:
    """(panel, stat, value) of one model under one decode rule."""
    run = v.run
    out: list[tuple[str, str, float | None]] = []
    log = sb.log_stats(run)
    out.append(("Loss", LOGGED, log["Final validation loss"]))
    tags = {f: f"{p}hclip_{tag}" for f, p in FORMATS.items()}  # queue_head_clip*.sh
    for f, t in tags.items():
        out.append(("Loss", f"Eval, {f} caches", eval_rows(run, t)["0-8192"]))
    bf16 = eval_rows(run, tags["bf16"])
    out += [("Loss", f"Positions {b}", bf16[b]) for b in BINS]
    for f, p in FORMATS.items():
        change: list[float | None] = [None, None]
        if tag != own:
            d = paired.deltas(run[:8], tags[f], f"{p}hclip_{own}")  # all, past 2048
            change = [d[0][2], max(x[2] for x in d[1:])]
        out.append(("Change vs trained rule", f"All positions, {f}", change[0]))
        out.append(("Change vs trained rule", f"Worst past 2048, {f}", change[1]))
    out.append(("Training", "Training time (h)", log["Training time (h)"]))
    out.append(("Training", "Peak memory (MiB)", log["Peak training memory (MiB)"]))
    for f, ver in (("bf16", v), ("FP8", fp8(v)), ("FP4", v4)):
        ver = with_means(ver) if "block_mean" in tag else ver
        for n in sb.LENGTHS:
            out.append(("KV cache (MiB)", f"{f}, {n // 1024}K", sb.cache_mib(ver, n)))
    for f, suffix in (("bf16", ""), ("FP8", ", FP8 caches")):
        for n in DECODE_LENGTHS:
            k = n // 1024
            name = f"{k}K" if k < 1024 else f"{k // 1024}M"
            value = us[dec + suffix, n] if dec else None
            out.append((f"Decode, {f} caches (us per token)", name, value))
    return out


def main() -> None:
    us = decode()
    table = []
    for name, v, v4, own, model_rules in MODELS:
        for picks, tag, dec in model_rules:
            for panel, stat, value in row(v, v4, own, tag, dec, us):
                table.append((name, picks, panel, stat, value))
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("model", "picks", "panel", "stat", "value"))
        w.writerows((*r[:4], "" if r[4] is None else f"{r[4]:.4f}") for r in table)
    print("wrote", OUT)
    for panel in dict.fromkeys(r[2] for r in table):
        stats = list(dict.fromkeys(r[3] for r in table if r[2] == panel))
        print(f"\n{panel}\n\n| Model | Picks | " + " | ".join(stats) + " |")
        print("|---" * (len(stats) + 2) + "|")
        for key in dict.fromkeys((r[0], r[1]) for r in table):
            cells = {r[3]: r[4] for r in table if (r[0], r[1]) == key and r[2] == panel}
            vals = ["–" if cells[s] is None else f"{cells[s]:.4g}" for s in stats]
            print(f"| {key[0]} | {key[1]} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
