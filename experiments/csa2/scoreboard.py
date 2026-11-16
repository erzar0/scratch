"""CSA2 vs our versions on every stat that must not get worse: final validation loss,
training time and peak memory (record 91 run logs), KV cache for decode at batch 1
(bytes from each layout) and decode time per token (picks + attention, Triton,
runs/nanogpt/long_context/decode_heat.csv, decode_block.csv, decode_ref_picks.csv).
Writes runs/nanogpt/csa2_scoreboard.csv with each value and its ratio to CSA2 (below 1 =
better than CSA2). Record 91's own window keys are cached at their 96 real channels (the
rest is zero padding). Eval loss also with the lm_head weights clipped to the training
head's FP8 range (eval/variant_eval.py HEAD_CLIP), where measured: the logged validation
loss uses them unclipped. A version's decode time is the fastest of its listed decode
rules at each length; listed rules pass the loss gate at eval (EXACT;
experiments/csa2/eval/rule_eval.py). Also writes runs/nanogpt/csa2_positions.csv:
validation loss by position (rule_eval.py, exact picks; CSA2 with its own indexer,
logs/rule_296a91a4_indexer.csv, recovered from that eval's printed output) and
runs/nanogpt/csa2_rules.csv: run 7's loss by position under each eval rule (picks shared
across groups, block prefilter; logs/rule_1b0702e3_<tag>.csv, oldest first) and
runs/nanogpt/csa2_ref_rules.csv: the same for the V4.1 reference port (65614cd9) with
our summed-query picks in place of its indexer (eval/ref_rule_eval.py).
    python experiments/csa2/scoreboard.py
"""

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

LOGS = Path.home() / "repos/modded-nanogpt-r91/logs"
RUNS = Path(__file__).resolve().parents[2] / "runs/nanogpt"
EVAL = Path(__file__).resolve().parent / "logs"
LAYERS, GROUPS, WINDOW, DIM, INDEX_DIM, HEADS, QK = 10, 3, 128, 128, 32, 6, 96
LENGTHS = (8192, 32768, 131072, 524288)


@dataclass(frozen=True)
class Version:
    """One trained model: its run, cache layout and decode row."""

    name: str
    run: str
    window_bytes: int  # per window token per layer
    entries: tuple[tuple[int, int], ...]  # per KV owner: tokens, bytes per entry
    decode: tuple[str, ...]  # decode_heat.csv versions (decode rules)
    rule: str = "exact"  # eval: logs/rule_<run[:8]>_<rule>.csv


REF_DECODE = "CSA2, V4.1 reference layout (TensorRT-LLM decode)"  # bench/decode_block
REF = Version(
    "CSA2, V4.1-Flash-1017 reference code",
    "65614cd9-0fdb-4adc-b897-1294ab8caf74",
    DIM * 2,  # one K = V head
    ((2, (DIM + INDEX_DIM) * 2), (1, (DIM + INDEX_DIM) * 2)),  # ratio 2, ratio 1 owner
    (REF_DECODE,),
    "indexer",
)
# TensorRT-LLM's cache rows (quantization.py): entry 72 B (FP4 + E4M3 scale per 16) +
# index key 17 B (FP4 + UE8M0 per 32); window 132 B (FP8 + UE8M0 per 32). Decode not
# timed: Triton on this GPU (SM89) has no FP4 type
REF_FP4 = Version(
    "CSA2, V4.1-Flash-1017 reference code, TensorRT-LLM FP4 caches",
    REF.run, 132, ((2, 72 + 17), (1, 72 + 17)), (), "indexer_fp4",
)  # fmt: skip
# round 4 (queue_round4.sh): the reference trained with our summed-query picks in
# place of its indexer (src/nanogpt/csa2_ours.py); no index keys. Block-mean 64
# blocks fails the gate on these runs (+0.03 at 4096-8192): decode by exact picks
OURS_REF = "V4.1 reference, trained with our"
OURS_REF_EXACT = "V4.1 reference layout, summed query, exact picks"  # decode row
C1 = Version(
    f"{OURS_REF} exact picks",
    "8ee7cd76-cdd7-4e9e-8fc5-356b9a9cc5ec",
    DIM * 2,
    ((2, DIM * 2), (1, DIM * 2)),
    (OURS_REF_EXACT,),
)
C1_FP4 = Version(
    f"{OURS_REF} exact picks, FP4 entries", C1.run, 132, ((2, 72), (1, 72)), (),
    "fp4_exact",
)  # fmt: skip
C2 = Version(
    f"{OURS_REF} tournament picks",
    "7f71fcf2-d7fc-49e7-9fc8-559c821b4a58",
    DIM * 2,
    ((2, DIM * 2), (1, DIM * 2)),
    # exact at eval: -0.0014, passes; block mean 256 blocks of 8: +0.0031 at
    # 4096-8192, passes (64 and 128 blocks fail)
    (
        OURS_REF_EXACT,
        "V4.1 reference layout, summed query, block-mean picks, best 256 blocks of 8",
    ),
    "tournament",
)
C2_FP4 = Version(
    f"{OURS_REF} tournament picks, FP4 entries", C2.run, 132, ((2, 72), (1, 72)), (),
    "fp4_tournament",
)  # fmt: skip
# C3: entries of 4 tokens at layers 1-9; block-mean 64 blocks of 8 passes (-0.0037):
# its means (one per 8 entries, both sources) are cached too
R4 = "V4.1 reference, entries of 4 tokens at layers 1-9"  # decode_ref_picks.py rows
C3 = Version(
    f"{OURS_REF} tournament picks, 4-token entries",
    "87d9531c-3007-4568-ac42-855982849471",
    DIM * 2,
    ((4, DIM * 2),) * 2 + ((32, DIM * 2),) * 2,
    (f"{R4}, summed query, exact picks",
     f"{R4}, summed query, block-mean picks, best 64 blocks of 8",
     f"{R4}, summed query, block-mean picks, best 64 blocks of 8, one launch"),
    "tournament",
)  # fmt: skip
C3_FP4 = Version(
    f"{C3.name}, FP4 entries, FP8 means", C3.run, 132,
    ((4, 72),) * 2 + ((32, 132),) * 2, (), "fp4_tournament",
)  # fmt: skip
# C3 retrained with the stage-switch fix (LR ramp at stage starts); block mean 64
# blocks of 8 passes (-0.0055 at 4096-8192), FP4 too
C3S = Version(f"{C3.name}, smooth stage switches",
              "9624b171-a72f-43a2-b4c5-223e1b41a764", C3.window_bytes, C3.entries,
              C3.decode, "tournament")  # fmt: skip
C3S_FP4 = Version(f"{C3S.name}, FP4 entries, FP8 means", C3S.run, 132,
                  C3_FP4.entries, (), "fp4_tournament")  # fmt: skip
# round 5: parameter-free block (src/nanogpt/csa2_pf.py): C2's layout, entries = mean
# of the layer's own kv, no compressor, indexer or sink
PF = "parameter-free CSA2, V4.1 layout, summed query"  # decode_ref_picks.py rows
C5 = Version(
    "Parameter-free V4.1 block, tournament picks",
    "fa3176f4-4264-4efe-accb-b49d7c9d7d9c",
    DIM * 2,
    ((2, DIM * 2), (1, DIM * 2)),
    # gates vs tournament as C2's: exact -0.0016, block mean 256 blocks of 8 -0.0002 at
    # 4096-8192 pass; 64 blocks +0.028 fails
    (f"{PF}, exact picks", f"{PF}, block-mean picks, best 256 blocks of 8"),
    "tournament",
)
C5_FP4 = Version(
    f"{C5.name}, FP4 entries", C5.run, 132, ((2, 72), (1, 72)), (), "fp4_tournament",
)  # fmt: skip
# round 6: C5 with a constant sink logit ln 128 in every head, no parameter
# (src/nanogpt/csa2_pf_sink.py); decode = C5's (a constant sink costs what none does)
C6 = replace(
    C5, name="Parameter-free V4.1 block, fixed sink ln 128, tournament picks",
    run="5be6bb81-dfbf-4690-a598-e1422bbbe1e1",
)  # fmt: skip
C6_FP4 = replace(C5_FP4, name=f"{C6.name}, FP4 entries", run=C6.run)
CSA2 = Version(
    "CSA2, our port of the V4.1 report",
    "296a91a4-55af-4b0b-be1a-3da00a57f95d",
    DIM * 2,  # one K = V head
    ((4, (DIM + INDEX_DIM) * 2),) * GROUPS,  # main KV + indexer K, 3 groups
    ("CSA2, learned indexer",),
    "indexer",
)
OURS_WINDOW = HEADS * (QK + DIM) * 2  # record 91's own K and V, 6 heads
MQA_WINDOW = (QK + DIM) * 2  # one K and one V head
KV_WINDOW = DIM * 2  # one shared K=V vector (csa2_window), per group of heads
OURS_ENTRIES = ((4, DIM * 2),) * GROUPS  # one K=V entry per 4 tokens, 3 groups
LAYOUT = "CSA2 KV layout in stripped record 91"
EXACT = ("Ours, exact", "Ours, prefilter keeps 512")
# prefilter listed only where it passes the gate (loss by position, paired per token:
# within 0.005 of exact over all positions and 0.01 in every bin past 2048)
ROUTED4 = "Ours, routed, block-major, picks every 4 layers"
WIN = {1: "1 shared K=V window head", 2: "2 shared K=V window heads",
       6: "per-head K=V window"}  # fmt: skip
VERSIONS = (  # the first is the baseline of every ratio
    REF,
    REF_FP4,
    C1,
    C1_FP4,
    C2,
    C2_FP4,
    C3,
    C3_FP4,
    C3S,
    C3S_FP4,
    C5,
    C5_FP4,
    C6,
    C6_FP4,
    CSA2,
    Version(f"{LAYOUT}, exact top 64 entries",
            "5f0cf619-be55-44ea-ad17-e182a1b0858a", OURS_WINDOW, OURS_ENTRIES, EXACT),
    Version(f"{LAYOUT}, tournament top 64 entries",
            "f7f5027f-86ca-47c3-8fc0-4b4c0447ba95", OURS_WINDOW, OURS_ENTRIES,
            ("Ours, exact",)),  # prefilter: +0.011 at 4096-8192
    Version(f"{LAYOUT}, MQA window, XSA on mean value, tournament top 64 entries",
            "60ac1e12-d0a5-4c6a-b660-8fc11d38fca8", MQA_WINDOW, OURS_ENTRIES, ()),
    Version(f"{LAYOUT}, MQA window, tournament top 64 entries",
            "17d3ab72-460c-4128-8655-e2cc8a5a6745", MQA_WINDOW, OURS_ENTRIES, ()),
    Version(f"{LAYOUT}, exact, {WIN[1]}",
            "d3c62030-839b-46fa-9cc3-4434d250d0fd", KV_WINDOW, OURS_ENTRIES,
            (f"Ours, exact, {WIN[1]}",)),  # prefilter: +0.032 at 4096-8192
    Version(f"{LAYOUT}, exact, {WIN[2]}",
            "a5c3510a-b8ee-4ed4-a36c-fb2d28d59cba", 2 * KV_WINDOW, OURS_ENTRIES,
            (f"Ours, exact, {WIN[2]}",)),
    Version(f"{LAYOUT}, exact, {WIN[6]}",
            "3f7b047d-2d9b-4349-ac8c-d42aa50ea257", 6 * KV_WINDOW, OURS_ENTRIES,
            (f"Ours, exact, {WIN[6]}",)),
    Version(f"{LAYOUT}, exact, picks every 4 layers",
            "be093c56-fd99-48c5-adc6-59a8421f8d2c", OURS_WINDOW, OURS_ENTRIES,
            ("Ours, exact, picks every 4 layers",)),
    # entries stored FP8 e4m3 at eval (b/make_fp8_script.py all; gate +0.0001); decode
    # not timed in FP8
    Version(f"{LAYOUT}, exact, picks every 4 layers, FP8 entries",
            "be093c56-fd99-48c5-adc6-59a8421f8d2c", OURS_WINDOW,
            ((4, DIM),) * GROUPS, (), "fp8all_exact"),
    Version(f"{LAYOUT}, routed 2 of 6 blocks, picks every 4 layers",
            "40c90993-6932-4f96-bd05-74cae7ca6cbc", OURS_WINDOW, OURS_ENTRIES,
            (ROUTED4,)),
    Version(f"{LAYOUT}, routed, picks every 4 layers, {WIN[1]}",
            "7fbb74fb-5f88-45be-be3c-fe4aaf590e59", KV_WINDOW, OURS_ENTRIES,
            (f"{ROUTED4}, {WIN[1]}",)),
    Version(f"{LAYOUT}, routed, picks every 4 layers, {WIN[1]}, seed 2",
            "0677f733-6c4d-4227-93d3-230ee74e41e8", KV_WINDOW, OURS_ENTRIES,
            (f"{ROUTED4}, {WIN[1]}",)),
    Version(f"{LAYOUT}, exact, picks every 4 layers, {WIN[1]}",
            "1b0702e3-0cc6-43cb-8fc1-a133899b90d0", KV_WINDOW, OURS_ENTRIES,
            (f"Ours, exact, picks every 4 layers, {WIN[1]}",  # prefilter: +0.016
             f"Ours, exact, picks at layers 0, 4, {WIN[1]}",  # loss -0.001
             # loss -0.0008 (rows of 8192 tokens: 512 of at most 2048 entries kept)
             f"Ours, block mean, 64 blocks of 8, picks at layers 0, 4, {WIN[1]}")),
    Version(f"{LAYOUT}, exact, picks at layers 0, 4, {WIN[1]}",
            "d9c7a25f-854f-4d35-b381-8384016ae22b", KV_WINDOW, OURS_ENTRIES,
            (f"Ours, exact, picks at layers 0, 4, {WIN[1]}",)),
)  # fmt: skip


def log_stats(run: str) -> dict[str, float]:
    """Final val loss, training hours and peak MiB from a record 91 log."""
    text = (LOGS / f"{run}.txt").read_text()
    loss = re.findall(r"val_loss:([\d.]+) train_time:(\d+)ms", text)[-1]
    peak = re.findall(r"peak memory allocated: (\d+) MiB", text)[-1]
    return {
        "Final validation loss": float(loss[0]),
        "Training time (h)": int(loss[1]) / 3.6e6,
        "Peak training memory (MiB)": float(peak),
    }


def cache_mib(v: Version, n: int) -> float:
    """Decode KV cache at n tokens: each layer's window, each KV owner's entries."""
    window = LAYERS * min(n, WINDOW) * v.window_bytes
    return (window + sum(n // m * size for m, size in v.entries)) / 2**20


def decode_us() -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for name in ("decode_heat", "decode_block", "decode_ref_picks"):
        with (RUNS / f"long_context/{name}.csv").open() as f:
            out |= {
                (r["version"], int(r["n"])): float(r["us_per_token"])
                for r in csv.DictReader(f)
                if r["part"] == "Picks + attention"
            }
    return out


def eval_csv(v: Version) -> list[dict[str, str]]:
    with (EVAL / f"rule_{v.run[:8]}_{v.rule}.csv").open() as f:
        return list(csv.DictReader(f))


def hclip(rule: str) -> str:
    """Eval tag of rule with lm_head clipped as in training (queue_head_clip.sh)."""
    base = rule.replace("fp4_", "").replace("_fp4", "")
    return "fp4_" * (base != rule) + "hclip_" + base


def stats(v: Version, decode: dict[tuple[str, int], float]) -> dict[str, float]:
    out = log_stats(v.run)
    (row,) = [r for r in eval_csv(v) if (r["start"], r["end"]) == ("0", "8192")]
    out["Eval loss (8192-token rows, eval rule)"] = float(row["loss"])
    clipped = replace(v, rule=hclip(v.rule))
    if (EVAL / f"rule_{v.run[:8]}_{clipped.rule}.csv").exists():
        (row,) = [r for r in eval_csv(clipped) if r["start"] + r["end"] == "08192"]
        out["Eval loss, lm_head clipped as in training"] = float(row["loss"])
    for n in LENGTHS:
        out[f"KV cache at {n // 1024}K tokens (MiB)"] = cache_mib(v, n)
    for n in LENGTHS:
        times = [decode[rule, n] for rule in v.decode if (rule, n) in decode]
        if times:
            out[f"Decode at {n // 1024}K tokens (us per token)"] = min(times)
    return out


def positions(v: Version) -> list[tuple[str, str, str]]:
    """Validation loss by position bin (the all-positions row left out)."""
    return [
        (v.name, f"positions {r['start']}-{r['end']}", r["loss"])
        for r in eval_csv(v)
        if (r["start"], r["end"]) != ("0", "8192")
    ]


RUN7 = "1b0702e3"
RULES = {
    "exact": "exact picks every 4 layers (as trained)",
    "p512": "prefilter keeps 512, then exact",
    "share0": "exact picks at layer 0 only, all groups reuse them",
    "share04": "exact picks at layers 0, 4; group 3 reuses group 2's",
    "share08": "exact picks at layers 0, 8; group 2 reuses group 1's",
}
SUMMARY = {"mean": "mean key", "max": "max key", "minmax": "min/max bound",
           "meanstd": "mean + std"}  # fmt: skip
FP8 = {"fp8picks": "FP8 pick keys", "fp8all": "FP8 entries"}


def rule_name(tag: str) -> str:
    """Plain name of a rule_eval tag: logs/rule_1b0702e3_[share04_][fp8<x>_]<rule>."""
    shared, tag = tag.startswith("share04_"), tag.removeprefix("share04_")
    fp8 = FP8.get(tag.split("_")[0])
    tag = tag.split("_", 1)[1] if fp8 else tag
    if shared and tag == "exact":
        name, shared = RULES["share04"], False
    elif tag in RULES:
        name = RULES[tag]
    else:
        _, summary, sk = tag.split("_")  # block_<summary>_<size>x<keep>
        size, keep = map(int, sk.split("x"))
        name = f"block {SUMMARY[summary]}, best {keep} blocks of {size} entries"
    return name + ", picks at layers 0, 4" * shared + f", {fp8}" * bool(fp8)


REF_RUN = "65614cd9"  # V4.1 reference port; eval/ref_rule_eval.py tags
REF_RULES = {
    "indexer": "its learned indexer (as trained)",
    "summed_exact": "summed query, exact picks",
    "summed_block1": "summed query, block-mean picks, best 64 blocks of 8",
    "summed_block4": "summed query, block-mean picks, best 256 blocks of 8",
    "indexer_fp4": "its learned indexer, FP4 caches",
    "summed_exact_fp4": "summed query, exact picks, FP4 entries",
    "summed_block1_fp4": "summed query, block-mean picks, 64 blocks, FP4 entries",
    "indexer_yarn4": "its learned indexer, YaRN 4x at eval",
    "summed_exact_yarn4": "summed query, exact picks, YaRN 4x at eval",
    "summed_exact_top128": "summed query, exact picks, 128 per query",
}  # not plotted (far off the axis): YaRN with its attention scale (+0.17 to
# +0.18), layer 9 reusing layer 5's picks (+0.22)


def rules(
    run: str = RUN7, name: Callable[[str], str | None] = rule_name
) -> list[tuple[str, str, str]]:
    """A run's loss by position under each eval rule (no retraining), oldest first;
    tags name() maps to None are left out."""
    out, bins = [], {"08192": "all positions"}
    for path in sorted(EVAL.glob(f"rule_{run}_*.csv"), key=lambda p: p.stat().st_mtime):
        tag = path.stem.removeprefix(f"rule_{run}_")
        if tag.startswith("lean"):  # code check (same model, lean script), not a rule
            continue
        label = name(tag)
        if label is None:
            continue
        with path.open() as f:
            out += [
                (
                    label,
                    bins.get(
                        r["start"] + r["end"], f"positions {r['start']}-{r['end']}"
                    ),
                    r["loss"],
                )
                for r in csv.DictReader(f)
            ]
    return out


def main() -> None:
    decode = decode_us()
    base = stats(VERSIONS[0], decode)
    rows = []
    for v in VERSIONS:
        for stat, value in stats(v, decode).items():
            rows.append((v.name, stat, f"{value:.4f}", f"{value / base[stat]:.4f}"))
    out = RUNS / "csa2_scoreboard.csv"
    with out.open("w", newline="") as f:
        csv.writer(f).writerows([("version", "stat", "value", "ratio"), *rows])
    for row in rows:
        print(*row, sep=" | ")
    with (RUNS / "csa2_positions.csv").open("w", newline="") as f:
        loss = [row for v in VERSIONS for row in positions(v)]
        csv.writer(f).writerows([("version", "positions", "loss"), *loss])
    with (RUNS / "csa2_rules.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("rule", "positions", "loss"), *rules()])
    with (RUNS / "csa2_ref_rules.csv").open("w", newline="") as f:
        ref = rules(REF_RUN, REF_RULES.get)
        csv.writer(f).writerows([("rule", "positions", "loss"), *ref])


if __name__ == "__main__":
    main()
