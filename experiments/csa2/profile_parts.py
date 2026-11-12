"""Where training time goes: the torch.profiler steps of make_profile_script.py
(logs/profile_<version>.csv, one step per stage) summed by part, from kernel names
-> runs/nanogpt/training/csa2_profile.csv (version, stage, row, ms per step).
Prints each version's top kernels (calls, us per call) and the unmatched ones, to
check the name rules. Inductor's "## Call CompiledFxGraph" ranges are not kernels
(they span them) and are dropped. Record 91 runs most backward kernels inside the
model call, so phases are named by call, not by pass. Idle = wall minus kernels
(the profiled step syncs after each phase, so idle is an upper bound).

    uv run python experiments/csa2/profile_parts.py
"""

from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "runs" / "nanogpt" / "training" / "csa2_profile.csv"
VERSIONS = {
    "csa2": "CSA2 stripped, learned indexer",
    "a_tournament": "CSA2 stripped, tournament top 64 instead of the indexer",
    "b_exact": "CSA2 KV layout in stripped record 91, exact top 64 entries",
    "b_tournament": "CSA2 KV layout in stripped record 91, tournament top 64 entries",
    "b_route": "CSA2 KV layout in stripped record 91, routed channels, exact top 64",
    "csa2_ref": "CSA2, V4.1-Flash-1017 reference code",
    "csa2_ref_sync": "CSA2, V4.1-Flash-1017 reference code, 4096-query indexer chunks",
    "b_win1_pick4": "CSA2 KV layout in stripped record 91, exact, picks every 4 layers,"
    " 1 shared K=V window head",
}
STAGES = {6: 1, 16: 2, 26: 3}
WALL = "Profiled step, wall clock (profiler syncs inflate it)"
IDLE = "Profiled step, GPU idle (wall clock minus kernels)"
OURS = "_picks|_windows|_sortable|_rounds|_redo|_top_sort|_order|_bits|_floor"
OURS += "|_select|_scores"  # our pick kernels (Triton function names)
MM = "_tem_fused_(mm|addmm|bmm)"  # inductor matmul templates
PARTS = (  # part -> kernel name regex, first match wins
    ("Attention, flex", "flex_attention"),
    ("Picks and top-k", f"^({OURS})$|TopK|topk|[Rr]adix|bitonic|Sort"),
    ("Matrix multiplies", f"gemm|nvjet|cutlass|xmma|cublas|splitK|{MM}"),
    ("Optimizer, Newton-Schulz kernels", "XXT|XTX|ba_plus_cAA"),
    ("MLP up projection + ReLU squared, one fused kernel", "relu_square"),
    ("Loss, fused cross-entropy kernel", "ce_fwd_bwd"),
    ("Q, K norm and RoPE kernels", "qk_norm_rope|qkv_norm_rope"),
    ("Memory copies and fills", "^Memcpy|^Memset"),
    ("Gather, scatter and index kernels", "gather|scatter|index"),
    ("Other fused elementwise and reductions", "^triton_(poi|red|per)_"),
    ("Eager PyTorch elementwise kernels", "at::native::.*elementwise"),
)  # fmt: skip
PHASES = {
    "forward": "Inside the model call (forward, most of backward)",
    "backward": "Inside loss.backward()",
    "optimizer": "Optimizer step",
}


def part(kernel: pl.Expr) -> pl.Expr:
    """The part a kernel name belongs to ("Other kernels" if none matches)."""
    out = pl.lit("Other kernels")
    for label, pattern in reversed(PARTS):
        out = pl.when(kernel.str.contains(pattern)).then(pl.lit(label)).otherwise(out)
    return out


def rows(version: str, raw: pl.DataFrame) -> pl.DataFrame:
    """ms per step by row label, one stage per profiled step."""
    wall = raw.filter((pl.col("kernel") == "(wall)") & (pl.col("phase") == "step"))
    gpu = raw.filter(
        ~pl.col("kernel").str.starts_with("## ") & (pl.col("kernel") != "(wall)")
    )
    gpu = gpu.with_columns(part=part(pl.col("kernel")))
    ms = (pl.col("us").sum() / 1000).alias("ms")
    phase = pl.col("phase").replace_strict(PHASES)
    whole, busy = wall.group_by("step").agg(ms), gpu.group_by("step").agg(ms)
    idle = whole.join(busy, on="step").select(
        "step", ms=pl.col("ms") - pl.col("ms_right")
    )
    tables = [
        whole.with_columns(row=pl.lit(WALL)),
        busy.with_columns(row=pl.lit("GPU busy (all kernels)")),
        idle.with_columns(row=pl.lit(IDLE)),
        gpu.filter(pl.col("phase").is_in(list(PHASES)))
        .group_by("step", "phase")
        .agg(ms)
        .with_columns(row=phase + ", all kernels"),
        gpu.group_by("step", "part").agg(ms).rename({"part": "row"}),
    ]
    return pl.concat(t.select("step", "row", "ms") for t in tables).with_columns(
        version=pl.lit(VERSIONS[version]), stage=pl.col("step").replace_strict(STAGES)
    )


def main() -> None:
    tables = []
    for key in VERSIONS:
        path = HERE / "logs" / f"profile_{key}.csv"
        if not path.exists():
            print("missing", path.name)
            continue
        raw = pl.read_csv(path)
        tables.append(rows(key, raw))
        kernels = (
            raw.filter(~pl.col("kernel").str.starts_with("## "))
            .filter(pl.col("kernel") != "(wall)")
            .group_by("kernel")
            .agg(
                ms=pl.col("us").sum() / 3000,
                calls=pl.len() / 3,
                us_per_call=pl.col("us").mean(),
            )
            .with_columns(part=part(pl.col("kernel")))
            .sort("ms", descending=True)
        )
        with pl.Config(tbl_rows=40, fmt_str_lengths=70, tbl_width_chars=160):
            print(f"\n{VERSIONS[key]}: top kernels, ms per step (mean of 3 stages)")
            print(kernels.head(25))
            print("unmatched:")
            print(kernels.filter(pl.col("part") == "Other kernels").head(10))
    order = [
        WALL,
        "GPU busy (all kernels)",
        IDLE,
        *(f"{x}, all kernels" for x in PHASES.values()),
        *(label for label, _ in PARTS),
        "Other kernels",
    ]
    table = pl.concat(tables).select("version", "stage", "row", "ms")
    grid = (  # every row for every version and stage, 0 where nothing ran
        table.select("version").unique(maintain_order=True)
        .join(pl.DataFrame({"stage": list(STAGES.values())}), how="cross")
        .join(pl.DataFrame({"row": order}), how="cross")
    )  # fmt: skip
    table = grid.join(table, on=["version", "stage", "row"], how="left").fill_null(0)
    table.write_csv(OUT)
    mean = table.group_by("version", "row", maintain_order=True).agg(
        pl.col("ms").mean()
    )
    with pl.Config(
        tbl_rows=60, fmt_str_lengths=50, tbl_width_chars=250, float_precision=1
    ):
        print("ms per step, mean of stages 1-3")
        print(mean.pivot("version", index="row", values="ms"))
        print(
            table.filter(pl.col("row") == WALL).pivot(
                "version", index="stage", values="ms"
            )
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
