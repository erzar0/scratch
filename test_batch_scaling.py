#!/usr/bin/env python3
"""Benchmark high batch size scaling: DeepSeek CSA2 vs Parameter-Free Attention.

Evaluates throughput (tokens/sec), latency (ms), and VRAM scaling across
batch sizes B in [1, 2, 4, 8, 16, 32, 64, 128] on NVIDIA H200.
Also triggers batched llama-bench runs to saturate GPU compute and power (700W).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


def _bootstrap_venv() -> None:
    """Switch to virtual environment if running under unconfigured python."""
    if not sys.argv or sys.argv[0] == "-c":
        return
    for venv_name in [".venv/bin/python", ".env/bin/python"]:
        cand = Path(__file__).resolve().parent / venv_name
        if cand.exists() and sys.executable != str(cand):
            try:
                import torch  # noqa: F401
            except ImportError:
                os.execv(str(cand), [str(cand), *sys.argv])


_bootstrap_venv()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from test_iq1m_h200 import (  # noqa: E402
    BLOCK_SIZE,
    DIM,
    HEAD_DIM,
    INDEX_HEAD_DIM,
    N_HEADS,
    WINDOW_SIZE,
    BenchmarkConfig,
    DeepSeekStandardAttention,
    ParameterFreeAttention,
    gather_hardware_metadata,
)

logger = logging.getLogger("iq1m_batch_benchmark")


@dataclass(frozen=True)
class BatchBenchmarkRow:
    """Performance metrics for a specific batch size configuration."""

    batch_size: int
    context_length: int
    ds_latency_ms: float
    pf_latency_ms: float
    speedup: float
    ds_throughput_tok_s: float
    pf_throughput_tok_s: float
    pf_vram_saved_pct: float
    cossim: float


def benchmark_batch(
    b: int,
    ctx: int,
    ds_module: DeepSeekStandardAttention,
    pf_module: ParameterFreeAttention,
    cfg: BenchmarkConfig,
    repeats: int = 20,
) -> BatchBenchmarkRow:
    """Benchmark one batch size configuration comparing DS and PF attention.

    Args:
        b: Batch size.
        ctx: Sequence context length.
        ds_module: DeepSeek standard attention module.
        pf_module: Parameter-free attention module.
        cfg: Benchmark configuration container.
        repeats: Measurement repetitions.

    Returns:
        Populated BatchBenchmarkRow dataclass.
    """
    device, dtype = cfg.device, cfg.dtype
    n_entries = max(128, int(ctx * 2.5 / 61 * 4))
    n_blocks = max(1, n_entries // BLOCK_SIZE)
    padded_len = n_blocks * BLOCK_SIZE

    # Input tensors batched by B
    q = torch.randn(b, N_HEADS, HEAD_DIM, device=device, dtype=dtype)
    x = torch.randn(b, DIM, device=device, dtype=dtype)
    window_kv = torch.randn(WINDOW_SIZE, HEAD_DIM, device=device, dtype=dtype)
    kv_cache = torch.randn(n_entries, HEAD_DIM, device=device, dtype=dtype)
    index_k = torch.randn(n_entries, INDEX_HEAD_DIM, device=device, dtype=dtype)
    block_means = kv_cache[:padded_len].view(n_blocks, BLOCK_SIZE, HEAD_DIM).mean(dim=1)

    # Warmup
    for _ in range(3):
        for i in range(b):
            _ = ds_module.decode_step(
                q[i : i + 1],
                x[i : i + 1],
                index_k,
                kv_cache,
                window_kv,
            )
            _ = pf_module.decode_step(
                q[i : i + 1],
                block_means,
                kv_cache,
                window_kv,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Measure DeepSeek Standard Attention
    t0 = time.perf_counter()
    out_ds_list = []
    for _ in range(repeats):
        for i in range(b):
            out_i = ds_module.decode_step(
                q[i : i + 1],
                x[i : i + 1],
                index_k,
                kv_cache,
                window_kv,
            )
            if len(out_ds_list) < b:
                assert isinstance(out_i, torch.Tensor)
                out_ds_list.append(out_i)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ds_total_s = (time.perf_counter() - t0) / repeats
    ds_lat_ms = ds_total_s * 1000.0

    # Measure Parameter-Free Attention
    t1 = time.perf_counter()
    out_pf_list = []
    for _ in range(repeats):
        for i in range(b):
            out_i = pf_module.decode_step(
                q[i : i + 1],
                block_means,
                kv_cache,
                window_kv,
            )
            if len(out_pf_list) < b:
                assert isinstance(out_i, torch.Tensor)
                out_pf_list.append(out_i)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pf_total_s = (time.perf_counter() - t1) / repeats
    pf_lat_ms = pf_total_s * 1000.0

    speedup = ds_lat_ms / max(1e-4, pf_lat_ms)
    ds_thru = float(b) / max(1e-6, ds_total_s)
    pf_thru = float(b) / max(1e-6, pf_total_s)

    # Compute CosSim between batch outputs
    out_ds = torch.cat(out_ds_list, dim=0)
    out_pf = torch.cat(out_pf_list, dim=0)
    sim_tensor = F.cosine_similarity(out_ds.mean(dim=1), out_pf.mean(dim=1), dim=-1)
    cossim = float(sim_tensor.mean().item())

    # Cache savings: (index_k + sink) eliminated
    ds_cache_bytes = n_entries * (HEAD_DIM + INDEX_HEAD_DIM)
    pf_cache_bytes = n_entries * HEAD_DIM + n_blocks * HEAD_DIM
    vram_saved = (1.0 - pf_cache_bytes / ds_cache_bytes) * 100.0

    return BatchBenchmarkRow(
        batch_size=b,
        context_length=ctx,
        ds_latency_ms=ds_lat_ms,
        pf_latency_ms=pf_lat_ms,
        speedup=speedup,
        ds_throughput_tok_s=ds_thru,
        pf_throughput_tok_s=pf_thru,
        pf_vram_saved_pct=vram_saved,
        cossim=cossim,
    )


def run_batched_llama_bench(
    bench_bin: Path,
    model_path: Path,
    batch_sizes: list[int],
    ngl: int = 99,
) -> None:
    """Run llama-bench with high batch sizes to measure hardware saturation.

    Args:
        bench_bin: Path to llama-bench executable.
        model_path: Path to GGUF model shard.
        batch_sizes: List of batch sizes to benchmark.
        ngl: GPU layers to offload.
    """
    if not bench_bin.exists():
        logger.warning("llama-bench binary not found at %s; skipping.", bench_bin)
        return

    b_str = ",".join(str(b) for b in batch_sizes)
    print("\n" + "=" * 94)
    print("RUNNING BATCHED LLAMA-BENCH ON NVIDIA H200 (HIGH COMPUTE SATURATION)")
    print(f"Batch Sizes: {b_str} | Layers Offloaded: {ngl}")
    print("=" * 94)

    cmd = [
        str(bench_bin),
        "-m",
        str(model_path),
        "-ngl",
        str(ngl),
        "-p",
        "512",
        "-n",
        "64",
        "-b",
        b_str,
    ]
    subprocess.run(cmd, check=False)


def main() -> None:
    """Entry point for high batch size benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark high batch size scaling: DeepSeek vs Parameter-Free.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Batch sizes to evaluate.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=32768,
        help="Context length for batched decode steps.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=25,
        help="Timing repetitions per batch size.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./model_iq1m"),
        help="Path where GGUF model shards reside.",
    )
    parser.add_argument(
        "--llama-dir",
        type=Path,
        default=Path("./llama.cpp"),
        help="Path to llama.cpp build directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs/benchmarks/iq1m_h200"),
        help="Directory to export benchmark results.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="deepseek-v4.1-batch-scaling",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--run-llama-bench",
        action="store_true",
        help="Also execute batched llama-bench sweep.",
    )
    args, _ = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfg = BenchmarkConfig(device=device, dtype=dtype)

    hw_info = gather_hardware_metadata(device)
    logger.info("Executing on %s (dtype: %s)", hw_info["gpu_name"], dtype)

    ds_attention = DeepSeekStandardAttention(device=device, dtype=dtype)
    pf_attention = ParameterFreeAttention(device=device, dtype=dtype)

    wandb_run = None
    if args.wandb and HAS_WANDB:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"batch_sweep_{ts}",
            config=vars(args),
        )

    print("\n" + "=" * 94)
    print(f"HIGH BATCH SIZE ATTENTION SCALING BENCHMARK on {hw_info['gpu_name']}")
    print(f"Context Length: {args.context} tokens | Repetitions: {args.repeats}")
    print("=" * 94)
    print(
        f"{'Batch':>7} | {'DS Lat (ms)':>12} | {'PF Lat (ms)':>12} | "
        f"{'Speedup':>9} | {'DS Tok/s':>11} | {'PF Tok/s':>11} | "
        f"{'CosSim':>7} | {'Saved':>6}"
    )
    print("-" * 94)

    rows: list[BatchBenchmarkRow] = []
    for b in args.batch_sizes:
        row = benchmark_batch(
            b,
            args.context,
            ds_attention,
            pf_attention,
            cfg,
            repeats=args.repeats,
        )
        rows.append(row)

        if wandb_run:
            wandb.log(
                {
                    "batch/size": row.batch_size,
                    "batch/ds_latency_ms": row.ds_latency_ms,
                    "batch/pf_latency_ms": row.pf_latency_ms,
                    "batch/speedup": row.speedup,
                    "batch/ds_throughput_tok_s": row.ds_throughput_tok_s,
                    "batch/pf_throughput_tok_s": row.pf_throughput_tok_s,
                    "batch/cossim": row.cossim,
                }
            )

        print(
            f"{row.batch_size:>7} | "
            f"{row.ds_latency_ms:>12.3f} | "
            f"{row.pf_latency_ms:>12.3f} | "
            f"{row.speedup:>8.2f}x | "
            f"{row.ds_throughput_tok_s:>11.1f} | "
            f"{row.pf_throughput_tok_s:>11.1f} | "
            f"{row.cossim:>7.4f} | "
            f"{row.pf_vram_saved_pct:>5.1f}%"
        )

    print("=" * 94 + "\n")

    if HAS_POLARS and rows:
        df = pl.DataFrame([asdict(r) for r in rows])
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        csv_path = args.output_dir / f"results_batch_scaling_{ts}.csv"
        df.write_csv(csv_path)
        logger.info("Saved batch scaling results to %s", csv_path)
        if wandb_run:
            cols = df.columns
            vals = df.to_numpy().tolist()
            wandb.log({"batch_scaling_table": wandb.Table(columns=cols, data=vals)})

    if args.run_llama_bench:
        bench_bin = args.llama_dir / "build" / "bin" / "llama-bench"
        if not bench_bin.exists():
            import shutil

            sys_bench = shutil.which("llama-bench")
            if sys_bench:
                bench_bin = Path(sys_bench)
        model_shard1 = (
            args.model_dir
            / "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf"
        )
        run_batched_llama_bench(
            bench_bin, model_shard1, [1, 4, 16, 64, 128, 512], ngl=99
        )

    if wandb_run:
        wandb_run.finish()

    logger.info("Batch size scaling benchmark completed successfully.")


if __name__ == "__main__":
    main()
