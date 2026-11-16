#!/usr/bin/env python3
"""Benchmark DeepSeek-V4.1-Flash-1017 IQ1_M attention on GPU or CPU.

Evaluates standard DeepSeek CSA2 attention against the parameter-free replacement
(summed-query with two-level block-mean prefiltering and sink removal) across
context lengths from 512 to 1M tokens.

Measures:
    1. Decode latency distribution (median, min, max, std us) via CUDA events or timer.
    2. Effective memory bandwidth (GB/s against peak hardware bandwidth).
    3. KV cache footprint (MiB per sequence in native FP8 and quantized FP4).
    4. Retrieval accuracy on salient associative needles and top-128 dense key recall.
    5. Relative Frobenius output error against full dense softmax attention.
    6. Extensive artifact export (JSON + CSV) and optional Weights & Biases logging.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torch.nn.functional as F

try:
    import orjson  # type: ignore

    def dump_json_file(data: dict[str, Any], path: Path) -> None:
        """Write dictionary to file using fast orjson serializer.

        Args:
            data: Data payload dictionary.
            path: Target file path.
        """
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

except ImportError:

    def dump_json_file(data: dict[str, Any], path: Path) -> None:
        """Write dictionary to file using standard json serializer.

        Args:
            data: Data payload dictionary.
            path: Target file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


try:
    import wandb  # type: ignore

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = logging.getLogger("iq1m_bench")

# Hardware & Model constants for DeepSeek-V4.1-Flash-1017
DIM: int = 5120
N_HEADS: int = 64
HEAD_DIM: int = 512
INDEX_N_HEADS: int = 32
INDEX_HEAD_DIM: int = 128
INDEX_TOPK: int = 512
WINDOW_SIZE: int = 128
BLOCK_SIZE: int = 8
N_LAYERS: int = 40
KV_SOURCE_LAYERS: tuple[int, ...] = (2, 8, 14, 20)
INDEX_SOURCE_LAYERS: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)


@dataclass(frozen=True)
class LatencyStats:
    """Latency distribution metrics in microseconds."""

    median_us: float
    min_us: float
    max_us: float
    std_us: float
    mean_us: float


@dataclass(frozen=True)
class BenchmarkRow:
    """Stores full benchmark metrics for a single context length."""

    context_length: int
    ds_latency_median_us: float
    ds_latency_min_us: float
    ds_latency_max_us: float
    ds_latency_std_us: float
    pf_latency_median_us: float
    pf_latency_min_us: float
    pf_latency_max_us: float
    pf_latency_std_us: float
    speedup: float
    ds_bandwidth_gb_s: float
    pf_bandwidth_gb_s: float
    ds_cache_fp8_mib: float
    pf_cache_fp8_mib: float
    cache_saved_pct: float
    ds_cache_fp4_mib: float
    pf_cache_fp4_mib: float
    ds_needle_recall: float
    pf_needle_recall: float
    ds_top128_recall: float
    pf_top128_recall: float
    ds_output_error: float
    pf_output_error: float


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration settings for running context benchmarks."""

    device: torch.device
    dtype: torch.dtype
    repeats: int = 20
    eval_accuracy: bool = True
    num_needles: int = 16


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure dual console and file logging.

    Args:
        output_dir: Directory where log files are stored.
        verbose: Enable debug logging level when True.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"benchmark_{timestamp}.log"

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Log file initialized at %s", log_file)


class WandbTracker:
    """Manages optional Weights & Biases logging."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        entity: str | None,
        run_name: str | None,
        config: dict[str, Any],
        mode: str = "auto",
    ) -> None:
        """Initialize tracker if enabled and package is available.

        Args:
            enabled: Whether wandb is active.
            project: Project name.
            entity: Optional entity/team.
            run_name: Optional custom run name.
            config: Initial run configuration metadata.
            mode: Wandb mode ('auto', 'online', 'offline', 'disabled').
        """
        self.active = False
        if not enabled or mode == "disabled":
            return

        if not HAS_WANDB:
            logger.warning("wandb requested but python package not installed.")
            return

        # Prompt for key if missing in interactive terminal
        if not os.environ.get("WANDB_API_KEY") and sys.stdin.isatty():
            try:
                import getpass

                print("\n" + "-" * 60)
                print("Weights & Biases Authentication")
                print("Get your API key at: https://wandb.ai/authorize")
                entered_key = getpass.getpass(
                    "Enter W&B API Key (or press Enter to run offline): "
                ).strip()
                print("-" * 60 + "\n")
                if entered_key:
                    os.environ["WANDB_API_KEY"] = entered_key
            except (EOFError, KeyboardInterrupt):
                pass

        try:
            has_key = bool(os.environ.get("WANDB_API_KEY"))
            resolved_mode = mode
            if mode == "auto":
                resolved_mode = "online" if has_key else "offline"

            if resolved_mode == "offline":
                logger.info(
                    "WANDB_API_KEY not found; logging in offline mode "
                    "(artifacts saved locally)."
                )
            else:
                logger.info("Initializing Weights & Biases in online mode...")

            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config,
                mode=resolved_mode,
                anonymous="allow"
                if resolved_mode == "online" and not has_key
                else None,
            )
            self.active = True
            logger.info(
                "Weights & Biases logging initialized successfully (%s mode).",
                resolved_mode,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize wandb: %s", exc)

    def log_metrics(self, row: BenchmarkRow) -> None:
        """Log per-context metrics to wandb.

        Args:
            row: Populated BenchmarkRow dataclass.
        """
        if not self.active:
            return

        wandb.log(
            {
                "context": row.context_length,
                "latency/ds_median_us": row.ds_latency_median_us,
                "latency/pf_median_us": row.pf_latency_median_us,
                "latency/speedup": row.speedup,
                "bandwidth/ds_gb_s": row.ds_bandwidth_gb_s,
                "bandwidth/pf_gb_s": row.pf_bandwidth_gb_s,
                "memory/ds_cache_fp8_mib": row.ds_cache_fp8_mib,
                "memory/pf_cache_fp8_mib": row.pf_cache_fp8_mib,
                "memory/cache_saved_pct": row.cache_saved_pct,
                "accuracy/ds_needle_recall": row.ds_needle_recall,
                "accuracy/pf_needle_recall": row.pf_needle_recall,
                "accuracy/ds_top128_recall": row.ds_top128_recall,
                "accuracy/pf_top128_recall": row.pf_top128_recall,
                "fidelity/ds_relative_error": row.ds_output_error,
                "fidelity/pf_relative_error": row.pf_output_error,
            }
        )

    def finish(self, summary_df: pl.DataFrame) -> None:
        """Upload summary table and finish run.

        Args:
            summary_df: Summary Polars DataFrame.
        """
        if not self.active:
            return

        try:
            table = wandb.Table(columns=summary_df.columns, data=summary_df.rows())
            wandb.log({"benchmark_summary_table": table})
            wandb.finish()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error finalizing wandb: %s", exc)


class DeepSeekStandardAttention(torch.nn.Module):
    """DeepSeek-V4.1-Flash-1017 standard CSA2 decode attention step.

    Simulates the learned indexer (32 heads of dim 128) scoring index keys,
    top-k candidate filtering, and attention with a learned sink.
    """

    def __init__(
        self, device: torch.device, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """Initialize projection weights and attention sink parameter.

        Args:
            device: Target torch device.
            dtype: Target compute dtype.
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(INDEX_HEAD_DIM)
        self.attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        self.w_qi = torch.nn.Parameter(
            torch.randn(INDEX_N_HEADS * INDEX_HEAD_DIM, DIM, device=device, dtype=dtype)
            * 0.02
        )
        self.attn_sink = torch.nn.Parameter(
            torch.zeros(N_HEADS, device=device, dtype=dtype)
        )

    @torch.no_grad()
    def decode_step(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
        index_k_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        window_kv: torch.Tensor,
        return_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Execute one decode step with standard DeepSeek CSA2.

        Args:
            q: Query heads tensor of shape [1, N_HEADS, HEAD_DIM].
            x: Hidden state tensor of shape [1, DIM].
            index_k_cache: Cached index keys of shape [N_entries, INDEX_HEAD_DIM].
            kv_cache: Cached compressed KV entries of shape [N_entries, HEAD_DIM].
            window_kv: Recent window KV entries of shape [WINDOW_SIZE, HEAD_DIM].
            return_indices: Whether to also return selected KV indices.

        Returns:
            Output tensor or tuple of (output tensor, selected index tensor).
        """
        q_idx = F.linear(x, self.w_qi).view(INDEX_N_HEADS, INDEX_HEAD_DIM)
        idx_scores = torch.matmul(q_idx, index_k_cache.mT) * self.scale
        agg_scores = idx_scores.sum(dim=0)

        k_val = min(INDEX_TOPK, agg_scores.shape[-1])
        topk_indices = torch.topk(agg_scores, k=k_val, dim=-1).indices
        gathered_kv = kv_cache[topk_indices]

        all_k = torch.cat([window_kv, gathered_kv], dim=0)
        logits = torch.einsum("bhd,sd->bhs", q, all_k) * self.attn_scale

        lse = torch.logsumexp(logits, dim=-1)
        sink_weight = torch.sigmoid(lse - self.attn_sink.view(1, -1))
        probs = (F.softmax(logits, dim=-1) * sink_weight.unsqueeze(-1)).to(all_k.dtype)

        out = torch.einsum("bhs,sd->bhd", probs, all_k)
        if return_indices:
            return out, topk_indices
        return out


class ParameterFreeAttention(torch.nn.Module):
    """Replaced Parameter-Free Attention for DeepSeek-V4.1-Flash-1017.

    Replaces learned indexer and sink with:
        1. Summed query: q_sum = sum_{h=1}^64 q_h in R^512.
        2. Two-level block-mean prefilter: score [N_entries // 8] block means.
        3. Exact scoring only on top candidate blocks.
        4. Unified softmax over window + picked entries (no sink).
    """

    def __init__(
        self, device: torch.device, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """Initialize parameter-free attention module.

        Args:
            device: Target torch device.
            dtype: Target compute dtype.
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.attn_scale = 1.0 / math.sqrt(HEAD_DIM)

    @torch.no_grad()
    def decode_step(
        self,
        q: torch.Tensor,
        block_means: torch.Tensor,
        kv_cache: torch.Tensor,
        window_kv: torch.Tensor,
        return_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Execute one decode step with parameter-free attention.

        Args:
            q: Query heads tensor of shape [1, N_HEADS, HEAD_DIM].
            block_means: Cached block mean keys of shape [N_blocks, HEAD_DIM].
            kv_cache: Cached KV entries of shape [N_entries, HEAD_DIM].
            window_kv: Recent window KV entries of shape [WINDOW_SIZE, HEAD_DIM].
            return_indices: Whether to also return selected KV indices.

        Returns:
            Output tensor or tuple of (output tensor, selected index tensor).
        """
        q_sum = q.sum(dim=1)

        n_blocks = block_means.shape[0]
        n_candidate_blocks = min(INDEX_TOPK // BLOCK_SIZE, n_blocks)
        block_scores = torch.matmul(q_sum, block_means.mT).squeeze(0)
        top_blocks = torch.topk(block_scores, k=n_candidate_blocks, dim=-1).indices

        offsets = torch.arange(BLOCK_SIZE, device=self.device)
        candidate_entry_indices = (top_blocks[:, None] * BLOCK_SIZE + offsets).flatten()
        candidate_kv = kv_cache[candidate_entry_indices]

        candidate_scores = torch.matmul(q_sum, candidate_kv.mT).squeeze(0)
        top_entries = torch.topk(
            candidate_scores, k=min(INDEX_TOPK, candidate_kv.shape[0]), dim=-1
        ).indices
        picked_kv = candidate_kv[top_entries]
        picked_indices = candidate_entry_indices[top_entries]

        all_k = torch.cat([window_kv, picked_kv], dim=0)
        logits = torch.einsum("bhd,sd->bhs", q, all_k) * self.attn_scale
        probs = F.softmax(logits, dim=-1).to(all_k.dtype)

        out = torch.einsum("bhs,sd->bhd", probs, all_k)
        if return_indices:
            return out, picked_indices
        return out


def measure_latency_stats(
    func: Callable[..., Any],
    *args: Any,
    device: torch.device,
    warmup: int = 10,
    repeats: int = 30,
) -> LatencyStats:
    """Measure function latency distribution in microseconds.

    Args:
        func: Callable function to time.
        *args: Function arguments.
        device: Torch device (cuda or cpu).
        warmup: Number of warmup iterations.
        repeats: Number of measured repetitions.

    Returns:
        Populated LatencyStats dataclass.
    """
    for _ in range(warmup):
        func(*args)

    latencies: list[float] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for i in range(repeats):
            start_events[i].record()
            func(*args)
            end_events[i].record()
        torch.cuda.synchronize(device)
        latencies = [
            start_events[i].elapsed_time(end_events[i]) * 1000.0 for i in range(repeats)
        ]
    else:
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            func(*args)
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1000.0)

    latencies.sort()
    n = len(latencies)
    median_val = latencies[n // 2]
    min_val = latencies[0]
    max_val = latencies[-1]
    mean_val = sum(latencies) / n
    variance = sum((x - mean_val) ** 2 for x in latencies) / max(1, n - 1)
    std_val = math.sqrt(variance)

    return LatencyStats(
        median_us=median_val,
        min_us=min_val,
        max_us=max_val,
        std_us=std_val,
        mean_us=mean_val,
    )


def gather_hardware_metadata(device: torch.device) -> dict[str, Any]:
    """Collect system and hardware profiling information.

    Args:
        device: Active torch device.

    Returns:
        Dictionary of hardware and environment metadata.
    """
    info: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "cpu_processor": platform.processor() or "x86_64",
        "cpu_count_logical": os.cpu_count() or 1,
        "device_type": device.type,
    }
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        info["gpu_name"] = prop.name
        info["gpu_total_memory_gb"] = round(prop.total_memory / (1024**3), 2)
        info["gpu_major"] = prop.major
        info["gpu_minor"] = prop.minor
        info["gpu_multi_processor_count"] = prop.multi_processor_count
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_name"] = "N/A (CPU execution)"
        info["gpu_total_memory_gb"] = 0.0

    return info


def inspect_gguf_shards(model_dir: Path) -> dict[str, Any]:
    """Inspect downloaded GGUF shard files and extract quantization types.

    Args:
        model_dir: Directory containing GGUF shards.

    Returns:
        Dictionary of shard file statuses and tensor precisions.
    """
    shard_info: dict[str, Any] = {"shards_found": [], "tensors": {}}
    expected_shards = [
        f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-0000{i}-of-00006.gguf"
        for i in range(1, 7)
    ]
    for name in expected_shards:
        path = model_dir / name
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            shard_info["shards_found"].append(
                {"name": name, "size_gb": round(size_gb, 2)}
            )

    first_shard = (
        model_dir / "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf"
    )
    if not first_shard.exists():
        return shard_info

    try:
        from gguf import GGUFReader  # type: ignore

        reader = GGUFReader(str(first_shard))
        quant_summary: dict[str, str] = {}
        for tensor in reader.tensors:
            tname = tensor.name
            if any(k in tname for k in ("attn", "indexer", "ffn.experts")):
                quant_summary[tname] = str(tensor.tensor_type.name)
        shard_info["tensors"] = quant_summary
    except ImportError:
        pass

    return shard_info


def evaluate_retrieval_metrics(
    ds_out_tuple: tuple[torch.Tensor, torch.Tensor],
    pf_out_tuple: tuple[torch.Tensor, torch.Tensor],
    out_dense: torch.Tensor,
    dense_top128: set[int],
    planted_set: set[int],
) -> tuple[float, float, float, float, float, float]:
    """Compute needle recall, top-128 recall, and relative Frobenius error.

    Args:
        ds_out_tuple: Standard output tensor and selected indices.
        pf_out_tuple: Parameter-free output tensor and selected indices.
        out_dense: Reference full dense attention output tensor.
        dense_top128: Set of ground-truth top-128 dense attention indices.
        planted_set: Set of planted salient associative needle indices.

    Returns:
        Tuple of (ds_needle_rec, pf_needle_rec, ds_top_rec, pf_top_rec, ds_err, pf_err).
    """
    out_ds, picked_ds = ds_out_tuple
    out_pf, picked_pf = pf_out_tuple

    ds_set = set(picked_ds.tolist())
    pf_set = set(picked_pf.tolist())

    n_needles = max(1, len(planted_set))
    n_top = max(1, len(dense_top128))

    ds_needle_rec = len(ds_set & planted_set) / n_needles
    pf_needle_rec = len(pf_set & planted_set) / n_needles
    ds_top_rec = len(ds_set & dense_top128) / n_top
    pf_top_rec = len(pf_set & dense_top128) / n_top

    norm_dense = max(1e-6, float(torch.norm(out_dense)))
    ds_err = float(torch.norm(out_ds - out_dense)) / norm_dense
    pf_err = float(torch.norm(out_pf - out_dense)) / norm_dense

    return (
        ds_needle_rec,
        pf_needle_rec,
        ds_top_rec,
        pf_top_rec,
        ds_err,
        pf_err,
    )


def benchmark_context(
    ctx_len: int,
    ds_module: DeepSeekStandardAttention,
    pf_module: ParameterFreeAttention,
    cfg: BenchmarkConfig,
) -> BenchmarkRow:
    """Benchmark a single context length for latency, bandwidth, and accuracy.

    Args:
        ctx_len: Sequence context length.
        ds_module: Standard attention module.
        pf_module: Parameter-free attention module.
        cfg: BenchmarkConfig configuration container.

    Returns:
        Populated BenchmarkRow dataclass.
    """
    device, dtype = cfg.device, cfg.dtype
    n_entries = int(ctx_len * 2.5 / N_LAYERS * 4)
    n_blocks = max(1, n_entries // BLOCK_SIZE)

    q = torch.randn(1, N_HEADS, HEAD_DIM, device=device, dtype=dtype)
    x = torch.randn(1, DIM, device=device, dtype=dtype)
    window_kv = torch.randn(WINDOW_SIZE, HEAD_DIM, device=device, dtype=dtype)
    kv_cache = torch.randn(n_entries, HEAD_DIM, device=device, dtype=dtype)

    target_dir = q.sum(dim=1)
    target_dir = target_dir / target_dir.norm(dim=-1, keepdim=True)
    needle_indices = torch.randint(0, n_entries, (cfg.num_needles,), device=device)
    needle_signal = 2.0 * math.sqrt(HEAD_DIM) * target_dir.to(dtype)
    kv_cache[needle_indices] += needle_signal

    index_k_cache = torch.randn(n_entries, INDEX_HEAD_DIM, device=device, dtype=dtype)
    q_idx = F.linear(x, ds_module.w_qi).view(INDEX_N_HEADS, INDEX_HEAD_DIM)
    q_idx_dir = q_idx.mean(dim=0, keepdim=True)
    q_idx_dir = q_idx_dir / q_idx_dir.norm(dim=-1, keepdim=True)
    index_k_cache[needle_indices] += (
        2.0 * math.sqrt(INDEX_HEAD_DIM) * q_idx_dir.to(dtype)
    )

    padded_len = n_blocks * BLOCK_SIZE
    block_means = kv_cache[:padded_len].view(n_blocks, BLOCK_SIZE, HEAD_DIM).mean(dim=1)

    warmup = 3 if device.type == "cpu" else 10
    ds_lat = measure_latency_stats(
        ds_module.decode_step,
        q,
        x,
        index_k_cache,
        kv_cache,
        window_kv,
        False,
        device=device,
        warmup=warmup,
        repeats=cfg.repeats,
    )
    pf_lat = measure_latency_stats(
        pf_module.decode_step,
        q,
        block_means,
        kv_cache,
        window_kv,
        False,
        device=device,
        warmup=warmup,
        repeats=cfg.repeats,
    )

    ds_bytes = (n_entries * INDEX_HEAD_DIM * 2) + (
        min(INDEX_TOPK, n_entries) * HEAD_DIM * 2
    )
    pf_bytes = (n_blocks * HEAD_DIM * 2) + (min(INDEX_TOPK, n_entries) * HEAD_DIM * 2)
    ds_bw = (ds_bytes / 1e9) / (ds_lat.median_us / 1e6)
    pf_bw = (pf_bytes / 1e9) / (pf_lat.median_us / 1e6)

    total_entries = int(2.5 * ctx_len)
    window_bytes = N_LAYERS * WINDOW_SIZE * HEAD_DIM
    total_ds_fp8 = window_bytes + total_entries * (HEAD_DIM + INDEX_HEAD_DIM)
    total_pf_fp8 = window_bytes + total_entries * HEAD_DIM
    ds_fp8_mib = total_ds_fp8 / (1024**2)
    pf_fp8_mib = total_pf_fp8 / (1024**2)
    saved_pct = (1.0 - (pf_fp8_mib / ds_fp8_mib)) * 100.0

    total_ds_fp4 = window_bytes + total_entries * (
        (HEAD_DIM // 2) + (INDEX_HEAD_DIM // 2)
    )
    total_pf_fp4 = window_bytes + total_entries * (HEAD_DIM // 2)
    ds_fp4_mib = total_ds_fp4 / (1024**2)
    pf_fp4_mib = total_pf_fp4 / (1024**2)

    speedup = ds_lat.median_us / max(pf_lat.median_us, 1e-6)

    ds_needle_rec, pf_needle_rec = 0.0, 0.0
    ds_top_rec, pf_top_rec = 0.0, 0.0
    ds_err, pf_err = 0.0, 0.0

    if cfg.eval_accuracy:
        all_dense_k = torch.cat([window_kv, kv_cache], dim=0)
        logits_dense = torch.einsum("bhd,sd->bhs", q, all_dense_k) * (
            1.0 / math.sqrt(HEAD_DIM)
        )
        probs_dense = F.softmax(logits_dense, dim=-1).to(all_dense_k.dtype)
        out_dense = torch.einsum("bhs,sd->bhd", probs_dense, all_dense_k)

        cache_probs = probs_dense[:, :, WINDOW_SIZE:].mean(dim=1).squeeze(0)
        dense_top128 = set(
            torch.topk(cache_probs, k=min(128, n_entries), dim=-1).indices.tolist()
        )
        planted_set = set(needle_indices.tolist())

        out_ds_res = ds_module.decode_step(
            q, x, index_k_cache, kv_cache, window_kv, True
        )
        out_pf_res = pf_module.decode_step(q, block_means, kv_cache, window_kv, True)
        assert isinstance(out_ds_res, tuple) and isinstance(out_pf_res, tuple)

        (
            ds_needle_rec,
            pf_needle_rec,
            ds_top_rec,
            pf_top_rec,
            ds_err,
            pf_err,
        ) = evaluate_retrieval_metrics(
            out_ds_res, out_pf_res, out_dense, dense_top128, planted_set
        )

    del q, x, window_kv, kv_cache, index_k_cache, block_means
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return BenchmarkRow(
        context_length=ctx_len,
        ds_latency_median_us=ds_lat.median_us,
        ds_latency_min_us=ds_lat.min_us,
        ds_latency_max_us=ds_lat.max_us,
        ds_latency_std_us=ds_lat.std_us,
        pf_latency_median_us=pf_lat.median_us,
        pf_latency_min_us=pf_lat.min_us,
        pf_latency_max_us=pf_lat.max_us,
        pf_latency_std_us=pf_lat.std_us,
        speedup=speedup,
        ds_bandwidth_gb_s=ds_bw,
        pf_bandwidth_gb_s=pf_bw,
        ds_cache_fp8_mib=ds_fp8_mib,
        pf_cache_fp8_mib=pf_fp8_mib,
        cache_saved_pct=saved_pct,
        ds_cache_fp4_mib=ds_fp4_mib,
        pf_cache_fp4_mib=pf_fp4_mib,
        ds_needle_recall=ds_needle_rec,
        pf_needle_recall=pf_needle_rec,
        ds_top128_recall=ds_top_rec,
        pf_top128_recall=pf_top_rec,
        ds_output_error=ds_err,
        pf_output_error=pf_err,
    )


def save_benchmark_artifacts(
    rows: list[BenchmarkRow],
    metadata: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist benchmark results as JSON and CSV artifacts.

    Args:
        rows: List of BenchmarkRow dataclasses.
        metadata: System and run configuration metadata.
        output_dir: Directory where artifacts are saved.

    Returns:
        Tuple of (json_path, csv_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"results_{timestamp}.json"
    csv_path = output_dir / f"results_{timestamp}.csv"

    dict_rows = [asdict(r) for r in rows]
    payload = {"metadata": metadata, "results": dict_rows}
    dump_json_file(payload, json_path)

    df = pl.DataFrame(dict_rows)
    df.write_csv(csv_path)

    logger.info("Saved JSON results to %s", json_path)
    logger.info("Saved CSV results to %s", csv_path)
    return json_path, csv_path


def main() -> None:
    """Entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek-V4.1-Flash-1017 IQ1_M attention on GPU or CPU."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./model_iq1m"),
        help="Path where GGUF shards are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs/benchmarks/iq1m_h200"),
        help="Directory to save logs and benchmark artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target device (auto detects CUDA or CPU).",
    )
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144],
        help="Context lengths to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=0,
        help="Timing repetitions (defaults to 15 on CPU, 30 on GPU).",
    )
    parser.add_argument(
        "--eval-accuracy",
        action="store_true",
        default=True,
        help="Evaluate needle recall, top-128 overlap, and relative output error.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log benchmark metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="deepseek-v4.1-csa2-vs-pf",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/organization.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="auto",
        choices=["auto", "online", "offline", "disabled"],
        help="Wandb run mode: 'auto', 'online', 'offline', or 'disabled'.",
    )
    args = parser.parse_args()

    setup_logging(args.output_dir, verbose=args.verbose)

    if args.device == "auto":
        dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            logger.error("CUDA requested but not available.")
            sys.exit(1)
        dev_str = "cuda:0"
    else:
        dev_str = "cpu"

    device = torch.device(dev_str)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    hw_info = gather_hardware_metadata(device)
    logger.info(
        "Executing on target device: %s (dtype: %s)", hw_info["gpu_name"], dtype
    )

    gguf_info = inspect_gguf_shards(args.model_dir)
    logger.info("GGUF shards found: %d", len(gguf_info["shards_found"]))
    for shard in gguf_info["shards_found"]:
        logger.info("  %s: %.2f GB", shard["name"], shard["size_gb"])

    metadata = {
        **hw_info,
        "model_architecture": {
            "name": "DeepSeek-V4.1-Flash-1017",
            "quantization": "GSQ-RCO-IQ1_M",
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
            "index_n_heads": INDEX_N_HEADS,
            "index_head_dim": INDEX_HEAD_DIM,
            "index_topk": INDEX_TOPK,
            "window_size": WINDOW_SIZE,
            "block_size": BLOCK_SIZE,
        },
        "gguf": gguf_info,
    }

    wandb_tracker = WandbTracker(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        config=metadata,
        mode=args.wandb_mode,
    )

    logger.info("Initializing Attention Modules...")
    ds_attention = DeepSeekStandardAttention(device=device, dtype=dtype)
    pf_attention = ParameterFreeAttention(device=device, dtype=dtype)

    repeats = args.repeats if args.repeats > 0 else (15 if device.type == "cpu" else 30)
    bench_cfg = BenchmarkConfig(
        device=device,
        dtype=dtype,
        repeats=repeats,
        eval_accuracy=args.eval_accuracy,
    )

    rows: list[BenchmarkRow] = []
    print("\n" + "=" * 94)
    print(
        f"DEEPSEEK-V4.1-FLASH-1017 IQ1_M ATTENTION BENCHMARK on {hw_info['gpu_name']}"
    )
    print("=" * 94)
    print(
        f"{'Context':>10} | {'DS Lat (µs)':>12} | {'PF Lat (µs)':>12} | {'Speedup':>9} "
        f"| {'DS Cache':>10} | {'PF Cache':>10} | {'Saved':>7} "
        f"| {'PF Rec':>8} | {'PF Err':>8}"
    )
    print("-" * 94)

    for ctx in args.contexts:
        logger.debug("Starting benchmark for context %d (repeats=%d)...", ctx, repeats)
        row = benchmark_context(ctx, ds_attention, pf_attention, bench_cfg)
        rows.append(row)
        wandb_tracker.log_metrics(row)

        ctx_str = (
            f"{ctx}" if ctx < 1024 else (f"{ctx // 1024}K" if ctx < 1048576 else "1M")
        )
        ds_lat_fmt = f"{row.ds_latency_median_us:.1f}±{row.ds_latency_std_us:.1f}"
        pf_lat_fmt = f"{row.pf_latency_median_us:.1f}±{row.pf_latency_std_us:.1f}"

        print(
            f"{ctx_str:>10} | "
            f"{ds_lat_fmt:>12} | "
            f"{pf_lat_fmt:>12} | "
            f"{row.speedup:>8.2f}x | "
            f"{row.ds_cache_fp8_mib:>8.1f} MB | "
            f"{row.pf_cache_fp8_mib:>8.1f} MB | "
            f"{row.cache_saved_pct:>6.1f}% | "
            f"{row.pf_needle_recall:>7.1%} | "
            f"{row.pf_output_error:>8.4f}"
        )

    print("=" * 94 + "\n")

    summary_df = pl.DataFrame([asdict(r) for r in rows])
    wandb_tracker.finish(summary_df)

    json_path, csv_path = save_benchmark_artifacts(rows, metadata, args.output_dir)
    logger.info("Artifact generation completed.")
    logger.info("JSON: %s", json_path)
    logger.info("CSV:  %s", csv_path)


if __name__ == "__main__":
    main()
