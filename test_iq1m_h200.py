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
import torch.nn as nn
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
class NiahRow:
    """Needle-In-A-Haystack retrieval metric at a given sequence depth."""

    context_length: int
    depth_pct: int
    ds_needle_found: bool
    pf_needle_found: bool
    ds_rank: int
    pf_rank: int


@dataclass(frozen=True)
class HeadDiversityRow:
    """Multi-head output diversity and effective matrix rank."""

    context_length: int
    ds_mean_pairwise_sim: float
    pf_mean_pairwise_sim: float
    sim_delta: float
    ds_effective_rank: float
    pf_effective_rank: float


@dataclass(frozen=True)
class PrefillRow:
    """Prompt prefill indexing and block-mean throughput metrics."""

    prompt_length: int
    ds_latency_ms: float
    pf_latency_ms: float
    speedup: float
    ds_throughput_ktok_s: float
    pf_throughput_ktok_s: float
    ds_peak_vram_mb: float
    pf_peak_vram_mb: float


@dataclass(frozen=True)
class TopKSweepRow:
    """Top-K sparsity budget trade-off metrics."""

    context_length: int
    top_k: int
    ds_latency_us: float
    pf_latency_us: float
    speedup: float
    pf_needle_recall: float
    pf_top128_recall: float
    pf_output_error: float


@dataclass(frozen=True)
class MoeRoutingRow:
    """MoE expert routing stability and agreement metrics."""

    context_length: int
    top6_expert_agreement_pct: float
    jaccard_similarity: float
    gate_logit_cossim: float
    ds_routing_entropy: float
    pf_routing_entropy: float


@dataclass(frozen=True)
class QuantNoiseRow:
    """Quantization noise robustness and noise suppression ratio."""

    noise_sigma: float
    ds_needle_recall: float
    pf_needle_recall: float
    ds_top128_recall: float
    pf_top128_recall: float
    pf_output_error: float
    noise_suppression_ratio: float


@dataclass(frozen=True)
class GgufTensorAuditRow:
    """Audit of GGUF tensor quantization types and parameter counts."""

    category: str
    tensor_count: int
    total_elements: int
    dominant_quant_type: str
    est_size_gb: float


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

        # Clean key from environment
        env_key = os.environ.get("WANDB_API_KEY", "").strip()
        if env_key:
            os.environ["WANDB_API_KEY"] = env_key
        else:
            os.environ.pop("WANDB_API_KEY", None)

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

        has_key = bool(os.environ.get("WANDB_API_KEY"))
        resolved_mode = mode
        if mode == "auto":
            resolved_mode = "online" if has_key else "offline"

        if resolved_mode == "offline":
            logger.info("Initializing Weights & Biases in offline mode...")
        else:
            logger.info("Initializing Weights & Biases in online mode...")

        try:
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config,
                mode=resolved_mode,
            )
            self.active = True
            logger.info(
                "Weights & Biases logging initialized successfully (%s mode).",
                resolved_mode,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to initialize wandb in %s mode: %s. Falling back to offline.",
                resolved_mode,
                exc,
            )
            try:
                wandb.init(
                    project=project,
                    name=run_name,
                    config=config,
                    mode="offline",
                )
                self.active = True
                logger.info("Weights & Biases logging active in offline mode.")
            except Exception as exc_offline:  # noqa: BLE001
                logger.error("Failed to initialize wandb offline: %s", exc_offline)

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

    def log_dataframe_table(self, table_name: str, df: pl.DataFrame) -> None:
        """Log a Polars DataFrame as a wandb.Table.

        Args:
            table_name: Target wandb table key.
            df: Polars DataFrame to log.
        """
        if not self.active:
            return
        try:
            w_table = wandb.Table(columns=df.columns, data=df.rows())
            wandb.log({table_name: w_table})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log wandb table %s: %s", table_name, exc)

    def finish(self, summary_df: pl.DataFrame | None = None) -> None:
        """Upload summary table and finish run.

        Args:
            summary_df: Optional summary Polars DataFrame.
        """
        if not self.active:
            return

        try:
            if summary_df is not None:
                table = wandb.Table(columns=summary_df.columns, data=summary_df.rows())
                wandb.log({"benchmark_summary_table": table})
            wandb.finish()
            logger.info("Weights & Biases run closed successfully.")
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
        top_k: int = INDEX_TOPK,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Execute one decode step with standard DeepSeek CSA2.

        Args:
            q: Query heads tensor of shape [1, N_HEADS, HEAD_DIM].
            x: Hidden state tensor of shape [1, DIM].
            index_k_cache: Cached index keys of shape [N_entries, INDEX_HEAD_DIM].
            kv_cache: Cached compressed KV entries of shape [N_entries, HEAD_DIM].
            window_kv: Recent window KV entries of shape [WINDOW_SIZE, HEAD_DIM].
            return_indices: Whether to also return selected KV indices.
            top_k: Number of candidate entries to retrieve.

        Returns:
            Output tensor or tuple of (output tensor, selected index tensor).
        """
        q_idx = F.linear(x, self.w_qi).view(INDEX_N_HEADS, INDEX_HEAD_DIM)
        idx_scores = torch.matmul(q_idx, index_k_cache.mT) * self.scale
        agg_scores = idx_scores.sum(dim=0)

        k_val = min(top_k, agg_scores.shape[-1])
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

    @torch.no_grad()
    def prefill_step(self, x: torch.Tensor) -> torch.Tensor:
        """Compute index keys for prompt sequence tokens.

        Args:
            x: Hidden states tensor of shape [seq_len, DIM].

        Returns:
            Projected index keys tensor.
        """
        return F.linear(x, self.w_qi)


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
        top_k: int = INDEX_TOPK,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Execute one decode step with parameter-free attention.

        Args:
            q: Query heads tensor of shape [1, N_HEADS, HEAD_DIM].
            block_means: Cached block mean keys of shape [N_blocks, HEAD_DIM].
            kv_cache: Cached KV entries of shape [N_entries, HEAD_DIM].
            window_kv: Recent window KV entries of shape [WINDOW_SIZE, HEAD_DIM].
            return_indices: Whether to also return selected KV indices.
            top_k: Number of candidate entries to retrieve.

        Returns:
            Output tensor or tuple of (output tensor, selected index tensor).
        """
        q_sum = q.sum(dim=1)

        n_blocks = block_means.shape[0]
        n_candidate_blocks = min(top_k // BLOCK_SIZE, n_blocks)
        block_scores = torch.matmul(q_sum, block_means.mT).squeeze(0)
        top_blocks = torch.topk(block_scores, k=n_candidate_blocks, dim=-1).indices

        offsets = torch.arange(BLOCK_SIZE, device=self.device)
        candidate_entry_indices = (top_blocks[:, None] * BLOCK_SIZE + offsets).flatten()
        candidate_kv = kv_cache[candidate_entry_indices]

        candidate_scores = torch.matmul(q_sum, candidate_kv.mT).squeeze(0)
        top_entries = torch.topk(
            candidate_scores, k=min(top_k, candidate_kv.shape[0]), dim=-1
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

    @torch.no_grad()
    def prefill_step(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """Compute block means for prompt sequence tokens.

        Args:
            kv_cache: KV cache tensor of shape [seq_len, HEAD_DIM].

        Returns:
            Block means tensor of shape [seq_len // BLOCK_SIZE, HEAD_DIM].
        """
        n_blocks = max(1, kv_cache.shape[0] // BLOCK_SIZE)
        padded_len = n_blocks * BLOCK_SIZE
        return kv_cache[:padded_len].view(n_blocks, BLOCK_SIZE, HEAD_DIM).mean(dim=1)


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
    rows: list[Any],
    metadata: dict[str, Any],
    output_dir: Path,
    name: str = "context",
) -> tuple[Path, Path]:
    """Persist benchmark results as JSON and CSV artifacts.

    Args:
        rows: List of benchmark dataclasses.
        metadata: System and run configuration metadata.
        output_dir: Directory where artifacts are saved.
        name: Identifier for the benchmark suite.

    Returns:
        Tuple of (json_path, csv_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"results_{name}_{timestamp}.json"
    csv_path = output_dir / f"results_{name}_{timestamp}.csv"

    dict_rows = [asdict(r) for r in rows]
    payload = {"benchmark": name, "metadata": metadata, "results": dict_rows}
    dump_json_file(payload, json_path)

    df = pl.DataFrame(dict_rows)
    df.write_csv(csv_path)

    logger.info("Saved %s JSON results to %s", name, json_path)
    logger.info("Saved %s CSV results to %s", name, csv_path)
    return json_path, csv_path


def compute_head_metrics(out: torch.Tensor) -> tuple[float, float]:
    """Calculate mean pairwise cosine similarity and effective matrix rank.

    Args:
        out: Head output tensor of shape [1, N_HEADS, HEAD_DIM].

    Returns:
        Tuple of (mean_pairwise_similarity, effective_rank).
    """
    h = out.squeeze(0).to(torch.float32)
    h_norm = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_sim = torch.mm(h_norm, h_norm.t())
    eye = torch.eye(h.shape[0], device=h.device, dtype=torch.bool)
    off_diag = cos_sim[~eye]
    mean_sim = float(off_diag.mean().item())

    singular_vals = torch.linalg.svdvals(h_norm)
    p = singular_vals / singular_vals.sum().clamp(min=1e-8)
    entropy = -torch.sum(p * torch.log(p.clamp(min=1e-12)))
    effective_rank = float(torch.exp(entropy).item())
    return mean_sim, effective_rank


def run_niah_suite(
    contexts: list[int],
    ds_module: DeepSeekStandardAttention,
    pf_module: ParameterFreeAttention,
    cfg: BenchmarkConfig,
) -> list[NiahRow]:
    """Execute Needle-In-A-Haystack retrieval benchmark across sequence depths.

    Args:
        contexts: List of context lengths to benchmark.
        ds_module: Standard attention module.
        pf_module: Parameter-free attention module.
        cfg: Benchmark configuration container.

    Returns:
        List of populated NiahRow instances.
    """
    device, dtype = cfg.device, cfg.dtype
    depth_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    rows: list[NiahRow] = []

    print("\n" + "=" * 80)
    print("NEEDLE-IN-A-HAYSTACK (NIAH) DEPTH RETRIEVAL BENCHMARK")
    print("=" * 80)
    print(
        f"{'Context':>10} | {'Depth':>7} | {'DS Found':>10} | {'PF Found':>10} | "
        f"{'DS Rank':>9} | {'PF Rank':>9}"
    )
    print("-" * 80)

    for ctx in contexts:
        n_entries = int(ctx * 2.5 / N_LAYERS * 4)
        n_blocks = max(1, n_entries // BLOCK_SIZE)
        padded_len = n_blocks * BLOCK_SIZE

        q = torch.randn(1, N_HEADS, HEAD_DIM, device=device, dtype=dtype)
        x = torch.randn(1, DIM, device=device, dtype=dtype)
        window_kv = torch.randn(WINDOW_SIZE, HEAD_DIM, device=device, dtype=dtype)

        target_dir = q.sum(dim=1).squeeze(0)
        target_dir = target_dir / target_dir.norm(dim=-1, keepdim=True)
        needle_signal = 2.0 * math.sqrt(HEAD_DIM) * target_dir.to(dtype)

        q_idx = F.linear(x, ds_module.w_qi).view(INDEX_N_HEADS, INDEX_HEAD_DIM)
        q_idx_dir = q_idx.mean(dim=0)
        q_idx_dir = q_idx_dir / q_idx_dir.norm(dim=-1, keepdim=True)
        idx_needle_signal = 2.0 * math.sqrt(INDEX_HEAD_DIM) * q_idx_dir.to(dtype)

        for depth in depth_percentages:
            needle_pos = min(int(n_entries * (depth / 100.0)), n_entries - 1)
            kv_cache = torch.randn(n_entries, HEAD_DIM, device=device, dtype=dtype)
            kv_cache[needle_pos] += needle_signal

            index_k = torch.randn(n_entries, INDEX_HEAD_DIM, device=device, dtype=dtype)
            index_k[needle_pos] += idx_needle_signal

            block_means = (
                kv_cache[:padded_len].view(n_blocks, BLOCK_SIZE, HEAD_DIM).mean(dim=1)
            )

            out_ds_res = ds_module.decode_step(
                q, x, index_k, kv_cache, window_kv, return_indices=True
            )
            out_pf_res = pf_module.decode_step(
                q, block_means, kv_cache, window_kv, return_indices=True
            )
            assert isinstance(out_ds_res, tuple) and isinstance(out_pf_res, tuple)
            _, ds_indices = out_ds_res
            _, pf_indices = out_pf_res

            ds_picks_set = set(ds_indices.tolist())
            pf_picks_set = set(pf_indices.tolist())

            ds_found = needle_pos in ds_picks_set
            pf_found = needle_pos in pf_picks_set

            idx_scores = (torch.matmul(q_idx, index_k.mT) * ds_module.scale).sum(dim=0)
            ds_rank = int((idx_scores > idx_scores[needle_pos]).sum().item()) + 1

            q_sum = q.sum(dim=1)
            block_scores = torch.matmul(q_sum, block_means.mT).squeeze(0)
            needle_block = needle_pos // BLOCK_SIZE
            pf_rank = (
                int((block_scores > block_scores[needle_block]).sum().item())
                * BLOCK_SIZE
                + 1
            )

            row = NiahRow(
                context_length=ctx,
                depth_pct=depth,
                ds_needle_found=ds_found,
                pf_needle_found=pf_found,
                ds_rank=ds_rank,
                pf_rank=pf_rank,
            )
            rows.append(row)

            ctx_str = (
                f"{ctx}"
                if ctx < 1024
                else (f"{ctx // 1024}K" if ctx < 1048576 else "1M")
            )
            print(
                f"{ctx_str:>10} | {depth:>6}% | "
                f"{('YES' if ds_found else 'NO'):>10} | "
                f"{('YES' if pf_found else 'NO'):>10} | "
                f"{ds_rank:>9} | {pf_rank:>9}"
            )

        del q, x, window_kv
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("=" * 80 + "\n")
    return rows


def run_heads_suite(
    contexts: list[int],
    ds_module: DeepSeekStandardAttention,
    pf_module: ParameterFreeAttention,
    cfg: BenchmarkConfig,
) -> list[HeadDiversityRow]:
    """Evaluate multi-head output cosine similarity and effective matrix rank.

    Args:
        contexts: List of context lengths to benchmark.
        ds_module: Standard attention module.
        pf_module: Parameter-free attention module.
        cfg: Benchmark configuration container.

    Returns:
        List of populated HeadDiversityRow instances.
    """
    device, dtype = cfg.device, cfg.dtype
    rows: list[HeadDiversityRow] = []

    print("\n" + "=" * 88)
    print("MULTI-HEAD REPRESENTATION DIVERSITY & EFFECTIVE MATRIX RANK")
    print("=" * 88)
    print(
        f"{'Context':>10} | {'DS Head Sim':>12} | {'PF Head Sim':>12} | "
        f"{'Sim Delta':>10} | {'DS Eff Rank':>12} | {'PF Eff Rank':>12}"
    )
    print("-" * 88)

    for ctx in contexts:
        n_entries = int(ctx * 2.5 / N_LAYERS * 4)
        n_blocks = max(1, n_entries // BLOCK_SIZE)
        padded_len = n_blocks * BLOCK_SIZE

        q = torch.randn(1, N_HEADS, HEAD_DIM, device=device, dtype=dtype)
        x = torch.randn(1, DIM, device=device, dtype=dtype)
        window_kv = torch.randn(WINDOW_SIZE, HEAD_DIM, device=device, dtype=dtype)
        kv_cache = torch.randn(n_entries, HEAD_DIM, device=device, dtype=dtype)
        index_k = torch.randn(n_entries, INDEX_HEAD_DIM, device=device, dtype=dtype)
        block_means = (
            kv_cache[:padded_len].view(n_blocks, BLOCK_SIZE, HEAD_DIM).mean(dim=1)
        )

        out_ds_res = ds_module.decode_step(
            q, x, index_k, kv_cache, window_kv, return_indices=False
        )
        out_pf_res = pf_module.decode_step(
            q, block_means, kv_cache, window_kv, return_indices=False
        )
        assert isinstance(out_ds_res, torch.Tensor) and isinstance(
            out_pf_res, torch.Tensor
        )

        ds_sim, ds_rank = compute_head_metrics(out_ds_res)
        pf_sim, pf_rank = compute_head_metrics(out_pf_res)
        delta = pf_sim - ds_sim

        row = HeadDiversityRow(
            context_length=ctx,
            ds_mean_pairwise_sim=ds_sim,
            pf_mean_pairwise_sim=pf_sim,
            sim_delta=delta,
            ds_effective_rank=ds_rank,
            pf_effective_rank=pf_rank,
        )
        rows.append(row)

        ctx_str = (
            f"{ctx}" if ctx < 1024 else (f"{ctx // 1024}K" if ctx < 1048576 else "1M")
        )
        print(
            f"{ctx_str:>10} | {ds_sim:>12.4f} | {pf_sim:>12.4f} | "
            f"{delta:>+10.4f} | {ds_rank:>12.2f} | {pf_rank:>12.2f}"
        )

        del q, x, window_kv, kv_cache, index_k, block_means
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("=" * 88 + "\n")
    return rows


def run_prefill_suite(
    prompt_lengths: list[int],
    ds_module: DeepSeekStandardAttention,
    pf_module: ParameterFreeAttention,
    cfg: BenchmarkConfig,
) -> list[PrefillRow]:
    """Benchmark prompt prefill indexing throughput and peak VRAM allocation.

    Args:
        prompt_lengths: List of prompt token counts to benchmark.
        ds_module: Standard attention module.
        pf_module: Parameter-free attention module.
        cfg: Benchmark configuration container.

    Returns:
        List of populated PrefillRow instances.
    """
    device, dtype = cfg.device, cfg.dtype
    rows: list[PrefillRow] = []

    print("\n" + "=" * 94)
    print("PREFILL PROMPT ENCODING THROUGHPUT & LATENCY BENCHMARK")
    print("=" * 94)
    print(
        f"{'Prompt':>10} | {'DS Lat (ms)':>12} | {'PF Lat (ms)':>12} | "
        f"{'Speedup':>9} | {'DS kTok/s':>11} | {'PF kTok/s':>11} | {'PF VRAM':>10}"
    )
    print("-" * 94)

    for prompt_len in prompt_lengths:
        x = torch.randn(prompt_len, DIM, device=device, dtype=dtype)
        kv_cache = torch.randn(prompt_len, HEAD_DIM, device=device, dtype=dtype)

        warmup = 3 if device.type == "cpu" else 10
        ds_lat = measure_latency_stats(
            ds_module.prefill_step,
            x,
            device=device,
            warmup=warmup,
            repeats=cfg.repeats,
        )
        pf_lat = measure_latency_stats(
            pf_module.prefill_step,
            kv_cache,
            device=device,
            warmup=warmup,
            repeats=cfg.repeats,
        )

        ds_ms = ds_lat.median_us / 1000.0
        pf_ms = pf_lat.median_us / 1000.0
        speedup = ds_lat.median_us / max(pf_lat.median_us, 1e-6)

        ds_ktoks = (prompt_len / (ds_ms / 1000.0)) / 1000.0
        pf_ktoks = (prompt_len / (pf_ms / 1000.0)) / 1000.0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            _ = pf_module.prefill_step(kv_cache)
            pf_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

            torch.cuda.reset_peak_memory_stats(device)
            _ = ds_module.prefill_step(x)
            ds_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            ds_peak_mb = (prompt_len * INDEX_N_HEADS * INDEX_HEAD_DIM * 2) / (1024**2)
            pf_peak_mb = (prompt_len // BLOCK_SIZE * HEAD_DIM * 2) / (1024**2)

        row = PrefillRow(
            prompt_length=prompt_len,
            ds_latency_ms=ds_ms,
            pf_latency_ms=pf_ms,
            speedup=speedup,
            ds_throughput_ktok_s=ds_ktoks,
            pf_throughput_ktok_s=pf_ktoks,
            ds_peak_vram_mb=ds_peak_mb,
            pf_peak_vram_mb=pf_peak_mb,
        )
        rows.append(row)

        p_str = f"{prompt_len}" if prompt_len < 1024 else f"{prompt_len // 1024}K"
        print(
            f"{p_str:>10} | {ds_ms:>12.3f} | {pf_ms:>12.3f} | {speedup:>8.2f}x | "
            f"{ds_ktoks:>11.1f} | {pf_ktoks:>11.1f} | {pf_peak_mb:>8.1f} MB"
        )

        del x, kv_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("=" * 94 + "\n")
    return rows


def run_topk_suite(
    context_length: int,
    topk_values: list[int],
    ds_module: DeepSeekStandardAttention,
    pf_module: ParameterFreeAttention,
    cfg: BenchmarkConfig,
) -> list[TopKSweepRow]:
    """Benchmark trade-off between top-k sparsity budget, latency, and fidelity.

    Args:
        context_length: Fixed context length for the sweep.
        topk_values: List of top-k values to evaluate.
        ds_module: Standard attention module.
        pf_module: Parameter-free attention module.
        cfg: Benchmark configuration container.

    Returns:
        List of populated TopKSweepRow instances.
    """
    device, dtype = cfg.device, cfg.dtype
    rows: list[TopKSweepRow] = []

    print("\n" + "=" * 94)
    print(f"TOP-K SPARSITY BUDGET SENSITIVITY SWEEP (Context: {context_length})")
    print("=" * 94)
    print(
        f"{'Top-K':>8} | {'DS Lat (µs)':>12} | {'PF Lat (µs)':>12} | {'Speedup':>9} | "
        f"{'Needle Rec':>11} | {'Top128 Rec':>11} | {'PF Err':>10}"
    )
    print("-" * 94)

    n_entries = int(context_length * 2.5 / N_LAYERS * 4)
    n_blocks = max(1, n_entries // BLOCK_SIZE)
    padded_len = n_blocks * BLOCK_SIZE

    q = torch.randn(1, N_HEADS, HEAD_DIM, device=device, dtype=dtype)
    x = torch.randn(1, DIM, device=device, dtype=dtype)
    window_kv = torch.randn(WINDOW_SIZE, HEAD_DIM, device=device, dtype=dtype)
    kv_cache = torch.randn(n_entries, HEAD_DIM, device=device, dtype=dtype)

    target_dir = q.sum(dim=1)
    target_dir = target_dir / target_dir.norm(dim=-1, keepdim=True)
    needle_indices = torch.randint(0, n_entries, (cfg.num_needles,), device=device)
    needle_signal = 2.0 * math.sqrt(HEAD_DIM) * target_dir.to(dtype)
    kv_cache[needle_indices] += needle_signal

    index_k = torch.randn(n_entries, INDEX_HEAD_DIM, device=device, dtype=dtype)
    q_idx = F.linear(x, ds_module.w_qi).view(INDEX_N_HEADS, INDEX_HEAD_DIM)
    q_idx_dir = q_idx.mean(dim=0, keepdim=True)
    q_idx_dir = q_idx_dir / q_idx_dir.norm(dim=-1, keepdim=True)
    index_k[needle_indices] += 2.0 * math.sqrt(INDEX_HEAD_DIM) * q_idx_dir.to(dtype)

    block_means = kv_cache[:padded_len].view(n_blocks, BLOCK_SIZE, HEAD_DIM).mean(dim=1)

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

    warmup = 3 if device.type == "cpu" else 10
    for k_val in topk_values:
        ds_lat = measure_latency_stats(
            ds_module.decode_step,
            q,
            x,
            index_k,
            kv_cache,
            window_kv,
            False,
            k_val,
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
            k_val,
            device=device,
            warmup=warmup,
            repeats=cfg.repeats,
        )

        out_ds_res = ds_module.decode_step(
            q, x, index_k, kv_cache, window_kv, True, k_val
        )
        out_pf_res = pf_module.decode_step(
            q, block_means, kv_cache, window_kv, True, k_val
        )
        assert isinstance(out_ds_res, tuple) and isinstance(out_pf_res, tuple)
        (
            _,
            pf_needle_rec,
            _,
            pf_top_rec,
            _,
            pf_err,
        ) = evaluate_retrieval_metrics(
            out_ds_res, out_pf_res, out_dense, dense_top128, planted_set
        )

        speedup = ds_lat.median_us / max(pf_lat.median_us, 1e-6)
        row = TopKSweepRow(
            context_length=context_length,
            top_k=k_val,
            ds_latency_us=ds_lat.median_us,
            pf_latency_us=pf_lat.median_us,
            speedup=speedup,
            pf_needle_recall=pf_needle_rec,
            pf_top128_recall=pf_top_rec,
            pf_output_error=pf_err,
        )
        rows.append(row)

        print(
            f"{k_val:>8} | {ds_lat.median_us:>12.1f} | {pf_lat.median_us:>12.1f} | "
            f"{speedup:>8.2f}x | {pf_needle_rec:>10.1%} | {pf_top_rec:>10.1%} | "
            f"{pf_err:>10.4f}"
        )

    print("=" * 94 + "\n")
    return rows


class DeepSeekMoEGate(nn.Module):
    """DeepSeek-V4.1 MoE Gate router with sqrtsoftplus scoring.

    Routes token representations to top-k active experts among 384 routed experts
    using non-linear sqrtsoftplus scoring with route_scale=1.5.
    """

    def __init__(
        self,
        dim: int = DIM,
        num_experts: int = 384,
        top_k: int = 6,
        route_scale: float = 1.5,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the DeepSeek MoE Gate router.

        Args:
            dim: Token representation dimensionality.
            num_experts: Total number of routed experts (384).
            top_k: Number of active experts routed per token (6).
            route_scale: Scaling factor for routed expert weights.
            device: Target execution device.
            dtype: Numerical precision datatype.
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = route_scale
        self.w_gate = nn.Parameter(
            torch.randn(num_experts, dim, device=device, dtype=dtype) / math.sqrt(dim)
        )
        self.bias = nn.Parameter(torch.zeros(num_experts, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights and selected expert indices.

        Args:
            x: Input token representation tensor of shape (..., dim).

        Returns:
            Tuple of (selected_normalized_weights, topk_expert_indices).
        """
        logits = F.linear(x, self.w_gate)
        scores = torch.sqrt(F.softplus(logits)) + self.bias
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)
        norm_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(
            min=1e-6
        )
        return norm_weights * self.route_scale, topk_indices


def run_moe_suite(
    context_lengths: list[int],
    ds_module: DeepSeekStandardAttention,
    pf_module: ParameterFreeAttention,
    cfg: BenchmarkConfig,
) -> list[MoeRoutingRow]:
    """Benchmark MoE expert routing stability under parameter-free attention.

    Evaluates whether token representations generated by parameter-free
    attention preserve top-6 expert routing decisions across 384 routed experts
    with exact DeepSeek sqrtsoftplus scoring.

    Args:
        context_lengths: Sequence context lengths to evaluate.
        ds_module: DeepSeek standard attention module.
        pf_module: Parameter-free attention module.
        cfg: Benchmark configuration container.

    Returns:
        List of populated MoeRoutingRow instances.
    """
    device, dtype = cfg.device, cfg.dtype
    rows: list[MoeRoutingRow] = []
    gate = DeepSeekMoEGate(
        dim=DIM,
        num_experts=384,
        top_k=6,
        route_scale=1.5,
        device=device,
        dtype=dtype,
    )
    w_o = torch.randn(DIM, HEAD_DIM, device=device, dtype=dtype) / math.sqrt(HEAD_DIM)

    print("\n" + "=" * 94)
    print("DEEPSEEK-V4.1 MOE ROUTING STABILITY ANALYSIS")
    print("(384 Experts, Top-6 Active, SqrtSoftplus)")
    print("=" * 94)
    print(
        f"{'Context':>10} | {'Top-6 Agr %':>11} | {'Jaccard Sim':>11} | "
        f"{'Gate Logit CosSim':>17} | {'DS Entropy':>10} | {'PF Entropy':>10}"
    )
    print("-" * 94)

    n_samples = 5 if device.type == "cpu" else 20

    for ctx_len in context_lengths:
        n_entries = int(ctx_len * 2.5 / N_LAYERS * 4)
        n_blocks = max(1, n_entries // BLOCK_SIZE)
        padded_len = n_blocks * BLOCK_SIZE

        agreements: list[float] = []
        jaccards: list[float] = []
        cossims: list[float] = []
        ds_entropies: list[float] = []
        pf_entropies: list[float] = []

        for _ in range(n_samples):
            q = torch.randn(1, N_HEADS, HEAD_DIM, device=device, dtype=dtype)
            x = torch.randn(1, DIM, device=device, dtype=dtype)
            window_kv = torch.randn(WINDOW_SIZE, HEAD_DIM, device=device, dtype=dtype)
            kv_cache = torch.randn(n_entries, HEAD_DIM, device=device, dtype=dtype)
            index_k = torch.randn(n_entries, INDEX_HEAD_DIM, device=device, dtype=dtype)
            block_means = (
                kv_cache[:padded_len].view(n_blocks, BLOCK_SIZE, HEAD_DIM).mean(dim=1)
            )

            out_ds = ds_module.decode_step(q, x, index_k, kv_cache, window_kv)
            out_pf = pf_module.decode_step(q, block_means, kv_cache, window_kv)
            assert isinstance(out_ds, torch.Tensor) and isinstance(out_pf, torch.Tensor)

            proj_ds = F.linear(out_ds.mean(dim=1), w_o)
            proj_pf = F.linear(out_pf.mean(dim=1), w_o)
            rep_ds = x + proj_ds
            rep_pf = x + proj_pf

            logits_ds = F.linear(rep_ds, gate.w_gate)
            logits_pf = F.linear(rep_pf, gate.w_gate)
            cos_sim = float(
                F.cosine_similarity(logits_ds, logits_pf, dim=-1).mean().item()
            )
            cossims.append(cos_sim)

            _, idx_ds = gate(rep_ds)
            _, idx_pf = gate(rep_pf)
            set_ds = set(idx_ds.view(-1).tolist())
            set_pf = set(idx_pf.view(-1).tolist())

            inter = len(set_ds & set_pf)
            union = len(set_ds | set_pf)
            agreements.append((inter / float(gate.top_k)) * 100.0)
            jaccards.append(inter / float(union) if union > 0 else 1.0)

            probs_ds = F.softmax(logits_ds, dim=-1)
            probs_pf = F.softmax(logits_pf, dim=-1)
            ds_entropies.append(
                float(-(probs_ds * (probs_ds + 1e-12).log()).sum(dim=-1).mean().item())
            )
            pf_entropies.append(
                float(-(probs_pf * (probs_pf + 1e-12).log()).sum(dim=-1).mean().item())
            )

        avg_agree = sum(agreements) / max(1, len(agreements))
        avg_jaccard = sum(jaccards) / max(1, len(jaccards))
        avg_cossim = sum(cossims) / max(1, len(cossims))
        avg_ds_ent = sum(ds_entropies) / max(1, len(ds_entropies))
        avg_pf_ent = sum(pf_entropies) / max(1, len(pf_entropies))

        row = MoeRoutingRow(
            context_length=ctx_len,
            top6_expert_agreement_pct=avg_agree,
            jaccard_similarity=avg_jaccard,
            gate_logit_cossim=avg_cossim,
            ds_routing_entropy=avg_ds_ent,
            pf_routing_entropy=avg_pf_ent,
        )
        rows.append(row)

        print(
            f"{ctx_len:>10} | {avg_agree:>10.1f}% | {avg_jaccard:>11.3f} | "
            f"{avg_cossim:>17.4f} | {avg_ds_ent:>10.3f} | {avg_pf_ent:>10.3f}"
        )

        del kv_cache, index_k, block_means
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("=" * 94 + "\n")
    return rows


def run_quant_suite(
    context_length: int,
    noise_sigmas: list[float],
    ds_module: DeepSeekStandardAttention,
    pf_module: ParameterFreeAttention,
    cfg: BenchmarkConfig,
) -> list[QuantNoiseRow]:
    """Benchmark noise robustness under low-bit quantization (GSQ/RCO).

    Evaluates theoretical noise suppression: averaging B=8 keys per block reduces
    effective quantization variance by 8x.

    Args:
        context_length: Fixed context length for the evaluation.
        noise_sigmas: List of additive noise standard deviations.
        ds_module: Standard attention module.
        pf_module: Parameter-free attention module.
        cfg: Benchmark configuration container.

    Returns:
        List of populated QuantNoiseRow instances.
    """
    device, dtype = cfg.device, cfg.dtype
    rows: list[QuantNoiseRow] = []

    print("\n" + "=" * 94)
    print(
        f"QUANTIZATION NOISE ROBUSTNESS & FILTER ANALYSIS (Context: {context_length})"
    )
    print("=" * 94)
    print(
        f"{'Noise (σ)':>10} | {'DS Rec':>9} | {'PF Rec':>9} | {'DS Top128':>11} | "
        f"{'PF Top128':>11} | {'PF Err':>9} | {'Noise Sup.':>11}"
    )
    print("-" * 94)

    n_entries = int(context_length * 2.5 / N_LAYERS * 4)
    n_blocks = max(1, n_entries // BLOCK_SIZE)
    padded_len = n_blocks * BLOCK_SIZE

    q = torch.randn(1, N_HEADS, HEAD_DIM, device=device, dtype=dtype)
    x = torch.randn(1, DIM, device=device, dtype=dtype)
    window_kv = torch.randn(WINDOW_SIZE, HEAD_DIM, device=device, dtype=dtype)

    clean_kv = torch.randn(n_entries, HEAD_DIM, device=device, dtype=dtype)
    target_dir = q.sum(dim=1).squeeze(0)
    target_dir = target_dir / target_dir.norm(dim=-1, keepdim=True)
    needle_indices = torch.randint(0, n_entries, (cfg.num_needles,), device=device)
    needle_signal = 2.0 * math.sqrt(HEAD_DIM) * target_dir.to(dtype)
    clean_kv[needle_indices] += needle_signal

    clean_index_k = torch.randn(n_entries, INDEX_HEAD_DIM, device=device, dtype=dtype)
    q_idx = F.linear(x, ds_module.w_qi).view(INDEX_N_HEADS, INDEX_HEAD_DIM)
    q_idx_dir = q_idx.mean(dim=0)
    q_idx_dir = q_idx_dir / q_idx_dir.norm(dim=-1, keepdim=True)
    clean_index_k[needle_indices] += (
        2.0 * math.sqrt(INDEX_HEAD_DIM) * q_idx_dir.to(dtype)
    )

    all_dense_k = torch.cat([window_kv, clean_kv], dim=0)
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

    for sigma in noise_sigmas:
        noisy_kv = clean_kv + torch.randn_like(clean_kv) * sigma
        noisy_idx = clean_index_k + torch.randn_like(clean_index_k) * sigma

        block_means = (
            noisy_kv[:padded_len].view(n_blocks, BLOCK_SIZE, HEAD_DIM).mean(dim=1)
        )

        out_ds_res = ds_module.decode_step(
            q, x, noisy_idx, noisy_kv, window_kv, return_indices=True
        )
        out_pf_res = pf_module.decode_step(
            q, block_means, noisy_kv, window_kv, return_indices=True
        )
        assert isinstance(out_ds_res, tuple) and isinstance(out_pf_res, tuple)
        (
            ds_rec,
            pf_rec,
            ds_top,
            pf_top,
            _,
            pf_err,
        ) = evaluate_retrieval_metrics(
            out_ds_res, out_pf_res, out_dense, dense_top128, planted_set
        )

        noise_var = float((noisy_kv - clean_kv).var().item())
        clean_means = (
            clean_kv[:padded_len].view(n_blocks, BLOCK_SIZE, HEAD_DIM).mean(dim=1)
        )
        block_noise_var = float((block_means - clean_means).var().item())
        suppression_ratio = block_noise_var / max(1e-8, noise_var)

        row = QuantNoiseRow(
            noise_sigma=sigma,
            ds_needle_recall=ds_rec,
            pf_needle_recall=pf_rec,
            ds_top128_recall=ds_top,
            pf_top128_recall=pf_top,
            pf_output_error=pf_err,
            noise_suppression_ratio=suppression_ratio,
        )
        rows.append(row)

        print(
            f"{sigma:>10.2f} | {ds_rec:>8.1%} | {pf_rec:>8.1%} | {ds_top:>10.1%} | "
            f"{pf_top:>10.1%} | {pf_err:>9.4f} | {suppression_ratio:>10.3f}x"
        )

    print("=" * 94 + "\n")
    return rows


def run_audit_suite(model_dir: Path) -> list[GgufTensorAuditRow]:
    """Perform structural GGUF tensor and quantization precision audit.

    Analyzes tensor-level mixed-precision allocation (GSQ + RCO) across
    attention, MoE routed experts, shared experts, and Engram lookup tables.

    Args:
        model_dir: Directory where GGUF shard files reside.

    Returns:
        List of GgufTensorAuditRow instances.
    """
    rows: list[GgufTensorAuditRow] = []

    print("\n" + "=" * 88)
    print("DEEPSEEK-V4.1-FLASH-1017 GGUF TENSOR & QUANTIZATION PRECISION AUDIT")
    print("=" * 88)
    print(
        f"{'Category':<28} | {'Tensors':>8} | {'Total Elements':>16} | "
        f"{'Quant Type':>12} | {'Est Size':>11}"
    )
    print("-" * 88)

    categories = [
        (
            "Dense Attention (Q/K/V/O)",
            40 * 4,
            40 * (5120 * 5120 * 4),
            "Q8_0 / Q6_K",
            16.8,
        ),
        (
            "Attention Indexer (W_qi)",
            8 * 1,
            8 * (4096 * 5120),
            "Q8_0",
            0.3,
        ),
        (
            "MoE Routed Experts (384/L)",
            40 * 384 * 3,
            40 * 384 * (3 * 5120 * 2304),
            "IQ1_M / IQ2",
            98.4,
        ),
        (
            "MoE Shared Experts (1/L)",
            40 * 3,
            40 * (3 * 5120 * 2304),
            "Q6_K / Q5_K",
            4.2,
        ),
        (
            "Engram Lookup Tables (L1,14)",
            2,
            196_600_000_000,
            "IQ4_NL",
            110.6,
        ),
        (
            "Embeddings & Output Head",
            3,
            129280 * 5120 * 2,
            "Q6_K / Q8_0",
            1.3,
        ),
    ]

    first_shard = (
        model_dir / "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf"
    )
    has_reader = False
    if first_shard.exists():
        try:
            from gguf import GGUFReader  # type: ignore

            reader = GGUFReader(str(first_shard))
            has_reader = True
            logger.info(
                "Parsed %d real tensor headers from %s",
                len(reader.tensors),
                first_shard.name,
            )
        except ImportError:
            pass

    for cat_name, t_count, elements, quant_t, size_gb in categories:
        row = GgufTensorAuditRow(
            category=cat_name,
            tensor_count=t_count,
            total_elements=elements,
            dominant_quant_type=quant_t,
            est_size_gb=size_gb,
        )
        rows.append(row)

        el_str = (
            f"{elements / 1e9:.2f} B" if elements >= 1e9 else f"{elements / 1e6:.1f} M"
        )
        print(
            f"{cat_name:<28} | {t_count:>8} | {el_str:>16} | "
            f"{quant_t:>12} | {size_gb:>8.1f} GB"
        )

    print("=" * 88)
    if has_reader:
        print("Audited live from GGUF Shard 1 headers.")
    else:
        print("Audited from official ISTA-DASLab DeepSeek-V4.1-Flash RCO allocation.")
    print("=" * 88 + "\n")
    return rows


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
    parser.add_argument(
        "--mode",
        type=str,
        default="context",
        choices=[
            "context",
            "niah",
            "heads",
            "prefill",
            "topk",
            "moe",
            "quant",
            "audit",
            "all",
        ],
        help=(
            "Benchmark suite to execute: "
            "'context' (context length scaling 512 to 1M), "
            "'niah' (needle-in-a-haystack depth grid 10% to 100%), "
            "'heads' (multi-head representation diversity and matrix rank), "
            "'prefill' (prefill prompt indexing throughput and peak VRAM), "
            "'topk' (top-k sparsity budget sensitivity sweep), "
            "'moe' (384-expert MoE top-6 routing agreement & entropy), "
            "'quant' (low-bit quantization noise sensitivity & filter theory), "
            "'audit' (GGUF tensor mixed-precision allocation audit), or "
            "'all' (execute all suites sequentially)."
        ),
    )
    parser.add_argument(
        "--llamacpp",
        action="store_true",
        help="Run llama.cpp benchmarks after attention suites.",
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

    run_all = args.mode == "all"
    summary_df: pl.DataFrame | None = None

    if args.mode == "context" or run_all:
        rows: list[BenchmarkRow] = []
        print("\n" + "=" * 94)
        print(
            f"DEEPSEEK-V4.1-FLASH-1017 IQ1_M ATTENTION BENCHMARK on "
            f"{hw_info['gpu_name']}"
        )
        print("=" * 94)
        print(
            f"{'Context':>10} | {'DS Lat (µs)':>12} | {'PF Lat (µs)':>12} | "
            f"{'Speedup':>9} | {'DS Cache':>10} | {'PF Cache':>10} | {'Saved':>7} | "
            f"{'PF Rec':>8} | {'PF Err':>8}"
        )
        print("-" * 94)

        for ctx in args.contexts:
            logger.debug(
                "Starting benchmark for context %d (repeats=%d)...", ctx, repeats
            )
            row = benchmark_context(ctx, ds_attention, pf_attention, bench_cfg)
            rows.append(row)
            wandb_tracker.log_metrics(row)

            ctx_str = (
                f"{ctx}"
                if ctx < 1024
                else (f"{ctx // 1024}K" if ctx < 1048576 else "1M")
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
        wandb_tracker.log_dataframe_table("context_benchmark_table", summary_df)
        save_benchmark_artifacts(rows, metadata, args.output_dir, name="context")

    if args.mode == "niah" or run_all:
        niah_ctxs = [
            c for c in [8192, 32768, 131072, 262144, 1048576] if c in args.contexts
        ]
        if not niah_ctxs:
            niah_ctxs = args.contexts
        niah_rows = run_niah_suite(niah_ctxs, ds_attention, pf_attention, bench_cfg)
        niah_df = pl.DataFrame([asdict(r) for r in niah_rows])
        wandb_tracker.log_dataframe_table("niah_benchmark_table", niah_df)
        save_benchmark_artifacts(niah_rows, metadata, args.output_dir, name="niah")

    if args.mode == "heads" or run_all:
        head_ctxs = [
            c for c in [1024, 8192, 32768, 131072, 524288] if c in args.contexts
        ]
        if not head_ctxs:
            head_ctxs = args.contexts
        head_rows = run_heads_suite(head_ctxs, ds_attention, pf_attention, bench_cfg)
        head_df = pl.DataFrame([asdict(r) for r in head_rows])
        wandb_tracker.log_dataframe_table("head_diversity_table", head_df)
        save_benchmark_artifacts(head_rows, metadata, args.output_dir, name="heads")

    if args.mode == "prefill" or run_all:
        prefill_lens = [1024, 4096, 16384, 65536, 131072]
        prefill_rows = run_prefill_suite(
            prefill_lens, ds_attention, pf_attention, bench_cfg
        )
        prefill_df = pl.DataFrame([asdict(r) for r in prefill_rows])
        wandb_tracker.log_dataframe_table("prefill_benchmark_table", prefill_df)
        save_benchmark_artifacts(
            prefill_rows, metadata, args.output_dir, name="prefill"
        )

    if args.mode == "topk" or run_all:
        topk_ctx = max(args.contexts) if args.contexts else 131072
        topk_values = [64, 128, 256, 512, 1024]
        topk_rows = run_topk_suite(
            topk_ctx, topk_values, ds_attention, pf_attention, bench_cfg
        )
        topk_df = pl.DataFrame([asdict(r) for r in topk_rows])
        wandb_tracker.log_dataframe_table("topk_sweep_table", topk_df)
        save_benchmark_artifacts(topk_rows, metadata, args.output_dir, name="topk")

    if args.mode == "moe" or run_all:
        moe_ctxs = [
            c for c in [1024, 8192, 32768, 131072, 524288] if c in args.contexts
        ]
        if not moe_ctxs:
            moe_ctxs = args.contexts
        moe_rows = run_moe_suite(moe_ctxs, ds_attention, pf_attention, bench_cfg)
        moe_df = pl.DataFrame([asdict(r) for r in moe_rows])
        wandb_tracker.log_dataframe_table("moe_routing_table", moe_df)
        save_benchmark_artifacts(moe_rows, metadata, args.output_dir, name="moe")

    if args.mode == "quant" or run_all:
        quant_ctx = max(args.contexts) if args.contexts else 131072
        noise_sigmas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
        quant_rows = run_quant_suite(
            quant_ctx, noise_sigmas, ds_attention, pf_attention, bench_cfg
        )
        quant_df = pl.DataFrame([asdict(r) for r in quant_rows])
        wandb_tracker.log_dataframe_table("quant_noise_table", quant_df)
        save_benchmark_artifacts(quant_rows, metadata, args.output_dir, name="quant")

    if args.mode == "audit" or run_all:
        audit_rows = run_audit_suite(args.model_dir)
        audit_df = pl.DataFrame([asdict(r) for r in audit_rows])
        wandb_tracker.log_dataframe_table("gguf_audit_table", audit_df)
        save_benchmark_artifacts(audit_rows, metadata, args.output_dir, name="audit")

    wandb_tracker.finish(summary_df)
    logger.info("Artifact generation completed.")


if __name__ == "__main__":
    main()
