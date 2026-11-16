"""Unit tests for high batch size attention scaling benchmark."""

from __future__ import annotations

import torch
from test_batch_scaling import benchmark_batch
from test_iq1m_h200 import (
    BenchmarkConfig,
    DeepSeekStandardAttention,
    ParameterFreeAttention,
)


def test_batch_scaling_execution() -> None:
    """Verify batched benchmark execution across multiple batch sizes."""
    device = torch.device("cpu")
    dtype = torch.float32
    cfg = BenchmarkConfig(device=device, dtype=dtype)

    ds_att = DeepSeekStandardAttention(device=device, dtype=dtype)
    pf_att = ParameterFreeAttention(device=device, dtype=dtype)

    for b in [1, 2, 4]:
        row = benchmark_batch(
            b,
            ctx=512,
            ds_module=ds_att,
            pf_module=pf_att,
            cfg=cfg,
            repeats=2,
        )
        assert row.batch_size == b
        assert row.ds_latency_ms > 0
        assert row.pf_latency_ms > 0
        assert row.speedup > 0
        assert row.cossim > 0.95
        assert row.pf_vram_saved_pct > 0
