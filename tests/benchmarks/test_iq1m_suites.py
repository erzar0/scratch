"""Unit tests for DeepSeek-V4.1-Flash IQ1_M benchmark suites."""

from pathlib import Path

import torch
from test_iq1m_h200 import (
    BLOCK_SIZE,
    DeepSeekMoEGate,
    run_audit_suite,
)


def test_moe_gate_dimensions_and_scaling() -> None:
    """Verify DeepSeekMoEGate output dimensions and route weight scaling."""
    # Arrange
    dim = 256
    num_experts = 32
    top_k = 6
    route_scale = 1.5
    gate = DeepSeekMoEGate(
        dim=dim,
        num_experts=num_experts,
        top_k=top_k,
        route_scale=route_scale,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    x = torch.randn(4, dim, dtype=torch.float32)

    # Act
    weights, indices = gate(x)

    # Assert
    assert weights.shape == (4, top_k)
    assert indices.shape == (4, top_k)
    assert torch.all(indices >= 0) and torch.all(indices < num_experts)
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.full((4,), route_scale, dtype=torch.float32),
        atol=1e-5,
    )


def test_quant_noise_suppression_ratio() -> None:
    """Verify block-mean prefiltering suppresses noise variance by ~8x (B=8)."""
    # Arrange
    n_blocks = 500
    head_dim = 128
    sigma = 0.25
    clean_kv = torch.randn(n_blocks * BLOCK_SIZE, head_dim)
    noise = torch.randn_like(clean_kv) * sigma
    noisy_kv = clean_kv + noise

    # Act
    clean_means = clean_kv.view(n_blocks, BLOCK_SIZE, head_dim).mean(dim=1)
    block_means = noisy_kv.view(n_blocks, BLOCK_SIZE, head_dim).mean(dim=1)
    noise_var = float(noise.var().item())
    block_noise_var = float((block_means - clean_means).var().item())
    suppression_ratio = block_noise_var / max(1e-8, noise_var)

    # Assert: Var(1/8 sum eps_i) = sigma^2 / 8 = 0.125 * sigma^2
    assert 0.10 <= suppression_ratio <= 0.15


def test_audit_suite_returns_expected_categories() -> None:
    """Verify GGUF structural audit outputs all expected model categories."""
    # Arrange
    dummy_dir = Path("/tmp/nonexistent_gguf_audit_dir")

    # Act
    rows = run_audit_suite(dummy_dir)

    # Assert
    categories = [r.category for r in rows]
    assert len(rows) == 6
    assert "Dense Attention (Q/K/V/O)" in categories
    assert "MoE Routed Experts (384/L)" in categories
    assert "Engram Lookup Tables (L1,14)" in categories
