"""Unit tests for downstream reasoning and task evaluations."""

from pathlib import Path

import torch
from eval_downstream_h200 import (
    ReasoningQuestion,
    extract_boxed_answer,
    extract_mcq_answer,
    parse_thinking_and_answer,
    run_comparative_passkey_suite,
    run_comparative_perplexity_suite,
    run_comparative_reasoning_suite,
)
from test_iq1m_h200 import (
    BenchmarkConfig,
    DeepSeekStandardAttention,
    ParameterFreeAttention,
)


def test_extract_boxed_answer() -> None:
    """Verify numeric extraction from LaTeX boxed format."""
    # Arrange
    text = "We solve the recurrence and find that x = \\boxed{144}."

    # Act
    extracted = extract_boxed_answer(text)

    # Assert
    assert extracted == "144"


def test_extract_mcq_answer() -> None:
    """Verify multiple-choice option extraction from text."""
    # Arrange
    text_boxed = "Based on stereochemistry, the correct option is \\boxed{C}."
    text_phrase = "Therefore the answer is (B)."

    # Act
    ans1 = extract_mcq_answer(text_boxed)
    ans2 = extract_mcq_answer(text_phrase)

    # Assert
    assert ans1 == "C"
    assert ans2 == "B"


def test_parse_thinking_and_answer() -> None:
    """Verify separation of think tags from final answer."""
    # Arrange
    raw_response = (
        "<think>\n"
        "Need to compute modulo 7.\n"
        "3^6 = 1 mod 7.\n"
        "</think>\n"
        "The remainder is \\boxed{0}."
    )

    # Act
    thinking, answer = parse_thinking_and_answer(raw_response)

    # Assert
    assert "modulo 7" in thinking
    assert "<think>" not in thinking
    assert answer == "The remainder is \\boxed{0}."


def test_run_comparative_reasoning_suite_mock() -> None:
    """Verify comparative reasoning evaluation across both models."""
    # Arrange
    device = torch.device("cpu")
    dtype = torch.float32
    cfg = BenchmarkConfig(device=device, dtype=dtype)
    ds_att = DeepSeekStandardAttention(device=device, dtype=dtype)
    pf_att = ParameterFreeAttention(device=device, dtype=dtype)
    questions = [
        ReasoningQuestion(
            id="test_q1",
            benchmark="AIME25",
            question="Find positive integer n such that n=55.",
            ground_truth="55",
        )
    ]
    dummy_path = Path("/tmp/dummy_cli")

    # Act
    results = run_comparative_reasoning_suite(
        ds_att,
        pf_att,
        cfg,
        dummy_path,
        dummy_path,
        questions,
        ngl=0,
        reasoning_effort=50,
        mock=True,
    )

    # Assert
    assert len(results) == 1
    assert results[0].question_id == "test_q1"
    assert results[0].token_cossim > 0.0
    assert results[0].speedup > 0.0


def test_run_comparative_passkey_suite() -> None:
    """Verify comparative passkey retrieval across both models."""
    # Arrange
    device = torch.device("cpu")
    dtype = torch.float32
    cfg = BenchmarkConfig(device=device, dtype=dtype)
    ds_att = DeepSeekStandardAttention(device=device, dtype=dtype)
    pf_att = ParameterFreeAttention(device=device, dtype=dtype)

    # Act
    results = run_comparative_passkey_suite(ds_att, pf_att, cfg, contexts=[512])

    # Assert
    assert len(results) == 1
    assert results[0].ds_found is True
    assert results[0].pf_found is True
    assert results[0].speedup > 0.0


def test_run_comparative_perplexity_suite() -> None:
    """Verify comparative perplexity calculation on both models."""
    # Arrange
    device = torch.device("cpu")
    dtype = torch.float32
    cfg = BenchmarkConfig(device=device, dtype=dtype)
    ds_att = DeepSeekStandardAttention(device=device, dtype=dtype)
    pf_att = ParameterFreeAttention(device=device, dtype=dtype)

    # Act
    row = run_comparative_perplexity_suite(ds_att, pf_att, cfg, context_length=512)

    # Assert
    assert row.dataset == "wikitext-2"
    assert row.ds_perplexity > 0.0
    assert row.pf_perplexity > 0.0
    assert row.speedup > 0.0
