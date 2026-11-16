"""Unit tests for downstream reasoning and task evaluations."""

from pathlib import Path

from eval_downstream_h200 import (
    ReasoningQuestion,
    extract_boxed_answer,
    extract_mcq_answer,
    parse_thinking_and_answer,
    run_passkey_suite,
    run_perplexity_suite,
    run_reasoning_suite,
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


def test_run_reasoning_suite_mock() -> None:
    """Verify mock execution of reasoning benchmark."""
    # Arrange
    questions = [
        ReasoningQuestion(
            id="test_q1",
            benchmark="AIME25",
            question="Compute 2+2.",
            ground_truth="55",
        )
    ]
    dummy_path = Path("/tmp/dummy_cli")

    # Act
    results = run_reasoning_suite(
        dummy_path, dummy_path, questions, ngl=0, reasoning_effort=50, mock=True
    )

    # Assert
    assert len(results) == 1
    assert results[0].question_id == "test_q1"
    assert results[0].is_correct is True


def test_run_passkey_suite_mock() -> None:
    """Verify mock execution of long-context passkey retrieval."""
    # Arrange
    dummy_path = Path("/tmp/dummy_cli")

    # Act
    results = run_passkey_suite(
        dummy_path, dummy_path, contexts=[1024], ngl=0, mock=True
    )

    # Assert
    assert len(results) == 1
    assert results[0].success is True


def test_run_perplexity_suite_mock(tmp_path: Path) -> None:
    """Verify mock execution of wikitext perplexity calculation."""
    # Arrange
    dummy_path = Path("/tmp/dummy_ppl")

    # Act
    res = run_perplexity_suite(dummy_path, dummy_path, tmp_path, ngl=0, mock=True)

    # Assert
    assert res.dataset == "wikitext-2"
    assert res.perplexity > 0.0
