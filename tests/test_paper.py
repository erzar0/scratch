"""plots/paper.py: CSV configs read as words in the paper's tables."""

import pytest

from plots.paper import setting


@pytest.mark.parametrize(
    ("config", "words"),
    [
        ("top_k=8", "8 entries"),
        ("S=16, s=4", "superblocks of 16, keep 4"),
        ("S=8/8, s=1/8", "superblocks of 8, keep 1 (needle layers: keep 8)"),
        (
            "S=16/32, s=2/16",
            "superblocks of 16, keep 2 (needle layers: superblocks of 32, keep 16)",
        ),
    ],
)
def test_setting(config: str, words: str) -> None:
    # Arrange: config as written by experiments/paired_se.py

    # Act
    result = setting(config)

    # Assert
    assert result == words
