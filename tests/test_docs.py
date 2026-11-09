"""docs/methods.md explains every registered version."""

from pathlib import Path

from attention.registry import VERSIONS


def test_every_version_is_explained() -> None:
    text = Path("docs/methods.md").read_text()
    missing = [v.id for v in VERSIONS.values() if f"`{v.id}`" not in text]
    missing += [v.name for v in VERSIONS.values() if v.name not in text]
    assert not missing, missing
