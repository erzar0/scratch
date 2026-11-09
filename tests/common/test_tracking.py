"""MLflow tracking (src/common/tracking.py) on a temporary store."""

from pathlib import Path

import pytest

from common import tracking


def test_run_done_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracking, "URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(tracking, "ARTIFACTS", tmp_path / "artifacts")
    config = {"lengths": [8, 16], "keep": 4}
    tags = {"version": "dense", "label": "Dense", "implementation": "PyTorch"}
    assert not tracking.done("latency", "smoke", "dense", config)

    with tracking.run("latency", "smoke", tags, config):
        tracking.log_rows([{"n": 8, "us": 1.0}, {"n": 16, "us": 2.0}])
    with pytest.raises(KeyboardInterrupt), tracking.run("latency", "smoke", tags, {}):
        raise KeyboardInterrupt  # a killed run is not finished

    assert tracking.done("latency", "smoke", "dense", config)
    assert not tracking.done("latency", "smoke", "dense", {"lengths": [8], "keep": 4})
    assert not tracking.done("latency", "other", "dense", config)
    rows = tracking.load("latency", "smoke")
    assert rows.sort("n")["us"].to_list() == [1.0, 2.0]
    assert set(rows["label"]) == {"Dense"}
