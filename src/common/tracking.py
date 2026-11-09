"""Experiment tracking: MLflow, local (runs/mlflow.db, artifacts in runs/mlartifacts).

One MLflow experiment per benchmark; one run per (suite, version, config). Tags
name the suite, the version (id, label, family, implementation) and the software;
MLflow adds the git commit. Params hold the config and its hash. A run's rows (one
per length and seed) are its logged table rows.json.

Browse: uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
"""

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

URI = "sqlite:///runs/mlflow.db"
ARTIFACTS = Path("runs/mlartifacts")
TABLE = "rows.json"


def experiment(benchmark: str) -> str:
    """The benchmark's experiment id, created on first use."""
    mlflow.set_tracking_uri(URI)
    found = mlflow.get_experiment_by_name(benchmark)
    if found is not None:
        return found.experiment_id
    return mlflow.create_experiment(benchmark, ARTIFACTS.resolve().as_uri())


def config_hash(config: dict[str, Any]) -> str:
    """12 hex digits of the config's sorted JSON."""
    text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def finished(benchmark: str, suite: str) -> pl.DataFrame:
    """Finished runs of a suite's benchmark, newest first: run_id, tags, params."""
    filter_string = f"tags.suite = '{suite}' and attributes.status = 'FINISHED'"
    runs = mlflow.search_runs(
        [experiment(benchmark)], filter_string, order_by=["attributes.start_time DESC"]
    )
    assert not isinstance(runs, list)
    return pl.from_pandas(runs)


def done(benchmark: str, suite: str, version: str, config: dict[str, Any]) -> bool:
    """Whether the suite already has a finished run of this version and config."""
    runs = finished(benchmark, suite)
    if runs.is_empty():
        return False
    match = (pl.col("tags.version") == version) & (
        pl.col("params.config_hash") == config_hash(config)
    )
    return not runs.filter(match).is_empty()


@contextmanager
def run(
    benchmark: str, suite: str, tags: dict[str, str], config: dict[str, Any]
) -> Iterator[None]:
    """An MLflow run; a run left unfinished (killed) is not done and reruns."""
    software = {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        **{name: package_version(name) for name in ("torch", "triton", "transformers")},
    }
    with mlflow.start_run(
        experiment_id=experiment(benchmark),
        run_name=tags["version"],
        tags={"suite": suite, **tags, **software},
    ):
        params = {key: json.dumps(value, default=str) for key, value in config.items()}
        mlflow.log_params(params | {"config_hash": config_hash(config)})
        yield


def log_rows(rows: list[dict[str, Any]]) -> None:
    """Log the run's rows (dicts with the same keys) as its table."""
    mlflow.log_table({key: [row[key] for row in rows] for key in rows[0]}, TABLE)


def load(benchmark: str, suite: str) -> pl.DataFrame:
    """Every row of the suite's newest finished run per version, with its tags."""
    runs = finished(benchmark, suite)
    if runs.is_empty():
        return pl.DataFrame()
    runs = runs.unique("tags.version", keep="first", maintain_order=True)
    tags = runs.select(
        "run_id", *(pl.col(c).alias(c[5:]) for c in runs.columns if c[:5] == "tags.")
    ).drop("mlflow.runName", "mlflow.source.name", strict=False)
    ids = runs["run_id"].to_list()
    table = mlflow.MlflowClient().load_table(
        experiment(benchmark), TABLE, ids, ["run_id"]
    )
    return pl.from_pandas(table).join(tags, on="run_id")
