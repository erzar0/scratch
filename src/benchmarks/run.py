"""Run a benchmark suite: every registered version through each benchmark section.

    uv run python -m benchmarks.run src/benchmarks/suites/standard.toml
        [--only latency fidelity tasks] [--versions dense sparq ...] [--force]

One MLflow run per (suite, benchmark, version, config); a finished one is skipped
unless --force, so a killed queue resumes where it stopped. Each run logs its rows
as a table and their means over seeds as metrics (step: context length).
"""

import argparse
import tomllib
from pathlib import Path

import mlflow
import polars as pl

from attention.registry import VERSIONS
from benchmarks import fidelity, latency, tasks
from common import tracking

BENCHMARKS = {"latency": latency.run, "fidelity": fidelity.run, "tasks": tasks.run}
KEYS = ("distribution", "task", "keep")


def log_means(rows: list[dict[str, object]]) -> None:
    """Each numeric column's mean over seeds, as metric <column>/<distribution>
    (or /<task>)/<keep>."""
    frame = pl.DataFrame(rows).drop("seed", strict=False)
    keys = [key for key in KEYS if key in frame.columns]
    for row in frame.group_by([*keys, "n"]).mean().iter_rows(named=True):
        where = "/".join(str(row[key]) for key in keys)
        for name, value in row.items():
            if name not in (*keys, "n") and value is not None:
                mlflow.log_metric(f"{name}/{where}", value, step=row["n"])


def main() -> None:
    """Parse the arguments and run the suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--only", nargs="+", choices=list(BENCHMARKS))
    parser.add_argument("--versions", nargs="+", choices=list(VERSIONS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    suite = tomllib.loads(args.suite.read_text())
    name = suite.pop("name")
    for benchmark, section in suite.items():
        if args.only and benchmark not in args.only:
            continue
        if benchmark == "tasks":
            tasks.train(section)
        for version_id in args.versions or VERSIONS:
            version = VERSIONS[version_id]
            if version.emulation is None and benchmark != "latency":
                print(f"{benchmark} {version_id}: latency only", flush=True)
                continue
            config = {"benchmark": section, "settings": version.settings}
            if not args.force and tracking.done(benchmark, name, version_id, config):
                print(f"{benchmark} {version_id}: done", flush=True)
                continue
            print(f"{benchmark} {version_id}", flush=True)
            with tracking.run(benchmark, name, version.tags(), config):
                rows = BENCHMARKS[benchmark](version, section)
                tracking.log_rows(rows)
                log_means(rows)


if __name__ == "__main__":
    main()
