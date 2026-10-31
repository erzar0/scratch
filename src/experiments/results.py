"""Result CSVs shared by the experiment scripts: rerunning a study replaces its rows.

Every value is stored as text, so studies with different columns share one file.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

type Row = Mapping[str, object]


class Results:
    """One CSV; the rows matching `replace` are dropped on open and rewritten by add.

    Attributes:
        path: The CSV.
        rows: Kept and added rows, as text.
    """

    def __init__(self, path: Path, replace: Row) -> None:
        """Load path, dropping rows whose values equal every item of replace.

        Args:
            path: The CSV; created on the first add.
            replace: Column -> value identifying this run's rows.
        """
        self.path = path
        self.rows: list[dict[str, str]] = []
        if path.exists():
            matches = pl.lit(True)
            for column, value in replace.items():
                matches &= pl.col(column).fill_null("") == str(value)
            old = pl.read_csv(path, infer_schema=False)
            if all(column in old.columns for column in replace):
                old = old.filter(~matches)
            self.rows = old.to_dicts()

    def add(self, row: Row) -> None:
        """Append a row and rewrite the file, so a crash keeps what is done."""
        self.rows.append({name: _text(value) for name, value in row.items()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(self.rows, infer_schema_length=None).write_csv(self.path)


def _text(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else str(value)


def read(path: str | Path) -> pl.DataFrame:
    """A results CSV with numeric columns parsed; "nan" (e.g. OOM) and "" (None)
    become null."""
    return pl.read_csv(path, infer_schema_length=None, null_values=["nan", ""])


def pareto_front(frame: pl.DataFrame, cost: str, loss: str) -> pl.DataFrame:
    """Rows no other row beats on both cost and loss (lower is better), by cost."""
    ordered = frame.sort([cost, loss])
    best = ordered[loss].cum_min()
    keep = ordered[loss] <= best.shift(1, fill_value=float("inf")) - 1e-12
    return ordered.filter(keep)


def paired_delta(losses: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Mean of losses - reference over the same batches, and its standard error."""
    diff = np.asarray(losses, np.float64) - np.asarray(reference, np.float64)
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))
