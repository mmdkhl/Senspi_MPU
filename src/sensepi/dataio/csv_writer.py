"""CSV writing helpers for recorded sensor data."""

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence


def write_rows(path: Path, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """
    Write a header row and all data rows to a CSV file.

    Directories are created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
