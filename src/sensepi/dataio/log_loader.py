"""Utilities for loading recorded CSV logs."""

from pathlib import Path
from typing import Iterable, List, Sequence
import io

import numpy as np


def _looks_numeric_csv_line(line: str) -> bool:
    """Heuristically decide if a CSV line is numeric-only (no header)."""
    stripped = line.strip()
    if not stripped:
        return False
    tokens = [t for t in stripped.split(",") if t]
    if not tokens:
        return False
    try:
        for t in tokens:
            float(t)
        return True
    except ValueError:
        return False


def load_csv(path: Path) -> np.ndarray:
    """
    Load a CSV file containing numeric data.

    The file may optionally include a single header row, which will be
    skipped automatically.
    """
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        rest = f.read()

    # Decide if the first line is header or data
    if _looks_numeric_csv_line(first_line):
        buffer = io.StringIO(first_line + rest)
    else:
        buffer = io.StringIO(rest)

    return np.loadtxt(buffer, delimiter=",")


def chunk_array(array: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    """Yield fixed-size chunks from an array."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")

    total = array.shape[0]
    for start in range(0, total, chunk_size):
        yield array[start : start + chunk_size]


def merge_logs(paths: Sequence[Path]) -> np.ndarray:
    """Load multiple CSV logs and concatenate them along the first axis."""
    arrays: List[np.ndarray] = [load_csv(path) for path in paths]
    if not arrays:
        return np.empty((0, 0))
    return np.concatenate(arrays, axis=0)
