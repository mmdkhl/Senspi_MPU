"""CSV writing helpers for recorded sensor data."""

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


# Locked PC-side recording schema (SF-5). The CSV header AND column ORDER are a hard
# contract: modal_session_loader reads columns by NAME, but log_loader/plotter read
# by POSITION — never insert, reorder, or drop a column.
SENSOR_CSV_HEADER: tuple[str, ...] = ("timestamp_ns", "t_s", "sensor_id", "ax", "ay", "gz")


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


class StreamingCsvWriter:
    """Append-mode CSV writer for PC-side live recording (Smart Recording, SF-3/SF-5).

    Unlike :func:`write_rows` (write-all-at-once), this opens a file, writes the locked
    header once, then appends rows one at a time as live samples arrive — so a stream
    can be persisted incrementally and the file is byte-compatible with the existing
    recordings (same header + column ORDER ``timestamp_ns,t_s,sensor_id,ax,ay,gz``).

    **Additive by design:** it does NOT touch ``write_rows`` or ``RecorderSession``
    (also used by the benchmark). Intended to be driven from the ingest worker thread
    (G4: never write to disk on the GUI thread).
    """

    def __init__(self, path: Path, header: Sequence[str] = SENSOR_CSV_HEADER) -> None:
        self._path = Path(path)
        self._header = list(header)
        self._fh = None
        self._writer = None
        self._rows = 0

    def open(self) -> "StreamingCsvWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self._header)
        return self

    def write_row(self, row: Sequence[Any]) -> None:
        if self._writer is None:
            raise RuntimeError("StreamingCsvWriter is not open")
        self._writer.writerow(row)
        self._rows += 1

    def write_sample(self, timestamp_ns: Any, t_s: Any, sensor_id: Any,
                     ax: Any, ay: Any, gz: Any) -> None:
        """Write one sample in the locked column order."""
        self.write_row([timestamp_ns, t_s, sensor_id, ax, ay, gz])

    @property
    def rows_written(self) -> int:
        return self._rows

    @property
    def path(self) -> Path:
        return self._path

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
        self._fh = None
        self._writer = None

    def __enter__(self) -> "StreamingCsvWriter":
        return self.open()

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


def write_recording_meta(meta_path: Path, **fields: Any) -> None:
    """Write the ``.csv.meta.json`` sidecar for a PC-side recording.

    Loaders ignore the meta (they read the CSV), so new provenance keys
    (``pc_start_utc``, ``actual_hz``, ``requested_hz``, …) are safe to add here.
    """
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(fields, fh, indent=2, default=str)
