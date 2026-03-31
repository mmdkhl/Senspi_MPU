#!/usr/bin/env python3
"""Helpers for decimating/smoothing SensePi time-series data for plotting.

The Offline/Recordings GUI tab imports this module directly so it can reuse
the same NumPy-based downsampling routines that power the standalone CLI
plotter. When executed as a script (see ``LocalPlotRunner``), it opens a
Matplotlib window that can either replay a log once (``--mode replay``) or
periodically reload a log for a faux-live view (``--mode follow``). If
``--file`` is omitted the newest ``*.csv``/``*.jsonl`` under ``data/raw/``,
``logs/``, or the project root is selected automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_TIME_FIELDS = {"timestamp_ns", "t_s", "t_rel_s", "timestamp"}
BASE_IGNORE_FIELDS = BASE_TIME_FIELDS | {"sensor_id"}


# --------------------------------------------------------------------------- # helpers
def find_latest_log(search_roots: Sequence[Path]) -> Optional[Path]:
    candidates: list[Path] = []
    patterns = ("*.csv", "*.jsonl")
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.glob(pattern))

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_meta_sidecar(path: Path) -> dict[str, Any] | None:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception as exc:
        raise ValueError(f"Failed to parse metadata {meta_path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"Metadata file {meta_path} must contain a JSON object")
    return meta


def _load_csv_with_meta(path: Path) -> tuple[np.ndarray, list[str], dict[str, Any] | None]:
    data, columns = load_csv(path)
    meta = _load_meta_sidecar(path)
    return data, columns, meta


def _classify_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if isinstance(value, str):
        return "str"
    return "object"


def _merge_kinds(existing: str | None, new_kind: str | None) -> str | None:
    if new_kind is None:
        return existing
    if existing is None:
        return new_kind
    if existing == new_kind:
        return existing
    numeric = {"int", "float"}
    if existing in numeric and new_kind in numeric:
        return "float"
    return "object"


def _dtype_for_kind(kind: str | None) -> str:
    if kind == "float":
        return "f8"
    if kind == "int":
        return "i8"
    if kind == "bool":
        return "?"
    # Strings and mixed/unknown types fall back to Python objects
    return "O"


def _records_to_structured_array(
    records: Sequence[dict[str, Any]], meta: dict[str, Any] | None
) -> tuple[np.ndarray, list[str]]:
    order: list[str] = []
    if meta:
        header = meta.get("header")
        if isinstance(header, list):
            for entry in header:
                if isinstance(entry, str) and entry not in order:
                    order.append(entry)

    for record in records:
        for key in record.keys():
            if key not in order:
                order.append(key)

    if not order:
        raise ValueError("JSONL log contains no columns")

    kinds: dict[str, str | None] = {col: None for col in order}
    for record in records:
        for key, value in record.items():
            if key not in kinds:
                kinds[key] = None
            kinds[key] = _merge_kinds(kinds.get(key), _classify_value(value))

    dtype = [(col, _dtype_for_kind(kinds.get(col))) for col in order]
    data = np.zeros(len(records), dtype=dtype)

    for col, kind in ((col_name, kinds.get(col_name)) for col_name in order):
        if kind == "float":
            data[col] = np.nan

    for row_idx, record in enumerate(records):
        for key, value in record.items():
            if value is None or key not in data.dtype.names:
                continue
            data[key][row_idx] = value

    columns = list(data.dtype.names or [])
    return data, columns


def _load_jsonl_with_meta(path: Path) -> tuple[np.ndarray, list[str], dict[str, Any] | None]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object")
            records.append(obj)

    if not records:
        raise ValueError(f"{path} contains no JSON objects")

    meta = _load_meta_sidecar(path)
    data, columns = _records_to_structured_array(records, meta)

    if not any(field in columns for field in BASE_TIME_FIELDS):
        raise ValueError(f"{path} is missing a timestamp field")
    plottable = [c for c in columns if c.lower() not in BASE_IGNORE_FIELDS]
    if not plottable:
        raise ValueError(f"{path} does not contain any data channels to plot")

    return data, columns, meta


def _load_log_with_meta(path: Path) -> tuple[np.ndarray, list[str], dict[str, Any] | None]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_with_meta(path)
    if suffix == ".jsonl":
        return _load_jsonl_with_meta(path)
    raise ValueError(f"Unsupported log file type: {path.suffix}")


def load_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load a CSV file with a header row into a structured NumPy array."""
    data = np.genfromtxt(path, delimiter=",", names=True)

    # When there's only a single row, genfromtxt may return a scalar; coerce.
    if data.size == 0:
        raise ValueError(f"File {path} contains no data rows")
    if data.ndim == 0:
        data = data.reshape(1)

    names = list(data.dtype.names or [])
    if not names:
        raise ValueError(f"File {path} has no header / column names")

    return data, names


def infer_sensor_type(columns: Sequence[str], meta: dict[str, Any] | None = None) -> str:
    """
    Heuristically infer sensor type from the available columns/metadata.

    Returns "mpu6050" if it finds typical MPU6050 columns, otherwise "generic".
    """
    if meta:
        declared = meta.get("sensor_type")
        if isinstance(declared, str) and declared.strip():
            return declared.strip().lower()

    lower = {c.lower() for c in columns}

    if {"sensor_id", "ax", "ay", "az", "gx", "gy", "gz"} <= lower:
        return "mpu6050"

    # Anything else is treated as generic.
    return "generic"


def build_time_axis(
    data: np.ndarray, columns: Sequence[str], meta: dict[str, Any] | None = None
) -> tuple[np.ndarray, str]:
    """Return (t, x_label)."""
    if "t_rel_s" in columns:
        t = data["t_rel_s"]
        return t, "Time [s]"
    if "t_s" in columns:
        t = data["t_s"]
        return t, "Time [s]"
    if "timestamp_ns" in columns:
        ns = data["timestamp_ns"]
        t0 = ns[0]
        t = (ns - t0) * 1e-9
        return t, "Time [s since start]"
    if "timestamp" in columns:
        ts = data["timestamp"]
        t0 = ts[0]
        t = ts - t0
        return t, "Time [relative units]"
    if meta:
        rate = meta.get("device_rate_hz")
        if isinstance(rate, (int, float)) and rate > 0:
            n = data.shape[0]
            t = np.arange(n, dtype=float) / float(rate)
            return t, f"Time [s @ {rate:g} Hz]"
    n = data.shape[0]
    t = np.arange(n, dtype=float)
    return t, "Sample index"


def _meta_sampling_rate(meta: dict[str, Any] | None) -> float | None:
    if not meta:
        return None
    for key in ("device_rate_hz", "requested_rate_hz", "clamped_rate_hz"):
        value = meta.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _preferred_data_columns(columns: Sequence[str], meta: dict[str, Any] | None) -> list[str]:
    cols = [c for c in columns if c]
    lookup = {c.lower(): c for c in cols}
    if meta:
        meta_channels = meta.get("channels")
        ordered: list[str] = []
        if isinstance(meta_channels, list):
            for entry in meta_channels:
                if not isinstance(entry, str):
                    continue
                resolved = lookup.get(entry.lower())
                if resolved and resolved not in ordered:
                    ordered.append(resolved)
        if ordered:
            return ordered

        header = meta.get("header")
        if isinstance(header, list):
            ordered = []
            for entry in header:
                if not isinstance(entry, str):
                    continue
                key = entry.lower()
                if key in BASE_IGNORE_FIELDS:
                    continue
                resolved = lookup.get(key)
                if resolved and resolved not in ordered:
                    ordered.append(resolved)
            if ordered:
                return ordered

    filtered = [c for c in cols if c.lower() not in BASE_IGNORE_FIELDS]
    if filtered:
        return filtered
    return list(cols)


def pick_data_columns(
    sensor_type: str, columns: Sequence[str], meta: dict[str, Any] | None = None
) -> tuple[list[str], list[str]]:
    """Return (acc_columns, gyro_columns) based on the header."""
    cols = [c for c in columns if c]
    lower_map = {c.lower(): c for c in cols}
    ordered = _preferred_data_columns(cols, meta)

    if sensor_type == "mpu6050":
        acc_cols: list[str] = []
        gyro_cols: list[str] = []
        for desired in ("ax", "ay", "az"):
            actual = lower_map.get(desired)
            if actual:
                acc_cols.append(actual)
        for desired in ("gx", "gy", "gz"):
            actual = lower_map.get(desired)
            if actual:
                gyro_cols.append(actual)
        if not acc_cols and ordered:
            acc_cols = ordered
        return acc_cols, gyro_cols

    if ordered:
        return ordered, []
    return cols, []


def setup_figure(
    path: Path,
    sensor_type: str,
    data: np.ndarray,
    columns: Sequence[str],
    meta: dict[str, Any] | None = None,
):
    """Create figure/axes/lines and return (fig, axes, lines, time_vector)."""
    t, x_label = build_time_axis(data, columns, meta)
    acc_cols, gyro_cols = pick_data_columns(sensor_type, columns, meta)

    if not acc_cols and not gyro_cols:
        raise ValueError(f"No plottable data columns found in {path}")

    n_axes = 2 if gyro_cols else 1
    fig, axes = plt.subplots(n_axes, 1, sharex=True)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    lines: dict[str, any] = {}

    # Acceleration subplot(s)
    if acc_cols:
        ax0 = axes[0]
        for col in acc_cols:
            y = data[col]
            (line,) = ax0.plot(t, y, label=col)
            lines[col] = line
        if sensor_type == "mpu6050":
            ax0.set_ylabel("Acceleration [m/s²]")
        else:
            ax0.set_ylabel("Value")
        ax0.legend(loc="upper right")
        rate = _meta_sampling_rate(meta)
        rate_str = f" @ {rate:g} Hz" if rate else ""
        ax0.set_title(f"{sensor_type}{rate_str} — {path.name}")

    # Gyro subplot
    if gyro_cols:
        ax1 = axes[1]
        for col in gyro_cols:
            y = data[col]
            (line,) = ax1.plot(t, y, label=col)
            lines[col] = line
        ax1.set_ylabel("Angular rate [deg/s]")
        ax1.legend(loc="upper right")

    axes[-1].set_xlabel(x_label)
    fig.tight_layout()

    return fig, axes, lines, t


def build_plot_for_file(path: Path, sensor_type: str = "auto"):
    """Return a Matplotlib Figure configured for the given log file."""

    data, columns, meta = _load_log_with_meta(path)
    if sensor_type == "auto":
        sensor_type = infer_sensor_type(columns, meta)

    fig, axes, lines, _t = setup_figure(path, sensor_type, data, columns, meta)
    return fig, axes, lines


class Plotter:
    """Prepare decimated/smoothed data for visualisation.

    This class is used by the offline viewer to downsample long time-series
    logs while preserving their overall shape and extrema.
    """

    def __init__(self, sensor_type: str = "auto") -> None:
        self.sensor_type = sensor_type

    def build_figure(
        self,
        path: Path,
        sensor_type: Optional[str] = None,
    ):
        """
        Return the ``(fig, axes, lines)`` tuple for ``path``.

        ``sensor_type`` can override the default detected value for a single
        call.
        """
        resolved = sensor_type or self.sensor_type
        return build_plot_for_file(path, sensor_type=resolved)

    def replay(self, path: Path, sensor_type: Optional[str] = None) -> None:
        """Render ``path`` once in Matplotlib."""
        resolved = sensor_type or self.sensor_type
        plot_replay(path, resolved)

    def follow(
        self,
        path: Path,
        interval_s: float = 0.5,
        sensor_type: Optional[str] = None,
    ) -> None:
        """Continuously reload ``path`` and update the plot."""
        resolved = sensor_type or self.sensor_type
        plot_follow(path, resolved, interval_s)


# --------------------------------------------------------------------------- # plotting modes
def plot_replay(path: Path, sensor_type: str) -> None:
    fig, _axes, _lines = build_plot_for_file(path, sensor_type)
    fig.canvas.manager.set_window_title(f"SensePi replay — {path.name}")
    plt.show()


def plot_follow(path: Path, sensor_type: str, interval_s: float) -> None:
    data, columns, meta = _load_log_with_meta(path)
    if sensor_type == "auto":
        sensor_type = infer_sensor_type(columns, meta)

    fig, axes, lines, _t = setup_figure(path, sensor_type, data, columns, meta)

    def _update(_frame):
        try:
            new_data, new_columns, new_meta = _load_log_with_meta(path)
        except Exception:
            # If the file temporarily disappears or is being written to, just
            # skip this frame.
            return list(lines.values())

        t_new, _x_label = build_time_axis(new_data, new_columns, new_meta)

        for name, line in lines.items():
            if name not in new_columns:
                continue
            y = new_data[name]
            line.set_data(t_new, y)

        for ax in axes:
            ax.relim()
            ax.autoscale_view()

        return list(lines.values())

    fig.canvas.manager.set_window_title(f"SensePi live — {path.name}")
    FuncAnimation(fig, _update, interval=interval_s * 1000.0, blit=False)
    plt.show()


# --------------------------------------------------------------------------- # CLI
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simple Matplotlib-based plotter for SensePi CSV/JSONL logs."
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help=(
            "Path to a log file (.csv or .jsonl). If omitted, the newest log "
            "found under data/raw, logs, or the project root is used."
        ),
    )
    parser.add_argument(
        "-s",
        "--sensor",
        type=str,
        choices=["auto", "mpu6050", "generic"],
        default="auto",
        help="Sensor type for plotting (default: auto-detect from columns).",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["follow", "replay"],
        default="follow",
        help=(
            "Plot mode: 'replay' for a static plot, 'follow' to periodically "
            "reload the file for a live view (default)."
        ),
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.5,
        help="Update interval in seconds when using --mode follow (default: 0.5).",
    )

    args = parser.parse_args(argv)

    if args.file:
        csv_path = Path(args.file).expanduser().resolve()
        if not csv_path.exists():
            parser.error(f"Log file not found: {csv_path}")
    else:
        csv_path = find_latest_log(
            [
                REPO_ROOT / "data" / "raw",
                REPO_ROOT / "logs",
                REPO_ROOT,
            ]
        )
        if csv_path is None:
            parser.error(
                "No log files found in data/raw, logs, or the project root.\n"
                "Specify a file explicitly with --file."
            )
        print(f"[INFO] Using latest log: {csv_path}")

    try:
        if args.mode == "replay":
            plot_replay(csv_path, args.sensor)
        else:
            plot_follow(csv_path, args.sensor, args.interval)
    except KeyboardInterrupt:
        # Allow clean exit on Ctrl+C
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
