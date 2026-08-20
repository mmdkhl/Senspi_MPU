#!/usr/bin/env python
# raspberrypi_scripts/debug_log_sample_rate.py
"""
Estimate recorded sampling rate from MPU6050 log files on the Raspberry Pi.

This script is intended to be run on the Pi inside the `raspberrypi_scripts/`
directory. It inspects CSV or JSONL log files produced by
`mpu6050_multi_logger.py` and estimates the sampling rate from timestamps.

Each MPU6050 sensor can provide up to six numeric channels:

    ax, ay, az, gx, gy, gz

Many deployments only use a subset (often ax, ay, gz). This script does NOT
look at channel values; it only cares about timestamps and sensor_id in order
to validate that the *recorded* rate on the Pi matches the configured
sample_rate_hz, independent of GUI streaming and --stream-every.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_rows_csv(path: Path) -> List[Dict[str, Any]]:
    """Load all rows from a CSV log into a list of dicts."""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _load_rows_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all rows from a JSONL log into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _load_meta(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load the .meta.json sidecar if present.

    For a log file /path/to/log.csv, the meta is expected at
    /path/to/log.csv.meta.json as written by mpu6050_multi_logger.py.
    """
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta, dict):
            return meta
    except Exception:
        return None
    return None


def _extract_times(
    rows: Iterable[Dict[str, Any]]
) -> Tuple[List[float], List[int]]:
    """
    Extract timestamps (seconds) and sensor_ids from rows.

    Preference order for time fields:
      1) t_s (float seconds or int nanoseconds, depending on logger)
      2) t_rel_s (float seconds, newer logs)
      3) timestamp_ns (int nanoseconds)

    We normalise everything to float seconds.
    """
    times: List[float] = []
    sensor_ids: List[int] = []

    for row in rows:
        t: Optional[float] = None

        if "t_s" in row and row["t_s"] not in ("", None):
            try:
                t_val = float(row["t_s"])
                # Heuristic: if it looks like nanoseconds, scale down
                if abs(t_val) > 1e12:
                    t = t_val * 1e-9
                else:
                    t = t_val
            except (TypeError, ValueError):
                t = None
        elif "t_rel_s" in row and row["t_rel_s"] not in ("", None):
            try:
                t = float(row["t_rel_s"])
            except (TypeError, ValueError):
                t = None
        elif "timestamp_ns" in row and row["timestamp_ns"] not in ("", None):
            try:
                t_ns = float(row["timestamp_ns"])
                t = t_ns * 1e-9
            except (TypeError, ValueError):
                t = None

        if t is None:
            continue

        times.append(t)

        sid_val = row.get("sensor_id")
        try:
            if sid_val is not None and sid_val != "":
                sensor_ids.append(int(sid_val))
        except (TypeError, ValueError):
            # Ignore unparsable sensor_id; report as unknown
            pass

    return times, sensor_ids


def _summarize_file(path: Path, explicit_sensor_id: Optional[int] = None) -> None:
    """Compute and print a sampling-rate summary for a single log file."""
    suffix = path.suffix.lower()
    if suffix.endswith(".csv"):
        rows = _load_rows_csv(path)
    elif suffix.endswith(".jsonl"):
        rows = _load_rows_jsonl(path)
    else:
        print(f"\n=== Sample rate check ===")
        print(f"File: {path}")
        print(f"  WARNING: unsupported extension {path.suffix!r}; skipping.")
        return

    print("\n=== Sample rate check ===")
    print(f"File: {path}")

    if not rows:
        print("  WARNING: file is empty; cannot estimate rate.")
        return

    times, sensor_ids = _extract_times(rows)
    if len(times) < 2:
        print(
            f"  WARNING: only {len(times)} timestamped samples; "
            "cannot estimate rate."
        )
        return

    t_first = times[0]
    t_last = times[-1]
    t_span = t_last - t_first
    n_samples = len(times)

    if t_span <= 0:
        print(
            f"  WARNING: non-positive time span ({t_span:.6f} s); "
            "cannot estimate rate."
        )
        return

    rate_est = n_samples / t_span

    # Determine sensor_id info
    sid_text = "(unknown)"
    if explicit_sensor_id is not None:
        sid_text = str(explicit_sensor_id)
    elif sensor_ids:
        unique_ids = sorted(set(sensor_ids))
        if len(unique_ids) == 1:
            sid_text = str(unique_ids[0])
        else:
            sid_text = f"mixed {unique_ids}"

    print(f"  sensor_id: {sid_text}")
    print(f"  samples: {n_samples}")
    print(f"  time_span: {t_span:.3f} s")
    print(f"  estimated_rate: {rate_est:.2f} Hz")

    meta = _load_meta(path)
    if meta:
        dev_rate = meta.get("device_rate_hz")
        requested = meta.get("requested_rate_hz", meta.get("sample_rate_hz"))
        stream_every = meta.get("stream_every")

        if dev_rate is not None:
            try:
                dev_rate_f = float(dev_rate)
                delta = rate_est - dev_rate_f
                pct = (delta / dev_rate_f * 100.0) if dev_rate_f != 0 else 0.0
                print(
                    f"  meta.device_rate_hz: {dev_rate_f:.2f} Hz "
                    f"(delta: {delta:+.2f} Hz, {pct:+.1f} %)")
            except (TypeError, ValueError):
                print(f"  meta.device_rate_hz: {dev_rate!r} (unparsable)")

        if requested is not None:
            try:
                req_f = float(requested)
                print(f"  meta.requested_rate_hz: {req_f:.2f} Hz")
            except (TypeError, ValueError):
                print(f"  meta.requested_rate_hz: {requested!r}")

        if stream_every is not None:
            print(f"  meta.stream_every: {stream_every}")
    else:
        print("  (no .meta.json sidecar found)")


def _iter_log_files(root: Path, pattern: str) -> Iterable[Path]:
    """Yield all files matching pattern under root, sorted by name."""
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate recorded sampling rate from MPU6050 log files.\n"
            "Intended to be run on the Raspberry Pi in raspberrypi_scripts/."
        )
    )
    parser.add_argument(
        "path",
        help="Path to a CSV/JSONL log file or to a directory of log files.",
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="Glob pattern when PATH is a directory (default: '*.csv').",
    )
    parser.add_argument(
        "--sensor-id",
        type=int,
        default=None,
        help=(
            "Optional sensor_id to associate with logs that do not contain a "
            "sensor_id column. If provided, it overrides any inferred ID."
        ),
    )
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        any_files = False
        for log_path in _iter_log_files(target, args.glob):
            any_files = True
            _summarize_file(log_path, explicit_sensor_id=args.sensor_id)
        if not any_files:
            print(f"No files matched {args.glob!r} in {target}")
    else:
        _summarize_file(target, explicit_sensor_id=args.sensor_id)


if __name__ == "__main__":
    main()
