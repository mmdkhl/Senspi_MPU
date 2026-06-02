"""Load a recorded sensor session into per-sensor ``ax`` arrays for modal ID.

The authoritative recording is the Pi logger's ``.jsonl`` / ``.csv`` files,
downloaded into ``data/raw/<session>/`` by ``remote.log_sync``. Each JSONL line
carries ``sensor_id``, ``ax`` and a timestamp. This module parses those lines
with the shared :func:`sensepi.sensors.mpu6050.parse_line`, groups samples by
sensor, trims to the requested window, and resamples every sensor onto one
uniform time grid so the FDD step in :mod:`sensepi.analysis.modal` gets aligned
channels.

Pure I/O + numpy — no Qt, no SSH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config.app_config import AppPaths
from ..sensors.mpu6050 import parse_line

_LOG_SUFFIXES = {".jsonl", ".csv", ".log", ".txt"}


@dataclass
class ModalSession:
    """Aligned per-sensor data ready for :func:`modal.identify_modes`."""

    data: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # (n_sensors, n_samples)
    fs: float = float("nan")
    sensor_ids: list[int] = field(default_factory=list)
    duration_s: float = 0.0
    nan_fraction: float = 0.0
    source: str = ""
    success: bool = False
    message: str = ""


def list_sessions(base: Path | None = None) -> list[Path]:
    """Return recorded-session directories under ``data/raw``, newest first."""
    root = base or AppPaths().raw_data
    if not root.is_dir():
        return []
    sessions = [p for p in root.iterdir() if p.is_dir()]
    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions


def find_session_files(session_dir: Path) -> list[Path]:
    """Find candidate log files within a session directory (recursively)."""
    if session_dir.is_file():
        return [session_dir]
    files = [p for p in session_dir.rglob("*") if p.is_file() and p.suffix.lower() in _LOG_SUFFIXES]
    files.sort()
    return files


def _collect_samples(paths: list[Path], axis: str) -> dict[int, list[tuple[float, float]]]:
    """Parse files into {sensor_id: [(t_seconds, axis_value), ...]}."""
    attr = axis.lower()
    per_sensor: dict[int, list[tuple[float, float]]] = {}
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    sample = parse_line(line)
                    if sample is None:
                        continue
                    sid = sample.sensor_id if sample.sensor_id is not None else 1
                    if sample.t_s is not None:
                        t = float(sample.t_s)
                    else:
                        t = float(sample.timestamp_ns) / 1e9
                    value = getattr(sample, attr, None)
                    if value is None:
                        continue
                    per_sensor.setdefault(int(sid), []).append((t, float(value)))
        except OSError:
            continue
    return per_sensor


def load_session(
    source: Path,
    *,
    axis: str = "ax",
    last_seconds: float | None = None,
    target_fs: float | None = None,
    use_index_time: bool = False,
) -> ModalSession:
    """Load a session directory (or single file) into aligned channels.

    Parameters
    ----------
    source : Path
        Session directory or a single log file.
    axis : str
        Acceleration channel to extract (default ``ax``).
    last_seconds : float, optional
        Keep only the most recent ``last_seconds`` of the recording.
    target_fs : float, optional
        Resampling rate. If None, estimated from the densest sensor's timestamps.
    """
    paths = find_session_files(Path(source))
    if not paths:
        return ModalSession(source=str(source), message="No log files found in session.")

    per_sensor = _collect_samples(paths, axis)
    if not per_sensor:
        return ModalSession(source=str(source), message="No parseable samples found.")

    series = {sid: list(pairs) for sid, pairs in per_sensor.items()}
    return align_per_sensor_series(
        series, last_seconds=last_seconds, target_fs=target_fs,
        use_index_time=use_index_time, source=str(source),
    )


def align_per_sensor_series(
    series: dict[int, list[tuple[float, float]]],
    *,
    last_seconds: float | None = None,
    target_fs: float | None = None,
    use_index_time: bool = False,
    source: str = "",
) -> ModalSession:
    """Align per-sensor ``(t_seconds, value)`` samples onto a common uniform grid.

    Shared by the disk loader and the live capture path so both produce an
    identical :class:`ModalSession` for the FDD step.

    When ``use_index_time`` is True, each sensor's timestamps are IGNORED and
    rebuilt from sample index at ``target_fs`` (i.e. "trust this rate, not the
    timestamps"). This rescues recordings whose timestamps are missing or
    unreliable; it requires ``target_fs`` to be set.
    """
    sensor_ids = sorted(series.keys())
    if not sensor_ids:
        return ModalSession(source=source, message="No sensors in series.")

    if use_index_time and (target_fs is None or target_fs <= 0):
        return ModalSession(source=source, sensor_ids=sensor_ids,
                            message="A sample rate is required when ignoring timestamps.")

    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid in sensor_ids:
        pairs = sorted(series[sid], key=lambda p: p[0])
        v = np.array([p[1] for p in pairs], dtype=float)
        if use_index_time:
            t = np.arange(v.size, dtype=float) / float(target_fs)
        else:
            t = np.array([p[0] for p in pairs], dtype=float)
        arrays[sid] = (t, v)

    if target_fs is None:
        densest = max(sensor_ids, key=lambda s: arrays[s][0].size)
        t = arrays[densest][0]
        if t.size < 2:
            return ModalSession(source=source, sensor_ids=sensor_ids,
                                message="Not enough samples to estimate sampling rate.")
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            return ModalSession(source=source, sensor_ids=sensor_ids,
                                message="Non-monotonic timestamps; cannot resample.")
        target_fs = float(1.0 / np.median(dt))

    t_start = max(arrays[s][0][0] for s in sensor_ids)
    t_end = min(arrays[s][0][-1] for s in sensor_ids)
    if t_end <= t_start:
        return ModalSession(source=source, sensor_ids=sensor_ids,
                            message="Sensors do not share an overlapping time window.")
    if last_seconds is not None and last_seconds > 0:
        t_start = max(t_start, t_end - float(last_seconds))

    n_samples = int(max(2, round((t_end - t_start) * target_fs)))
    grid = t_start + np.arange(n_samples) / target_fs

    rows = []
    total_nan = 0
    for sid in sensor_ids:
        t, v = arrays[sid]
        finite = np.isfinite(t) & np.isfinite(v)
        total_nan += int(np.sum(~np.isfinite(v)))
        t_f, v_f = t[finite], v[finite]
        if t_f.size < 2:
            return ModalSession(source=source, sensor_ids=sensor_ids,
                                message=f"Sensor {sid} has too few valid samples.")
        rows.append(np.interp(grid, t_f, v_f))

    data = np.vstack(rows)
    total_points = sum(arrays[s][0].size for s in sensor_ids)
    nan_fraction = total_nan / total_points if total_points else 0.0

    return ModalSession(
        data=data,
        fs=target_fs,
        sensor_ids=sensor_ids,
        duration_s=float(n_samples / target_fs),
        nan_fraction=nan_fraction,
        source=source,
        success=True,
        message=(
            f"Loaded {len(sensor_ids)} sensors, {n_samples} samples @ {target_fs:.1f} Hz "
            f"({n_samples / target_fs:.1f} s)."
        ),
    )
