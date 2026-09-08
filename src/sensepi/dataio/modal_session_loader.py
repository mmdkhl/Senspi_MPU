"""Load a recorded sensor session into per-sensor ``ax`` arrays for modal ID.

The authoritative recording is the Pi logger's ``.jsonl`` / ``.csv`` files,
downloaded into ``output/sensor_recordings/<session>/`` by ``remote.log_sync``.
Each JSONL line
carries ``sensor_id``, ``ax`` and a timestamp. This module parses those lines
with the shared :func:`sensepi.sensors.mpu6050.parse_line`, groups samples by
sensor, trims to the requested window, and resamples every sensor onto one
uniform time grid so the FDD step in :mod:`sensepi.analysis.modal` gets aligned
channels.

Pure I/O + numpy — no Qt, no SSH.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config.app_config import AppPaths
from ..sensors.mpu6050 import parse_line

_LOG_SUFFIXES = {".jsonl", ".csv", ".log", ".txt"}
_SID_RE = re.compile(r"_S(\d+)_")


def _sid_from_filename(path: Path) -> int | None:
    """Recover a sensor id from a per-sensor filename like ``mpu_S2_...``."""
    m = _SID_RE.search(path.name)
    return int(m.group(1)) if m else None


def _is_header_line(line: str) -> bool:
    """A CSV header has a first token that is not a number (e.g. 'timestamp_ns')."""
    first = line.split(",", 1)[0].strip()
    if not first:
        return False
    try:
        float(first)
        return False
    except ValueError:
        return True


def _row_time_seconds(row: dict) -> float | None:
    """Prefer ``t_s`` (seconds since start); fall back to ``timestamp_ns``."""
    for key, scale in (("t_s", 1.0), ("timestamp_ns", 1e-9)):
        val = row.get(key)
        if val not in (None, ""):
            try:
                return float(val) * scale
            except (TypeError, ValueError):
                return None
    return None


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


def sliced_session(session: "ModalSession", rows) -> "ModalSession":
    """A ModalSession restricted to some of its sensor rows.

    Used to drop the floor-0 (shaker) row before an output-only identification:
    it measures the input, and feeding it in as a response biases the picked
    modes toward the excitation. Keeps ``sensor_ids`` aligned with the rows so
    the story mapping downstream still refers to the right sensors.
    """
    rows = [int(r) for r in rows]
    ids = list(session.sensor_ids)
    return ModalSession(
        source=session.source, fs=float(session.fs),
        data=np.asarray(session.data, dtype=float)[rows, :],
        sensor_ids=[ids[i] for i in rows],
        duration_s=float(session.duration_s),
        nan_fraction=float(getattr(session, "nan_fraction", 0.0)),
        success=bool(session.success), message=str(session.message),
    )


def _is_audit_path(path: Path) -> bool:
    """Smart Record's per-recording audit copy (``audit_<stamp>/raw_S<n>_….csv``).

    It holds the same samples as the main files, before decimation. Loading it
    alongside them doubled every sensor's series — harmless only while
    decimation was 1, which is why it went unnoticed.
    """
    return any(part.startswith("audit_") for part in path.parts) \
        or path.name.startswith("raw_S")


def _has_log_files(directory: Path) -> bool:
    try:
        return any(p.is_file() and p.suffix.lower() in _LOG_SUFFIXES
                   and not _is_audit_path(p) for p in directory.iterdir())
    except OSError:
        return False


def list_sessions(base: Path | None = None) -> list[Path]:
    """Recording folders under the recordings root, newest first.

    A *session* is a directory that directly contains log files. Smart Record
    writes one such folder per recording (``<host>/mpu/<stamp>[_<name>]/``), and
    the log-sync path writes ``<name>/`` or ``<host>/<prefix>/``; both are found
    by the same rule. This used to list only the root's immediate children — the
    host folders — so every recording under a host was globbed into one session
    and two tests were silently averaged together.
    """
    root = base or AppPaths().sensor_recordings
    if not root.is_dir():
        return []
    sessions: list[Path] = []
    for directory in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if any(part.startswith("audit_") for part in directory.relative_to(root).parts):
            continue
        if _has_log_files(directory):
            sessions.append(directory)

    def newest(directory: Path) -> float:
        try:
            return max((p.stat().st_mtime for p in directory.iterdir() if p.is_file()),
                       default=directory.stat().st_mtime)
        except OSError:
            return 0.0

    sessions.sort(key=newest, reverse=True)
    return sessions


def find_session_files(session_dir: Path) -> list[Path]:
    """The log files of ONE recording.

    Files directly inside the folder when there are any; only then does it look
    deeper (a legacy host folder with recordings loose underneath). The audit
    copy is excluded either way — see :func:`_is_audit_path`.
    """
    if session_dir.is_file():
        return [session_dir]

    def wanted(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in _LOG_SUFFIXES and not _is_audit_path(p)

    files = [p for p in session_dir.iterdir() if wanted(p)] if session_dir.is_dir() else []
    if not files and session_dir.is_dir():
        files = [p for p in session_dir.rglob("*") if wanted(p)]
    files.sort()
    return files


def _collect_samples(paths: list[Path], axis: str) -> dict[int, list[tuple[float, float]]]:
    """Parse files into ``{sensor_id: [(t_seconds, axis_value), ...]}``.

    Handles three on-disk formats produced by the Pi logger:
    - **JSONL** (lines starting with ``{``) — via :func:`parse_line`.
    - **Header CSV** (named columns, e.g. ``timestamp_ns,t_s,sensor_id,ax,ay,gz``)
      — read by column NAME so it works with any channel subset.
    - **Legacy positional CSV** (no header, ``ts,ax,ay,az,gx,gy,gz``) — via
      :func:`parse_line`, with sensor id recovered from the filename.
    """
    attr = axis.lower()
    per_sensor: dict[int, list[tuple[float, float]]] = {}
    for path in paths:
        sid_default = _sid_from_filename(path) or 1
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                first = ""
                while first == "":
                    line = fh.readline()
                    if not line:
                        break
                    first = line.strip()
                fh.seek(0)
                if not first:
                    continue

                if first.startswith("{"):
                    _collect_jsonl(fh, attr, sid_default, per_sensor)
                elif _is_header_line(first):
                    _collect_header_csv(fh, attr, sid_default, per_sensor)
                else:
                    _collect_positional(fh, attr, sid_default, per_sensor)
        except OSError:
            continue
    return per_sensor


def _collect_jsonl(fh, attr, sid_default, per_sensor) -> None:
    for line in fh:
        sample = parse_line(line)
        if sample is None:
            continue
        value = getattr(sample, attr, None)
        if value is None or (isinstance(value, float) and value != value):  # skip NaN
            continue
        t = float(sample.t_s) if sample.t_s is not None else float(sample.timestamp_ns) / 1e9
        sid = sample.sensor_id if sample.sensor_id is not None else sid_default
        per_sensor.setdefault(int(sid), []).append((t, float(value)))


def _collect_header_csv(fh, attr, sid_default, per_sensor) -> None:
    reader = csv.DictReader(fh)
    if not reader.fieldnames or attr not in reader.fieldnames:
        return  # this file doesn't carry the requested axis
    for row in reader:
        raw = row.get(attr)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        t = _row_time_seconds(row)
        if t is None:
            continue
        sid_raw = row.get("sensor_id")
        try:
            sid = int(float(sid_raw)) if sid_raw not in (None, "") else sid_default
        except (TypeError, ValueError):
            sid = sid_default
        per_sensor.setdefault(sid, []).append((t, value))


def _collect_positional(fh, attr, sid_default, per_sensor) -> None:
    for line in fh:
        sample = parse_line(line)
        if sample is None:
            continue
        value = getattr(sample, attr, None)
        if value is None or (isinstance(value, float) and value != value):
            continue
        t = float(sample.t_s) if sample.t_s is not None else float(sample.timestamp_ns) / 1e9
        sid = sample.sensor_id if sample.sensor_id is not None else sid_default
        per_sensor.setdefault(int(sid), []).append((t, float(value)))


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
