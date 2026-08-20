#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_actual_rate.py
=====================
Reports the *actually achieved* per-sensor sample rate from recorded CSV
data, as opposed to the configured/theoretical rate.

Why this is needed: the rate shown on the OLED (and written into each
session's ``.meta.json`` as ``device_rate_hz``) comes from the MPU6050's
register math at startup (``SMPLRT_DIV``) -- it's what the sensor is
*configured* to output, computed once before the sampling loop starts. It
is NOT a live measurement, so it says nothing about whether the Pi's
software loop (I2C reads across up to 4 sensors, per-sample writes, disk
I/O) is actually able to keep up with that rate in practice. Two runs
could both report "100 Hz configured" while one silently only *achieves*
80 Hz because of overruns -- the configured-rate number can't tell you
that; only the recorded timestamps can.

This script computes the real rate directly from each sensor's own
recorded timestamps: actual_hz = (n_samples - 1) / (last_t_s - first_t_s).

Usage (on the Pi, or anywhere the log directory is reachable):
    python3 check_actual_rate.py [--dir /home/verwalter/logs/mpu] [--session SESSION_PREFIX]

With no arguments, it finds the most recent session in the default log
directory (matched by the timestamp embedded in the filenames) and reports
every sensor found in it.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG_DIR = "/home/verwalter/logs/mpu"

# Matches both filename shapes produced by build_log_file_paths()
# (see src/sensepi/config/log_paths.py):
#   no --session-name:      mpu_S1_2026-08-18_12-44-35.csv
#   with --session-name X:  x_mpu_S1_2026-08-18_12-44-35.csv  (slug prefixed, not appended)
# The timestamp is anchored to its known YYYY-MM-DD_HH-MM-SS shape so parsing
# is unambiguous regardless of what the session-name slug itself contains.
FILENAME_RE = re.compile(
    r"^(?:(?P<prefix>.+)_)?mpu_S(?P<sid>\d+)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$"
)


def find_sessions(log_dir: Path) -> dict[str, dict[int, Path]]:
    """
    Group CSV files by run, keyed by timestamp (unique per run, shared across
    that run's sensors regardless of session-name prefix).

    Searches recursively: a run started with --session-name writes into a
    subdirectory (build_pi_session_dir() nests it under log_dir/<slug>/),
    not directly in log_dir, so a plain top-level glob misses it.
    """
    sessions: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in log_dir.rglob("*mpu_S*.csv"):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        sensor_id = int(m.group("sid"))
        ts = m.group("ts")
        sessions[ts][sensor_id] = path
    return sessions


def session_prefix(sessions: dict[str, dict[int, Path]], ts: str) -> str | None:
    """Return the session-name slug (if any) used for the given run, from its filenames."""
    for path in sessions.get(ts, {}).values():
        m = FILENAME_RE.match(path.name)
        if m and m.group("prefix"):
            return m.group("prefix")
        return None
    return None


def resolve_session(sessions: dict[str, dict[int, Path]], requested: str) -> str | None:
    """Resolve a user-provided --session value against either the raw timestamp or the session-name slug."""
    if requested in sessions:
        return requested
    for ts in sessions:
        if session_prefix(sessions, ts) == requested:
            return ts
    return None


def latest_session(sessions: dict[str, dict[int, Path]]) -> str | None:
    if not sessions:
        return None
    # Timestamps are "YYYY-MM-DD_HH-MM-SS" -- lexicographic sort == chronological.
    return sorted(sessions.keys())[-1]


def configured_rate_hz(csv_path: Path) -> float | None:
    """Read the theoretical configured rate from the matching .meta.json, if present."""
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return float(meta.get("device_rate_hz"))
    except Exception:
        return None


def actual_rate_hz(csv_path: Path) -> tuple[int, float, float] | None:
    """Return (n_samples, duration_s, actual_hz) computed from real t_s timestamps."""
    first_t = last_t = None
    n = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "t_s" not in (reader.fieldnames or []):
            return None
        for row in reader:
            try:
                t = float(row["t_s"])
            except (TypeError, ValueError):
                continue
            if first_t is None:
                first_t = t
            last_t = t
            n += 1
    if n < 2 or first_t is None or last_t is None:
        return None
    duration = last_t - first_t
    if duration <= 0:
        return None
    return n, duration, (n - 1) / duration


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=DEFAULT_LOG_DIR, help=f"Log directory (default: {DEFAULT_LOG_DIR})")
    ap.add_argument(
        "--session",
        default=None,
        help="Session timestamp (YYYY-MM-DD_HH-MM-SS) or --session-name slug to check (default: most recent)",
    )
    args = ap.parse_args()

    log_dir = Path(args.dir)
    if not log_dir.is_dir():
        raise SystemExit(f"ERROR: log directory not found: {log_dir}")

    sessions = find_sessions(log_dir)
    if not sessions:
        raise SystemExit(f"ERROR: no mpu_S*.csv files found under {log_dir}")

    if args.session:
        session = resolve_session(sessions, args.session)
        if session is None:
            names = [f"{ts} ({session_prefix(sessions, ts)})" if session_prefix(sessions, ts) else ts
                     for ts in sorted(sessions.keys())[-10:]]
            raise SystemExit(f"ERROR: session '{args.session}' not found. Recent sessions: {', '.join(names)}")
    else:
        session = latest_session(sessions)

    prefix = session_prefix(sessions, session)
    label = f"{session} (--session-name '{prefix}')" if prefix else session
    print(f"Session: {label}  (dir: {log_dir})\n")
    header = f"{'Sensor':<8}{'Samples':<10}{'Duration(s)':<14}{'Actual Hz':<12}{'Configured Hz':<15}{'Delta':<10}"
    print(header)
    print("-" * len(header))

    for sid in sorted(sessions[session].keys()):
        csv_path = sessions[session][sid]
        result = actual_rate_hz(csv_path)
        cfg = configured_rate_hz(csv_path)
        if result is None:
            print(f"S{sid:<7}{'--':<10}{'--':<14}{'(too few samples)':<12}")
            continue
        n, duration, hz = result
        cfg_str = f"{cfg:.3f}" if cfg is not None else "?"
        delta_str = f"{hz - cfg:+.3f}" if cfg is not None else "?"
        print(f"S{sid:<7}{n:<10}{duration:<14.3f}{hz:<12.3f}{cfg_str:<15}{delta_str:<10}")

    print(
        "\n'Actual Hz' is measured directly from recorded timestamps "
        "(ground truth). 'Configured Hz' is the sensor's register-level "
        "setting reported at startup, not a live measurement -- it will "
        "read ~100 regardless of whether the Pi actually kept up. A large "
        "negative delta (actual noticeably below configured) means the "
        "sampling loop was falling behind (I2C/disk overhead across "
        "however many sensors are active)."
    )


if __name__ == "__main__":
    main()
