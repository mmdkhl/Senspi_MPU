#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_sensor_mapping.py
========================
Diagnose whether logical sensor IDs match physical placement, and whether all
sensors are oriented consistently -- the two things that most often produce a
mode shape that "can't be right" (e.g. floor 2 reading lower than floor 1 in
the first mode).

Sensor IDs are NOT assigned by physical labels or cable order. They come purely
from I2C wiring (see default_mapping() in mpu6050_multi_logger.py):

    ID 1 = bus 1, 0x68 (AD0 low)      ID 3 = bus 0, 0x68 (AD0 low)
    ID 2 = bus 1, 0x69 (AD0 high)     ID 4 = bus 0, 0x69 (AD0 high)

So a sensor mounted on floor 2 but plugged into bus 0 is logged as ID 3 or 4.

What this reports, per sensor:

  RMS        Excitation amplitude on the chosen axis, AC-coupled (mean removed,
             so gravity/DC offset doesn't dominate). In a first mode, RMS should
             INCREASE monotonically with height. A sensor that breaks that trend
             is either on a different floor than assumed, misoriented, or loose.

  Corr       Pearson correlation of that sensor against a reference sensor.
             In a first mode all floors move essentially in phase, so this
             should be strongly POSITIVE (typically > 0.7) for every sensor.
             A strongly NEGATIVE value means that sensor's axis points the
             opposite way -- it's mounted rotated 180 deg about vertical. That
             flips the sign of its mode-shape contribution and, if it shares a
             floor with another sensor, partially cancels it in the average.
             A near-ZERO value means it's sensing a perpendicular direction
             (rotated ~90 deg) or is just measuring noise.

Usage:
    python3 check_sensor_mapping.py [--dir DIR] [--session NAME] [--axis ax]

Run it on a recording where the structure was actually excited (a shake-table
run, or just push-and-release the top floor). Static/ambient data won't show a
usable mode.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG_DIR = "/home/verwalter/logs/mpu"

FILENAME_RE = re.compile(
    r"^(?:(?P<prefix>.+)_)?mpu_S(?P<sid>\d+)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$"
)


def find_sessions(log_dir: Path) -> dict[str, dict[int, Path]]:
    sessions: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in log_dir.rglob("*mpu_S*.csv"):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        sessions[m.group("ts")][int(m.group("sid"))] = path
    return sessions


def session_prefix(sessions: dict[str, dict[int, Path]], ts: str) -> str | None:
    for path in sessions.get(ts, {}).values():
        m = FILENAME_RE.match(path.name)
        return m.group("prefix") if m and m.group("prefix") else None
    return None


def resolve_session(sessions: dict[str, dict[int, Path]], requested: str) -> str | None:
    if requested in sessions:
        return requested
    for ts in sessions:
        if session_prefix(sessions, ts) == requested:
            return ts
    return None


def read_axis(csv_path: Path, axis: str) -> list[float]:
    vals: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if axis not in (reader.fieldnames or []):
            return []
        for row in reader:
            try:
                vals.append(float(row[axis]))
            except (TypeError, ValueError):
                continue
    return vals


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def rms_ac(xs: list[float]) -> float:
    """RMS after removing the mean (strips gravity/DC bias)."""
    if not xs:
        return 0.0
    mu = mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check sensor ID <-> placement/orientation consistency")
    ap.add_argument("--dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--session", default=None, help="Session timestamp or --session-name slug (default: most recent)")
    ap.add_argument("--axis", default="ax", help="Axis to analyze (default: ax, the one mode shapes use)")
    ap.add_argument("--ref", type=int, default=None, help="Reference sensor id for correlation (default: lowest id)")
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
            names = sorted(sessions.keys())[-10:]
            raise SystemExit(f"ERROR: session '{args.session}' not found. Recent: {', '.join(names)}")
    else:
        session = sorted(sessions.keys())[-1]

    prefix = session_prefix(sessions, session)
    label = f"{session} (--session-name '{prefix}')" if prefix else session
    print(f"Session: {label}")
    print(f"Axis:    {args.axis}\n")

    series: dict[int, list[float]] = {}
    for sid in sorted(sessions[session]):
        vals = read_axis(sessions[session][sid], args.axis)
        if vals:
            series[sid] = vals

    if not series:
        raise SystemExit(f"ERROR: no '{args.axis}' column found in this session's files.")

    ref = args.ref if args.ref in series else min(series)
    ref_vals = series[ref]

    header = f"{'Sensor':<9}{'Samples':<10}{'RMS (AC)':<14}{f'Corr vs S{ref}':<15}{'Note':<34}"
    print(header)
    print("-" * len(header))

    rms_by_sid: dict[int, float] = {}
    for sid, vals in series.items():
        r = rms_ac(vals)
        rms_by_sid[sid] = r
        c = 1.0 if sid == ref else pearson(vals, ref_vals)
        if sid == ref:
            note = "(reference)"
        elif c < -0.5:
            note = "FLIPPED? mounted 180 deg rotated"
        elif abs(c) < 0.3:
            note = "PERPENDICULAR/noise? check axis"
        elif c > 0.5:
            note = "in phase (expected)"
        else:
            note = "weak correlation - inspect"
        print(f"S{sid:<8}{len(vals):<10}{r:<14.5f}{c:<15.3f}{note:<34}")

    print("\nInterpretation")
    print("  - First mode: RMS should increase with height (floor 1 < floor 2 < floor 3).")
    print("  - Any sensor breaking that trend is likely on a different floor than")
    print("    assumed, misoriented, or not rigidly mounted.")
    print("  - Corr should be strongly positive for all sensors in a first mode.")
    print("    Strongly negative = rotated 180 deg (sign-flipped contribution).")
    print("    Near zero = sensing a perpendicular axis, or no real excitation.")

    ordered = sorted(rms_by_sid.items(), key=lambda kv: kv[1])
    ranking = " < ".join(f"S{sid}({r:.4f})" for sid, r in ordered)
    print(f"\n  RMS ranking (low to high): {ranking}")
    print("  Compare this against your physical floor order. If the sensor you")
    print("  mounted lowest is not lowest here, the ID<->floor mapping is wrong:")
    print("  IDs come from I2C wiring (bus + AD0), not from cable or label order.")


if __name__ == "__main__":
    main()
