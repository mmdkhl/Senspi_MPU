#!/usr/bin/env python3
"""Generate a faithful FAKE recorded session for plumbing-testing the
sensor → OpenSees model-updating workflow WITHOUT live hardware.

It reproduces exactly what the Live Signals recording leaves on disk after a
Pi capture is synced down:

    data/raw/<session>/mpu/<slug>_mpu_S<id>_<ts>.jsonl   (one file per sensor)
    data/raw/<session>/mpu/<slug>_mpu_S<id>_<ts>.jsonl.meta.json

Each JSONL line matches the real wire schema
(``timestamp_ns, t_s, sensor_id, ax, ay, az, gx, gy, gz``; ax/ay/az in m/s²).

The acceleration is NOT pure sines — each mode is a lightly damped resonance
excited by broadband noise (band-pass-filtered white noise), so the PSD has
realistic peaks with width, the way a real shaker test would. This exercises
the FDD peak-picking under noise.

Physical model: a 3-story shear frame, sensor S1=floor 1, S2=floor 2,
S3=floor 3 (one sensor per floor → full mode-shape coverage). For this file,
use the **"Fully instrumented (1 per floor)"** preset in the Sensor
Configuration panel.

Usage:
    python tools/make_fake_recording.py            # 30 s @ 200 Hz into data/raw
    python tools/make_fake_recording.py --duration 60 --rate 200
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal

# Truth used to synthesize the response (close to the project's target data).
TRUE_FREQS = [2.32, 6.5, 9.1]            # Hz
ZETA = [0.018, 0.015, 0.012]             # modal damping ratios
MODE_SHAPES = [                          # signed, per floor (story 1, 2, 3)
    [0.30, 0.75, 1.00],
    [-1.00, 0.10, 0.95],
    [1.00, -0.85, 0.30],
]
MODE_PARTICIPATION = [1.0, 0.45, 0.30]   # lower modes dominate the response
ACCEL_SCALE = 0.6                        # m/s² overall response level
SENSOR_NOISE = 0.01                      # m/s² broadband measurement noise


def _modal_coordinate(f_n: float, zeta: float, n: int, fs: float, rng) -> np.ndarray:
    """White noise passed through a narrow band-pass around f_n (a resonance)."""
    white = rng.standard_normal(n)
    bw = max(0.15, 2.0 * zeta * f_n)          # bandwidth ~ 2*zeta*f_n
    lo = max(0.02, (f_n - bw)) / (fs / 2.0)
    hi = min(0.98, (f_n + bw) / (fs / 2.0))
    b, a = signal.butter(2, [lo, hi], btype="band")
    q = signal.filtfilt(b, a, white)
    return q / (np.std(q) + 1e-12)


def generate(fs: float, duration: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration))
    t = np.arange(n) / fs

    q = np.vstack([
        MODE_PARTICIPATION[m] * _modal_coordinate(TRUE_FREQS[m], ZETA[m], n, fs, rng)
        for m in range(len(TRUE_FREQS))
    ])                                         # (n_modes, n)
    phi = np.array(MODE_SHAPES)                # (n_modes, n_floors)
    ax = ACCEL_SCALE * (phi.T @ q)             # (n_floors=3, n)
    ax += SENSOR_NOISE * rng.standard_normal(ax.shape)
    return t, ax


def write_session(out_root: Path, fs: float, duration: float, seed: int = 0) -> Path:
    t, ax = generate(fs, duration, seed)
    n = t.size
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"synthetic_shake_{ts_str}"
    session_dir = out_root / slug
    mpu_dir = session_dir / "mpu"          # mirrors the synced Pi layout
    mpu_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + 99)
    for sid in (1, 2, 3):
        fpath = mpu_dir / f"{slug}_mpu_S{sid}_{ts_str}.jsonl"
        floor = sid - 1
        with fpath.open("w", encoding="utf-8") as fh:
            for i in range(n):
                row = {
                    "timestamp_ns": int(t[i] * 1e9),
                    "t_s": round(float(t[i]), 6),
                    "sensor_id": sid,
                    "ax": round(float(ax[floor, i]), 6),
                    "ay": round(float(0.02 * rng.standard_normal()), 6),
                    "az": round(float(9.81 + 0.02 * rng.standard_normal()), 6),
                    "gx": round(float(0.05 * rng.standard_normal()), 6),
                    "gy": round(float(0.05 * rng.standard_normal()), 6),
                    "gz": round(float(0.05 * rng.standard_normal()), 6),
                }
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        meta = {
            "sensor_id": sid,
            "rate_hz": fs,
            "samples": n,
            "duration_s": duration,
            "format": "jsonl",
            "note": "SYNTHETIC plumbing-test data (tools/make_fake_recording.py)",
        }
        fpath.with_suffix(fpath.suffix + ".meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    return session_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", type=float, default=200.0, help="Sampling rate Hz")
    ap.add_argument("--duration", type=float, default=30.0, help="Seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="", help="Output raw-data root (default: data/raw)")
    args = ap.parse_args()

    if args.out:
        out_root = Path(args.out).expanduser().resolve()
    else:
        # Default to the project's data/raw via AppPaths.
        from sensepi.config.app_config import AppPaths
        out_root = AppPaths().raw_data
    out_root.mkdir(parents=True, exist_ok=True)

    session_dir = write_session(out_root, args.rate, args.duration, args.seed)
    print(f"Wrote synthetic session: {session_dir}")
    print("In the app: Model Updating ▸ Calibration ▸ source 'From Sensors' ▸")
    print("  preset 'Fully instrumented (1 per floor)' ▸ 'Use latest' ▸ Load & Identify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
