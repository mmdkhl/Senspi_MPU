#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpu6050_multi_logger.py
=======================
Raspberry Pi Zero — Multi‑MPU6050 local logger (no MQTT)

Changes in this version
-----------------------
- Less intrusive file flushing:
  * Periodic flush thresholds increased (defaults: 2000 rows, 2.0 s).
  * os.fsync() is optional during periodic flushes (default: disabled),
    but still performed once at shutdown.
- New CLI knobs: --flush-every, --flush-seconds, --fsync-each-flush

Features
--------
- Supports up to **three** MPU‑6050 sensors (default mapping below).
- Select which sensors to enable: --sensors 1,2,3
- Select which channels to record: --channels acc|gyro|both|default
  * default = AX, AY, and GZ (XY acceleration + yaw-rate around Z)
- Control sampling rate (Hz): --rate
- Local CSV or JSONL logging (one file per sensor) + metadata JSON.
- Drift‑corrected sampling loop using time.monotonic_ns().
- Per‑sensor writer thread for low‑latency I/O.
- Resilient to I²C hiccups; keeps other sensors running.
- Device scan: --list prints addresses seen on bus 0 and 1.

Default mapping
---------------
Sensor 1 → bus 1, address 0x68
Sensor 2 → bus 1, address 0x69
Sensor 3 → bus 0, address 0x68
(Override with --map "1:1-0x68,2:1-0x69,3:0-0x68")

Scaling (matches prior reference)
---------------------------------
Accel raw → g = raw / 16384.0, then * 9.80665  → m/s² (±2 g range)
Gyro  raw → dps = raw / 131.0                 → deg/s (±250 °/s range)

Install (on Raspberry Pi OS)
----------------------------
sudo apt-get update
sudo apt-get install -y python3-pip python3-smbus i2c-tools
pip3 install smbus2
sudo raspi-config nonint do_i2c 0   # ensure I2C enabled

Examples
--------
# List devices on bus 0 and 1
python3 mpu6050_multi_logger.py --list

# Two sensors, 100 Hz, both acc+gyro, log 10 s
python3 mpu6050_multi_logger.py --rate 100 --sensors 1,2 --channels both --duration 10 --out ./logs

# Gyro‑only from sensor 3 at 200 Hz until Ctrl‑C
python3 mpu6050_multi_logger.py --rate 200 --sensors 3 --channels gyro --out ./logs

# "default" selection — AX, AY and GZ
python3 mpu6050_multi_logger.py --rate 100 --channels default

Configuration via YAML
----------------------
This logger can read defaults from a small YAML file on the Pi. Use
``--config /path/to/pi_config.yaml`` explicitly, or omit ``--config``
and a ``pi_config.yaml`` that lives next to this script will be used if
present.

The merge strategy is:

1. Read defaults from the ``mpu6050`` section.
2. Apply explicit command-line options on top (CLI overrides config).
3. For boolean flags such as ``--temp``, ``--no-record`` and
   ``--stream-stdout``, the config controls the default state and the
   CLI can only enable additional behaviour.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEBUG_STREAM = False

from pi_logger_common import load_config
from sensepi.config.log_paths import (
    LOG_SUBDIR_MPU,
    build_log_file_paths,
    build_pi_session_dir,
    slugify_session_name,
)

try:
    from smbus2 import SMBus
except Exception as e:
    print("ERROR: smbus2 is required. Install with: pip3 install smbus2", file=sys.stderr)
    raise

# ---------------------------
# MPU6050 register constants
# ---------------------------
WHO_AM_I       = 0x75
PWR_MGMT_1     = 0x6B
SMPLRT_DIV     = 0x19
CONFIG         = 0x1A
GYRO_CONFIG    = 0x1B
ACCEL_CONFIG   = 0x1C

ACCEL_XOUT_H   = 0x3B
ACCEL_YOUT_H   = 0x3D
ACCEL_ZOUT_H   = 0x3F
GYRO_XOUT_H    = 0x43
GYRO_YOUT_H    = 0x45
GYRO_ZOUT_H    = 0x47
TEMP_OUT_H     = 0x41  # (optional) on-die temperature

# Scale factors for ±2g and ±250 dps
ACC_SF = 16384.0           # LSB/g
GYR_SF = 131.0             # LSB/(deg/s)
G_TO_MS2 = 9.80665

# DLPF bandwidth mapping (datasheet)
# index: (gyro_bw_Hz, accel_bw_Hz)
DLPF_BW = {
    0: (260, 256),
    1: (184, 188),
    2: (94, 98),
    3: (44, 42),
    4: (21, 20),
    5: (10, 10),
    6: (5, 5),
}
DLPF_DEFAULT = 3

# With DLPF enabled, internal rate is 1 kHz → SampleRate = 1000/(1+SMPLRT_DIV)
INTERNAL_RATE_HZ = 1000.0


@dataclass
class SensorMap:
    bus: int
    addr: int


class MPU6050:
    """Minimal MPU6050 driver using smbus2 (no DMP)."""
    def __init__(self, bus: SMBus, addr: int):
        self.bus = bus
        self.addr = addr

    def _write_u8(self, reg: int, val: int) -> None:
        self.bus.write_byte_data(self.addr, reg, val & 0xFF)

    def _read_u8(self, reg: int) -> int:
        return self.bus.read_byte_data(self.addr, reg)

    def _read_i16(self, reg_h: int) -> int:
        hi = self.bus.read_byte_data(self.addr, reg_h)
        lo = self.bus.read_byte_data(self.addr, reg_h + 1)
        v = (hi << 8) | lo
        if v & 0x8000:
            v = -((~v & 0xFFFF) + 1)
        return v

    def who_am_i(self) -> int:
        return self._read_u8(WHO_AM_I)

    def initialize(self, dlpf_cfg: int, fs_accel: int = 0, fs_gyro: int = 0, rate_hz: float = 100.0) -> Tuple[int, float]:
        """
        fs_accel: 0→±2g, 1→±4g, 2→±8g, 3→±16g
        fs_gyro : 0→±250dps, 1→±500, 2→±1000, 3→±2000
        Returns (smplrt_div, actual_rate_hz)
        """
        # Wake up and select PLL with X‑gyro as clock source (datasheet §5.5)
        self._write_u8(PWR_MGMT_1, 0x01)
        time.sleep(0.05)

        # DLPF
        self._write_u8(CONFIG, dlpf_cfg & 0x07)

        # Full-scale ranges
        self._write_u8(GYRO_CONFIG, (fs_gyro & 0x03) << 3)
        self._write_u8(ACCEL_CONFIG, (fs_accel & 0x03) << 3)

        # Sample rate divider (with DLPF: internal = 1 kHz)
        div = int(round(INTERNAL_RATE_HZ / max(1.0, rate_hz)) - 1)
        if div < 0: div = 0
        if div > 255: div = 255
        self._write_u8(SMPLRT_DIV, div)
        actual = INTERNAL_RATE_HZ / (1.0 + div)
        return div, actual

    def read_accel(self) -> Tuple[int, int, int]:
        ax = self._read_i16(ACCEL_XOUT_H)
        ay = self._read_i16(ACCEL_YOUT_H)
        az = self._read_i16(ACCEL_ZOUT_H)
        return ax, ay, az

    def read_gyro(self) -> Tuple[int, int, int]:
        gx = self._read_i16(GYRO_XOUT_H)
        gy = self._read_i16(GYRO_YOUT_H)
        gz = self._read_i16(GYRO_ZOUT_H)
        return gx, gy, gz

    def read_temp_c(self) -> float:
        # Optional: T(°C) = (TEMP_OUT / 340) + 36.53
        raw = self._read_i16(TEMP_OUT_H)
        return (raw / 340.0) + 36.53


class AsyncWriter:
    """Async CSV/JSONL writer with periodic flush.

    Changes:
    - Defaults to flush_every=2000 rows, flush_seconds=2.0 seconds
    - Periodic flush doesn't fsync by default (can be enabled)
    - Final fsync is always performed at stop()

    One AsyncWriter instance runs in a background thread per sensor. The main
    sampling loop only enqueues rows into a queue, so slow disks cannot stall
    time-critical sensor reads.
    """
    def __init__(self, filepath: Path, fmt: str, header: List[str],
                 flush_every: int = 2000, flush_seconds: float = 2.0,
                 fsync_each_flush: bool = False):
        self.filepath = filepath
        self.meta_path = filepath.with_suffix(filepath.suffix + ".meta.json")
        self.fmt = fmt
        self.header = header
        self.flush_every = flush_every
        self.flush_seconds = flush_seconds
        self.fsync_each_flush = fsync_each_flush
        self._q: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._fh = None
        self._writer = None
        self._lines_since_flush = 0
        self._last_flush = time.monotonic()
        self._stopping = False

    def start(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        # Be explicit about encoding; matches aggregated log file behavior.
        self._fh = open(self.filepath, "w", newline="", encoding="utf-8")
        if self.fmt == "csv":
            self._writer = csv.DictWriter(self._fh, fieldnames=self.header)
            self._writer.writeheader()
        self._t.start()

    def write(self, row: dict):
        self._q.put(row)

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                # None is a sentinel pushed by stop(): flush any pending rows
                # and exit the writer thread cleanly.
                break
            if self.fmt == "csv":
                self._writer.writerow(item)
            else:
                self._fh.write(json.dumps(item, separators=(",", ":")) + "\n")
            self._lines_since_flush += 1
            now = time.monotonic()
            if (
                self._lines_since_flush >= self.flush_every
                or (now - self._last_flush) >= self.flush_seconds
            ):
                self._fh.flush()
                if self.fsync_each_flush:
                    os.fsync(self._fh.fileno())
                self._lines_since_flush = 0
                self._last_flush = now

        # final flush – guard against rare cases where start() failed
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())  # ensure data hits storage at stop
            self._fh.close()

    def stop(self):
        if not self._stopping:
            self._stopping = True
            self._q.put(None)
            self._t.join()

    def write_metadata(self, meta: dict):
        with open(self.meta_path, "w", encoding="utf-8") as mfh:
            json.dump(meta, mfh, indent=2)


def parse_sensor_map(s: str) -> Dict[int, 'SensorMap']:
    """
    Parse --map like: "1:1-0x68,2:1-0x69,3:0-0x68"
    Returns dict {1: SensorMap(bus, addr), ...}
    """
    out: Dict[int, SensorMap] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            left, right = part.split(":")
            sid = int(left)
            bus_str, addr_str = right.split("-")
            bus = int(bus_str)
            addr = int(addr_str, 16) if addr_str.lower().startswith("0x") else int(addr_str)
            if sid not in (1, 2, 3):
                print(f"[WARN] Ignoring invalid sensor id in --map: {sid}", file=sys.stderr)
                continue
            out[sid] = SensorMap(bus=bus, addr=addr)
        except Exception:
            print(f"[WARN] Could not parse mapping entry '{part}', expected like 2:1-0x69", file=sys.stderr)
    return out


def default_mapping() -> Dict[int, 'SensorMap']:
    return {
        1: SensorMap(bus=1, addr=0x68),
        2: SensorMap(bus=1, addr=0x69),
        3: SensorMap(bus=0, addr=0x68),
    }


def scan_buses() -> None:
    print("Scanning I²C buses 0 and 1 for MPU6050 (0x68/0x69)...")
    for bus_id in (0, 1):
        try:
            with SMBus(bus_id) as bus:
                found = []
                for addr in (0x68, 0x69):
                    try:
                        who = bus.read_byte_data(addr, WHO_AM_I)
                        found.append((addr, who))
                    except Exception:
                        pass
                if found:
                    for addr, who in found:
                        print(f"  Bus {bus_id}: addr 0x{addr:02X} WHO_AM_I=0x{who:02X}")
                else:
                    print(f"  Bus {bus_id}: (no 0x68/0x69 detected)")
        except FileNotFoundError:
            print(f"  Bus {bus_id}: not available (skip)")


def monotonic_controller(rate_hz: float):
    """Yield target monotonic_ns timestamps for a fixed sampling rate.

    Each step adds a fixed period to the *previous target* time, which keeps the
    long-term rate stable and avoids drift from small sleep() errors.
    """
    period = int(1e9 / rate_hz)
    next_t = time.monotonic_ns()
    while True:
        next_t += period
        yield next_t


def main():
    def _normalize_out_path(path: str) -> str:
        if "\\" in path and "/" not in path:
            print(f"[WARN] Normalizing Windows-style path in --out: {path}", file=sys.stderr)
            return path.replace("\\", "/")
        return path

    ap = argparse.ArgumentParser(description="Multi‑MPU6050 local logger (CSV/JSONL). No MQTT.")
    ap.add_argument("--list", action="store_true", help="List detected 0x68/0x69 on bus 0 and 1 and exit")
    ap.add_argument(
        "--sample-rate-hz",
        type=int,
        default=None,
        help="Preferred sampling rate in Hz (4–1000, clamped on-device)",
    )
    ap.add_argument("--rate", type=float, help="Sampling rate in Hz (e.g., 10, 20, 50, 100, 200)", required=False)
    ap.add_argument("--sensors", type=str, default="1,2,3", help="Comma‑separated sensor ids to enable (subset of 1,2,3)")
    ap.add_argument("--map", type=str, default="", help="Override mapping like '1:1-0x68,2:1-0x69,3:0-0x68'")
    ap.add_argument(
        "--channels",
        type=str,
        choices=["acc", "gyro", "both", "default"],
        default="both",
        help=(
            "Which channels to record: 'acc', 'gyro', 'both', or 'default' "
            "(AX, AY, and GZ). Default is 'both' so that all six axes are "
            "available for streaming."
        ),
    )
    ap.add_argument("--duration", type=float, default=None, help="Duration in seconds (optional)")
    ap.add_argument("--samples", type=int, default=None, help="Number of samples to capture (optional)")
    ap.add_argument("--out", type=str, default="./logs", help="Output folder")
    ap.add_argument("--log-file", type=str, default=None,
                    help="Optional JSONL log file that receives every sample across sensors")
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config file with defaults "
            "(falls back to 'pi_config.yaml' next to this script if omitted)."
        ),
    )
    ap.add_argument("--format", type=str, choices=["csv", "jsonl"], default="csv", help="Output format")
    ap.add_argument("--prefix", type=str, default="mpu", help="Output filename prefix")
    ap.add_argument("--session-name", type=str, default="", help="Optional session label used as a directory/prefix")
    ap.add_argument("--dlpf", type=int, default=DLPF_DEFAULT, help="DLPF cfg 0..6 (default 3≈44 Hz)")
    ap.add_argument("--temp", action="store_true", help="Also log on‑die temperature (°C)")
    # NEW: flushing controls
    ap.add_argument("--flush-every", type=int, default=2000,
                    help="Flush to file every N rows (default 2000)")
    ap.add_argument("--flush-seconds", type=float, default=2.0,
                    help="Also flush if this many seconds passed (default 2.0)")
    ap.add_argument("--fsync-each-flush", action="store_true",
                    help="Call os.fsync() on each periodic flush (slower; default: final only)")
    ap.add_argument(
        "--no-record",
        action="store_true",
        help="Disable file output (no CSV/JSONL or metadata files)."
    )
    ap.add_argument(
        "--stream-stdout",
        action="store_true",
        help="Stream each sensor sample to stdout as JSON lines for a remote GUI."
    )
    ap.add_argument(
        "--stream-every",
        type=int,
        default=1,
        help=(
            "Deprecated: stream every sample; this flag is kept for compatibility and forced to 1."
        ),
    )
    ap.add_argument(
        "--gui-mode",
        action="store_true",
        help=(
            "Enable streaming to stdout and continue running even if file output fails. "
            "Convenience flag for desktop GUI integrations."
        ),
    )
    ap.add_argument(
        "--timing-warnings",
        action="store_true",
        help="Print overrun warnings to stderr (debugging only).",
    )
    # Default streaming fields:
    #   - timestamp_ns (int): monotonic time in nanoseconds
    #   - t_s          (float): seconds since the run started
    #   - sensor_id    (int): logical sensor index (1, 2, or 3)
    #   - ax, ay, az   (float): linear acceleration in m/s^2
    #   - gx, gy, gz   (float): angular rate in deg/s
    # `--stream-fields` controls which of the measured channels (ax..gz, temp_c)
    # are added on top of the always-present timestamp_ns, t_s and sensor_id.
    ap.add_argument(
        "--stream-fields",
        type=str,
        default="ax,ay,az,gx,gy,gz",
        help=(
            "Comma-separated list of data fields to include in streamed JSON, chosen from: "
            "ax, ay, az, gx, gy, gz, temp_c. Always includes timestamp_ns, t_s, and sensor_id."
        ),
    )

    args = ap.parse_args()

    # Normalize common Windows-style paths that might be passed from a GUI.
    args.out = _normalize_out_path(args.out)

    if args.list:
        scan_buses()
        return 0

    # ------------------------------------------------------------------
    # YAML configuration merge
    # ------------------------------------------------------------------
    # Strategy:
    #   1. Load optional ``mpu6050`` section from a config file
    #      (explicit --config or pi_config.yaml next to this script).
    #   2. Treat those values as defaults.
    #   3. Any explicit CLI option overrides the corresponding config
    #      value. For boolean flags like --temp, --no-record and
    #      --stream-stdout, the config controls the default state and
    #      CLI can only enable additional behaviour.
    script_dir = Path(__file__).resolve().parent
    default_cfg_path = script_dir / "pi_config.yaml"

    if args.config:
        cfg_path = Path(args.config)
    elif default_cfg_path.exists():
        cfg_path = default_cfg_path
    else:
        cfg_path = None

    cfg = {}
    section = {}
    if cfg_path is not None:
        try:
            cfg = load_config(cfg_path)
            section = cfg.get("mpu6050", {}) or {}
        except Exception as exc:
            print(f"[WARN] Failed to load config {cfg_path}: {exc}", file=sys.stderr)
            section = {}

    argv = sys.argv[1:]

    def _flag_present(name: str) -> bool:
        prefix = name + "="
        return any(a == name or a.startswith(prefix) for a in argv)

    # sample rate (Hz) via --sample-rate-hz / --rate / config fallback
    cfg_rate = section.get("sample_rate_hz")
    resolved_rate = None
    if args.sample_rate_hz is not None:
        resolved_rate = float(args.sample_rate_hz)
    elif args.rate is not None:
        resolved_rate = float(args.rate)
    elif cfg_rate is not None:
        try:
            resolved_rate = float(cfg_rate)
        except Exception:
            print(f"[WARN] Invalid mpu6050.sample_rate_hz in config: {cfg_rate!r}", file=sys.stderr)

    # sensors list (e.g. [1, 2, 3])
    cfg_sensors = section.get("sensors")
    if not _flag_present("--sensors") and cfg_sensors is not None:
        if isinstance(cfg_sensors, (list, tuple)):
            args.sensors = ",".join(str(s) for s in cfg_sensors)
        else:
            args.sensors = str(cfg_sensors)

    # channels: acc|gyro|both|default
    cfg_channels = section.get("channels")
    if not _flag_present("--channels") and cfg_channels is not None:
        args.channels = str(cfg_channels)

    # DLPF 0..6
    cfg_dlpf = section.get("dlpf")
    if not _flag_present("--dlpf") and cfg_dlpf is not None:
        try:
            args.dlpf = int(cfg_dlpf)
        except Exception:
            print(f"[WARN] Invalid mpu6050.dlpf in config: {cfg_dlpf!r}", file=sys.stderr)

    # include_temperature -> --temp
    if section.get("include_temperature") and not args.temp:
        args.temp = True

    # output_dir -> --out
    cfg_out = section.get("output_dir") or section.get("out")
    if cfg_out is not None and not _flag_present("--out"):
        args.out = str(cfg_out)

    # Normalize again after config overrides
    args.out = _normalize_out_path(args.out)

    # Optional behaviour flags
    if section.get("no_record") and not args.no_record:
        args.no_record = True
    if section.get("stream_stdout") and not args.stream_stdout:
        args.stream_stdout = True

    cfg_stream_every = section.get("stream_every")
    if cfg_stream_every not in (None, 1):
        print(
            f"[WARN] Ignoring mpu6050.stream_every={cfg_stream_every!r}; streaming uses every sample.",
            file=sys.stderr,
        )

    cfg_stream_fields = section.get("stream_fields")
    if cfg_stream_fields is not None and not _flag_present("--stream-fields"):
        # Accept either a comma-separated string or a YAML list like [ax, gz]
        if isinstance(cfg_stream_fields, (list, tuple)):
            args.stream_fields = ",".join(str(f) for f in cfg_stream_fields)
        else:
            args.stream_fields = str(cfg_stream_fields)

    cfg_duration = section.get("duration_s")
    if cfg_duration is not None and args.duration is None and not _flag_present("--duration"):
        try:
            args.duration = float(cfg_duration)
        except Exception:
            print(f"[WARN] Invalid mpu6050.duration_s in config: {cfg_duration!r}", file=sys.stderr)

    cfg_samples = section.get("samples")
    if cfg_samples is not None and args.samples is None and not _flag_present("--samples"):
        try:
            args.samples = int(cfg_samples)
        except Exception:
            print(f"[WARN] Invalid mpu6050.samples in config: {cfg_samples!r}", file=sys.stderr)

    cfg_session = section.get("session_name")
    if cfg_session and not _flag_present("--session-name") and not args.session_name:
        args.session_name = str(cfg_session)

    if resolved_rate is None or resolved_rate <= 0:
        print(
            "ERROR: provide a positive --sample-rate-hz/--rate (or set mpu6050.sample_rate_hz in pi_config.yaml).",
            file=sys.stderr,
        )
        return 2

    args.rate = resolved_rate
    args.sample_rate_hz = resolved_rate

    if args.stream_every != 1:
        print(
            f"[WARN] Forcing --stream-every from {args.stream_every} to 1 (single-rate streaming)",
            file=sys.stderr,
        )
    args.stream_every = 1

    if args.gui_mode:
        if not args.stream_stdout:
            args.stream_stdout = True
    session_name = (args.session_name or "").strip()
    session_slug = slugify_session_name(session_name) if session_name else ""
    session_name_for_paths = session_name or None

    # Clamp requested rate to practical 4..1000 Hz with DLPF enabled (datasheet)
    req_rate = max(4.0, min(float(resolved_rate), 1000.0))

    # ------------------------------------------------------------------
    # Rates overview (single-rate pipeline)
    # ------------------------------------------------------------------
    # - args.rate / mpu6050.sample_rate_hz:
    #     Device sampling + recording + streaming rate (Hz) on the Pi. Every
    #     enabled MPU6050 sensor runs at this cadence and records/streams
    #     every sample.
    # - GUI refresh rate:
    #     Controlled on the desktop side (SignalsTab QTimer). This timer
    #     controls how frequently the plots redraw and is independent from
    #     the Pi sampling rate unless the GUI is in "follow sampling rate"
    #     mode.

    try:
        enabled = sorted({int(x) for x in args.sensors.split(",") if x.strip()})
        enabled = [s for s in enabled if s in (1, 2, 3)]
    except Exception:
        print("ERROR: Could not parse --sensors. Use e.g. '1,3'", file=sys.stderr)
        return 2
    if not enabled:
        print("ERROR: No valid sensors selected.", file=sys.stderr)
        return 2

    # Build mapping
    mapping = default_mapping()
    mapping.update(parse_sensor_map(args.map))

    # Open bus handles per bus id (share across sensors on same bus)
    bus_handles: Dict[int, SMBus] = {}
    devices: Dict[int, MPU6050] = {}
    who_values: Dict[int, int] = {}
    smplrt_divs: Dict[int, int] = {}
    actual_rates: Dict[int, float] = {}
    errors: Dict[int, int] = {sid: 0 for sid in enabled}
    samples_written: Dict[int, int] = {sid: 0 for sid in enabled}
    overruns = 0

    # File writers per sensor
    base_out_dir = Path(args.out).expanduser().resolve()
    helper_base_dir = base_out_dir
    helper_sensor_prefix = LOG_SUBDIR_MPU
    if base_out_dir.name != LOG_SUBDIR_MPU:
        helper_sensor_prefix = ""
    else:
        helper_base_dir = base_out_dir.parent
    out_dir = build_pi_session_dir(
        sensor_prefix=helper_sensor_prefix,
        session_name=session_name_for_paths,
        base_dir=helper_base_dir,
    )
    writers: Dict[int, AsyncWriter] = {}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(
            f"[WARN] Unable to create output directory {out_dir}: {exc}",
            file=sys.stderr,
        )
        if not args.stream_stdout:
            # No streaming requested: this is a real failure
            return 1
        # Streaming is requested: continue in streaming-only mode
        print(
            "[WARN] Falling back to streaming-only mode (--no-record forced).",
            file=sys.stderr,
        )
        args.no_record = True
        writers = {}

    # Optional aggregated JSONL log file (single file for all sensors)
    log_file_handle = None
    log_file_path: Optional[Path] = None
    log_file_error = False
    if args.log_file:
        try:
            log_file_path = Path(args.log_file).expanduser()
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            log_file_handle = open(log_file_path, "a", buffering=1, encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] Unable to open log file {args.log_file}: {exc}", file=sys.stderr)
            if not args.stream_stdout:
                return 1
            print("[WARN] Continuing without aggregated log file.", file=sys.stderr)
            log_file_error = True

    # Header now includes time vector `t_s` (seconds since start)
    header = ["timestamp_ns", "t_s", "sensor_id"]
    # Mapping of --channels to exported axes:
    #   acc     -> ax, ay, az
    #   gyro    -> gx, gy, gz
    #   both    -> ax, ay, az, gx, gy, gz
    #   default -> ax, ay, gz  (NO az; helps keep streaming light-weight)
    ch_mode = args.channels.lower()
    if ch_mode == "acc":
        header += ["ax", "ay", "az"]
    elif ch_mode == "gyro":
        header += ["gx", "gy", "gz"]
    elif ch_mode == "both":
        header += ["ax", "ay", "az", "gx", "gy", "gz"]
    elif ch_mode == "default":
        header += ["ax", "ay", "gz"]
    else:
        print("ERROR: invalid channels", file=sys.stderr); return 2
    if args.temp:
        header += ["temp_c"]

    # Compute stream_fields: fields added on top of timestamp_ns, t_s, sensor_id.
    # The stream payload always includes:
    #   timestamp_ns : int  (monotonic time in nanoseconds)
    #   t_s          : float (seconds since the run started)
    #   sensor_id    : int  (logical sensor index: 1, 2 or 3)
    # `stream_fields` then selects which *measured* channels (ax..gz, temp_c,
    # etc.) are added on top.
    user_fields = [
        s.strip()
        for s in (getattr(args, "stream_fields", "") or "").split(",")
        if s.strip()
    ]

    # Valid data fields are everything in header except the time/sensor_id trio
    base_fields = {"timestamp_ns", "t_s", "sensor_id"}
    valid_fields = [c for c in header if c not in base_fields]

    if not user_fields:
        stream_fields = valid_fields
    else:
        stream_fields = [f for f in user_fields if f in valid_fields]
        if not stream_fields:
            print(
                "[WARN] --stream-fields did not match any known columns; "
                f"falling back to default {valid_fields}",
                file=sys.stderr,
            )
            stream_fields = valid_fields

    if args.no_record and not args.stream_stdout:
        print(
            "[WARN] --no-record specified without --stream-stdout; "
            "run will produce no output files and no streaming data.",
            file=sys.stderr,
        )

    hostname = socket.gethostname()
    start_iso = datetime.utcnow().isoformat() + "Z"
    start_mono_ns = time.monotonic_ns()
    start_dt = datetime.now()
    format_ext = "csv" if args.format == "csv" else "jsonl"

    # Initialize sensors
    for sid in enabled:
        bus_id = mapping[sid].bus
        addr = mapping[sid].addr
        try:
            # Each sensor is initialised in its own try/except block so that a
            # single failing device or I2C bus does not abort the entire run.
            if bus_id not in bus_handles:
                bus_handles[bus_id] = SMBus(bus_id)
            dev = MPU6050(bus_handles[bus_id], addr)
            who = dev.who_am_i()
            if who not in (0x68, 0x69):
                print(f"[WARN] Sensor {sid} WHO_AM_I=0x{who:02X} (expected 0x68/0x69). Continuing.", file=sys.stderr)
            div, actual = dev.initialize(dlpf_cfg=args.dlpf, fs_accel=0, fs_gyro=0, rate_hz=req_rate)
            devices[sid] = dev
            who_values[sid] = who
            smplrt_divs[sid] = div
            actual_rates[sid] = actual

            # Prepare writer only if recording is enabled
            if not args.no_record:
                try:
                    paths = build_log_file_paths(
                        sensor_prefix=args.prefix,
                        session_name=session_name_for_paths,
                        sensor_id=sid,
                        start_dt=start_dt,
                        format_ext=format_ext,
                        out_dir=out_dir,
                    )
                    writer = AsyncWriter(
                        paths.data_path, args.format, header,
                        flush_every=args.flush_every,
                        flush_seconds=args.flush_seconds,
                        fsync_each_flush=args.fsync_each_flush
                    )
                    writer.meta_path = paths.meta_path
                    writer.start()
                    gyro_bw, acc_bw = DLPF_BW.get(args.dlpf, (None, None))
                    meta = {
                        "start_utc": start_iso,
                        "hostname": hostname,
                        "sensor_id": sid,
                        "bus": bus_id,
                        "address_hex": f"0x{addr:02X}",
                        "who_am_i_hex": f"0x{who:02X}",
                        "requested_rate_hz": float(args.rate),
                        "clamped_rate_hz": req_rate,
                        "dlpf_cfg": args.dlpf,
                        "dlpf_gyro_bw_hz": gyro_bw,
                        "dlpf_accel_bw_hz": acc_bw,
                        "fs_accel": "±2g",
                        "fs_gyro": "±250dps",
                        "smplrt_div": div,
                        "device_rate_hz": round(actual, 6),
                        "record_rate_hz": round(actual, 6),
                        "stream_rate_hz": round(actual, 6),
                        "channels": ch_mode,
                        "format": args.format,
                        "header": header,
                        "start_monotonic_ns": start_mono_ns,
                        "session_name": session_name,
                        "session_slug": session_slug,
                        "session_dir": str(out_dir),
                        "stream_every": 1,
                        "stream_fields": list(stream_fields),
                        "version": 3
                    }
                    meta["pi_device_sample_rate_hz"] = meta["device_rate_hz"]
                    meta["pi_stream_decimation"] = 1
                    meta["pi_stream_rate_hz"] = meta["pi_device_sample_rate_hz"]
                    writer.write_metadata(meta)
                    writers[sid] = writer
                    bw_str = f"DLPF={args.dlpf} (gyro≈{gyro_bw}Hz, accel≈{acc_bw}Hz)" if gyro_bw else f"DLPF={args.dlpf}"
                    print(f"[INFO] Sensor {sid}: bus={bus_id} addr=0x{addr:02X} WHO=0x{who:02X} div={div} device_rate≈{actual:.3f} Hz {bw_str}")
                except Exception as exc:
                    print(
                        f"[WARN] Failed to initialize output at {out_dir} for sensor {sid}: {exc}",
                        file=sys.stderr,
                    )
                    if not args.stream_stdout:
                        return 1
                    print(
                        f"[WARN] Sensor {sid}: disabling file recording; streaming only.",
                        file=sys.stderr,
                    )
                    # Do not add a writer for this sensor, but keep the device
                    continue
            # else:
            #     # In GUI streaming mode this is just noise, so keep it silent by default.
            #     # print(
            #     #     f"[INFO] Sensor {sid}: no-record mode (CSV files disabled); streaming only.",
            #     #     file=sys.stderr,
            #     # )
        except FileNotFoundError:
            print(f"[WARN] Bus {bus_id} not available; sensor {sid} skipped.", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to init sensor {sid} on bus {bus_id} @0x{addr:02X}: {e}", file=sys.stderr)

    if not devices:
        print("ERROR: No sensors initialized. Exiting.", file=sys.stderr)
        return 2

    if args.stream_stdout:
        pi_stream_decimation = 1

        for sid in sorted(devices.keys()):
            # Actual per-sensor rate if available, otherwise fall back to requested
            pi_device_sample_rate_hz = float(actual_rates.get(sid, req_rate))
            pi_stream_rate_hz = pi_device_sample_rate_hz

            print(
                "[INFO][PI] streaming: "
                f"sensor={sid} "
                f"pi_device_sample_rate_hz={pi_device_sample_rate_hz:.1f} "
                f"pi_stream_decimation={pi_stream_decimation} "
                f"pi_stream_rate_hz={pi_stream_rate_hz:.1f} "
                f"stream_fields={stream_fields}",
                file=sys.stderr,
            )

        # Build a single meta header line for the whole run
        avg_device_rate = (
            sum(actual_rates.values()) / len(actual_rates)
            if actual_rates
            else float(req_rate)
        )
        meta_header = {
            "meta": "mpu6050_stream_config",
            "sensor_ids": sorted(int(sid) for sid in devices.keys()),
            "pi_device_sample_rate_hz": float(avg_device_rate),
            "pi_stream_decimation": pi_stream_decimation,
        }
        meta_header["pi_stream_rate_hz"] = meta_header["pi_device_sample_rate_hz"]

        # Advertise the Pi-side stream configuration as an initial JSON header so
        # the desktop GUI knows the device rate and how many samples are skipped
        # by --stream-every when estimating the live stream rate.
        # Emit once on stdout before any samples
        print(json.dumps(meta_header), file=sys.stdout, flush=True)

    # Sampling control
    controller = monotonic_controller(req_rate)
    target_next = next(controller)

    # Graceful stop flags
    stop_flag = {"stop": False}

    def _handle_sigint(signum, frame):
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    # Determine stopping condition
    deadline_ns = None
    if args.duration is not None and args.duration > 0:
        deadline_ns = time.monotonic_ns() + int(args.duration * 1e9)
    max_samples = args.samples if (args.samples and args.samples > 0) else None

    try:
        n = 0
        warn_every = 50
        while True:
            # Sleep until the next *target* tick from monotonic_controller so the
            # loop tracks the requested sample rate, counting overruns instead of
            # silently drifting when iterations run late.
            now_ns = time.monotonic_ns()
            sleep_ns = target_next - now_ns
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
            else:
                overruns += 1
                if args.timing_warnings and overruns % warn_every == 1:
                    over_ms = -sleep_ns / 1e6
                    print(
                        f"[WARN] Overrun: loop behind by {over_ms:.3f} ms (count={overruns})",
                        file=sys.stderr,
                    )

            # timestamp each read individually
            for sid, dev in list(devices.items()):
                try:
                    wall_ns = time.time_ns()
                    ts_ns = time.monotonic_ns()
                    t_s = (ts_ns - start_mono_ns) / 1e9
                    row = {"timestamp_ns": ts_ns, "t_s": t_s, "sensor_id": sid}

                    if ch_mode == "acc":
                        ax, ay, az = dev.read_accel()
                        row.update({
                            "ax": (ax / ACC_SF) * G_TO_MS2,
                            "ay": (ay / ACC_SF) * G_TO_MS2,
                            "az": (az / ACC_SF) * G_TO_MS2,
                        })
                    elif ch_mode == "gyro":
                        gx, gy, gz = dev.read_gyro()
                        row.update({
                            "gx": gx / GYR_SF,
                            "gy": gy / GYR_SF,
                            "gz": gz / GYR_SF,
                        })
                    elif ch_mode == "both":
                        ax, ay, az = dev.read_accel()
                        gx, gy, gz = dev.read_gyro()
                        row.update({
                            "ax": (ax / ACC_SF) * G_TO_MS2,
                            "ay": (ay / ACC_SF) * G_TO_MS2,
                            "az": (az / ACC_SF) * G_TO_MS2,
                            "gx": gx / GYR_SF,
                            "gy": gy / GYR_SF,
                            "gz": gz / GYR_SF,
                        })
                    elif ch_mode == "default":
                        # AX, AY and GZ (matches "original script" behavior)
                        ax, ay, _ = dev.read_accel()
                        _, _, gz = dev.read_gyro()
                        row.update({
                            "ax": (ax / ACC_SF) * G_TO_MS2,
                            "ay": (ay / ACC_SF) * G_TO_MS2,
                            "gz": gz / GYR_SF,
                        })

                    if args.temp:
                        try:
                            row["temp_c"] = dev.read_temp_c()
                        except Exception:
                            row["temp_c"] = float("nan")

                    # 1) Optional aggregated JSONL logging (one file for all samples)
                    if log_file_handle is not None and not log_file_error:
                        log_payload = {
                            "sensor_id": sid,
                            "wall_ns": wall_ns,
                            "timestamp_ns": ts_ns,
                            "t_s": t_s,
                        }
                        for key in ("ax", "ay", "az", "gx", "gy", "gz", "temp_c"):
                            if key in row:
                                log_payload[key] = row[key]
                        try:
                            log_file_handle.write(json.dumps(log_payload, separators=(",", ":")) + "\n")
                        except Exception as exc:
                            log_file_error = True
                            print(f"[WARN] Failed to write to log file {log_file_path}: {exc}", file=sys.stderr)

                    # 2) Optional per-sensor file output
                    w = writers.get(sid)
                    if w is not None:
                        w.write(row)

                    # 3) Update per-sensor sample counter
                    samples_written[sid] += 1

                    # 4) Optional stdout streaming (every sample)
                    if args.stream_stdout:
                        out_obj = {
                            "timestamp_ns": ts_ns,
                            "t_s": t_s,
                            "sensor_id": sid,
                        }
                        for key in stream_fields:
                            if key in row:
                                out_obj[key] = row[key]
                        line = json.dumps(out_obj, separators=(",", ":"))
                        if DEBUG_STREAM and samples_written[sid] % 50 == 0:
                            print(f"[DEBUG][PI] sample sid={sid} t_s={t_s:.3f}", file=sys.stderr)
                        print(line, flush=True)
                except Exception as e:
                    errors[sid] += 1
                    if errors[sid] <= 10 or (errors[sid] % 100) == 0:
                        print(f"[WARN] Read error on sensor {sid}: {e} (count={errors[sid]})", file=sys.stderr)
                    continue

            n += 1
            if max_samples is not None and n >= max_samples:
                break
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                break
            if stop_flag["stop"]:
                break
            target_next = next(controller)

    finally:
        # Stop writers and close buses
        for sid, w in writers.items():
            w.stop()
        for bus in bus_handles.values():
            try:
                bus.close()
            except Exception:
                pass
        if log_file_handle is not None:
            try:
                log_file_handle.close()
            except Exception:
                pass

        # Summary
        print("\n=== Run summary ===")
        print(f"Host: {hostname}")
        print(f"Started: {start_iso}")
        if session_name:
            if session_slug:
                print(f"Session: {session_name} (slug: {session_slug})")
            else:
                print(f"Session: {session_name}")
        for sid in enabled:
            if sid in samples_written:
                print(f" Sensor {sid}: samples={samples_written[sid]}, errors={errors.get(sid, 0)}")
        print(f" Overruns: {overruns}")
        print("Output directory:", out_dir)
        if log_file_path is not None:
            print("Aggregated log file:", log_file_path)


if __name__ == "__main__":
    sys.exit(main())
