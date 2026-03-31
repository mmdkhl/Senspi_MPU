"""Shared helpers that codify the SensePi log directory conventions.

These helpers keep the Raspberry Pi recorders, offline sync tools, and
GUI in agreement about where logs are written and how files are named.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import datetime as _dt
import re

from .app_config import AppPaths as _AppPaths

# ---- Constants ---------------------------------------------------------

# The defaults mirror pi_config.yaml so the Pi loggers behave as before.
DEFAULT_PI_LOG_ROOT = Path("~").expanduser() / "logs"

# Match the desktop application's AppPaths raw-data directory. Fall back
# to a repo-relative path so imports succeed even if app_config breaks.
try:
    DEFAULT_PC_RAW_ROOT = _AppPaths().raw_data
except Exception:  # pragma: no cover - extremely defensive
    DEFAULT_PC_RAW_ROOT = Path("data") / "raw"

# Sensor-specific subdirectory names. These names appear on both Pi and PC.
LOG_SUBDIR_MPU = "mpu"


# ---- Session slugging --------------------------------------------------

def slugify_session_name(name: str) -> str:
    """Return a filesystem-safe slug for a user-provided session name."""

    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "session"


# ---- Pi-side helpers ---------------------------------------------------

def build_pi_session_dir(
    sensor_prefix: str,
    session_name: Optional[str],
    base_dir: Path | str | None = None,
) -> Path:
    """Return the directory on the Pi where this session's logs will be stored."""

    root = Path(base_dir).expanduser() if base_dir is not None else DEFAULT_PI_LOG_ROOT

    # Keep behaviour in sync with build_pc_session_root
    sensor_prefix = sensor_prefix.strip("/ ")
    sensor_root = root / sensor_prefix if sensor_prefix else root

    if not session_name:
        return sensor_root

    slug = slugify_session_name(session_name)
    return sensor_root / slug


# ---- PC-side helpers ---------------------------------------------------

def build_pc_session_root(
    raw_root: Path | str,
    host_slug: str,
    session_name: Optional[str],
    sensor_prefix: str,
) -> Path:
    """Return the local root directory for logs downloaded from the Pi."""

    raw_root = Path(raw_root)

    if session_name:
        session_slug = slugify_session_name(session_name)
        return raw_root / session_slug

    slug = slugify_session_name(host_slug or "host")
    sensor_prefix = sensor_prefix.strip("/ ")
    if sensor_prefix:
        return raw_root / slug / sensor_prefix
    return raw_root / slug


# ---- Filename helpers --------------------------------------------------

def _format_start_ts(start_dt: _dt.datetime) -> str:
    """Return the canonical timestamp string used in log filenames."""

    return start_dt.strftime("%Y-%m-%d_%H-%M-%S")


@dataclass(frozen=True)
class LogFilePaths:
    """Container with the generated data path and .meta.json sidecar path."""

    data_path: Path
    meta_path: Path


def build_log_file_paths(
    sensor_prefix: str,
    session_name: Optional[str],
    sensor_id: int,
    start_dt: _dt.datetime,
    format_ext: str,
    out_dir: Path,
) -> LogFilePaths:
    """Return full paths for a log file and its metadata companion."""

    ts_str = _format_start_ts(start_dt)
    session_slug = slugify_session_name(session_name) if session_name else ""

    if session_slug:
        stem = f"{session_slug}_{sensor_prefix}_S{sensor_id}_{ts_str}"
    else:
        stem = f"{sensor_prefix}_S{sensor_id}_{ts_str}"

    format_ext = format_ext.lstrip(".")
    filename = f"{stem}.{format_ext}"

    data_path = out_dir / filename
    meta_path = data_path.with_suffix(data_path.suffix + ".meta.json")
    return LogFilePaths(data_path=data_path, meta_path=meta_path)
