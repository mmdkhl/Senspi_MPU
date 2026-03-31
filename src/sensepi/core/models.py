"""Shared dataclasses for SensePi sessions and samples."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SessionInfo:
    name: str
    sensor_type: str
    sample_rate_hz: float
    started_at: datetime
    output_path: Path


@dataclass
class LiveSample:
    timestamp_ns: int
    values: tuple[float, ...]
    sensor: Optional[str] = None
