from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

from sensepi.config.sampling import SamplingConfig

CalibrationKey = Tuple[int, str]  # (sensor_id, channel_name), e.g. (1, "ax")


@dataclass
class CalibrationOffsets:
    """
    Per-sensor, per-channel calibration offsets in physical units.

    Keys are (sensor_id, channel) tuples where channel names match the
    existing MPU6050 fields such as "ax", "ay", "az", "gx", "gy", "gz".
    """

    per_sensor_channel_offset: Dict[CalibrationKey, float] = field(
        default_factory=dict
    )
    description: str | None = None
    timestamp: datetime | None = None

    def offset_for(self, sensor_id: int, channel: str) -> float:
        """Return the stored offset, or 0.0 if none is present."""

        return self.per_sensor_channel_offset.get((int(sensor_id), channel), 0.0)

    def is_empty(self) -> bool:
        return not self.per_sensor_channel_offset

    def to_dict(self) -> dict:
        return {
            "per_sensor_channel_offset": {
                f"{sensor_id}:{channel}": offset
                for (sensor_id, channel), offset in self.per_sensor_channel_offset.items()
            },
            "description": self.description,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationOffsets":
        raw = data.get("per_sensor_channel_offset", {})
        mapping: Dict[CalibrationKey, float] = {}
        for key, value in raw.items():
            sensor_str, channel = key.split(":", 1)
            mapping[(int(sensor_str), channel)] = float(value)
        ts_raw = data.get("timestamp")
        ts = datetime.fromisoformat(ts_raw) if ts_raw else None
        return cls(
            per_sensor_channel_offset=mapping,
            description=data.get("description"),
            timestamp=ts,
        )


@dataclass
class SensorSelectionConfig:
    """
    GUI-level description of which sensors and channels are active.

    - active_sensors: list of sensor IDs (e.g. [1, 2, 3])
    - active_channels: list of channel names, e.g.:
        ["ax", "ay", "az", "gx", "gy", "gz"]
    """

    active_sensors: List[int] = field(default_factory=list)
    active_channels: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Compact, deterministic human-readable summary for logging."""
        sensors = sorted({int(s) for s in self.active_sensors})
        channels = sorted(set(self.active_channels))

        sensors_str = f"sensors={sensors or '[]'}"
        channels_str = f"channels={','.join(channels) or '(none)'}"
        return f"{sensors_str}, {channels_str}"

    def is_empty(self) -> bool:
        """Return True if no sensors or channels are selected."""
        return not self.active_sensors or not self.active_channels

    def to_dict(self) -> dict:
        return {
            "active_sensors": list(self.active_sensors),
            "active_channels": list(self.active_channels),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SensorSelectionConfig":
        return cls(
            active_sensors=[int(s) for s in data.get("active_sensors", [])],
            active_channels=list(data.get("active_channels", [])),
        )


@dataclass
class GuiAcquisitionConfig:
    """
    Full acquisition configuration as seen from the GUI.

    This is what RecorderTab / SignalsTab / FftTab will use to
    configure streaming and recording.
    """

    sampling: SamplingConfig
    stream_rate_hz: float
    record_only: bool = False
    sensor_selection: SensorSelectionConfig = field(
        default_factory=SensorSelectionConfig
    )
    calibration: CalibrationOffsets | None = None

    def summary(self) -> str:
        if self.sensor_selection.is_empty():
            selection_summary = "selection=(none)"
        else:
            selection_summary = self.sensor_selection.summary()

        return (
            f"sampling={self.sampling!r}, "
            f"stream={self.stream_rate_hz:.2f} Hz, "
            f"record_only={self.record_only}, "
            f"{selection_summary}"
        )

    def validate(self) -> None:
        """
        Raise ValueError if the configuration is obviously inconsistent.

        This is intentionally lightweight; deeper checks (e.g. channel names
        vs. hardware capabilities) should live closer to the hardware code.
        """
        if self.stream_rate_hz < 0:
            raise ValueError(
                f"stream_rate_hz must be >= 0, got {self.stream_rate_hz}"
            )

        if not self.record_only and self.stream_rate_hz == 0:
            raise ValueError("stream_rate_hz is 0.0 but record_only is False")

        if not self.sensor_selection.active_sensors:
            raise ValueError("No active sensors configured in sensor_selection")

        if not self.sensor_selection.active_channels:
            raise ValueError("No active channels configured in sensor_selection")

    def is_streaming_enabled(self) -> bool:
        """Return True if live streaming (not just recording) is configured."""
        return not self.record_only and self.stream_rate_hz > 0.0

    def to_dict(self) -> dict:
        return {
            "sampling": (
                self.sampling.to_dict()
                if hasattr(self.sampling, "to_dict")
                else dict(self.sampling)
            ),
            "stream_rate_hz": self.stream_rate_hz,
            "record_only": self.record_only,
            "sensor_selection": self.sensor_selection.to_dict(),
            "calibration": self.calibration.to_dict() if self.calibration else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GuiAcquisitionConfig":
        sampling_data = data["sampling"]
        if hasattr(SamplingConfig, "from_dict"):
            sampling = SamplingConfig.from_dict(sampling_data)
        else:
            sampling = SamplingConfig(**sampling_data)

        return cls(
            sampling=sampling,
            stream_rate_hz=float(data["stream_rate_hz"]),
            record_only=bool(data.get("record_only", False)),
            sensor_selection=SensorSelectionConfig.from_dict(
                data.get("sensor_selection", {})
            ),
            calibration=(
                CalibrationOffsets.from_dict(data["calibration"])
                if data.get("calibration") is not None
                else None
            ),
        )
