"""Unified sampling configuration and helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class RecordingMode:
    """User-facing presets (labels only; sampling is single-rate)."""

    key: str
    label: str


RECORDING_MODES: Dict[str, RecordingMode] = {
    "low_fidelity": RecordingMode(
        key="low_fidelity",
        label="Low fidelity (single-rate)",
    ),
    "high_fidelity": RecordingMode(
        key="high_fidelity",
        label="High fidelity (single-rate)",
    ),
    "raw": RecordingMode(
        key="raw",
        label="Raw (device rate)",
    ),
}


@dataclass
class SamplingConfig:
    """
    Single source of truth for sampling.

    device_rate_hz: what the sensor is *actually* sampled at on the Pi.
    mode_key: selects a RecordingMode from RECORDING_MODES.
    """

    device_rate_hz: float
    mode_key: str = "high_fidelity"

    @property
    def mode(self) -> RecordingMode:
        """Return the resolved recording mode (defaults to high_fidelity)."""
        return RECORDING_MODES.get(self.mode_key, RECORDING_MODES["high_fidelity"])

    @property
    def record_decimate(self) -> int:
        """Recording decimation is fixed to 1 (single sampling rate)."""

        return 1

    @property
    def stream_decimate(self) -> int:
        """Streaming decimation is fixed to 1 (single sampling rate)."""

        return 1

    @property
    def record_rate_hz(self) -> float:
        """Alias for the single sampling rate used for recording."""

        return float(self.device_rate_hz)

    @property
    def stream_rate_hz(self) -> float:
        """Alias for the single sampling rate used for streaming."""

        return float(self.device_rate_hz)

    def compute_decimation(self) -> dict:
        """
        Legacy helper returning decimation/rate info.

        All decimations are forced to 1 so recording and streaming use the
        same physical sampling rate as the device.
        """

        return {
            "record_decimate": self.record_decimate,
            "record_rate_hz": self.record_rate_hz,
            "stream_decimate": self.stream_decimate,
            "stream_rate_hz": self.stream_rate_hz,
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        default_device_rate: float = 200.0,
        default_mode: str = "high_fidelity",
    ) -> "SamplingConfig":
        """
        Construct a SamplingConfig from a mapping such as sensors.yaml.

        Supported shape::

            sampling:
              device_rate_hz: 200
              mode: high_fidelity
        """
        # Guard against None / non-mapping inputs early
        payload: Mapping[str, Any] = mapping or {}

        sampling_block = (
            payload.get("sampling") if isinstance(payload, Mapping) else None
        )

        device_rate: Any = default_device_rate
        mode_value: Any = default_mode

        if isinstance(sampling_block, Mapping):
            device_rate = sampling_block.get("device_rate_hz", device_rate)
            mode_value = sampling_block.get("mode", mode_value)

        # Coerce rate to float with a safe fallback
        try:
            rate = float(device_rate)
        except (TypeError, ValueError):
            rate = float(default_device_rate)

        # Normalize mode: case-insensitive, hyphens vs underscores, common aliases
        raw_mode = str(mode_value or default_mode).strip().lower().replace("-", "_")

        # Simple aliases for convenience
        if raw_mode in {"low", "low_fid"}:
            raw_mode = "low_fidelity"
        elif raw_mode in {"high", "high_fid"}:
            raw_mode = "high_fidelity"
        elif raw_mode in {"device", "raw_device"}:
            raw_mode = "raw"

        mode_key_str = raw_mode if raw_mode in RECORDING_MODES else default_mode

        return cls(device_rate_hz=rate, mode_key=mode_key_str)

    def to_mapping(self) -> dict:
        """
        Serialize the sampling config back into a mapping suitable for YAML.
        """
        return {
            "sampling": {
                "device_rate_hz": float(self.device_rate_hz),
                "mode": self.mode.key,
            }
        }


@dataclass
class GuiSamplingDisplay:
    device_rate_hz: float
    record_rate_hz: float
    stream_rate_hz: float
    mode_label: str

    @classmethod
    def from_sampling(cls, sampling: SamplingConfig) -> "GuiSamplingDisplay":
        return cls(
            device_rate_hz=sampling.device_rate_hz,
            record_rate_hz=sampling.record_rate_hz,
            stream_rate_hz=sampling.stream_rate_hz,
            mode_label=sampling.mode.label,
        )
