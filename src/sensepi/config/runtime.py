"""Runtime configuration helpers for the ingestion/plotting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


@dataclass(slots=True)
class SensePiConfig:
    """
    Tuning knobs for how samples are recorded, streamed, and visualized.

    The defaults assume ~500 Hz sensors feeding a ~50 Hz UI and network stream.
    """

    sensor_fs: float = 500.0
    recording_enabled: bool = True
    recording_chunk_seconds: float = 1.0

    streaming_enabled: bool = True
    stream_fs: float = 50.0

    plotting_enabled: bool = True
    plot_fs: float = 50.0
    plot_window_seconds: float = 10.0
    smoothing_alpha: float = 0.2
    use_envelope: bool = True
    spike_threshold: float = 0.5

    # Thread bridge sizing
    stream_queue_size: int = 8
    plot_queue_size: int = 8

    def sanitized(self) -> SensePiConfig:
        """Return a copy with derived limits applied."""
        alpha = self.smoothing_alpha
        if alpha is not None:
            alpha = max(1e-6, min(1.0, float(alpha)))
        return SensePiConfig(
            sensor_fs=max(1.0, float(self.sensor_fs)),
            recording_enabled=bool(self.recording_enabled),
            recording_chunk_seconds=max(0.01, float(self.recording_chunk_seconds)),
            streaming_enabled=bool(self.streaming_enabled),
            stream_fs=max(1.0, float(self.stream_fs)),
            plotting_enabled=bool(self.plotting_enabled),
            plot_fs=max(1.0, float(self.plot_fs)),
            plot_window_seconds=max(0.5, float(self.plot_window_seconds)),
            smoothing_alpha=alpha,
            use_envelope=bool(self.use_envelope),
            spike_threshold=float(self.spike_threshold),
            stream_queue_size=max(1, int(self.stream_queue_size)),
            plot_queue_size=max(1, int(self.plot_queue_size)),
        )


def _recognized_fields() -> set[str]:
    """Return the dataclass field names accepted by :class:`SensePiConfig`."""
    return {f.name for f in fields(SensePiConfig)}


def _normalize_mapping(data: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Flatten known nesting patterns (e.g. top-level ``pipeline`` key)."""
    if "pipeline" in data and isinstance(data["pipeline"], Mapping):
        merged: MutableMapping[str, Any] = {}
        for key, value in data.items():
            if key == "pipeline":
                merged.update(value)
            else:
                merged[key] = value
        return merged
    return dict(data)


def config_from_mapping(data: Mapping[str, Any] | None) -> SensePiConfig:
    """Build :class:`SensePiConfig` from ``data`` (ignoring unknown keys)."""
    if not data:
        return SensePiConfig()
    normalized = _normalize_mapping(data)
    known = _recognized_fields()
    payload = {key: normalized[key] for key in normalized.keys() & known}
    return SensePiConfig(**payload).sanitized()


def load_config(path: str | Path | None) -> SensePiConfig:
    """
    Load configuration from ``path``.

    Missing files fall back to default :class:`SensePiConfig`.
    """
    if path is None:
        return SensePiConfig()
    cfg_path = Path(path)
    if not cfg_path.exists():
        return SensePiConfig()
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected mapping in {cfg_path}, got {type(raw).__name__}")
    return config_from_mapping(raw)


__all__ = ["SensePiConfig", "config_from_mapping", "load_config"]
