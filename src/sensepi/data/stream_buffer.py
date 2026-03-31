"""Central ring buffer for recent streaming samples."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
from typing import Deque, Iterable, Iterator, List, MutableMapping, Optional

from ..sensors.mpu6050 import MpuSample

logger = logging.getLogger(__name__)

SensorKey = int | str


@dataclass
class BufferConfig:
    """Configuration for :class:`StreamingDataBuffer`."""

    max_seconds: float = 6.0
    sample_rate_hz: float = 200.0
    capacity_margin: float = 1.2
    max_samples_per_sensor: int | None = None

    def capacity(self) -> int:
        """
        Return the maximum number of samples to retain per sensor.

        This value is used as a backstop to avoid unbounded growth when
        timestamps are missing or invalid.
        """
        if self.max_samples_per_sensor is not None:
            return max(1, int(self.max_samples_per_sensor))

        seconds = max(0.1, float(self.max_seconds))
        rate = max(1.0, float(self.sample_rate_hz))
        margin = max(1.0, float(self.capacity_margin))
        estimate = seconds * rate * margin
        return max(1, int(math.ceil(estimate)))


class StreamingDataBuffer:
    """
    Multi-sensor ring buffer for :class:`MpuSample` instances.

    Instances are expected to be owned and mutated from the Qt main thread so
    simple Python containers are sufficient.

    Per-sensor sample timestamps are assumed to be monotonically increasing.
    If they are not, time-window queries (e.g. ``max_seconds`` windows) may
    behave in unexpected ways because the buffer assumes that newer samples
    always have timestamps greater than or equal to older ones.
    """

    def __init__(self, config: BufferConfig | None = None) -> None:
        self._config = config or BufferConfig()
        self._buffers: MutableMapping[SensorKey, Deque[MpuSample]] = {}

    # ------------------------------------------------------------------ ingest
    def add_samples(self, samples: Iterable[MpuSample]) -> None:
        """Append samples to per-sensor deques, enforcing a sliding time window.

        Each sensor_id gets its own ring-like deque; `_truncate` keeps the
        buffer size bounded so recent data is available without unbounded growth.
        """
        count = 0
        sensor_ids: set[SensorKey] = set()
        for sample in samples:
            if sample is None:
                continue
            sensor_id = self._sensor_key_from_sample(sample)
            sensor_ids.add(sensor_id)
            buf = self._buffers.setdefault(sensor_id, deque())
            buf.append(sample)
            self._truncate(sensor_id)
            count += 1

        if count:
            logger.debug(
                "StreamingDataBuffer.add_samples: n=%d, sensors=%s",
                count,
                sorted(sensor_ids, key=str),
            )

    # ------------------------------------------------------------------- query
    def get_sensor_ids(self) -> List[SensorKey]:
        """Return a snapshot of all sensor IDs currently present in the buffer.

        IDs are returned in a deterministic order to make calling code and
        tests easier to reason about.
        """
        return sorted(self._buffers.keys(), key=str)

    def get_recent_samples(
        self,
        sensor_id: SensorKey,
        seconds: float | None = None,
        max_samples: int | None = None,
    ) -> List[MpuSample]:
        """Return recent samples for ``sensor_id`` ordered by time.

        The optional ``seconds`` limit trims older samples using their
        timestamps, while ``max_samples`` caps how many points are returned.

        Parameters
        ----------
        sensor_id:
            Sensor identifier to query.
        seconds:
            Maximum age of samples to return. Falls back to ``config.max_seconds``
            when ``None``.
        max_samples:
            Optional hard limit on the number of samples returned.
        """
        buf = self._buffers.get(self._normalize_sensor_id(sensor_id))
        if not buf:
            return []

        window_s = self._resolve_window(seconds)

        # Treat ``max_samples`` <= 0 as "no explicit limit".
        if max_samples is not None and max_samples > 0:
            max_items = int(max_samples)
        else:
            max_items = 0

        latest_time = self._sample_time(buf[-1])
        threshold = None if latest_time is None else latest_time - window_s

        result: List[MpuSample] = []
        for sample in reversed(buf):
            if max_items and len(result) >= max_items:
                break
            sample_time = self._sample_time(sample)
            if threshold is not None and sample_time is not None:
                if sample_time < threshold:
                    break
            result.append(sample)
        result.reverse()
        return result

    def iter_all_samples(self, seconds: float | None = None) -> Iterator[MpuSample]:
        """Yield samples for all sensors ordered by sensor ID."""
        for sensor_id in sorted(self._buffers.keys(), key=str):
            for sample in self.get_recent_samples(sensor_id, seconds=seconds):
                yield sample

    def latest_timestamp(self, sensor_id: SensorKey | None = None) -> Optional[float]:
        """Return the latest timestamp in seconds for a sensor or across all sensors."""
        if sensor_id is not None:
            buf = self._buffers.get(self._normalize_sensor_id(sensor_id))
            if not buf:
                return None
            return self._sample_time(buf[-1])

        latest: Optional[float] = None
        for buf in self._buffers.values():
            if not buf:
                continue
            ts = self._sample_time(buf[-1])
            if ts is None:
                continue
            if latest is None or ts > latest:
                latest = ts
        return latest

    def get_axis_series(
        self,
        sensor_id: SensorKey,
        axis: str,
        seconds: float | None = None,
        max_samples: int | None = None,
    ) -> tuple[List[float], List[float]]:
        """
        Return synchronized timestamps and axis values for ``sensor_id``.

        Samples with missing timestamps or axis values (``None``) are skipped.
        NaN values are preserved so the caller can decide how to handle them.
        """
        attr = axis.lower()
        samples = self.get_recent_samples(sensor_id, seconds=seconds, max_samples=max_samples)
        timestamps: List[float] = []
        values: List[float] = []
        for sample in samples:
            ts = self._sample_time(sample)
            value = getattr(sample, attr, None)
            if ts is None or value is None:
                continue
            timestamps.append(ts)
            values.append(float(value))
        return timestamps, values

    def clear(self, sensor_id: SensorKey | None = None) -> None:
        """Drop samples for ``sensor_id`` or the entire buffer when omitted."""
        if sensor_id is None:
            self._buffers.clear()
            return
        self._buffers.pop(self._normalize_sensor_id(sensor_id), None)

    # ----------------------------------------------------------------- helpers
    def _truncate(self, sensor_id: SensorKey) -> None:
        buf = self._buffers.get(sensor_id)
        if not buf:
            return

        # Always clamp by capacity to protect against unbounded growth.
        capacity = self._config.capacity()
        while len(buf) > capacity:
            buf.popleft()

        # Additionally enforce the max_seconds window when timestamps are valid.
        max_seconds = float(self._config.max_seconds)
        if max_seconds <= 0 or not buf:
            return

        latest_time = self._sample_time(buf[-1])
        if latest_time is None:
            return

        threshold = latest_time - max_seconds
        while buf:
            oldest_time = self._sample_time(buf[0])
            if oldest_time is None or oldest_time >= threshold:
                break
            buf.popleft()

    def _resolve_window(self, seconds: float | None) -> float:
        if seconds is None:
            return max(0.0, float(self._config.max_seconds))
        try:
            return max(0.0, float(seconds))
        except (TypeError, ValueError):
            return max(0.0, float(self._config.max_seconds))

    @staticmethod
    def _normalize_sensor_id(sensor_id: SensorKey | None) -> SensorKey:
        if sensor_id is None:
            return 0
        try:
            return int(sensor_id)
        except (TypeError, ValueError):
            return str(sensor_id)

    def _sensor_key_from_sample(self, sample: MpuSample) -> SensorKey:
        sensor_value = getattr(sample, "sensor_id", None)
        return self._normalize_sensor_id(sensor_value)

    @staticmethod
    def _sample_time(sample: MpuSample | None) -> Optional[float]:
        if sample is None:
            return None
        t_s = getattr(sample, "t_s", None)
        if t_s is not None:
            try:
                return float(t_s)
            except (TypeError, ValueError):
                pass
        timestamp_ns = getattr(sample, "timestamp_ns", None)
        if timestamp_ns is None:
            return None
        try:
            return float(timestamp_ns) * 1e-9
        except (TypeError, ValueError):
            return None
