from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import Dict, Tuple

import numpy as np

from .ringbuffer import RingBuffer

NS_PER_SECOND = 1_000_000_000

TimeSeriesSample = tuple[int, float]
BufferKey = Tuple[int, str]


def calculate_capacity(
    window_seconds: float,
    max_rate_hz: float,
    *,
    margin: float = 1.1,
) -> int:
    """
    Compute how many samples are needed to cover ``window_seconds`` at
    ``max_rate_hz`` with an optional ``margin``.

    Raises
    ------
    ValueError
        If any of the inputs are non-positive.
    """
    if window_seconds <= 0.0:
        raise ValueError("window_seconds must be positive")
    if max_rate_hz <= 0.0:
        raise ValueError("max_rate_hz must be positive")
    if margin <= 0.0:
        raise ValueError("margin must be positive")

    samples = window_seconds * max_rate_hz * margin
    return max(1, int(math.ceil(samples)))


def initialize_buffers_for_channels(
    sensor_ids: Iterable[int],
    channels: Iterable[str],
    *,
    window_seconds: float,
    max_rate_hz: float,
    margin: float = 1.1,
) -> Dict[BufferKey, "TimeSeriesBuffer"]:
    """
    Pre-create :class:`TimeSeriesBuffer` objects for expected (sensor, channel)
    combinations so buffers are ready when samples start arriving.
    """
    capacity = calculate_capacity(window_seconds, max_rate_hz, margin=margin)
    return {
        (int(sensor_id), channel): TimeSeriesBuffer(capacity)
        for sensor_id in sensor_ids
        for channel in channels
    }


def ns_to_seconds(ts_ns: np.ndarray) -> np.ndarray:
    """Convert nanosecond timestamps into floating-point seconds."""
    if ts_ns.size == 0:
        return np.empty(0, dtype=np.float64)
    return ts_ns.astype(np.float64) / float(NS_PER_SECOND)


class TimeSeriesBuffer:
    """
    Convenience wrapper around :class:`RingBuffer` to manage (t_ns, value)
    samples and retrieve arbitrary time windows.
    """

    __slots__ = ("_buffer",)

    def __init__(self, capacity: int) -> None:
        self._buffer: RingBuffer[TimeSeriesSample] = RingBuffer(capacity)

    def append(self, timestamp_ns: int, value: float) -> None:
        """Append a ``(timestamp_ns, value)`` tuple to the buffer."""
        self._buffer.append((int(timestamp_ns), float(value)))

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[TimeSeriesSample]:
        return iter(self._buffer)

    def latest_timestamp_ns(self) -> int | None:
        """Return the newest timestamp in nanoseconds."""
        if len(self._buffer) == 0:
            return None
        return int(self._buffer[-1][0])

    def get_window(self, start_ns: int, end_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return timestamps and values in the interval ``[start_ns, end_ns]``.

        Both arrays are NumPy ``float64``/``int64`` for efficient downstream use.
        """
        if end_ns < start_ns:
            start_ns, end_ns = end_ns, start_ns

        data = list(self._buffer)
        if not data:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        count = len(data)
        times = np.fromiter((sample[0] for sample in data), dtype=np.int64, count=count)
        values = np.fromiter((sample[1] for sample in data), dtype=np.float64, count=count)

        start_idx = np.searchsorted(times, start_ns, side="left")
        end_idx = np.searchsorted(times, end_ns, side="right")
        if end_idx <= start_idx:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        return times[start_idx:end_idx], values[start_idx:end_idx]

