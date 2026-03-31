from __future__ import annotations

"""
Utilities for ingesting JSONL sensor streams and dispatching them into
thread-safe channel ring buffers that the GUI can consume.
"""

import json
import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .ringbuffer import RingBuffer

logger = logging.getLogger(__name__)

DEFAULT_RINGBUFFER_CAPACITY = 5000
_SKIP_FIELDS = {
    "sensor_id",
    "t_s",
    "timestamp_ns",
    "timestamp",
    "ts",
}

Number = float
ChannelName = str
SensorId = str
BufferKey = Tuple[SensorId, ChannelName]
ChannelSample = Tuple[Number, Number]


class ChannelBuffer:
    """Ring buffer plus lock for one channel of streaming data.

    The RLock allows a producer thread to append samples while consumer
    threads take snapshots without corrupting the underlying RingBuffer.
    """

    def __init__(self, capacity: int = DEFAULT_RINGBUFFER_CAPACITY) -> None:
        self._buffer: RingBuffer[ChannelSample] = RingBuffer(capacity)
        self._lock = threading.RLock()

    def append(self, timestamp: Number, value: Number) -> None:
        """Append a new sample (timestamp, value)."""
        with self._lock:
            self._buffer.append((float(timestamp), float(value)))

    def snapshot(self) -> List[ChannelSample]:
        """Return a thread-safe copy of the logical contents for read-only use."""
        with self._lock:
            return list(self._buffer)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def latest(self) -> Optional[ChannelSample]:
        """Return the newest sample, or ``None`` if the buffer is empty."""
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]


class ChannelBufferStore:
    """Mapping of (sensor_id, channel) -> ChannelBuffer used by the stream reader.

    A single writer thread appends samples, while readers take snapshots
    for plotting or analysis without blocking the ingest loop.
    """

    def __init__(self, capacity: int = DEFAULT_RINGBUFFER_CAPACITY) -> None:
        self._capacity = capacity
        self._buffers: Dict[BufferKey, ChannelBuffer] = {}
        self._lock = threading.RLock()

    def append(self, sensor_id: SensorId, channel: ChannelName, timestamp: Number, value: Number) -> None:
        buf = self.get_or_create(sensor_id, channel)
        buf.append(timestamp, value)

    def get_or_create(self, sensor_id: SensorId, channel: ChannelName) -> ChannelBuffer:
        key = (sensor_id, channel)
        with self._lock:
            # Lazily create per-channel buffers so only channels that actually
            # appear in the stream consume memory.
            buf = self._buffers.get(key)
            if buf is None:
                buf = ChannelBuffer(self._capacity)
                self._buffers[key] = buf
            return buf

    def get(self, sensor_id: SensorId, channel: ChannelName) -> Optional[ChannelBuffer]:
        with self._lock:
            return self._buffers.get((sensor_id, channel))

    def items(self) -> List[Tuple[BufferKey, ChannelBuffer]]:
        """Return a snapshot list of (key, buffer) pairs."""
        with self._lock:
            return list(self._buffers.items())

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()


def reader_loop(
    stream: Iterable[str],
    buffers: ChannelBufferStore,
    *,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Read JSONL records from a line-oriented stream and fill channel buffers.

    This is intended to run in a background thread: it stops when the
    input stream is exhausted or when an optional ``stop_event`` is set.
    """
    for raw_line in stream:
        if stop_event is not None and stop_event.is_set():
            break

        line = raw_line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Dropping malformed JSON line: %s (%s)", line, exc)
            continue

        if not isinstance(record, Mapping):
            logger.debug("Skipping non-object JSON payload: %r", record)
            continue

        try:
            _dispatch_record(record, buffers)
        except Exception:
            logger.exception("Failed to dispatch record: %r", record)


def _dispatch_record(record: Mapping[str, Any], buffers: ChannelBufferStore) -> None:
    sensor_id = record.get("sensor_id")
    if sensor_id is None:
        logger.debug("Record missing sensor_id: %r", record)
        return
    sensor_id_str = str(sensor_id)

    timestamp = _extract_timestamp(record)
    if timestamp is None:
        logger.debug("Record missing usable timestamp: %r", record)
        return

    appended = False
    for key, value in record.items():
        if key in _SKIP_FIELDS:
            continue
        numeric_value = _coerce_number(value)
        if numeric_value is None:
            continue
        buffers.append(sensor_id_str, str(key), timestamp, numeric_value)
        appended = True

    if not appended:
        logger.debug("No numeric channels found in record: %r", record)


def _extract_timestamp(record: Mapping[str, Any]) -> Optional[Number]:
    """
    Extract a timestamp in seconds from a JSON record.

    Supported fields, in order of preference:
    - t_s:        seconds (float or castable to float)
    - timestamp:  seconds (float or castable to float)
    - ts:         seconds (float or castable to float)
    - timestamp_ns: nanoseconds (int/float, converted to seconds)
    """
    # Preferred: explicit seconds field
    for key in ("t_s", "timestamp", "ts"):
        raw = record.get(key)
        if raw is not None:
            ts = _coerce_number(raw)
            if ts is not None:
                return ts

    # Fallback: nanoseconds -> seconds
    ts_ns = record.get("timestamp_ns")
    if ts_ns is not None:
        ns_val = _coerce_number(ts_ns)
        if ns_val is not None:
            return ns_val * 1e-9

    return None


def _coerce_number(value: Any) -> Optional[Number]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class StreamReaderHandle:
    thread: threading.Thread
    stop_event: threading.Event
    buffers: ChannelBufferStore

    def stop(self, *, join: bool = False, timeout: Optional[float] = None) -> None:
        self.stop_event.set()
        if join:
            self.thread.join(timeout)

    def is_alive(self) -> bool:
        return self.thread.is_alive()


def start_reader(
    stream: Iterable[str],
    *,
    buffers: Optional[ChannelBufferStore] = None,
    capacity: int = DEFAULT_RINGBUFFER_CAPACITY,
    thread_name: Optional[str] = None,
) -> StreamReaderHandle:
    """
    Start a background thread that ingests JSON lines from *stream*.
    """

    store = buffers or ChannelBufferStore(capacity=capacity)
    stop_event = threading.Event()

    def _target() -> None:
        reader_loop(stream, store, stop_event=stop_event)

    thread = threading.Thread(
        target=_target,
        name=thread_name or "SensePiStreamReader",
        daemon=True,
    )
    thread.start()
    return StreamReaderHandle(thread=thread, stop_event=stop_event, buffers=store)


def start_reader_on_stdin(
    *,
    capacity: int = DEFAULT_RINGBUFFER_CAPACITY,
    thread_name: Optional[str] = None,
) -> StreamReaderHandle:
    """
    Convenience wrapper that starts the reader on ``sys.stdin``.
    """
    import sys

    return start_reader(
        sys.stdin,
        capacity=capacity,
        thread_name=thread_name or "SensePiStreamReader(stdin)",
    )
