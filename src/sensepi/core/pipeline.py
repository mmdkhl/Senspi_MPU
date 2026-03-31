"""Sensor sample fan-out pipeline for recording, streaming, and plotting."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Callable, Optional, Protocol, Sequence, Tuple
from queue import Empty, Full, Queue

import numpy as np

from decimation import DecimationConfig, Decimator

__all__ = [
    "SampleSink",
    "Recorder",
    "Streamer",
    "PlotUpdate",
    "Plotter",
    "Pipeline",
    "NullSink",
]

SampleArray = np.ndarray
StreamPayload = Tuple[np.ndarray, np.ndarray]


class SampleSink(Protocol):
    """Common interface implemented by Recorder/Streamer/Plotter."""

    def handle_samples(self, t: np.ndarray, x: np.ndarray) -> None:  # pragma: no cover - protocol
        ...


class SampleBlockWriter(Protocol):
    """Protocol for recorder backends."""

    def write_samples(self, t: np.ndarray, x: np.ndarray) -> None:  # pragma: no cover - protocol
        ...


def _call_writer(writer: SampleBlockWriter | Callable[[np.ndarray, np.ndarray], None], t: np.ndarray, x: np.ndarray) -> None:
    if hasattr(writer, "write_samples"):
        writer.write_samples(t, x)  # type: ignore[attr-defined]
    else:
        writer(t, x)


def _offer_queue(queue: Queue, item: object) -> None:
    """Best-effort put used for GUI-facing queues.

    When the queue is full we drop the oldest item and keep the newest one
    instead, which keeps the UI responsive under back-pressure.
    """
    try:
        queue.put_nowait(item)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        queue.put_nowait(item)


@dataclass(slots=True)
class Recorder(SampleSink):
    """Stores raw samples to disk or any callable writer."""

    writer: SampleBlockWriter | Callable[[np.ndarray, np.ndarray], None]
    sensor_fs: float
    chunk_seconds: float = 1.0
    copy_blocks: bool = True

    _chunk_size: int = field(init=False)

    def __post_init__(self) -> None:
        if self.sensor_fs <= 0.0:
            raise ValueError("sensor_fs must be positive.")
        self._chunk_size = max(1, int(round(max(0.001, float(self.chunk_seconds)) * self.sensor_fs)))

    def handle_samples(self, t: np.ndarray, x: np.ndarray) -> None:
        times = np.asarray(t, dtype=np.float64).reshape(-1)
        values = np.asarray(x).reshape(-1)
        if times.size != values.size:
            raise ValueError("timestamps and samples must have the same length.")
        if times.size == 0:
            return
        chunk = self._chunk_size
        start = 0
        while start < times.size:
            end = min(times.size, start + chunk)
            t_view = times[start:end]
            x_view = values[start:end]
            if self.copy_blocks:
                t_view = np.array(t_view, copy=True)
                x_view = np.array(x_view, copy=True)
            _call_writer(self.writer, t_view, x_view)
            start = end


@dataclass(slots=True)
class Streamer(SampleSink):
    """Decimates and sends samples over the network."""

    sensor_fs: float
    stream_fs: float
    transport: Callable[[np.ndarray, np.ndarray], None] | None = None
    queue: Queue[StreamPayload] | None = None

    def __post_init__(self) -> None:
        if self.sensor_fs <= 0.0 or self.stream_fs <= 0.0:
            raise ValueError("sensor_fs and stream_fs must be positive.")
        cfg = DecimationConfig(
            sensor_fs=self.sensor_fs,
            plot_fs=self.stream_fs,
            use_envelope=False,
            smoothing_alpha=None,
        )
        self._decimator = Decimator(cfg)

    def handle_samples(self, t: np.ndarray, x: np.ndarray) -> None:
        if x is None or t is None:
            return
        values = np.asarray(x, dtype=np.float32).reshape(-1)
        times = np.asarray(t, dtype=np.float64).reshape(-1)
        if values.size == 0 or times.size == 0:
            return
        start_time = float(times[0])
        # Convert the high-rate sensor stream into a lighter, decimated stream
        # before shipping it across the network or into the GUI queue.
        t_dec, y_mean, _, _ = self._decimator.process_block(values, start_time=start_time)
        if t_dec.size == 0:
            return
        payload = (t_dec, y_mean)
        # The same decimated payload can be sent immediately over the transport
        # and/or handed to a queue that another thread (e.g. Qt) will drain.
        if self.transport is not None:
            self.transport(t_dec, y_mean)
        if self.queue is not None:
            _offer_queue(self.queue, payload)


@dataclass(slots=True)
class PlotUpdate:
    """Decimated plot data (mean/envelope/spikes) for the Signals tab."""

    timestamps: np.ndarray
    mean: np.ndarray
    y_min: Optional[np.ndarray] = None
    y_max: Optional[np.ndarray] = None
    spike_mask: Optional[np.ndarray] = None


@dataclass(slots=True)
class Plotter(SampleSink):
    """Prepares decimated/smoothed data for live plotting."""

    sensor_fs: float
    plot_fs: float
    smoothing_alpha: Optional[float] = 0.2
    use_envelope: bool = True
    spike_threshold: float = 0.5
    queue: Queue[PlotUpdate] | None = None

    _latest_update: Optional[PlotUpdate] = field(init=False, default=None, repr=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.sensor_fs <= 0.0 or self.plot_fs <= 0.0:
            raise ValueError("sensor_fs and plot_fs must be positive.")
        cfg = DecimationConfig(
            sensor_fs=self.sensor_fs,
            plot_fs=self.plot_fs,
            use_envelope=self.use_envelope,
            smoothing_alpha=self.smoothing_alpha,
        )
        self._decimator = Decimator(cfg)

    def handle_samples(self, t: np.ndarray, x: np.ndarray) -> None:
        if x is None or t is None:
            return
        values = np.asarray(x, dtype=np.float32).reshape(-1)
        times = np.asarray(t, dtype=np.float64).reshape(-1)
        if values.size == 0 or times.size == 0:
            return
        # Second decimation stage: compress the raw stream into a small
        # mean/envelope representation that the GUI can draw every refresh.
        t_dec, y_mean, y_min, y_max = self._decimator.process_block(values, start_time=float(times[0]))
        if t_dec.size == 0:
            return
        update = PlotUpdate(
            timestamps=t_dec,
            mean=y_mean,
            y_min=y_min,
            y_max=y_max,
            spike_mask=self._compute_spike_mask(y_mean, y_max),
        )
        with self._lock:
            self._latest_update = update
        if self.queue is not None:
            _offer_queue(self.queue, update)

    def latest_update(self) -> Optional[PlotUpdate]:
        with self._lock:
            return self._latest_update

    def drain_queue(self) -> list[PlotUpdate]:
        if self.queue is None:
            return []
        items: list[PlotUpdate] = []
        while True:
            try:
                items.append(self.queue.get_nowait())
            except Empty:
                break
        if items:
            with self._lock:
                self._latest_update = items[-1]
        return items

    def _compute_spike_mask(self, y_mean: np.ndarray, y_max: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if y_max is None:
            return None
        threshold = float(self.spike_threshold)
        if threshold <= 0:
            return None
        diff = y_max - y_mean
        return diff > threshold


@dataclass(slots=True)
class NullSink(SampleSink):
    """No-op sink used when a pipeline stage is disabled."""

    def handle_samples(self, t: np.ndarray, x: np.ndarray) -> None:  # pragma: no cover - trivial
        return


@dataclass(slots=True)
class Pipeline:
    """Fan out raw samples to recorder/streamer/plotter sinks."""

    recorder: SampleSink = field(default_factory=NullSink)
    streamer: SampleSink = field(default_factory=NullSink)
    plotter: SampleSink = field(default_factory=NullSink)

    def handle_samples(self, t: Sequence[float] | np.ndarray, x: Sequence[float] | np.ndarray) -> None:
        times = np.asarray(t, dtype=np.float64).reshape(-1)
        values = np.asarray(x).reshape(-1)
        if times.size != values.size:
            raise ValueError("t and x must have the same number of elements.")
        # Fan out the same block of samples to each sink; each sink can apply
        # its own decimation or buffering policy (recording, streaming, plotting).
        self.recorder.handle_samples(times, values)
        self.streamer.handle_samples(times, values)
        self.plotter.handle_samples(times, values)

    def on_new_sample(self, timestamp: float, value: float) -> None:
        """Append a single sample (convenience helper)."""
        t_arr = np.asarray([timestamp], dtype=np.float64)
        x_arr = np.asarray([value])
        self.handle_samples(t_arr, x_arr)
