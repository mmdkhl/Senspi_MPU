"""Factory helpers that wire a :class:`Pipeline` from configuration."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Callable, Optional

import numpy as np

from ..config import SensePiConfig
from .pipeline import PlotUpdate, Plotter, Recorder, StreamPayload, Streamer, NullSink, Pipeline, SampleBlockWriter


RecorderWriter = SampleBlockWriter | Callable[[np.ndarray, np.ndarray], None]
TransportFn = Callable[[np.ndarray, np.ndarray], None]


@dataclass(slots=True)
class PipelineHandles:
    """Return value from :func:`build_pipeline` containing ready-to-use pieces."""

    pipeline: Pipeline
    # Queue drained by the networking / Qt layer for live streaming (optional).
    stream_queue: Queue[StreamPayload] | None = None
    # Queue drained by the plotting layer to update live plots (optional).
    plot_queue: Queue[PlotUpdate] | None = None


def build_pipeline(
    cfg: SensePiConfig,
    *,
    recorder_writer: Optional[RecorderWriter] = None,
    stream_transport: Optional[TransportFn] = None,
    stream_queue: Queue[StreamPayload] | None = None,
    plot_queue: Queue[PlotUpdate] | None = None,
) -> PipelineHandles:
    """Construct a :class:`Pipeline` that fans out raw samples to the configured sinks.

    Recording sees the full device sample rate, while streaming and plotting
    use decimated views to keep network and GUI load manageable.

    Parameters
    ----------
    cfg:
        Runtime configuration (usually loaded from YAML).
    recorder_writer:
        Callable that persists raw samples. When omitted, the recorder becomes a
        :class:`NullSink`, even if recording is enabled in the config.
    stream_transport:
        Function invoked with decimated stream data (e.g. network sender).
    stream_queue:
        Optional :class:`queue.Queue` for streaming payloads. When ``None`` and
        streaming is enabled, a bounded queue sized via ``cfg.stream_queue_size``
        is created automatically.
    plot_queue:
        Optional :class:`queue.Queue`` receiving :class:`PlotUpdate` objects.
        Created automatically using ``cfg.plot_queue_size`` when omitted.
    """

    normalized = cfg.sanitized()

    # Recorder
    if normalized.recording_enabled and recorder_writer is not None:
        recorder = Recorder(
            writer=recorder_writer,
            sensor_fs=normalized.sensor_fs,
            chunk_seconds=normalized.recording_chunk_seconds,
        )
    else:
        recorder = NullSink()

    # Streamer
    if normalized.streaming_enabled:
        stream_queue = stream_queue or Queue(maxsize=normalized.stream_queue_size)
        streamer = Streamer(
            sensor_fs=normalized.sensor_fs,
            stream_fs=normalized.stream_fs,
            transport=stream_transport,
            queue=stream_queue,
        )
    else:
        stream_queue = None
        streamer = NullSink()

    # Plotter
    if normalized.plotting_enabled:
        plot_queue = plot_queue or Queue(maxsize=normalized.plot_queue_size)
        plotter = Plotter(
            sensor_fs=normalized.sensor_fs,
            plot_fs=normalized.plot_fs,
            smoothing_alpha=normalized.smoothing_alpha,
            use_envelope=normalized.use_envelope,
            spike_threshold=normalized.spike_threshold,
            queue=plot_queue,
        )
    else:
        plot_queue = None
        plotter = NullSink()

    pipeline = Pipeline(recorder=recorder, streamer=streamer, plotter=plotter)
    return PipelineHandles(pipeline=pipeline, stream_queue=stream_queue, plot_queue=plot_queue)


__all__ = ["PipelineHandles", "build_pipeline", "RecorderWriter", "TransportFn"]
