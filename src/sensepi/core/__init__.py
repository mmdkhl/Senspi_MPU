"""Core streaming pipeline: sessions, buffers, and data flow.

This package sits between remote ingest and the GUI by coordinating recorder
sessions, in-memory buffers, and the fan-out pipeline that feeds plots,
streaming sockets, and disk writers.
"""

# Data structures shared by the pipeline (historically imported from here)
from .ringbuffer import RingBuffer
from .timeseries_buffer import TimeSeriesBuffer

# High-level controllers and wiring helpers
from .pipeline import (
    NullSink,
    Pipeline,
    PlotUpdate,
    Plotter,
    Recorder,
    SampleSink,
    Streamer,
)
from .pipeline_wiring import PipelineHandles, build_pipeline
from .recorder_session import RecorderSession

__all__ = [
    "RingBuffer",
    "TimeSeriesBuffer",
    "Pipeline",
    "Plotter",
    "Recorder",
    "Streamer",
    "PlotUpdate",
    "NullSink",
    "SampleSink",
    "PipelineHandles",
    "build_pipeline",
    "RecorderSession",
]
