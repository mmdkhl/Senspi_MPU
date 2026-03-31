"""Generic streaming data buffers and queues for sensor samples.

Lightweight containers in this package (e.g. :mod:`stream_buffer`) provide
in-memory storage and fan-out helpers that shuttle sensor samples between
threads. They stay decoupled from Qt or network code so they can be reused in
the GUI, remote workers, and offline scripts.
"""

from __future__ import annotations

__all__ = [
    "BufferConfig",
    "StreamingDataBuffer",
]

from .stream_buffer import BufferConfig, StreamingDataBuffer

# Backwards-compatible re-exports for historical buffer classes in sensepi.core.
try:
    from ..core import RingBuffer, TimeSeriesBuffer
except ImportError:  # pragma: no cover - optional dependency during docs builds
    pass
else:  # pragma: no cover - simple namespace wiring
    __all__ += ["RingBuffer", "TimeSeriesBuffer"]

