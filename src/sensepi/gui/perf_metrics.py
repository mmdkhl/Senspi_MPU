"""Lightweight performance metrics used by plotting widgets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

MAX_SAMPLES_PERF = 300


@dataclass
class PlotPerfStats:
    """Ring-buffer style tracking of recent redraw performance."""

    frame_times: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES_PERF))
    frame_durations: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES_PERF))
    sample_to_draw_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES_PERF))

    def record_frame(self, start_ts: float, end_ts: float) -> None:
        """Add a frame timestamp and its duration."""
        self.frame_times.append(end_ts)
        self.frame_durations.append(end_ts - start_ts)

    def record_latency(self, latency_s: float) -> None:
        """Track latency between sample arrival and it being drawn."""
        self.sample_to_draw_latencies.append(latency_s)

    def compute_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        dt = self.frame_times[-1] - self.frame_times[0]
        if dt <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / dt

    def avg_frame_ms(self) -> float:
        if not self.frame_durations:
            return 0.0
        return 1000.0 * sum(self.frame_durations) / len(self.frame_durations)

    def avg_latency_ms(self) -> float:
        if not self.sample_to_draw_latencies:
            return 0.0
        return 1000.0 * sum(self.sample_to_draw_latencies) / len(self.sample_to_draw_latencies)

    def max_latency_ms(self) -> float:
        if not self.sample_to_draw_latencies:
            return 0.0
        return 1000.0 * max(self.sample_to_draw_latencies)

    def as_dict(self) -> dict[str, float]:
        """
        Return a snapshot of commonly-used metrics.

        This is useful for a perf HUD or structured logging.
        """
        return {
            "fps": self.compute_fps(),
            "avg_frame_ms": self.avg_frame_ms(),
            "avg_latency_ms": self.avg_latency_ms(),
            "max_latency_ms": self.max_latency_ms(),
        }

    def reset(self) -> None:
        """Clear all stored samples."""
        self.frame_times.clear()
        self.frame_durations.clear()
        self.sample_to_draw_latencies.clear()
