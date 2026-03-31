from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Literal, Optional


RateQuality = Literal["status_only", "ts_only", "fused"]


@dataclass
class RateEstimate:
    """Container for streaming rate estimate."""

    hz_effective: float
    hz_raw_status: Optional[float]
    hz_ts_window: Optional[float]
    quality: RateQuality


class RateController:
    """
    Estimate streaming rate from sample timestamps and optional status updates.

    Notes
    -----
    - Timestamps are assumed to be in seconds (monotonic increasing).
    - You can fuse:
        * a recent status-reported rate (e.g., device-reported),
        * and a windowed estimate from timestamps.

    The 'effective' rate is a simple average of status and timestamp
    estimates; you can easily change that if needed.
    """

    def __init__(self, window_size: int = 100, default_hz: float = 0.0) -> None:
        if window_size <= 1:
            raise ValueError("window_size must be > 1")
        self._times: Deque[float] = deque(maxlen=window_size)
        self.default_hz = float(default_hz)
        self._hz_status: Optional[float] = None

    def add_sample_time(self, t: float) -> None:
        """
        Append a new sample timestamp.

        Parameters
        ----------
        t:
            Sample timestamp in seconds (monotonic increasing).
        """
        self._times.append(float(t))

    def update_from_status(self, hz: float | None = None) -> RateEstimate:
        """
        Update controller from a device- or status-reported rate.

        Parameters
        ----------
        hz:
            Device-reported rate in Hz. If None, falls back to default_hz.

        Returns
        -------
        RateEstimate
            Estimate based only on status info.
        """
        if hz is None:
            hz = self.default_hz
        hz_f = float(hz)
        if hz_f < 0:
            raise ValueError(f"status rate must be >= 0, got {hz_f}")

        self._hz_status = hz_f
        return RateEstimate(
            hz_effective=hz_f,
            hz_raw_status=hz_f,
            hz_ts_window=None,
            quality="status_only",
        )

    @property
    def estimated_hz(self) -> float:
        """Estimate Hz from the current timestamp window only."""
        if len(self._times) < 2:
            return self.default_hz
        t0 = self._times[0]
        t1 = self._times[-1]
        span = t1 - t0
        if span <= 0:
            return self.default_hz
        count = len(self._times) - 1
        return count / span

    def estimate(self) -> RateEstimate:
        """
        Return fused rate estimate (status + timestamps if both are available).
        """
        hz_ts = self.estimated_hz
        if self._hz_status is not None:
            hz_eff = 0.5 * self._hz_status + 0.5 * hz_ts
            return RateEstimate(
                hz_effective=hz_eff,
                hz_raw_status=self._hz_status,
                hz_ts_window=hz_ts,
                quality="fused",
            )
        return RateEstimate(
            hz_effective=hz_ts,
            hz_raw_status=None,
            hz_ts_window=hz_ts,
            quality="ts_only",
        )

    @property
    def buffer_span_s(self) -> float:
        """Time span (seconds) covered by the current timestamp window."""
        if len(self._times) < 2:
            return 0.0
        return self._times[-1] - self._times[0]

    @property
    def buffer_size(self) -> int:
        """Number of timestamps currently in the window."""
        return len(self._times)

    def reset(self) -> None:
        """Clear all timestamps and status information."""
        self._times.clear()
        self._hz_status = None

    def feed_times(self, times: Iterable[float]) -> None:
        """Convenience method to bulk-add timestamps."""
        for t in times:
            self.add_sample_time(t)
