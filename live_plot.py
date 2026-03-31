"""Matplotlib helper that displays decimated samples with optional envelopes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
    runtime_checkable,
)

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from envelope_plot import (
    init_envelope_plot,
    init_spike_markers,
    update_envelope_plot,
    update_spike_markers,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sensepi.config import SensePiConfig


log = logging.getLogger(__name__)


@runtime_checkable
class PlotChunkLike(Protocol):  # pragma: no cover - structural typing helper
    """Subset of the PlotUpdate interface used by LivePlot."""

    timestamps: np.ndarray
    mean: np.ndarray
    y_min: Optional[np.ndarray]
    y_max: Optional[np.ndarray]
    spike_mask: Optional[np.ndarray]


PlotTuple = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
FetchResult = Union[PlotChunkLike, PlotTuple, Sequence[PlotTuple], Sequence[PlotChunkLike], None]


def _as_1d_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return ``values`` as a contiguous 1D float array."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr


@dataclass
class LivePlot:
    """
    Manage a scrolling Matplotlib display for decimated sensor data.

    ``update_plot`` accepts either a :class:`PlotUpdate` instance or a tuple of
    ``(t_dec, y_mean, y_min, y_max)``.  The class stores the last
    ``window_seconds`` worth of data and reuses Matplotlib artists so that draw
    calls remain cheap enough for 20–60 Hz refresh rates on a Raspberry Pi.
    """

    window_seconds: float = 10.0
    spike_threshold: float = 0.5
    autoscale_margin: float = 0.05

    # Optional injection of an existing Matplotlib Figure / Axes
    fig: plt.Figure | None = None
    ax: plt.Axes | None = None

    # Matplotlib artists, initialised in __post_init__
    line: Any = field(init=False, repr=False)
    envelope_coll: Any = field(init=False, repr=False)
    spike_scatter: Any = field(init=False, repr=False)

    _t: Deque[float] = field(init=False, default_factory=deque)
    _y_mean: Deque[float] = field(init=False, default_factory=deque)
    _y_min: Deque[float] = field(init=False, default_factory=deque)
    _y_max: Deque[float] = field(init=False, default_factory=deque)
    _animation: Optional[FuncAnimation] = field(init=False, default=None, repr=False)
    _envelope_enabled: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.window_seconds = max(0.1, float(self.window_seconds))
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            assert isinstance(self.fig, plt.Figure)
            assert isinstance(self.ax, plt.Axes)

        self.line, self.envelope_coll = init_envelope_plot(
            self.ax,
            color="C0",
            alpha=0.2,
        )
        self.spike_scatter = init_spike_markers(
            self.ax,
            color="red",
            marker="x",
            size=30.0,
        )

        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Sensor value")
        self.ax.grid(True)

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_config(
        cls,
        cfg: "SensePiConfig",
        **overrides: Any,
    ) -> "LivePlot":
        """Build a :class:`LivePlot` using the relevant values from ``cfg``.

        ``overrides`` can supply extra keyword arguments (e.g. ``fig`` / ``ax``)
        that are forwarded to :class:`LivePlot`'s constructor.
        """

        window_seconds = float(getattr(cfg, "plot_window_seconds", 10.0))
        spike_threshold = float(getattr(cfg, "spike_threshold", 0.5))
        params = dict(overrides)
        params.setdefault("window_seconds", window_seconds)
        params.setdefault("spike_threshold", spike_threshold)
        return cls(**params)

    # ------------------------------------------------------------------ buffers
    def _trim_window(self) -> None:
        if not self._t:
            return
        t_latest = self._t[-1]
        t_min = t_latest - self.window_seconds
        dropped = 0
        while self._t and self._t[0] < t_min:
            self._t.popleft()
            self._y_mean.popleft()
            self._y_min.popleft()
            self._y_max.popleft()
            dropped += 1
        if dropped and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "LivePlot._trim_window: dropped %d samples older than %.3f s",
                dropped,
                float(t_min),
            )

    def _extend_deque(self, dest: Deque[float], values: Iterable[float]) -> None:
        dest.extend(float(v) for v in values)

    def add_data(
        self,
        t_dec: np.ndarray,
        y_mean: np.ndarray,
        y_min: Optional[np.ndarray],
        y_max: Optional[np.ndarray],
    ) -> None:
        """Append a decimated block into the rolling buffer."""
        if t_dec.size == 0:
            return
        has_envelope = y_min is not None and y_max is not None

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "LivePlot.add_data: %d samples, t=[%.3f, %.3f], envelope=%s",
                int(t_dec.size),
                float(t_dec[0]),
                float(t_dec[-1]),
                has_envelope,
            )

        y_min_vals = y_min if has_envelope else y_mean
        y_max_vals = y_max if has_envelope else y_mean

        self._envelope_enabled = bool(has_envelope)
        self._extend_deque(self._t, t_dec)
        self._extend_deque(self._y_mean, y_mean)
        self._extend_deque(self._y_min, y_min_vals)
        self._extend_deque(self._y_max, y_max_vals)
        self._trim_window()

    # ---------------------------------------------------------------- redraw
    def redraw(self) -> None:
        """Update artists to reflect the current buffers."""
        if not self._t:
            return

        t_arr = np.fromiter(self._t, dtype=float)
        y_mean_arr = np.fromiter(self._y_mean, dtype=float)
        y_min_arr = np.fromiter(self._y_min, dtype=float)
        y_max_arr = np.fromiter(self._y_max, dtype=float)

        envelope_min = y_min_arr if self._envelope_enabled else None
        envelope_max = y_max_arr if self._envelope_enabled else None

        new_coll = update_envelope_plot(
            self.line,
            self.envelope_coll,
            t_arr,
            y_mean_arr,
            envelope_min,
            envelope_max,
        )
        if new_coll is not None:
            self.envelope_coll = new_coll

        update_spike_markers(
            self.spike_scatter,
            t_arr,
            y_mean_arr,
            envelope_max,
            self.spike_threshold,
        )

        t_end = t_arr[-1]
        t_start = max(t_end - self.window_seconds, t_arr[0])
        self.ax.set_xlim(t_start, t_end)

        # Autoscale with a configurable margin; clamp using element-wise min/max
        y_low = float(np.nanmin(np.minimum(y_min_arr, y_mean_arr)))
        y_high = float(np.nanmax(np.maximum(y_max_arr, y_mean_arr)))
        if not np.isfinite(y_low) or not np.isfinite(y_high):
            y_low, y_high = -1.0, 1.0
        if y_high <= y_low:
            pad = max(1e-3, abs(y_high) * 0.05 + self.autoscale_margin)
            y_low -= pad
            y_high += pad
        else:
            pad = (y_high - y_low) * float(self.autoscale_margin)
            y_low -= pad
            y_high += pad
        self.ax.set_ylim(y_low, y_high)

        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- control
    def update_plot(self, data_chunk: PlotChunkLike | PlotTuple | None) -> None:
        """
        Primary entry point used by animation/timer callbacks.

        ``data_chunk`` can be ``None`` (no update), a :class:`PlotUpdate` instance,
        or a tuple ``(t_dec, y_mean, y_min, y_max)``.  Supplying a two-element tuple
        ``(t_dec, y_mean)`` is also supported.
        """
        if data_chunk is None:
            return

        t_dec, y_mean, y_min, y_max = self._parse_chunk(data_chunk)
        if t_dec.size == 0:
            return
        self.add_data(t_dec, y_mean, y_min, y_max)
        self.redraw()

    def _parse_chunk(self, chunk: PlotChunkLike | PlotTuple) -> PlotTuple:
        if hasattr(chunk, "timestamps"):
            t_dec = _as_1d_array(chunk.timestamps)  # type: ignore[attr-defined]
            y_mean = _as_1d_array(chunk.mean)  # type: ignore[attr-defined]
            y_min = None
            y_max = None
            if getattr(chunk, "y_min", None) is not None:
                y_min = _as_1d_array(chunk.y_min)  # type: ignore[attr-defined]
            if getattr(chunk, "y_max", None) is not None:
                y_max = _as_1d_array(chunk.y_max)  # type: ignore[attr-defined]
            return t_dec, y_mean, y_min, y_max

        if not isinstance(chunk, tuple):
            raise TypeError("Unsupported data chunk type passed to LivePlot.")
        if len(chunk) == 2:
            t_dec, y_mean = chunk
            return (
                _as_1d_array(t_dec),
                _as_1d_array(y_mean),
                None,
                None,
            )
        if len(chunk) >= 4:
            t_dec, y_mean, y_min, y_max = chunk[:4]
            return (
                _as_1d_array(t_dec),
                _as_1d_array(y_mean),
                None if y_min is None else _as_1d_array(y_min),
                None if y_max is None else _as_1d_array(y_max),
            )
        raise ValueError("Expected tuple with 2 or 4 elements for plot data.")

    def start_animation(
        self,
        fetch_data: Callable[[], FetchResult],
        interval_ms: int = 40,
    ) -> FuncAnimation:
        """
        Drive ``update_plot`` via a Matplotlib ``FuncAnimation`` timer.

        ``fetch_data`` should return one of the accepted chunk formats or a
        sequence (e.g. list) of chunks.  Returning ``None`` results in a no-op,
        which makes it easy to poll ``queue.Queue`` objects with ``get_nowait``.
        """

        def _tick(_frame: int) -> None:
            result = fetch_data()
            if result is None:
                return
            if (
                isinstance(result, Sequence)
                and result
                and not isinstance(result, (tuple, PlotChunkLike))
            ):
                for item in result:
                    self.update_plot(item)
            else:
                self.update_plot(result)  # type: ignore[arg-type]

        self._animation = FuncAnimation(
            self.fig,
            _tick,
            interval=max(1, int(interval_ms)),
            blit=False,
        )
        return self._animation

    def stop_animation(self) -> None:
        """Cancel the animation timer if one was created."""
        if self._animation is not None:
            self._animation.event_source.stop()
            self._animation = None
