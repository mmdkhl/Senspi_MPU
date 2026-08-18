"""Signals tab and live plotting widgets for SensePi."""

from __future__ import annotations

import logging
import math
import queue
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from PySide6.QtCore import QSignalBlocker, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg
import numpy as np

from ..perf_metrics import PlotPerfStats
from ..widgets import (
    AcquisitionSettings,
    AcquisitionSettingsWidget,
    CollapsibleSection,
)
from ..config.acquisition_state import (
    CalibrationOffsets,
    GuiAcquisitionConfig,
    SensorSelectionConfig,
)
from ...config.sampling import SamplingConfig
from ...config.app_config import AppConfig, PlotPerformanceConfig
from ...config.constants import ENABLE_PLOT_PERF_METRICS
from ...core.timeseries_buffer import (
    NS_PER_SECOND,
    TimeSeriesBuffer,
    calculate_capacity,
    initialize_buffers_for_channels,
    ns_to_seconds,
)
from ...data import BufferConfig, StreamingDataBuffer
from ...baseline import BaselineState, collect_baseline_samples
from ...sensors.mpu6050 import MpuSample
from ...tools.debug import debug_enabled
from . import LayoutSignature, SampleKey

if TYPE_CHECKING:
    from ..recorder_controller import RecorderController

DEFAULT_REFRESH_MODE = "fixed"
DEFAULT_REFRESH_INTERVAL_MS = 33  # ~30 Hz default for live traces
# Hard lower bound on the GUI timer interval. Updating more often than ~50 Hz
# brings little perceptual benefit but can chew CPU when many traces are shown.
MIN_REFRESH_INTERVAL_MS = 20
REFRESH_PROFILE_CUSTOM_LABEL = "Custom"
REFRESH_PRESETS: list[tuple[str, int]] = [
    ("10 Hz", 100),
    ("30 Hz", 33),
    ("50 Hz", 20),
]
REFRESH_FOLLOW_DEVICE = "__device_rate__"  # QComboBox itemData sentinel
STREAM_STALL_THRESHOLD_S = 2.0
DEFAULT_DISPLAY_SLACK_NS = int(0.05 * NS_PER_SECOND)
MANUAL_STATUS_HOLD_S = 1.5

logger = logging.getLogger(__name__)

class SignalPlotWidgetBase(QWidget):
    """Shared data management and rendering helpers for signal plot widgets."""

    def __init__(self, parent: Optional[QWidget] = None, max_seconds: float = 10.0) -> None:
        super().__init__(parent)

        self._max_seconds = float(max_seconds)
        self._max_rate_hz: float = 500.0
        self._buffer_margin: float = 1.2
        self._buffer_capacity = calculate_capacity(
            self._max_seconds,
            self._max_rate_hz,
            margin=self._buffer_margin,
        )
        self._buffers: Dict[SampleKey, TimeSeriesBuffer] = self._create_buffer_store()
        self._sensor_ids = self._extract_sensor_ids()

        self._visible_channels: Set[str] = set()
        self._channel_order: list[str] = []

        # Per-line visibility filter (keeps backend objects alive while hiding traces)
        self._visible_line_keys: Set[SampleKey] = set()
        self._visible_channel_filter: Set[str] = set()
        self._visibility_filter_active: bool = False

        self._line_width: float = 0.8
        self._max_points_per_trace: int = 2000

        self._base_correction_enabled: bool = False
        self._baseline_offsets: Dict[SampleKey, float] = {}
        self._display_slack_ns: int = 0
        self._latest_timestamp_ns: Optional[int] = None

        self._perf: Optional[PlotPerfStats] = PlotPerfStats() if ENABLE_PLOT_PERF_METRICS else None
        self._latest_gui_receive_ts: Optional[float] = None
        self._target_refresh_hz: Optional[float] = None

        self._lines: Dict[SampleKey, Any] = {}
        self._layout_signature: LayoutSignature | tuple[()] = ()
        self._needs_layout: bool = True

        self._nominal_sample_rate_hz: float = 200.0
        self._plot_window_samples: int = self._compute_window_samples(
            self._nominal_sample_rate_hz
        )
        self._time_axis: np.ndarray = self._compute_time_axis(
            self._plot_window_samples,
            self._nominal_sample_rate_hz,
        )
        self._plot_buffers: Dict[SampleKey, np.ndarray] = {}
        self._plot_write_counts: Dict[SampleKey, int] = {}
        self._max_subplots: int | None = None
        self._max_lines_per_subplot: int | None = None

    def _create_buffer_store(self) -> Dict[SampleKey, TimeSeriesBuffer]:
        # Physical sensors are 1-4. Using 0 here created a phantom S0 row.
        return initialize_buffers_for_channels(
            sensor_ids=(1, 2, 3, 4),
            channels=("ax", "ay", "az", "gx", "gy", "gz"),
            window_seconds=self._max_seconds,
            max_rate_hz=self._max_rate_hz,
            margin=self._buffer_margin,
        )

    def _extract_sensor_ids(self) -> list[int]:
        return sorted({sensor_id for sensor_id, _ in self._buffers.keys()})

    def _make_key(self, sensor_id: int, channel: str) -> SampleKey:
        return int(sensor_id), str(channel)

    def _get_buffer(self, key: SampleKey) -> TimeSeriesBuffer | None:
        return self._buffers.get(key)

    def _ensure_buffer(self, key: SampleKey) -> TimeSeriesBuffer:
        buf = self._buffers.get(key)
        if buf is None:
            buf = TimeSeriesBuffer(self._buffer_capacity)
            self._buffers[key] = buf
            sensor_id = int(key[0])
            if sensor_id not in self._sensor_ids:
                self._sensor_ids.append(sensor_id)
                self._sensor_ids.sort()
                self._needs_layout = True
        return buf

    @property
    def window_seconds(self) -> float:
        """Length of the sliding time window shown in the plots."""
        return self._max_seconds

    def _compute_window_samples(self, sample_rate_hz: float) -> int:
        rate = max(1.0, float(sample_rate_hz))
        samples = int(math.ceil(rate * self._max_seconds))
        return max(1, samples)

    def _compute_time_axis(self, window_samples: int, sample_rate_hz: float) -> np.ndarray:
        rate = max(1.0, float(sample_rate_hz))
        count = max(1, int(window_samples))
        return np.arange(count, dtype=np.float64) / rate

    def _time_axis_domain(self) -> tuple[float, float]:
        """Return the (xmin, xmax) bounds in seconds for the rolling window."""

        time_axis = getattr(self, "_time_axis", None)
        if isinstance(time_axis, np.ndarray) and time_axis.size:
            return float(time_axis[0]), float(time_axis[-1])
        return 0.0, self._max_seconds

    @staticmethod
    def _channel_units(channel: str) -> str:
        """Return a human-readable unit for a channel name."""
        ch = channel.lower()
        if ch in {"ax", "ay", "az"}:
            return "m/s²"
        if ch in {"gx", "gy", "gz"}:
            return "deg/s"
        return ""

    # --------------------------------------------------------------- public API
    def clear(self) -> None:
        """Clear all buffered data and reset the plot."""
        self._buffers = self._create_buffer_store()
        self._sensor_ids = self._extract_sensor_ids()
        self._reset_plot_buffers()
        self._baseline_offsets.clear()
        self._latest_timestamp_ns = None
        self._latest_gui_receive_ts = None
        self._clear_canvas()

    def set_display_slack_ns(self, slack_ns: int) -> None:
        """Configure how much to lag the display behind new data to absorb jitter."""
        self._display_slack_ns = max(0, int(slack_ns))

    def set_target_refresh_rate(self, hz: Optional[float]) -> None:
        """Configure the intended refresh rate for FPS/drop metrics."""
        if hz is None:
            self._target_refresh_hz = None
            return
        try:
            value = float(hz)
        except (TypeError, ValueError):
            self._target_refresh_hz = None
            return
        if value <= 0.0:
            self._target_refresh_hz = None
            return
        self._target_refresh_hz = value

    def set_max_points_per_trace(self, max_points: int) -> None:
        """Cap how many samples are rendered per line in the live plots."""
        try:
            value = int(max_points)
        except (TypeError, ValueError):
            return
        self._max_points_per_trace = max(100, value)

    def set_nominal_sample_rate(self, hz: Optional[float]) -> None:
        """Update the nominal sample rate used for the rolling plot buffers."""
        try:
            value = float(hz) if hz is not None else float("nan")
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value) or value <= 0.0:
            value = 200.0
        value = max(1.0, value)
        if math.isclose(value, self._nominal_sample_rate_hz, rel_tol=0.05, abs_tol=0.5):
            return
        self._nominal_sample_rate_hz = value
        self._plot_window_samples = self._compute_window_samples(value)
        self._time_axis = self._compute_time_axis(self._plot_window_samples, value)
        self._reset_plot_buffers()
        self._refresh_axes_limits()

    def latest_timestamp_ns(self) -> Optional[int]:
        """Return the last timestamp appended to any buffer."""
        if self._latest_timestamp_ns is not None:
            return self._latest_timestamp_ns
        latest: Optional[int] = None
        for buf in self._buffers.values():
            candidate = buf.latest_timestamp_ns()
            if candidate is None:
                continue
            if latest is None or candidate > latest:
                latest = candidate
        self._latest_timestamp_ns = latest
        return latest

    def live_sensor_ids(self) -> list[int]:
        """
        Return sensor IDs that currently have buffered samples.

        This allows other widgets (e.g. FFT) to know which sensors are active
        when sharing the same ring buffers.
        """
        sensor_ids: Set[int] = set()
        for (sensor_id, _), buf in self._buffers.items():
            if len(buf) > 0:
                sensor_ids.add(sensor_id)
        return sorted(sensor_ids)

    def get_time_series_window(
        self,
        sensor_id: int,
        channel: str,
        window_seconds: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(times_s, values)`` arrays for the requested channel.

        The timestamps are converted to floating-point seconds so downstream
        consumers (FFT tab) can compute sample spacing without duplicating the
        ring-buffer plumbing.
        """
        key = self._make_key(sensor_id, channel)
        buf = self._get_buffer(key)
        if buf is None or len(buf) == 0:
            return self._empty_series()

        latest_ns = buf.latest_timestamp_ns()
        if latest_ns is None:
            return self._empty_series()

        try:
            window = float(window_seconds)
        except (TypeError, ValueError):
            window = self._max_seconds
        if window <= 0.0:
            window = self._max_seconds

        window_ns = int(window * NS_PER_SECOND)
        start_ns = max(0, latest_ns - window_ns)
        ts_ns, values = buf.get_window(start_ns, latest_ns)
        if ts_ns.size == 0 or values.size == 0:
            return self._empty_series()

        times_s = ns_to_seconds(ts_ns)
        return times_s, values.copy()

    @staticmethod
    def _empty_series() -> tuple[np.ndarray, np.ndarray]:
        """Return empty arrays for shared buffer queries."""
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    def _reset_plot_buffers(self) -> None:
        """Reset cached rolling buffers used for display rendering."""
        self._plot_buffers.clear()
        self._plot_write_counts.clear()

    def get_perf_snapshot(self) -> dict[str, float]:
        """Return current performance metrics if tracking is enabled."""
        target = float(self._target_refresh_hz) if self._target_refresh_hz else 0.0
        perf = self._perf
        snapshot = {
            "fps": 0.0,
            "achieved_fps": 0.0,
            "avg_frame_ms": 0.0,
            "avg_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "target_fps": target,
            "approx_dropped_frames_per_sec": 0.0,
        }
        if not (ENABLE_PLOT_PERF_METRICS and perf):
            return snapshot

        fps = perf.compute_fps()
        approx_drop = 0.0
        if target > 0.0:
            approx_drop = max(0.0, target - fps)
        snapshot.update(
            {
                "fps": fps,
                "achieved_fps": fps,
                "avg_frame_ms": perf.avg_frame_ms(),
                "avg_latency_ms": perf.avg_latency_ms(),
                "max_latency_ms": perf.max_latency_ms(),
                "target_fps": target,
                "approx_dropped_frames_per_sec": approx_drop,
            }
        )
        return snapshot

    def set_channel_layout(self, channels: Iterable[str]) -> None:
        """Configure which channel columns should be present in the grid."""
        channels_list = [str(ch) for ch in channels]
        self._visible_channels = set(channels_list)
        self._channel_order = channels_list
        self._needs_layout = True
        self._ensure_layout(self._get_visible_channels())

    def set_visible_channels(
        self,
        visible: Optional[Iterable[SampleKey | str]],
    ) -> None:
        """
        Hide or show individual sensor/channel traces without rebuilding plots.

        ``visible`` may contain ``(sensor_id, channel)`` tuples for explicit
        control or raw channel strings to apply visibility across all sensors.
        Pass ``None`` to reset the filter and show all traces.
        """
        if visible is None:
            self._visibility_filter_active = False
            self._visible_line_keys.clear()
            self._visible_channel_filter.clear()
            self._apply_visibility_to_all_lines()
            return

        explicit_keys: Set[SampleKey] = set()
        channel_filter: Set[str] = set()
        for item in visible:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                sensor_id, channel = item
                explicit_keys.add(self._make_key(sensor_id, channel))
            else:
                channel_filter.add(str(item))

        self._visible_line_keys = explicit_keys
        self._visible_channel_filter = channel_filter
        self._visibility_filter_active = True
        self._apply_visibility_to_all_lines()

    def set_subplot_limits(
        self,
        *,
        max_subplots: Optional[int] = None,
        max_lines_per_subplot: Optional[int] = None,
    ) -> None:
        """Clamp how many subplot slots and traces may be created."""
        normalized_subplots = self._normalize_positive_int(max_subplots)
        normalized_lines = self._normalize_positive_int(max_lines_per_subplot)
        if normalized_lines is None:
            normalized_lines = None
        if normalized_subplots == self._max_subplots and normalized_lines == self._max_lines_per_subplot:
            return
        self._max_subplots = normalized_subplots
        self._max_lines_per_subplot = normalized_lines
        self._needs_layout = True

    @staticmethod
    def _normalize_positive_int(value: Optional[int]) -> int | None:
        if value is None:
            return None
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return None
        if normalized <= 0:
            return None
        return normalized

    def add_sample(self, sample: MpuSample) -> None:
        """Append a sample from the MPU6050 sensor."""
        if ENABLE_PLOT_PERF_METRICS:
            gui_ts = getattr(sample, "gui_receive_ts", None)
            if gui_ts is not None:
                try:
                    self._latest_gui_receive_ts = float(gui_ts)
                except (TypeError, ValueError):
                    pass

        # Use sensor_id as row index; default to 1 if missing
        sensor_id = int(sample.sensor_id) if sample.sensor_id is not None else 1
        if sample.t_s is not None:
            t_ns = int(round(float(sample.t_s) * NS_PER_SECOND))
        else:
            t_ns = int(sample.timestamp_ns)
        for ch in ("ax", "ay", "az", "gx", "gy", "gz"):
            val = getattr(sample, ch, None)
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if math.isnan(v):
                continue
            self._append_point(sensor_id, ch, t_ns, v)

    def add_samples(self, samples: Iterable[MpuSample]) -> None:
        """Append multiple samples sequentially."""

        for sample in samples:
            if sample is None:
                continue
            self.add_sample(sample)

    def redraw(self) -> None:
        """Refresh the live plots (intended to be driven by a QTimer)."""
        perf = self._perf if ENABLE_PLOT_PERF_METRICS else None
        start_ts = time.perf_counter() if perf is not None else None
        try:
            visible_channels = self._get_visible_channels()
            if not visible_channels:
                self._clear_canvas()
                return

            self._ensure_layout(visible_channels)
            if not self._lines:
                return

            slack_samples = self._display_slack_samples()
            time_axis = self._time_axis
            for key in self._lines.keys():
                window_values = self._get_plot_window(key, slack_samples)
                if window_values.size == 0:
                    self._clear_line_data(key)
                    continue
                if self._base_correction_enabled:
                    offset = self._baseline_offsets.get(key, 0.0)
                    if offset:
                        window_values = window_values - offset

                finite_mask = np.isfinite(window_values)
                if not finite_mask.any():
                    self._clear_line_data(key)
                    continue

                times = time_axis[finite_mask]
                values = window_values[finite_mask]

                times_decimated, values_decimated = self._decimate_for_plot(
                    times,
                    values,
                    self._max_points_per_trace,
                )
                if not times_decimated:
                    self._clear_line_data(key)
                    continue
                self._set_line_data(key, times_decimated, values_decimated)

            self._refresh_axes_limits()
        finally:
            if perf is not None and start_ts is not None:
                end_ts = time.perf_counter()
                perf.record_frame(start_ts, end_ts)
                gui_ts = self._latest_gui_receive_ts
                if gui_ts is not None:
                    latency_s = end_ts - gui_ts
                    if 0.0 <= latency_s < 60.0:
                        perf.record_latency(latency_s)

        self._finalize_redraw()

    def _decimate_for_plot(
        self,
        times: Sequence[float],
        values: Sequence[float],
        max_points: int,
    ) -> Tuple[list[float], list[float]]:
        """
        Downsample ``times``/``values`` to at most ``max_points`` samples.

        Uses a simple envelope approach (per-chunk min/max) to preserve spikes
        without incurring heavy computation.
        """
        n = min(len(times), len(values))
        if n == 0:
            return [], []

        try:
            limit = int(max_points)
        except (TypeError, ValueError):
            limit = self._max_points_per_trace
        if limit <= 0:
            limit = 1

        if limit == 1:
            return [float(times[0])], [float(values[0])]

        if n <= limit:
            return (
                [float(times[i]) for i in range(n)],
                [float(values[i]) for i in range(n)],
            )

        last_index = n - 1
        mid_budget = max(0, limit - 2)
        selected_indices: list[int] = [0]

        if mid_budget == 0:
            if last_index != 0:
                selected_indices.append(last_index)
            return (
                [float(times[idx]) for idx in selected_indices],
                [float(values[idx]) for idx in selected_indices],
            )

        chunk_budget = max(1, math.ceil(mid_budget / 2))
        step = max(1, math.ceil(n / chunk_budget))
        mid_added = 0

        for start in range(0, n, step):
            if mid_added >= mid_budget:
                break
            end = min(n, start + step)
            if end - start <= 0:
                continue

            min_idx = start
            max_idx = start
            min_val = float(values[start])
            max_val = min_val

            for idx in range(start + 1, end):
                v = float(values[idx])
                if v < min_val:
                    min_val = v
                    min_idx = idx
                if v > max_val:
                    max_val = v
                    max_idx = idx

            for idx in sorted({min_idx, max_idx}):
                if idx == 0 or idx == last_index:
                    continue
                if idx <= selected_indices[-1]:
                    continue
                selected_indices.append(idx)
                mid_added += 1
                if mid_added >= mid_budget:
                    break

        if selected_indices[-1] != last_index:
            selected_indices.append(last_index)

        times_out = [float(times[idx]) for idx in selected_indices]
        values_out = [float(values[idx]) for idx in selected_indices]
        return times_out, values_out

    def _get_visible_channels(self) -> list[str]:
        return [ch for ch in self._channel_order if ch in self._visible_channels]

    def _apply_subplot_limits(
        self,
        visible_channels: list[str],
    ) -> tuple[list[int], list[str], bool, bool]:
        sensors = list(self._sensor_ids)
        channels = [str(ch) for ch in visible_channels]
        if not sensors or not channels:
            return sensors, channels, False, False
        max_subplots = self._max_subplots
        if not max_subplots or max_subplots <= 0:
            return sensors, channels, False, False
        total = len(sensors) * len(channels)
        if total <= max_subplots:
            return sensors, channels, False, False

        limited_channels = list(channels)
        max_channels = max(1, max_subplots // len(sensors))
        trimmed_channels = False
        if len(limited_channels) > max_channels:
            limited_channels = limited_channels[:max_channels]
            trimmed_channels = True
        if not limited_channels and channels:
            limited_channels = [channels[0]]

        limited_sensors = list(sensors)
        max_sensors = max(1, max_subplots // len(limited_channels))
        trimmed_sensors = False
        if len(limited_sensors) > max_sensors:
            limited_sensors = limited_sensors[:max_sensors]
            trimmed_sensors = True
        return limited_sensors, limited_channels, trimmed_channels, trimmed_sensors

    def _apply_visibility_to_all_lines(self) -> None:
        if not self._lines:
            return
        for key, line in self._lines.items():
            self._line_set_visible(line, self._is_key_visible(key))

    def _is_key_visible(self, key: SampleKey) -> bool:
        if not self._visibility_filter_active:
            return True
        if key in self._visible_line_keys:
            return True
        channel = key[1]
        if channel in self._visible_channel_filter:
            return True
        return False

    def _clear_canvas(self) -> None:
        self._backend_clear()
        self._lines.clear()
        self._layout_signature = ()
        self._needs_layout = True

    def _refresh_axes_limits(self) -> None:
        self._backend_refresh_axes_limits()

    def _ensure_layout(self, visible_channels: list[str]) -> None:
        if not visible_channels:
            self._clear_canvas()
            return
        if not self._sensor_ids:
            self._needs_layout = True
            return
        original_sensor_count = len(self._sensor_ids)
        original_channel_count = len(visible_channels)
        (
            sensor_ids,
            limited_channels,
            trimmed_channels,
            trimmed_sensors,
        ) = self._apply_subplot_limits(visible_channels)
        if not sensor_ids or not limited_channels:
            self._clear_canvas()
            return
        signature: LayoutSignature = (
            tuple(sensor_ids),
            tuple(limited_channels),
        )
        should_log_limits = (
            (trimmed_channels or trimmed_sensors)
            and (self._needs_layout or signature != self._layout_signature)
        )
        if should_log_limits:
            limit = self._max_subplots
            if trimmed_channels and original_channel_count > len(limited_channels):
                logger.warning(
                    "SignalsTab: reducing visible channels from %d to %d to honor max subplot limit (%s).",
                    original_channel_count,
                    len(limited_channels),
                    limit,
                )
            if trimmed_sensors and original_sensor_count > len(sensor_ids):
                logger.warning(
                    "SignalsTab: reducing visible sensor rows from %d to %d to honor max subplot limit (%s).",
                    original_sensor_count,
                    len(sensor_ids),
                    limit,
                )
        if not self._needs_layout and signature == self._layout_signature:
            return
        self._lines.clear()
        self._backend_rebuild_layout(sensor_ids, limited_channels)
        self._layout_signature = signature
        self._needs_layout = False
        self._apply_visibility_to_all_lines()

    # --------------------------------------------------------------- base correction API
    def enable_base_correction(self, enabled: bool) -> None:
        """Enable or disable baseline subtraction."""
        self._base_correction_enabled = bool(enabled)

    def reset_calibration(self) -> None:
        """Clear all stored baseline offsets."""
        self._baseline_offsets.clear()

    def calibrate_from_buffer(self, window_s: float | None = None) -> None:
        """
        Compute per-channel baseline from the most recent time window.

        For each (sensor_id, channel) we take the mean over the same sliding
        window that is used for plotting (self._max_seconds) unless an explicit
        ``window_s`` is provided.
        """
        if not self._buffers:
            return

        latest_times = [
            ts
            for buf in self._buffers.values()
            if len(buf) > 0 and (ts := buf.latest_timestamp_ns()) is not None
        ]
        if not latest_times:
            return

        try:
            window = float(window_s) if window_s is not None else self._max_seconds
        except (TypeError, ValueError):
            window = self._max_seconds
        if window <= 0.0:
            window = self._max_seconds

        window_ns = int(window * NS_PER_SECOND)
        latest_ns = max(latest_times)
        cutoff_ns = max(0, latest_ns - window_ns)

        new_offsets: Dict[SampleKey, float] = {}
        for key, buf in self._buffers.items():
            if not buf:
                continue
            _, values = buf.get_window(cutoff_ns, latest_ns)
            if values.size == 0:
                continue
            new_offsets[key] = float(values.mean())

        self._baseline_offsets = new_offsets

    # --------------------------------------------------------------- internals
    def _append_point(self, sensor_id: int, channel: str, t_ns: int, value: float) -> None:
        key = self._make_key(sensor_id, channel)
        buf = self._ensure_buffer(key)
        buf.append(t_ns, value)
        self._append_plot_value(key, value)
        if self._latest_timestamp_ns is None or t_ns > self._latest_timestamp_ns:
            self._latest_timestamp_ns = t_ns

    def _append_plot_value(self, key: SampleKey, value: float) -> None:
        """Append a sample to the rolling buffer used for live rendering."""
        window = self._plot_window_samples
        if window <= 0:
            return
        buf = self._ensure_plot_buffer(key)
        write_count = self._plot_write_counts.get(key, 0)
        idx = write_count % window
        buf[idx] = float(value)
        self._plot_write_counts[key] = write_count + 1

    def _ensure_plot_buffer(self, key: SampleKey) -> np.ndarray:
        buf = self._plot_buffers.get(key)
        if buf is not None and buf.shape[0] == self._plot_window_samples:
            return buf
        new_buf = np.full(self._plot_window_samples, np.nan, dtype=np.float64)
        self._plot_buffers[key] = new_buf
        self._plot_write_counts[key] = 0
        return new_buf

    def _display_slack_samples(self) -> int:
        if self._display_slack_ns <= 0:
            return 0
        slack_seconds = self._display_slack_ns / float(NS_PER_SECOND)
        samples = int(round(slack_seconds * self._nominal_sample_rate_hz))
        if samples <= 0:
            return 0
        return min(samples, self._plot_window_samples)

    def _get_plot_window(self, key: SampleKey, slack_samples: int) -> np.ndarray:
        buf = self._plot_buffers.get(key)
        if buf is None:
            return np.empty(0, dtype=np.float64)
        window = self._plot_window_samples
        if window <= 0:
            return np.empty(0, dtype=np.float64)
        write_count = self._plot_write_counts.get(key, 0)
        if write_count <= 0:
            return np.empty(0, dtype=np.float64)
        idx = write_count % window
        window_values = np.roll(buf, -idx).copy()
        if write_count < window:
            invalid = window - int(write_count)
            if invalid > 0:
                window_values[:invalid] = np.nan
        if slack_samples > 0:
            shift = min(slack_samples, window)
            window_values = np.roll(window_values, -shift)
            window_values[-shift:] = np.nan
        return window_values

    # ------------------------------------------------------------------ backend hooks
    def _set_line_data(self, key: SampleKey, times: Sequence[float], values: Sequence[float]) -> None:
        line = self._lines.get(key)
        if line is None:
            return
        self._line_set_data(line, times, values)

    def _clear_line_data(self, key: SampleKey) -> None:
        line = self._lines.get(key)
        if line is None:
            return
        self._line_clear_data(line)

    def _line_set_data(self, line: Any, times: Sequence[float], values: Sequence[float]) -> None:
        raise NotImplementedError

    def _line_clear_data(self, line: Any) -> None:
        raise NotImplementedError

    def _line_set_visible(self, line: Any, visible: bool) -> None:
        raise NotImplementedError

    def _backend_clear(self) -> None:
        raise NotImplementedError

    def _backend_refresh_axes_limits(self) -> None:
        raise NotImplementedError

    def _backend_rebuild_layout(self, sensor_ids: list[int], visible_channels: list[str]) -> None:
        raise NotImplementedError

    def _finalize_redraw(self) -> None:
        """Hook for subclasses that need to trigger an explicit redraw."""
        return


class SignalPlotWidgetPyQtGraph(SignalPlotWidgetBase):
    """PyQtGraph implementation of the signal plot widget."""

    def __init__(self, parent: Optional[QWidget] = None, max_seconds: float = 10.0) -> None:
        super().__init__(parent=parent, max_seconds=max_seconds)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._glw = pg.GraphicsLayoutWidget(self)
        layout.addWidget(self._glw)

        self._plots: Dict[SampleKey, pg.PlotItem] = {}
        self._line_pen = pg.mkPen(width=max(1.0, float(self._line_width)))

    def _time_axis_domain(self) -> tuple[float, float]:
        """Expose the base-class time axis helper for Qt's attribute lookup."""
        return super()._time_axis_domain()

    def _line_set_data(self, line: pg.PlotDataItem, times: Sequence[float], values: Sequence[float]) -> None:
        line.setData(times, values)

    def _line_clear_data(self, line: pg.PlotDataItem) -> None:
        line.setData([], [])

    def _line_set_visible(self, line: pg.PlotDataItem, visible: bool) -> None:
        line.setVisible(visible)

    def _backend_clear(self) -> None:
        self._glw.clear()
        self._plots.clear()

    def _backend_refresh_axes_limits(self) -> None:
        xmin, xmax = self._time_axis_domain()
        for plot in self._plots.values():
            plot.setXRange(xmin, xmax, padding=0.0)

    def _backend_rebuild_layout(self, sensor_ids: list[int], visible_channels: list[str]) -> None:
        self._glw.clear()
        self._plots.clear()

        nrows = len(sensor_ids)
        xmin, xmax = self._time_axis_domain()

        for row_idx, sid in enumerate(sensor_ids):
            for col_idx, ch in enumerate(visible_channels):
                plot = self._glw.addPlot(row=row_idx, col=col_idx)
                plot.setMenuEnabled(False)
                plot.hideButtons()
                plot.showGrid(x=True, y=True, alpha=0.3)
                plot.setXRange(xmin, xmax, padding=0.0)
                plot.enableAutoRange(x=False, y=True)

                if row_idx == nrows - 1:
                    plot.setLabel("bottom", "Time", units="s")

                unit = self._channel_units(ch)
                base_label = ch.upper()
                if unit:
                    base_label = f"{base_label} [{unit}]"

                if col_idx == 0:
                    # Label rows as S0, S1, S2... regardless of underlying sensor_id
                    plot.setLabel("left", f"S{row_idx}\n{base_label}")
                else:
                    plot.setLabel("left", base_label)

                line = plot.plot([], [], pen=self._line_pen)
                key = self._make_key(sid, ch)
                self._plots[key] = plot
                self._lines[key] = line


# Backwards compatibility alias: existing imports that expect ``SignalPlotWidget``
# receive the default PyQtGraph implementation used by the Qt GUI.
SignalPlotWidget = SignalPlotWidgetPyQtGraph


class SignalsTab(QWidget):
    """
    Live time-series dashboard for streaming MPU6050 data.

    Responsibilities:
    - Subscribe to :class:`RecorderTab.sample_received` and maintain
      time-series buffers for selected sensors/channels.
    - Drive the Qt-native live plot widget for plotting and
      expose refresh cadence controls (sampling vs. plotting rate decoupling).
    - Request start/stop of streaming or recording runs, propagating refresh
      hints to :class:`FftTab` so spectrum updates stay aligned.
    - Focused on live data; offline playback remains in :class:`OfflineTab`.
    """

    start_stream_requested = Signal(str)  # session name
    stop_stream_requested = Signal()
    record_requested = Signal(str, int)   # Smart Recording: (session name, duration_s)
    fft_refresh_interval_changed = Signal(int)
    acquisitionConfigChanged = Signal(GuiAcquisitionConfig)
    calibrationChanged = Signal(CalibrationOffsets)
    sync_logs_requested = Signal()

    BASELINE_DURATION_SEC = 3.0

    def __init__(
        self,
        recorder_tab: "RecorderController | None" = None,
        parent: Optional[QWidget] = None,
        plot_widget: SignalPlotWidgetBase | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        super().__init__(parent)

        self._recorder_tab: RecorderController | None = recorder_tab
        self._app_config: AppConfig = app_config or AppConfig()
        plot_perf = getattr(self._app_config, "plot_performance", None)
        if not isinstance(plot_perf, PlotPerformanceConfig):
            plot_perf = PlotPerformanceConfig()
        self._plot_perf_config: PlotPerformanceConfig = plot_perf
        default_refresh_ms = max(
            MIN_REFRESH_INTERVAL_MS,
            int(self._plot_perf_config.signal_refresh_interval_ms()),
        )

        # Refresh configuration
        self.refresh_mode: str = DEFAULT_REFRESH_MODE
        self.refresh_interval_ms: int = default_refresh_ms
        self._sampling_rate_hz: Optional[float] = None
        self._last_follow_interval_ms: Optional[int] = None

        max_window_seconds = self._plot_perf_config.normalized_time_window_s()
        if plot_widget is None:
            plot_widget = SignalPlotWidgetPyQtGraph(max_seconds=max_window_seconds)
        if plot_widget.parent() is None:
            plot_widget.setParent(self)
        self._plot: SignalPlotWidgetBase = plot_widget
        self._plot.set_subplot_limits(
            max_subplots=self._plot_perf_config.normalized_max_subplots(),
            max_lines_per_subplot=self._plot_perf_config.normalized_max_lines(),
        )
        self._plot.set_max_points_per_trace(
            self._plot_perf_config.normalized_max_points()
        )
        self._fallback_data_buffer = StreamingDataBuffer(
            BufferConfig(
                max_seconds=self._plot.window_seconds,
                sample_rate_hz=self._sampling_rate_hz or 200.0,
            )
        )
        self.baseline_state = BaselineState()
        self._baseline_buffer: list[np.ndarray] = []
        self._baseline_timer = QTimer(self)
        self._baseline_timer.setSingleShot(True)
        self._baseline_timer.timeout.connect(self._finish_baseline)
        self._base_correction_enabled: bool = False
        self._data_buffer: StreamingDataBuffer | None = None
        self._buffer_cursors: Dict[int | str, float] = {}
        self._synthetic_active = False
        # Provide a sensible default before SettingsTab sends anything.
        self._current_sensor_selection = SensorSelectionConfig(
            active_sensors=[1, 2, 3],
            active_channels=["ax", "ay", "az", "gx", "gy", "gz"],
        )
        self._current_gui_acquisition_config: GuiAcquisitionConfig | None = None
        self._active_sensors: list[int] = list(
            self._current_sensor_selection.active_sensors or []
        )
        self._active_channels: list[str] = list(
            self._current_sensor_selection.active_channels or []
        )
        if recorder_tab is not None:
            try:
                self.set_data_buffer(recorder_tab.data_buffer())
            except AttributeError:
                self._data_buffer = None
        self._target_refresh_hz: Optional[float] = None
        self._last_refresh_timestamp: float | None = None
        self._smoothed_refresh_hz: float | None = None
        self._timer_tick_counter = 0
        self._timer_measure_window_start = time.perf_counter()
        self._timer_stats_hz = 0.0
        self._channel_checkboxes: Dict[str, QCheckBox] = {}
        self._recording_section: CollapsibleSection | None = None
        self._host_section: CollapsibleSection | None = None
        self._acquisition_section: CollapsibleSection | None = None
        self._acquisition_widget = AcquisitionSettingsWidget(
            self,
            show_device_rate=False,
            show_mode=False,
        )
        self._acquisition_widget.hide()
        record_only_checkbox = getattr(self._acquisition_widget, "record_only_checkbox", None)
        if record_only_checkbox is not None:
            record_only_checkbox.stateChanged.connect(self._on_record_only_toggled)
        self._acquisition_widget.samplingChanged.connect(
            self._on_acquisition_widget_changed
        )
        self._acquisition_widget.signalsRefreshChanged.connect(
            self._on_acquisition_refresh_changed
        )
        self._acquisition_widget.fftRefreshChanged.connect(
            self._emit_fft_refresh_interval_changed
        )
        self._acquisition_widget.signalsRefreshChanged.connect(
            self._on_acquisition_widget_changed
        )
        self._acquisition_widget.fftRefreshChanged.connect(
            self._on_acquisition_widget_changed
        )
        self._acquisition_widget.set_signals_refresh_interval(self.refresh_interval_ms)
        fft_spin_min = int(self._acquisition_widget.fft_refresh_spin.minimum())
        default_fft_interval = max(
            fft_spin_min,
            int(self._plot_perf_config.fft_refresh_interval_ms()),
        )
        self._acquisition_widget.set_fft_refresh_interval(default_fft_interval)

        # Initialize the acquisition widget's sampling config from AppConfig so
        # the displayed rates reflect the canonical configuration loaded at
        # startup.
        try:
            sampling_cfg = getattr(self._app_config, "sampling_config", None)
        except Exception:
            sampling_cfg = None

        if sampling_cfg is not None:
            try:
                self._acquisition_widget.set_sampling_config(sampling_cfg)
            except Exception:
                logger.exception(
                    "SignalsTab: failed to apply sampling_config from AppConfig to acquisition widget"
                )

        self._rebuild_gui_acquisition_config()
        self._plot.set_display_slack_ns(DEFAULT_DISPLAY_SLACK_NS)

        self._stream_active = False
        self._awaiting_first_sample = False
        self._stream_stalled = False
        self._stall_threshold_s = STREAM_STALL_THRESHOLD_S
        self._last_data_monotonic: float = 0.0
        self._manual_status_hold_s = MANUAL_STATUS_HOLD_S
        self._status_source: str = "stream"
        self._last_status_change_monotonic: float = 0.0
        self._synthetic_timer: QTimer | None = None
        self._synthetic_rate_hz: float = 0.0
        self._synthetic_phase: float = 0.0
        self._synthetic_phase_step: float = 0.0
        self._synthetic_sensor_ids: List[int] = [1]
        self._synthetic_interval_ns: int = int(self.refresh_interval_ms * 1_000_000)
        self._synthetic_next_timestamp_ns: int = time.monotonic_ns()
        self._synthetic_start_timestamp_ns: int = self._synthetic_next_timestamp_ns
        self._last_redraw_ms: float = 0.0
        self._redraw_ema_ms: float = 0.0
        self._adaptive_slow_cycles: int = 0
        self._adaptive_fast_cycles: int = 0
        self._queue_ingest_time_acc_ms: float = 0.0
        self._queue_ingest_batches: int = 0
        self._queue_ingest_samples: int = 0
        now = time.perf_counter()
        self._queue_ingest_last_log: float = now
        self._redraw_debug_ema_ms: float = 0.0
        self._redraw_debug_last_log: float = now

        layout = QVBoxLayout(self)

        # Top controls ---------------------------------------------------------
        top_row_group = QGroupBox(self)
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        sensor_layout = QHBoxLayout()
        sensor_layout.setSpacing(8)
        self.sensor_label = QLabel("Sensor: MPU6050", top_row_group)
        sensor_layout.addWidget(self.sensor_label)
        sensor_layout.addStretch()

        self._refresh_profile_custom_index: int | None = None
        self._active_refresh_profile_label: str | None = None
        self._refresh_profile_label = QLabel("Refresh:", top_row_group)
        top_row.addWidget(self._refresh_profile_label)
        self.refresh_profile_combo = QComboBox(top_row_group)
        self.refresh_profile_combo.setToolTip(
            "Preset GUI refresh intervals for the time-domain plots."
        )
        for label, interval in REFRESH_PRESETS:
            self.refresh_profile_combo.addItem(label, int(interval))
        self.refresh_profile_combo.currentIndexChanged.connect(
            self._on_refresh_profile_changed
        )
        top_row.addWidget(self.refresh_profile_combo)

        self.record_only_check = QCheckBox(
            "Record only (no live streaming)", top_row_group
        )
        self.record_only_check.setToolTip(
            "When enabled, data is recorded on the Pi but not streamed live to this GUI."
        )
        # "Stop after" + its own duration box were removed: the recording length is
        # now driven by the single "Rec length" field for BOTH methods (see
        # duration_limit_enabled/seconds, which derive from record-only + Rec length).
        self._session_name_edit = QLineEdit(top_row_group)
        self._session_name_edit.setPlaceholderText("Session name (optional)")
        self._session_name_edit.setClearButtonEnabled(True)
        self._session_name_edit.setMaximumWidth(200)
        # Calibration UI has been removed from the Signals tab.

        # Start/stop. Start = live feed; if "Record only" is checked it instead runs
        # the ORIGINAL Pi-side recording for the Rec-length duration (auto-stops).
        self.start_button = QPushButton("Start", top_row_group)
        self.start_button.setToolTip(
            "Live feed. If 'Record only' is checked, records via the original Pi-side "
            "method for the 'Rec length' duration, then auto-stops.")
        self.stop_button = QPushButton("Stop", top_row_group)
        self.stop_button.setEnabled(False)

        # Rec length drives BOTH methods: the original record-only (Start) and Smart
        # Record. Smart Record probes the real rate for ~5 s then writes a PC-clock,
        # fixed-duration, byte-compatible recording on the PC.
        self.record_length_spin = QSpinBox(top_row_group)
        self.record_length_spin.setRange(1, 600)
        self.record_length_spin.setValue(60)
        self.record_length_spin.setSuffix(" s")
        self.record_length_spin.setToolTip(
            "Recording length — used by BOTH Smart Record and 'Record only' (Start).")
        # Requested sampling rate (Hz). This is THE single device rate (G3),
        # surfaced here for convenience and kept in sync with the Acquisition
        # section's "Sampling rate [Hz]". Smart Record asks the Pi for this rate and
        # then probes the actual delivered rate.
        self.record_rate_spin = QSpinBox(top_row_group)
        self.record_rate_spin.setRange(1, 1000)
        self.record_rate_spin.setSingleStep(10)
        try:
            _init_rate = int(round(
                self._acquisition_widget.current_sampling_config().device_rate_hz))
        except Exception:
            _init_rate = 100
        self.record_rate_spin.setValue(max(1, _init_rate))
        self.record_rate_spin.setSuffix(" Hz")
        self.record_rate_spin.setToolTip(
            "Requested sampling rate sent to the Pi for recording (the single device "
            "rate). Smart Record probes the actual delivered rate.")
        self.record_rate_spin.valueChanged.connect(self._on_record_rate_changed)
        self.record_button = QPushButton("Smart Record", top_row_group)
        self.record_button.setToolTip(
            "Smart Record: probe the real rate (~5 s) -> record for the Rec length "
            "with a PC clock -> save (cap/decimate to the true rate).")
        self.record_button.clicked.connect(self._on_record_clicked)

        # Small info labels: stream rate + plot refresh
        self._stream_rate_label = QLabel("Stream rate: -- Hz", top_row_group)
        self._stream_rate_label.setToolTip(
            "Estimated rate at which samples arrive in this GUI tab after any "
            "Pi-side stream decimation (mpu6050 --stream-every)."
        )
        self._plot_refresh_label = QLabel("Plot refresh: -- Hz", top_row_group)
        self._plot_refresh_label.setToolTip(
            "Approximate refresh/FPS rate achieved by the GUI plot."
        )
        self._stream_rate_label.setVisible(False)
        self._plot_refresh_label.setVisible(False)

        self.sync_logs_button = QPushButton("Sync logs from Pi", top_row_group)
        self.sync_logs_button.setToolTip(
            "Download recorded log files from the selected Raspberry Pi into the local raw data folder."
        )

        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.record_only_check.stateChanged.connect(self._on_record_only_toggled)
        self.sync_logs_button.clicked.connect(self._on_sync_logs_clicked)

        top_row.addWidget(self.record_only_check)
        top_row.addWidget(QLabel("Session:", top_row_group))
        top_row.addWidget(self._session_name_edit)
        top_row.addWidget(self.start_button)
        top_row.addWidget(self.stop_button)
        top_row.addWidget(QLabel("Rec length:", top_row_group))
        top_row.addWidget(self.record_length_spin)
        top_row.addWidget(QLabel("Rate:", top_row_group))
        top_row.addWidget(self.record_rate_spin)
        top_row.addWidget(self.record_button)
        top_row.addWidget(self.sync_logs_button)
        top_row.addStretch()

        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        group_layout.addLayout(sensor_layout)
        group_layout.addLayout(top_row)

        top_row_group.setLayout(group_layout)

        # No calibration controls in this tab anymore.

        recording_section = CollapsibleSection("Recording / stream controls", self)
        recording_layout = QVBoxLayout()
        recording_layout.setContentsMargins(0, 0, 0, 0)
        recording_layout.addWidget(top_row_group)
        recording_section.setContentLayout(recording_layout)
        layout.addWidget(recording_section)
        self._recording_section = recording_section

        # Plot widget -----------------------------------------------------------
        layout.addWidget(self._plot, stretch=1)

        # Status label ----------------------------------------------------------
        self._status_label = QLabel("", self)
        layout.addWidget(self._status_label)
        self._set_stream_status("Waiting for stream...", force=True)

        self._perf_summary_label = QLabel("", self)
        layout.addWidget(self._perf_summary_label)
        self._update_perf_summary()
        self.update_sensor_selection(self._current_sensor_selection)

        # NOTE: Legacy sample ingestion and redraw timers are decoupled to
        # respect the current buffer management behavior. Revisit during the
        # new GUI refactor.
        # periodic sample ingestion (decoupled from redraw refresh)
        self._ingest_timer = QTimer(self)
        self._ingest_timer.setInterval(20)
        self._ingest_timer.timeout.connect(self._drain_samples)
        self._refresh_ingest_timer()

        # periodic redraw of the plot
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_redraw_timer)
        self._apply_refresh_settings()
        self._update_refresh_profile_enabled()

    # --------------------------------------------------------------- slots
    @Slot()
    def _on_start_clicked(self) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_stream_status("Streaming...", force=True)
        session = self.session_name()
        self.start_stream_requested.emit(session)
        if self._recording_section is not None:
            self._recording_section.setCollapsed(True)
        if self._acquisition_section is not None:
            self._acquisition_section.setCollapsed(True)

    @Slot()
    def _on_record_clicked(self) -> None:
        """Smart Recording: probe -> PC-clock fixed-duration recording (Front C)."""
        self.start_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_stream_status("Recording (probing rate)...", force=True)
        session = self.session_name()
        self.record_requested.emit(session, int(self.record_length_spin.value()))
        if self._recording_section is not None:
            self._recording_section.setCollapsed(True)
        if self._acquisition_section is not None:
            self._acquisition_section.setCollapsed(True)

    @Slot()
    def _on_stop_clicked(self) -> None:
        self.start_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_manual_status("Stopping...")
        self.stop_stream_requested.emit()
        if self._recording_section is not None:
            self._recording_section.setCollapsed(False)
        if self._acquisition_section is not None:
            self._acquisition_section.setCollapsed(False)
        self._refresh_mode_hint()

    @Slot()
    def _on_sync_logs_clicked(self) -> None:
        """
        Emit a request to sync logs from the currently selected Pi.

        The actual sync work is delegated to whichever slot is connected in
        the main window (e.g. OfflineTab.sync_logs_from_pi).
        """

        self.sync_logs_requested.emit()

    @Slot(int)
    def _get_record_only_checked(self) -> bool:
        """Return the record-only checkbox state, preferring the acquisition widget."""

        widget = getattr(self, "_acquisition_widget", None)
        cb = getattr(widget, "record_only_checkbox", None) if widget is not None else None
        if cb is None:
            cb = getattr(self, "record_only_check", None)
        try:
            return bool(cb.isChecked()) if cb is not None else False
        except Exception:
            return False

    def _sync_record_only_checkboxes(self, checked: bool) -> None:
        """Keep any duplicate record-only checkboxes in sync without emitting signals."""

        checked = bool(checked)
        widgets: list[QCheckBox] = []
        widget = getattr(self, "_acquisition_widget", None)
        cb = getattr(widget, "record_only_checkbox", None) if widget is not None else None
        if isinstance(cb, QCheckBox):
            widgets.append(cb)
        tab_cb = getattr(self, "record_only_check", None)
        if isinstance(tab_cb, QCheckBox) and tab_cb not in widgets:
            widgets.append(tab_cb)

        for checkbox in widgets:
            with QSignalBlocker(checkbox):
                checkbox.setChecked(checked)

    def _on_record_only_toggled(self, _state: int) -> None:
        """Keep acquisition settings in sync with record-only controls."""

        enabled = self._get_record_only_checked()
        self._sync_record_only_checkboxes(enabled)
        self.set_record_only_mode(enabled)
        self._rebuild_gui_acquisition_config()
        self._refresh_mode_hint()

    def current_acquisition_settings(self) -> AcquisitionSettings:
        """Return the current sampling / refresh settings.

        The SamplingConfig inside AcquisitionSettings is now the single
        source of truth for device rate and stream decimation. The
        Recorder tab will derive --stream-every from it, so we don't
        need to tweak anything here.
        """
        return self._acquisition_widget.settings()

    def session_name(self) -> str:
        """Return the normalized session label provided by the user."""
        if not hasattr(self, "_session_name_edit"):
            return ""
        return self._session_name_edit.text().strip()

    def current_session_name(self) -> str:
        """Alias for :meth:`session_name` to mirror other tab APIs."""

        return self.session_name()

    def duration_limit_enabled(self) -> bool:
        """Bridge for the original recording path: 'Record only' acts as the old
        'Stop after' — when it's checked, the recording auto-stops after the
        Rec-length duration (main_window arms the PC timer + passes the Pi duration).
        A plain live view (record-only unchecked) runs until Stop.
        """
        cb = getattr(self, "record_only_check", None)
        return bool(cb is not None and cb.isChecked())

    def duration_limit_seconds(self) -> int:
        """Single source of recording duration for BOTH methods = the Rec-length
        field (replaces the removed 'Stop after' box)."""
        if not hasattr(self, "record_length_spin"):
            return 60
        return int(self.record_length_spin.value())

    def calibrate_from_buffer(self, window_s: float | None = None) -> None:
        self._plot.calibrate_from_buffer(window_s=window_s)

    def reset_calibration(self) -> None:
        self._plot.reset_calibration()

    def apply_calibration_to_recording(self) -> bool:
        """
        Calibration controls have been removed from the Signals tab,
        so calibration is never applied to recorded data.
        """

        return False

    def set_sampling_config(self, sampling_config: SamplingConfig) -> None:
        """Update the acquisition widget with a new sampling configuration."""

        self._acquisition_widget.set_sampling_config(sampling_config)

    def set_sensor_selection(self, selection: SensorSelectionConfig) -> None:
        """
        Called by MainWindow when the Device/Sensors tab changes selection.

        Cache the selection, rebuild the GuiAcquisitionConfig and
        auto-adjust the view preset so all selected channels are visible.
        """

        self.update_sensor_selection(selection)

    def update_sensor_selection(self, selection: SensorSelectionConfig) -> None:
        """
        Receive the latest sensor/channel selection so Live Signals
        can configure its layout and plots.
        """
        self._current_sensor_selection = selection

        active_sensors = selection.active_sensors or []
        active_channels = selection.active_channels or []

        self._active_sensors = list(active_sensors)
        self._active_channels = list(active_channels)

        if active_sensors and active_channels:
            # Also propagate the channel layout directly to the plot widget.
            self._plot.set_channel_layout(active_channels)
            # Show all configured channels by default.
            self._plot.set_visible_channels(active_channels)

        self._rebuild_gui_acquisition_config()

    def apply_gui_acquisition_config(self, cfg: GuiAcquisitionConfig) -> None:
        """
        Called from MainWindow when a new config is about to be used
        for streaming/recording.

        The device sampling rate (cfg.sampling.device_rate_hz) is the single
        source of truth for the SignalsTab x-axis. Stream rate measurements
        are only used for display and refresh timing, not for the x-axis.
        """

        self._current_gui_acquisition_config = cfg
        widget = getattr(self, "_acquisition_widget", None)
        if widget is not None:
            try:
                widget.set_sampling_config(cfg.sampling)
            except Exception:
                logger.exception(
                    "SignalsTab: failed to apply sampling from GuiAcquisitionConfig to acquisition widget"
                )
        self._active_channels = list(cfg.sensor_selection.active_channels or [])
        try:
            self._sampling_rate_hz = float(cfg.sampling.device_rate_hz)
        except (TypeError, ValueError):
            self._sampling_rate_hz = None
        self._sync_record_only_checkboxes(bool(cfg.record_only))
        if self._sampling_rate_hz:
            self._plot.set_nominal_sample_rate(self._sampling_rate_hz)
        self._apply_refresh_settings()
        try:
            self.update_stream_rate("mpu6050", float(cfg.stream_rate_hz))
        except Exception:
            pass
        self._refresh_mode_hint()

    def current_acquisition_config(self) -> GuiAcquisitionConfig:
        """
        High-level GUI acquisition config. This is what backend wiring will use in later phases.
        """

        if self._current_gui_acquisition_config is None:
            self._rebuild_gui_acquisition_config()
        assert self._current_gui_acquisition_config is not None
        return self._current_gui_acquisition_config

    def _on_acquisition_widget_changed(self, *_args) -> None:
        self._sync_record_rate_spin()
        self._rebuild_gui_acquisition_config()

    @Slot(int)
    def _on_record_rate_changed(self, hz: int) -> None:
        """Push the recording-row rate into THE single device rate (G3): update the
        acquisition widget's SamplingConfig so both stream and recording use it."""
        aw = getattr(self, "_acquisition_widget", None)
        if aw is None:
            return
        cur = aw.current_sampling_config()
        if abs(float(cur.device_rate_hz) - float(hz)) < 1e-6:
            return
        aw.set_sampling_config(
            SamplingConfig(device_rate_hz=float(hz), mode_key=cur.mode_key))
        self._rebuild_gui_acquisition_config()

    def _sync_record_rate_spin(self) -> None:
        """Keep the recording-row Rate box in step with the Acquisition section's
        device rate (one rate, surfaced in two places — G3)."""
        spin = getattr(self, "record_rate_spin", None)
        aw = getattr(self, "_acquisition_widget", None)
        if spin is None or aw is None:
            return
        hz = float(aw.current_sampling_config().device_rate_hz)
        if abs(spin.value() - hz) > 1e-6:
            with QSignalBlocker(spin):
                spin.setValue(int(round(hz)))

    def set_record_only_mode(self, enabled: bool) -> None:
        """
        Enable/disable 'record only (no streaming)' behavior for the live plot.

        When enabled:
        - Stop the GUI refresh timer.
        - Show a status message indicating live streaming is disabled.
        """

        enabled = bool(enabled)
        if enabled:
            self._stream_active = False
            if hasattr(self, "_timer"):
                self._timer.stop()
            self._refresh_ingest_timer()
            self._set_status_text(
                "Record-only mode: live streaming disabled.", source="manual"
            )
        else:
            self._refresh_ingest_timer()
            self._refresh_timer_state()
            self._refresh_mode_hint()

    def _rebuild_gui_acquisition_config(self) -> None:
        widget = self._acquisition_widget
        if widget is None:
            return

        sel = self._current_sensor_selection

        # If no sensors are selected, don't emit a bogus "empty" config.
        if not sel.active_sensors:
            return

        acq = widget.current_settings()
        cfg = GuiAcquisitionConfig(
            sampling=acq.sampling,
            stream_rate_hz=acq.stream_rate_hz,
            record_only=self._get_record_only_checked(),
            limit_duration=self.duration_limit_enabled(),
            duration_s=float(self.duration_limit_seconds()),
            sensor_selection=sel,
        )
        self._current_gui_acquisition_config = cfg
        self._active_channels = list(cfg.sensor_selection.active_channels or [])
        logger.debug("SignalsTab: rebuilt GuiAcquisitionConfig: %s", asdict(cfg))
        self.acquisitionConfigChanged.emit(cfg)

    def set_data_buffer(
        self, data_buffer: Optional[StreamingDataBuffer]
    ) -> None:
        """Set the StreamingDataBuffer used for timer-driven updates."""
        self._data_buffer = data_buffer
        self._buffer_cursors.clear()

    def live_sensor_ids(self) -> list[int]:
        """
        Expose the sensor IDs with buffered data (shared with FFT tab).
        """
        return self._plot.live_sensor_ids()

    def get_time_series_window(
        self,
        sensor_id: int,
        channel: str,
        window_seconds: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(times_s, values)`` arrays for ``sensor_id``/``channel``.

        Delegates to the underlying SignalPlotWidget so other tabs can reuse
        the exact same ring buffers used for the PyQtGraph plots.
        """
        return self._plot.get_time_series_window(sensor_id, channel, window_seconds)

    def _active_data_buffer(self) -> Optional[StreamingDataBuffer]:
        """
        Return the current data buffer (external or fallback).

        Prefer the RecorderTab's shared StreamingDataBuffer, so that when
        RecorderTab recreates the buffer for a new stream, the Live Signals
        tab automatically sees the new data.
        """
        # If we have a RecorderTab, always ask it for the latest buffer.
        recorder = getattr(self, "_recorder_tab", None)
        if recorder is not None:
            try:
                buf = recorder.data_buffer()
            except Exception:
                buf = None
            if buf is not None:
                return buf

        # Fallback: use any explicitly injected buffer, or the local fallback.
        return self._data_buffer or self._fallback_data_buffer

    def get_perf_snapshot(self) -> dict[str, float]:
        """
        Combine plot metrics and timer statistics for overlays or logging.
        """
        snap = self._plot.get_perf_snapshot()
        timer_hz = self._timer_stats_hz if ENABLE_PLOT_PERF_METRICS else 0.0
        target = self._target_refresh_hz
        if target is None:
            target = snap.get("target_fps", 0.0) or 0.0
        achieved = snap.get("achieved_fps")
        if achieved is None:
            achieved = snap.get("fps", 0.0)
        if achieved is None:
            achieved = 0.0
        achieved = float(achieved)
        approx_drop = snap.get("approx_dropped_frames_per_sec", 0.0) or 0.0
        if ENABLE_PLOT_PERF_METRICS:
            effective_timer = timer_hz if timer_hz > 0.0 else target
            if effective_timer > 0.0:
                approx_drop = max(0.0, effective_timer - achieved)
        snap.update(
            {
                "target_fps": target,
                "achieved_fps": achieved,
                "timer_hz": timer_hz,
                "approx_dropped_frames_per_sec": approx_drop,
            }
        )
        return snap

    def set_perf_hud_visible(self, visible: bool) -> None:
        """Legacy no-op (perf HUD overlay removed)."""
        _ = visible

    # --------------------------------------------------------------- helpers
    def _set_status_text(self, text: str, *, source: str) -> None:
        self._status_label.setText(text)
        self._status_source = source
        self._last_status_change_monotonic = time.monotonic()

    def _set_stream_status(self, text: str, *, force: bool = False) -> None:
        if (
            not force
            and self._status_source == "manual"
            and (time.monotonic() - self._last_status_change_monotonic)
            < self._manual_status_hold_s
        ):
            return
        self._set_status_text(text, source="stream")

    def _set_manual_status(self, text: str) -> None:
        self._set_status_text(text, source="manual")

    def _remote_destination_text(self) -> str:
        """Return a user-facing description of the remote recording folder."""
        recorder = getattr(self, "_recorder_tab", None)
        if recorder is None:
            return ""
        try:
            remote_dir = recorder.current_remote_data_dir()
        except AttributeError:
            return ""
        if remote_dir is None:
            return ""
        return remote_dir.as_posix()

    def _recording_requested(self) -> bool:
        recorder = getattr(self, "_recorder_tab", None)
        if recorder is None:
            return False
        getter = getattr(recorder, "recording_requested", None)
        if callable(getter):
            try:
                return bool(getter())
            except Exception:
                logger.exception("RecorderTab.recording_requested() raised.")
        return False

    def _refresh_mode_hint(self) -> None:
        label = getattr(self, "_mode_hint_label", None)
        if label is None:
            return
        dest = self._remote_destination_text()
        session = " ".join(self.session_name().split())
        record_only = self._get_record_only_checked()
        recording_requested = self._recording_requested()
        if record_only or recording_requested:
            hint = "Recording enabled. "
            if dest:
                hint += f"Data is written to {dest} on the Pi. "
            elif recording_requested:
                hint += "Data is written to the configured Pi logs directory. "
        else:
            hint = "Streaming only (samples are not stored on the Pi). "
            if dest:
                hint += f"Enable recording to write into {dest}. "
        hint += "Recording is controlled from the Recording / stream controls tab."
        if session:
            hint += f" Session name: {session}."
        label.setText(hint.strip())

    def _extract_sample_timestamp_ns(self, sample: MpuSample) -> Optional[int]:
        if sample.t_s is not None:
            try:
                return int(round(float(sample.t_s) * NS_PER_SECOND))
            except (TypeError, ValueError):
                return None
        try:
            return int(sample.timestamp_ns)
        except (TypeError, ValueError):
            return None

    def _sample_time_seconds(self, sample: MpuSample) -> Optional[float]:
        ts_ns = self._extract_sample_timestamp_ns(sample)
        if ts_ns is None:
            return None
        return ts_ns / float(NS_PER_SECOND)

    @Slot(str, float)
    def update_stream_rate(self, sensor_type: str, hz: float) -> None:
        """
        Update the GUI-side *stream* rate shown in the Signals tab.

        NOTE:
            The live plot x-axis should be driven only by the device sampling
            rate configured in Settings (SamplingConfig.device_rate_hz).
            Therefore, this method must NOT modify the SignalPlotWidget's
            nominal sample rate. It only updates display and (optionally)
            the refresh behavior when in follow_sampling_rate mode.
        """
        if sensor_type != "mpu6050":
            return

        # Display the effective GUI stream rate after decimation.
        #self._stream_rate_label.setText(f"Stream ≈ {hz:5.1f} Hz")

        # Keep this for perf/timer purposes only.
        try:
            self._sampling_rate_hz = float(hz)
        except (TypeError, ValueError):
            self._sampling_rate_hz = None

        # In follow_sampling_rate mode, we still adjust the GUI timer interval
        # based on the measured stream rate, but we do NOT touch the x-axis rate.
        if self.refresh_mode == "follow_sampling_rate":
            self._apply_refresh_settings()

        self._update_perf_summary()

    def set_sampling_rate_hz(self, hz: float) -> None:
        """
        Manually set the nominal sampling/stream rate used by the
        live plots and timers.

        This is used when we already know the target stream/plot rate
        from the GUI (GuiAcquisitionConfig) before RecorderTab has
        measured and reported a rate.

        Internally it just routes through update_stream_rate().
        """
        try:
            value = float(hz)
        except (TypeError, ValueError):
            # Ignore invalid values; keep existing rate
            return

        # Reuse the existing logic that already updates labels, timers, etc.
        self.update_stream_rate("mpu6050", value)

    @property
    def fixed_interval_ms(self) -> int:
        return self.refresh_interval_ms

    @fixed_interval_ms.setter
    def fixed_interval_ms(self, value: int) -> None:
        interval = max(MIN_REFRESH_INTERVAL_MS, int(value))
        self.refresh_interval_ms = interval
        if self.refresh_mode == "fixed":
            self._apply_refresh_settings()

    @Slot(int)
    def _on_acquisition_refresh_changed(self, interval_ms: int) -> None:
        """Update the timer interval when the acquisition widget spin changes."""
        self.fixed_interval_ms = max(MIN_REFRESH_INTERVAL_MS, int(interval_ms))

    @Slot(str)
    def _on_acquisition_mode_changed(self, mode: str) -> None:
        """Switch between fixed and adaptive refresh modes from the UI."""
        stream_rate = self._get_sampling_rate_hz()
        self.set_refresh_mode(mode, stream_rate)

    @Slot(int)
    def _emit_fft_refresh_interval_changed(self, interval_ms: int) -> None:
        """Relay FFT refresh adjustments so MainWindow can reconfigure FftTab."""
        self.fft_refresh_interval_changed.emit(int(interval_ms))

    @Slot(int)
    def _on_refresh_profile_changed(self, index: int) -> None:
        combo = getattr(self, "refresh_profile_combo", None)
        if combo is None or index < 0:
            return
        interval_data = combo.itemData(index)
        if interval_data == REFRESH_FOLLOW_DEVICE:
            self.set_refresh_mode("follow_sampling_rate")
            return
        if interval_data is None:
            # Custom entry - editing happens via the acquisition widget spin box.
            return
        interval_ms = max(MIN_REFRESH_INTERVAL_MS, int(interval_data))
        self._acquisition_widget.set_signals_refresh_interval(interval_ms)
        was_fixed = self.refresh_mode == "fixed"
        self.fixed_interval_ms = interval_ms
        if not was_fixed:
            self.set_refresh_mode("fixed")

    def _rebuild_channel_checkboxes(self) -> None:
        # Channel toggles are driven by SettingsTab; this is now a no-op.
        self._channel_checkboxes.clear()

    @Slot()
    def _on_view_mode_changed(self) -> None:
        return

    @Slot()
    def _on_channel_toggles_changed(self) -> None:
        # Plot visibility follows SettingsTab channel selection.
        self._plot.set_visible_channels(None)

    # ------------------------------------------------------------------ baseline
    def start_baseline_correction(self) -> None:
        """
        Collect raw samples for a short window before computing the baseline.
        """

        self.baseline_state.active = False
        self.baseline_state.offset = None
        self._baseline_buffer.clear()
        self._baseline_timer.start(int(self.BASELINE_DURATION_SEC * 1000))

    def _finish_baseline(self) -> None:
        """Compute and store baseline offset after capture window."""

        if not self._baseline_buffer:
            return

        offset = collect_baseline_samples(self._baseline_buffer)
        self.baseline_state.offset = offset
        self.baseline_state.active = True
        self._baseline_buffer.clear()

    def _sample_to_array(self, sample: MpuSample) -> np.ndarray:
        values: list[float] = []
        for axis in ("ax", "ay", "az", "gx", "gy", "gz"):
            val = getattr(sample, axis, None)
            if val is None:
                values.append(float("nan"))
            else:
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    values.append(float("nan"))
        return np.asarray(values, dtype=float)

    def _apply_baseline_to_sample(self, sample: MpuSample) -> MpuSample:
        raw_values = self._sample_to_array(sample)

        if self._baseline_timer.isActive():
            self._baseline_buffer.append(raw_values)
            corrected_values = raw_values
        else:
            # Base correction UI is disabled; pass through raw values.
            corrected_values = raw_values

        ax, ay, az, gx, gy, gz = corrected_values.tolist()
        return MpuSample(
            timestamp_ns=sample.timestamp_ns,
            ax=float(ax),
            ay=float(ay),
            az=float(az),
            gx=float(gx),
            gy=float(gy),
            gz=float(gz),
            sensor_id=sample.sensor_id,
            t_s=sample.t_s,
        )

    def set_refresh_mode(
        self, mode: str, stream_rate_hz: Optional[float] = None
    ) -> None:
        """
        Configure how often the plot is refreshed.

        ``mode`` accepts ``"fixed"`` (use ``self.refresh_interval_ms``) or
        ``"adaptive"``/``"follow_sampling_rate"`` which uses ``stream_rate_hz``.
        """
        normalized = self._normalize_refresh_mode(mode)
        self.refresh_mode = normalized
        if stream_rate_hz is not None:
            self._sampling_rate_hz = stream_rate_hz
            self._plot.set_nominal_sample_rate(stream_rate_hz)
        self._apply_refresh_settings()
        self._update_refresh_profile_enabled()

    def _ingest_buffer_data(self) -> None:
        """Append new samples from the active StreamingDataBuffer to the plot buffers."""
        data_buffer = self._active_data_buffer()
        if data_buffer is None:
            return

        latest_ts = data_buffer.latest_timestamp()
        if latest_ts is None:
            logger.debug("SignalsTab: _ingest_buffer_data buffer has no data yet")
            return

        sensor_ids = data_buffer.get_sensor_ids()
        if not sensor_ids:
            logger.warning("SignalsTab: _ingest_buffer_data no sensor IDs available")
            return

        channels = self._active_channels or [
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        ]

        logger.debug(
            "SignalsTab: _ingest_buffer_data sensor_ids=%s channels=%s latest_ts=%.6f",
            sensor_ids,
            channels,
            latest_ts,
        )

        window_s = self._plot.window_seconds
        for sensor_id in sensor_ids:
            samples = data_buffer.get_recent_samples(sensor_id, seconds=window_s)
            logger.debug(
                "SignalsTab: sensor=%s recent_samples=%d window_s=%.3f",
                sensor_id,
                len(samples),
                window_s,
            )
            if not samples:
                continue
            last_seen = self._buffer_cursors.get(sensor_id)
            updated_last = last_seen
            for sample in samples:
                ts_s = self._sample_time_seconds(sample)
                if ts_s is None:
                    continue
                if updated_last is not None and ts_s <= updated_last:
                    continue
                if ENABLE_PLOT_PERF_METRICS:
                    try:
                        sample.gui_receive_ts = time.perf_counter()
                    except Exception:
                        pass
                corrected_sample = self._apply_baseline_to_sample(sample)
                self._plot.add_sample(corrected_sample)
                updated_last = ts_s
                self._handle_ingested_sample(corrected_sample)
            if updated_last is not None:
                self._buffer_cursors[sensor_id] = updated_last

    def _handle_ingested_sample(self, sample: MpuSample) -> None:
        """Update stream state tracking when new data arrives."""
        if not self._stream_active:
            return

        timestamp_ns = self._extract_sample_timestamp_ns(sample)
        if timestamp_ns is None:
            return

        self._last_data_monotonic = time.monotonic()
        if self._awaiting_first_sample:
            self._awaiting_first_sample = False
            self._set_stream_status("Streaming...")
        if self._stream_stalled:
            self._stream_stalled = False
            self._set_stream_status("Streaming...")

    @Slot()
    def _on_redraw_timer(self) -> None:
        start = time.perf_counter()
        self.update_plot()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._last_redraw_ms = elapsed_ms
        alpha = 0.2
        if self._redraw_ema_ms <= 0.0:
            self._redraw_ema_ms = elapsed_ms
        else:
            self._redraw_ema_ms = alpha * elapsed_ms + (1.0 - alpha) * self._redraw_ema_ms

    @Slot()
    def update_plot(self) -> None:
        """Timer slot that refreshes the SignalPlotWidget."""
        # First update the stream status (waiting / stalled / streaming),
        # then ask the plotting backend to redraw the latest buffered data.
        debug_on = debug_enabled()
        if ENABLE_PLOT_PERF_METRICS:
            self._record_timer_tick()
        if self._stream_active:
            if self._awaiting_first_sample:
                self._set_stream_status("Waiting for data...")
            else:
                stalled = False
                if self._last_data_monotonic > 0.0:
                    stalled = (time.monotonic() - self._last_data_monotonic) >= self._stall_threshold_s

                if stalled:
                    if not self._stream_stalled:
                        self._stream_stalled = True
                        self._set_stream_status("No recent data (stream paused?)", force=True)
                elif self._stream_stalled:
                    self._stream_stalled = False
                    self._set_stream_status("Streaming...")
        redraw_start = time.perf_counter() if debug_on else 0.0
        self._plot.redraw()
        if debug_on:
            redraw_ms = (time.perf_counter() - redraw_start) * 1000.0
            alpha = 0.2
            if self._redraw_debug_ema_ms <= 0.0:
                self._redraw_debug_ema_ms = redraw_ms
            else:
                self._redraw_debug_ema_ms = (
                    alpha * redraw_ms + (1.0 - alpha) * self._redraw_debug_ema_ms
                )
            now_perf = time.perf_counter()
            if now_perf - self._redraw_debug_last_log >= 5.0:
                interval_ms = self._timer.interval() if hasattr(self, "_timer") else 0
                logger.debug(
                    "SignalsTab: redraw interval=%d ms ema≈%.2f ms",
                    interval_ms,
                    self._redraw_debug_ema_ms,
                )
                self._redraw_debug_last_log = now_perf
        self._record_refresh_tick()

    def _adaptive_step(self) -> None:
        """Periodic background check that nudges refresh/stream fidelity.

        If the adaptive checkbox is absent (Signals tab UI), adaptive mode is
        effectively disabled.
        """
        self._update_perf_summary()
        checkbox = getattr(self, "adaptive_mode_check", None)
        if checkbox is None or not checkbox.isChecked():
            self._adaptive_slow_cycles = 0
            self._adaptive_fast_cycles = 0
            return
        if not (self._stream_active or self._synthetic_active):
            return
        redraw_ms = self._redraw_ema_ms
        if redraw_ms <= 0.0:
            return
        target_interval_ms = self._compute_refresh_interval()
        if target_interval_ms <= 0:
            return
        stream_rate = float(self._sampling_rate_hz or 0.0)
        target_stream = float(self._target_stream_rate_from_recorder() or 0.0)
        max_frame_ms = target_interval_ms * 0.9
        redraw_slow = redraw_ms > max_frame_ms
        overloaded_stream = False
        if target_stream > 0.0 and stream_rate > 0.0:
            overloaded_stream = stream_rate > target_stream * 1.15
        comfortable = redraw_ms < target_interval_ms * 0.6
        more_stream_allowed = (
            target_stream > 0.0
            and stream_rate > 0.0
            and stream_rate < target_stream * 0.8
        )

        if redraw_slow or overloaded_stream:
            self._adaptive_slow_cycles += 1
            self._adaptive_fast_cycles = 0
            if self._adaptive_slow_cycles >= 2:
                if self._lower_fidelity():
                    self._adaptive_slow_cycles = 0
            return

        self._adaptive_slow_cycles = 0
        if comfortable:
            self._adaptive_fast_cycles += 1
        else:
            self._adaptive_fast_cycles = 0

        if comfortable and self._adaptive_fast_cycles >= 3:
            if self._maybe_increase_fidelity(allow_stream_tuning=more_stream_allowed):
                self._adaptive_fast_cycles = 0

    def _lower_fidelity(self) -> bool:
        """Try to reduce GUI/stream demand when frames are too slow."""
        adjusted = False
        widget = getattr(self, "_acquisition_widget", None)
        if self.refresh_mode == "fixed" and widget is not None:
            current = int(self.refresh_interval_ms)
            new_interval = min(500, int(max(MIN_REFRESH_INTERVAL_MS, math.ceil(current * 1.5))))
            if new_interval > current:
                widget.set_signals_refresh_interval(int(new_interval))
                self.fixed_interval_ms = int(new_interval)
                adjusted = True
        if adjusted:
            self._update_perf_summary()
            return True
        recorder = getattr(self, "_recorder_tab", None)
        if recorder is not None and hasattr(recorder, "request_coarser_streaming"):
            try:
                recorder.request_coarser_streaming()
                return True
            except Exception:
                logger.exception("SignalsTab: failed to request coarser streaming")
        return False

    def _maybe_increase_fidelity(self, *, allow_stream_tuning: bool) -> bool:
        """If redraws are comfortably fast, cautiously improve fidelity."""
        adjusted = False
        widget = getattr(self, "_acquisition_widget", None)
        if self.refresh_mode == "fixed" and widget is not None:
            current = int(self.refresh_interval_ms)
            if current > MIN_REFRESH_INTERVAL_MS:
                reduced = max(MIN_REFRESH_INTERVAL_MS, int(current * 0.8))
                if reduced < current:
                    widget.set_signals_refresh_interval(int(reduced))
                    self.fixed_interval_ms = int(reduced)
                    adjusted = True
        if adjusted:
            self._update_perf_summary()
            return True
        if not allow_stream_tuning:
            return False
        recorder = getattr(self, "_recorder_tab", None)
        if recorder is not None and hasattr(recorder, "request_finer_streaming"):
            try:
                recorder.request_finer_streaming()
                return True
            except Exception:
                logger.exception("SignalsTab: failed to request finer streaming")
        return False

    def _drain_samples(self) -> None:
        """
        Periodically called by _ingest_timer to transfer samples into the plot.

        Newer architecture: we prefer the shared StreamingDataBuffer that
        RecorderTab fills in _on_samples_batch. If that buffer is not available
        for some reason, we fall back to the legacy sample_queue path.
        """
        logger.debug("SignalsTab: _drain_samples tick active=%s", self._stream_active)
        if not self._stream_active:
            return

        # --- Preferred path: use StreamingDataBuffer ------------------------
        buf = self._active_data_buffer()
        if buf is not None:
            logger.debug("SignalsTab: draining from StreamingDataBuffer")
            # This pulls all new samples since the last cursor position for each
            # sensor/channel and forwards them to the plot widget.
            self._ingest_buffer_data()
            return

        # --- Fallback: legacy queue-based ingestion -------------------------
        queue_obj = self._recorder_sample_queue()
        if queue_obj is None:
            return

        # Fallback path kept for older ingestion flows when no buffer exists.
        logger.debug("SignalsTab: draining from legacy sample_queue")

        drained: list[MpuSample] = []
        start = time.perf_counter()
        try:
            while True:
                drained.append(queue_obj.get_nowait())
        except queue.Empty:
            pass

        if not drained:
            return

        corrected: list[MpuSample] = [
            self._apply_baseline_to_sample(sample) for sample in drained
        ]
        self._plot.add_samples(corrected)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._record_ingest_stats(len(drained), elapsed_ms)

    def _recorder_sample_queue(self) -> queue.Queue[object] | None:
        recorder = getattr(self, "_recorder_tab", None)
        if recorder is None:
            return None
        try:
            queue_obj = recorder.sample_queue
        except AttributeError:
            return None
        return queue_obj

    def _normalize_refresh_mode(self, mode: Optional[str]) -> str:
        if mode in {"fixed", "follow_sampling_rate"}:
            return str(mode)
        if mode == "adaptive":
            return "follow_sampling_rate"
        return DEFAULT_REFRESH_MODE

    def _get_sampling_rate_hz(self) -> Optional[float]:
        if self._sampling_rate_hz is not None:
            return self._sampling_rate_hz
        return None

    def _target_stream_rate_from_recorder(self) -> Optional[float]:
        recorder = getattr(self, "_recorder_tab", None)
        if recorder is None:
            return None
        getter = getattr(recorder, "target_stream_rate_hz", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        widget = getattr(recorder, "mpu_target_stream_rate", None)
        if widget is not None:
            try:
                return float(widget.value())
            except (TypeError, ValueError):
                return None
        return None

    def _sync_refresh_profile_combo(self, interval_ms: int) -> None:
        combo = getattr(self, "refresh_profile_combo", None)
        if combo is None:
            return
        if self.refresh_mode == "follow_sampling_rate":
            idx = combo.findData(REFRESH_FOLLOW_DEVICE)
            with QSignalBlocker(combo):
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            return
        idx = combo.findData(int(interval_ms))
        with QSignalBlocker(combo):
            if idx >= 0:
                combo.setCurrentIndex(idx)
                interval_data = combo.itemData(idx)
                self._active_refresh_profile_label = self._preset_name_for_interval(
                    int(interval_data)
                )
            elif self._refresh_profile_custom_index is not None:
                self._set_custom_refresh_profile_label(interval_ms)
                combo.setCurrentIndex(self._refresh_profile_custom_index)
                self._active_refresh_profile_label = REFRESH_PROFILE_CUSTOM_LABEL
            else:
                self._active_refresh_profile_label = REFRESH_PROFILE_CUSTOM_LABEL

    def _set_custom_refresh_profile_label(self, interval_ms: int) -> None:
        if self._refresh_profile_custom_index is None:
            return
        combo = getattr(self, "refresh_profile_combo", None)
        if combo is None:
            return
        combo.setItemText(
            self._refresh_profile_custom_index,
            self._format_custom_refresh_profile_label(interval_ms),
        )

    def _format_custom_refresh_profile_label(self, interval_ms: int) -> str:
        if interval_ms <= 0:
            return REFRESH_PROFILE_CUSTOM_LABEL
        return f"{REFRESH_PROFILE_CUSTOM_LABEL} ({interval_ms} ms)"

    def _update_refresh_profile_enabled(self) -> None:
        combo = getattr(self, "refresh_profile_combo", None)
        label = getattr(self, "_refresh_profile_label", None)
        enabled = self.refresh_mode in ("fixed", "follow_sampling_rate")
        tooltip = (
            "Preset refresh rates are available only when using a fixed interval or device sampling rate mode."
            if not enabled
            else "Preset GUI refresh rates for the time-domain plots."
        )
        if combo is not None:
            combo.setEnabled(enabled)
            combo.setToolTip(tooltip)
        if label is not None:
            label.setEnabled(enabled)
        if not enabled:
            self._active_refresh_profile_label = None

    def _refresh_profile_descriptor(self) -> str:
        if self.refresh_mode == "follow_sampling_rate":
            return "Device sampling rate"
        if self._active_refresh_profile_label:
            return self._active_refresh_profile_label
        return REFRESH_PROFILE_CUSTOM_LABEL

    @staticmethod
    def _preset_name_for_interval(interval_ms: int) -> str | None:
        for name, preset_interval in REFRESH_PRESETS:
            if int(preset_interval) == int(interval_ms):
                return name
        return None

    def _compute_refresh_interval(self) -> int:
        if self.refresh_mode == "follow_sampling_rate":
            rate_hz = self._get_sampling_rate_hz()
            if not rate_hz or rate_hz <= 0:
                return DEFAULT_REFRESH_INTERVAL_MS

            interval_ms = int(1000.0 / rate_hz)
            if interval_ms < MIN_REFRESH_INTERVAL_MS:
                # Guard against extremely fast redraws when the device samples
                # faster than humans can meaningfully perceive.
                interval_ms = MIN_REFRESH_INTERVAL_MS
            return interval_ms

        return int(self.refresh_interval_ms)

    def _reset_timer_stats(self) -> None:
        self._timer_tick_counter = 0
        self._timer_measure_window_start = time.perf_counter()
        self._timer_stats_hz = 0.0

    def _record_timer_tick(self) -> None:
        now = time.perf_counter()
        # Initialize the window start if this is the first tick.
        if self._timer_measure_window_start <= 0.0:
            self._timer_measure_window_start = now
        self._timer_tick_counter += 1
        elapsed = now - self._timer_measure_window_start
        if elapsed >= 1.0:
            self._timer_stats_hz = self._timer_tick_counter / elapsed
            self._timer_tick_counter = 0
            self._timer_measure_window_start = now

    def _reset_refresh_measurement(self) -> None:
        self._last_refresh_timestamp = None
        self._smoothed_refresh_hz = None
        self._update_plot_refresh_label()

    def _record_refresh_tick(self) -> None:
        now = time.monotonic()
        last = self._last_refresh_timestamp
        if last is not None:
            dt = now - last
            if dt > 0.0:
                inst_hz = 1.0 / dt
                smoothed = self._smoothed_refresh_hz
                if smoothed is None:
                    self._smoothed_refresh_hz = inst_hz
                else:
                    alpha = 0.2
                    self._smoothed_refresh_hz = alpha * inst_hz + (1.0 - alpha) * smoothed
        self._last_refresh_timestamp = now
        self._update_plot_refresh_label()

    def _record_ingest_stats(self, samples: int, elapsed_ms: float | None = None) -> None:
        if not ENABLE_PLOT_PERF_METRICS or samples <= 0:
            return
        self._queue_ingest_samples += int(samples)
        self._queue_ingest_batches += 1
        if elapsed_ms is not None:
            self._queue_ingest_time_acc_ms += float(elapsed_ms)

        now = time.perf_counter()
        if now - self._queue_ingest_last_log >= 5.0:
            batches = max(1, self._queue_ingest_batches)
            avg_samples = self._queue_ingest_samples / batches
            avg_time_ms = self._queue_ingest_time_acc_ms / batches
            logger.debug(
                "SignalsTab: ingest avg %.1f samples/batch avg_time≈%.2f ms",
                avg_samples,
                avg_time_ms,
            )
            self._queue_ingest_samples = 0
            self._queue_ingest_batches = 0
            self._queue_ingest_time_acc_ms = 0.0
            self._queue_ingest_last_log = now

    def _update_perf_summary(self) -> None:
        label = getattr(self, "_perf_summary_label", None)
        if label is None:
            return
        stream_hz = float(self._sampling_rate_hz or 0.0)
        redraw_ms = self._redraw_ema_ms if self._redraw_ema_ms > 0.0 else self._last_redraw_ms
        if redraw_ms <= 0.0:
            redraw_ms = 0.0
        interval_ms = 0.0
        timer = getattr(self, "_timer", None)
        if timer is not None:
            try:
                interval_ms = float(timer.interval())
            except (TypeError, ValueError):
                interval_ms = 0.0
        if interval_ms <= 0.0:
            interval_ms = float(self._compute_refresh_interval())
        if interval_ms <= 0.0:
            interval_ms = float(self.refresh_interval_ms)
        #label.setText(
        #    f"Perf: Stream ≈ {stream_hz:4.1f} Hz  |  "
        #    f"Redraw EMA ≈ {redraw_ms:4.1f} ms  |  "
        #    f"Refresh interval ≈ {interval_ms:.0f} ms"
        #)

    def _update_plot_refresh_label(self) -> None:
        label = getattr(self, "_plot_refresh_label", None)
        if label is None:
            return
        timer = getattr(self, "_timer", None)
        timer_active = True
        interval_ms = int(self.refresh_interval_ms)
        if timer is not None:
            timer_active = timer.isActive()
            try:
                interval_ms = int(timer.interval())
            except TypeError:
                pass
        descriptor = self._refresh_profile_descriptor()
        hz = self._smoothed_refresh_hz
        suffix = ""
        interval_parts: list[str] = []
        if interval_ms > 0:
            interval_parts.append(f"{interval_ms} ms")
        if descriptor:
            interval_parts.append(descriptor)
        if interval_parts:
            suffix = f" ({' – '.join(interval_parts)})"
        if hz is not None:
            label.setText(f"Plot refresh: {hz:4.1f} Hz{suffix}")
            return
        target = self._target_refresh_hz
        if not timer_active:
            label.setText("Plot refresh: paused")
            return
        if target and target > 0.0:
            label.setText(f"Plot refresh (target): {target:4.1f} Hz{suffix}")
        else:
            label.setText(f"Plot refresh: -- Hz{suffix}")

    def _refresh_ingest_timer(self) -> None:
        """Start/stop the ingest timer based on stream activity."""
        if not hasattr(self, "_ingest_timer"):
            return
        should_run = self._stream_active or self._synthetic_active
        if should_run and not self._ingest_timer.isActive():
            logger.debug("SignalsTab: starting ingest timer")
            self._ingest_timer.start()
        elif not should_run and self._ingest_timer.isActive():
            logger.debug("SignalsTab: stopping ingest timer")
            self._ingest_timer.stop()

    def _refresh_timer_state(self) -> None:
        """Start/stop the redraw timer based on live or synthetic stream activity."""
        if not hasattr(self, "_timer"):
            return
        # Only tick the GUI timer while we have real or synthetic data;
        # pausing it avoids wasting CPU when no stream is active.
        should_run = self._stream_active or self._synthetic_active
        if should_run:
            if not self._timer.isActive():
                self._timer.start()
                self._reset_refresh_measurement()
        else:
            if self._timer.isActive():
                self._timer.stop()
                self._reset_refresh_measurement()
        self._update_plot_refresh_label()

    def _apply_refresh_settings(self) -> None:
        interval_ms = self._compute_refresh_interval()
        if self.refresh_mode == "follow_sampling_rate":
            previous = self._last_follow_interval_ms
            self._last_follow_interval_ms = interval_ms
            if (
                previous
                and previous > 0
                and abs(interval_ms - previous) / float(previous) > 0.10
            ):
                logger.debug(
                    "SignalsTab: adjusting follow-mode refresh interval from %d ms to %d ms",
                    previous,
                    interval_ms,
                )
        else:
            self._last_follow_interval_ms = None
        target_hz = 1000.0 / interval_ms if interval_ms > 0 else None
        self._target_refresh_hz = target_hz
        self._plot.set_target_refresh_rate(target_hz)
        self._sync_refresh_profile_combo(interval_ms)
        self._reset_refresh_measurement()
        if ENABLE_PLOT_PERF_METRICS:
            self._reset_timer_stats()
        if hasattr(self, "_timer"):
            self._timer.setInterval(interval_ms)
            if self._timer.isActive():
                self._timer.start(interval_ms)
            else:
                self._refresh_timer_state()
        self._update_perf_summary()

    # --------------------------------------------------------------- synthetic data helpers
    def start_synthetic_stream(
        self,
        rate_hz: float,
        sensor_ids: Iterable[int] | None = None,
    ) -> None:
        interval_ms = max(1, int(round(1000.0 / max(1.0, rate_hz))))
        interval_ns = max(1, int(1_000_000_000 / max(1.0, rate_hz)))
        self.stop_synthetic_stream()
        self._synthetic_rate_hz = float(rate_hz)
        self._synthetic_sensor_ids = list(sensor_ids) if sensor_ids else [1]
        self._synthetic_phase = 0.0
        self._synthetic_phase_step = 2.0 * math.pi * (1.0 / max(1.0, rate_hz))
        self._synthetic_interval_ns = interval_ns
        self._synthetic_start_timestamp_ns = time.monotonic_ns()
        self._synthetic_next_timestamp_ns = self._synthetic_start_timestamp_ns
        self._fallback_data_buffer.clear()
        self._buffer_cursors.clear()
        timer = QTimer(self)
        timer.setTimerType(Qt.PreciseTimer)
        timer.setInterval(interval_ms)
        timer.timeout.connect(self._on_synthetic_tick)
        timer.start()
        self._synthetic_timer = timer
        self._synthetic_active = True
        self._refresh_ingest_timer()
        self._refresh_timer_state()

    def stop_synthetic_stream(self) -> None:
        timer = self._synthetic_timer
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        self._synthetic_timer = None
        self._synthetic_active = False
        self._refresh_ingest_timer()
        self._refresh_timer_state()

    def _on_synthetic_tick(self) -> None:
        if self._synthetic_timer is None:
            return
        timestamp_ns = self._synthetic_next_timestamp_ns
        interval_ns = self._synthetic_interval_ns
        start_ns = self._synthetic_start_timestamp_ns
        phase = self._synthetic_phase
        phase_step = self._synthetic_phase_step
        per_sensor_offset_ns = max(1, interval_ns // max(1, len(self._synthetic_sensor_ids)))
        generated: List[MpuSample] = []
        for idx, sensor_id in enumerate(self._synthetic_sensor_ids):
            offset = idx * 0.5
            val_sin = math.sin(phase + offset)
            val_cos = math.cos(phase + offset)
            sample_ns = int(timestamp_ns + idx * per_sensor_offset_ns)
            sample = MpuSample(
                timestamp_ns=sample_ns,
                ax=val_sin,
                ay=math.sin(phase + offset + 0.5),
                az=math.sin(phase + offset + 1.0),
                gx=val_cos,
                gy=math.cos(phase + offset + 0.5),
                gz=math.cos(phase + offset + 1.0),
                sensor_id=int(sensor_id),
                t_s=float((sample_ns - start_ns) / 1_000_000_000),
            )
            generated.append(sample)
        self._synthetic_phase = phase + phase_step
        self._synthetic_next_timestamp_ns = timestamp_ns + interval_ns
        if generated:
            self._fallback_data_buffer.add_samples(generated)

    @Slot()
    def on_stream_started(self) -> None:
        logger.info("SignalsTab: stream started")
        # Clear plot data and cursors, but keep the nominal sample rate
        # that was configured from SettingsTab via apply_gui_acquisition_config().
        self._plot.clear()
        self._buffer_cursors.clear()

        # Reset the *measured* stream rate estimate (used for display/perf),
        # but do NOT touch the plot's nominal sample rate.
        self._sampling_rate_hz = None
        self._stream_rate_label.setText("Stream rate: -- Hz")
        self._stream_active = True
        self._awaiting_first_sample = True
        self._stream_stalled = False
        self._last_data_monotonic = 0.0
        self._set_stream_status("Streaming...", force=True)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._refresh_ingest_timer()
        self._refresh_timer_state()
        self._refresh_mode_hint()

    def set_view_mode_by_channels(self, charts: int) -> None:
        return

    def set_view_mode_preset(self, preset: str) -> None:
        _ = preset

    @Slot()
    def on_stream_stopped(self) -> None:
        logger.info("SignalsTab: stream stopped")
        self._stream_active = False
        self._awaiting_first_sample = False
        self._stream_stalled = False
        self._last_data_monotonic = 0.0
        self._set_stream_status("Stopped.", force=True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._plot.clear()
        self._buffer_cursors.clear()
        self._refresh_ingest_timer()
        self._refresh_timer_state()
        self._refresh_mode_hint()

    @Slot(str)
    def handle_error(self, message: str) -> None:
        self._set_manual_status(message)

    @Slot(int)
    def _on_base_correction_toggled(self, state: int) -> None:
        enabled = state == Qt.Checked
        self._base_correction_enabled = enabled
        self._plot.enable_base_correction(enabled)
        if enabled:
            self._set_manual_status("Base correction enabled.")
        else:
            self._set_manual_status("Base correction disabled.")

