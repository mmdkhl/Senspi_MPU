"""Live FFT / spectrum tab for MPU6050 samples."""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...analysis import filters
from ...analysis import modal as modal_id
from ..config.acquisition_state import (
    CalibrationOffsets,
    GuiAcquisitionConfig,
    SensorSelectionConfig,
)
from ...config.app_config import AppConfig, PlotPerformanceConfig
from ...core.ringbuffer import RingBuffer
from ...data import StreamingDataBuffer
# (eigen capture now uses RecorderController.snapshot_modal_capture, which aligns
# internally — no direct align_per_sensor_series call here.)
from ...tools.debug import debug_enabled
from . import LayoutSignature, SampleKey

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from ..recorder_controller import RecorderController
    from .tab_signals import SignalsTab

DEFAULT_FFT_WINDOW_S = 2.0
MIN_FFT_WINDOW_S = 0.5
MAX_FFT_WINDOW_S = 10.0

DEFAULT_FFT_UPDATE_MS = 500  # fallback if config missing
MIN_FFT_UPDATE_MS = 50
MAX_FFT_UPDATE_MS = 2000

DEFAULT_MAX_FREQUENCY_HZ = 200.0  # cap plotted frequency if useful

# The spectrum shows only the structural horizontal axes (T11.1).
SPECTRUM_CHANNELS: tuple[str, ...] = ("ax", "ay")
# Window 2 eigen-frequency identification uses a rolling batch of this length.
# It must exceed identify_modes' MIN_DURATION_S = 10 s with margin (alignment can
# trim a little), so 12 s. Captured from the controller's 120 s modal buffer (NOT
# the 6 s display buffer — that was why the panel stayed empty).
EIGEN_BATCH_S = 12.0
# How often Window 2 recomputes (rolling/overlapping → faster perceived updates;
# "as fast as possible" once the first 12 s of data has accumulated).
EIGEN_UPDATE_S = 4.0
# Distinct colours for the (up to) three identified natural frequencies — also
# reused to mark those frequencies on the per-sensor grid (Window 1).
_EIGEN_COLORS = ("#ff5252", "#448aff", "#69f0ae")

logger = logging.getLogger(__name__)


class _EigenFreqWorker(QObject):
    """Identify the structure's natural frequencies from a live modal-capture batch.

    Pure compute off the GUI thread (G4); emits plain floats only (G1). Captures a
    rolling batch from the controller's **120 s modal buffer** via the supplied
    ``capture_fn`` (``RecorderController.snapshot_modal_capture`` — the SAME source
    Model Updating's Mode B uses; the 6 s display buffer could never supply the
    >= 10 s identify_modes needs). Reuses the shared ``identify_modes`` engine (D2);
    the ``method`` ("fdd"|"fft") chooses the algorithm. Guards the degenerate
    FDD-with-one-sensor case.
    """

    result = Signal(object)      # dict: freqs + identification-spectrum curve
    collecting = Signal(float)   # seconds available so far (< MIN_DURATION_S)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, capture_fn, fs, method, f_min, f_max, n_modes, batch_s,
                 placement=None, n_story=0):
        super().__init__()
        self._capture_fn = capture_fn
        self._fs = fs
        self._method = method
        self._f_min = f_min
        self._f_max = f_max
        self._n_modes = n_modes
        self._batch_s = batch_s
        self._placement = dict(placement or {})  # {sensor_id: floor}
        self._n_story = int(n_story)

    @Slot()
    def run(self) -> None:
        try:
            session = self._capture_fn(
                axis="ax", last_seconds=self._batch_s, target_fs=self._fs)
            data = getattr(session, "data", None)
            if data is None or data.size == 0 or data.shape[0] == 0:
                self.collecting.emit(0.0)
                return
            if session.duration_s < modal_id.MIN_DURATION_S:
                # Not enough buffered yet — report progress, not an error.
                self.collecting.emit(float(session.duration_s))
                return
            if self._method == "fdd" and data.shape[0] < 2:
                self.failed.emit("FDD needs ≥ 2 sensors; switch to FFT")
                return
            res = modal_id.identify_modes(
                data, session.fs, method=self._method, n_modes=self._n_modes,
                f_min=self._f_min, f_max=self._f_max)
            if res.success:
                payload = {
                    "freqs": [float(f) for f in res.frequencies_hz],
                    "spec_f": np.asarray(res.fdd_freqs, dtype=float),
                    "spec_v": np.asarray(res.fdd_spectrum, dtype=float),
                    "signed": self._method == "fdd",
                }
                # Map the per-sensor shapes onto floors when a placement + floor
                # count are available (mode-shape view). map_to_stories is pure/cheap.
                if self._placement and self._n_story > 0:
                    story_map = [int(self._placement.get(int(sid), 0))
                                 for sid in session.sensor_ids]
                    story_data = modal_id.map_to_stories(res, story_map, self._n_story)
                    payload["mode_shapes_ux"] = dict(story_data.mode_shapes_ux)
                    payload["coverage_stories"] = list(story_data.coverage_stories)
                    payload["n_story"] = self._n_story
                    payload["full_coverage"] = bool(story_data.mode_shapes_available)
                self.result.emit(payload)
            else:
                self.failed.emit(res.message or "Identification failed")
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class FftTab(QWidget):
    """
    Live spectrum/FFT view for the streaming MPU6050 channels.

    Responsibilities:
    - Pull sliding windows of samples from the shared
      :class:`StreamingDataBuffer` (owned by :class:`RecorderController`).
    - Render frequency-domain plots that complement :class:`SignalsTab`'s time
      series, using matching sensor/channel layouts.
    - Track stream rate updates and refresh interval hints emitted by the
      signals tab so spectral analysis follows live data pacing.
    """

    def __init__(
        self,
        recorder_tab: RecorderController,
        signals_tab: "SignalsTab | None" = None,
        parent: Optional[QWidget] = None,
        app_config: AppConfig | None = None,
    ) -> None:
        super().__init__(parent)

        self._recorder_tab = recorder_tab
        self._signals_tab: SignalsTab | None = signals_tab
        self._app_config: AppConfig = app_config or AppConfig()
        self._gui_acq_config: GuiAcquisitionConfig | None = None
        self._sensor_selection: SensorSelectionConfig | None = None
        self._calibration_offsets: CalibrationOffsets | None = None
        plot_perf = getattr(self._app_config, "plot_performance", None)
        if not isinstance(plot_perf, PlotPerformanceConfig):
            plot_perf = PlotPerformanceConfig()
        self._plot_perf_config: PlotPerformanceConfig = plot_perf
        self._max_subplots = self._plot_perf_config.normalized_max_subplots()
        self._device_rate_hz: float = 0.0
        self._measured_rate_hz: float = 0.0
        self._refresh_interval_ms: int = self._clamp_fft_interval(
            self._plot_perf_config.fft_refresh_interval_ms()
        )
        self._stream_active: bool = False
        self._last_rendered_latest_ts: Optional[float] = None
        self._force_next_update: bool = True
        self._psd_plots: Dict[SampleKey, "pg.PlotItem"] = {}
        self._psd_curves: Dict[SampleKey, "pg.PlotDataItem"] = {}
        self._current_layout: tuple | None = None
        # Bound how many samples each FFT uses so the GUI stays responsive.
        self._max_fft_samples = 4096
        self._fft_decimation_target = 2048
        self._fft_size = 512
        self._fft_sample_rate_hz: float = 1.0
        self._fft_freqs = np.fft.rfftfreq(self._fft_size, 1.0 / self._fft_sample_rate_hz)
        self._fft_window = np.hanning(self._fft_size)

        # Window 1: per-sensor PSD grid (pyqtgraph — D1, matches Live Signals) ---
        # Black background + bright white spectra, like the Live Signals tab
        # (pyqtgraph's default dark theme; the previous white bg + thin blue line
        # looked faded).
        pg.setConfigOptions(antialias=True)
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("k")
        self._psd_pen = pg.mkPen("w", width=1.3)
        # Identified-mode markers overlaid on each per-sensor cell (Window 1),
        # driven by the FDD/FFT selector — so the method visibly changes the peaks
        # shown on the individual sensors too.
        self._grid_eig_lines: list = []
        self._last_eigen_freqs: list[float] = []

        # Window 2: identification spectrum + identified eigen-frequencies -------
        self._eig_glw = pg.GraphicsLayoutWidget()
        self._eig_glw.setBackground("k")
        self._eig_plot = self._eig_glw.addPlot()
        self._eig_plot.setLabel("bottom", "Frequency", units="Hz")
        self._eig_plot.setLabel("left", "Response")
        self._eig_plot.showGrid(x=True, y=True, alpha=0.3)
        self._eig_plot.setMouseEnabled(x=True, y=True)
        self._eig_plot.enableAutoRange(y=True)
        # The identification spectrum curve (FDD = 1st singular value of the CSD;
        # FFT = sensor-averaged Welch PSD) — white like Live Signals; the coloured
        # mode lines are drawn on top. This curve is what visibly differs between
        # FDD and FFT even when the picked peaks coincide.
        self._eig_curve = self._eig_plot.plot([], [], pen=pg.mkPen("w", width=1.3))
        self._eig_items: list = []

        # Window 2 (alt view): per-floor mode shapes (optional, toggled). amplitude
        # (x) vs floor (y); one coloured profile per mode. Built from the identified
        # per-sensor shapes via map_to_stories (needs the sensor→floor placement).
        self._shape_glw = pg.GraphicsLayoutWidget()
        self._shape_glw.setBackground("k")
        self._shape_plot = self._shape_glw.addPlot()
        self._shape_plot.setLabel("bottom", "Modal amplitude (|max|=1)")
        self._shape_plot.setLabel("left", "Floor")
        self._shape_plot.showGrid(x=True, y=True, alpha=0.3)
        self._shape_plot.addLine(x=0.0, pen=pg.mkPen("#888", width=1.0))  # zero ref
        self._shape_items: list = []
        self._last_shape_data: dict | None = None
        # Sensor→floor placement (only meaningful for mode shapes). Fixed rig set.
        self._shape_sensor_ids = (1, 2, 3)
        # Rolling/overlapping cadence for Window 2 (recompute every EIGEN_UPDATE_S
        # using the last EIGEN_BATCH_S of data from the 120 s modal buffer).
        self._eig_timer = QTimer(self)
        self._eig_timer.setInterval(int(EIGEN_UPDATE_S * 1000))
        self._eig_timer.timeout.connect(self._launch_eigen_compute)
        self._eig_thread: QThread | None = None
        self._eig_worker: _EigenFreqWorker | None = None

        # Controls --------------------------------------------------------------
        # T11.7 (compact): FDD and FFT share every input, so ALL params live in one
        # compact band at the top (two short rows) — this keeps the figures large.
        # The method selector still picks the Window-2 algorithm (FDD = SVD of the
        # cross-spectral-density matrix → signed shapes; FFT = averaged Welch PSD →
        # magnitude shapes); the PSD grid (Window 1) is unaffected (U1).
        controls_group = QGroupBox("Spectrum settings")
        controls_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        controls_v = QVBoxLayout(controls_group)

        # Row 1 — method selector + eigen-identification params (Window 2).
        self.method_combo = QComboBox()
        self.method_combo.addItems(["FDD", "FFT"])  # FDD default (M2)
        self.method_combo.setToolTip(
            "FDD: SVD of the cross-spectral-density matrix — signed mode shapes.\n"
            "FFT: sensor-averaged Welch PSD peak-picking — magnitude-only shapes.\n"
            "Both use the same frequency band + mode count; only the algorithm differs."
        )
        self._eig_fmin = QDoubleSpinBox()
        self._eig_fmin.setRange(0.05, 500.0)
        self._eig_fmin.setDecimals(2)
        self._eig_fmin.setValue(0.5)
        self._eig_fmax = QDoubleSpinBox()
        self._eig_fmax.setRange(0.10, 1000.0)
        self._eig_fmax.setDecimals(2)
        self._eig_fmax.setValue(20.0)  # structural band default (modes are low-freq)
        self._eig_nmodes = QSpinBox()
        self._eig_nmodes.setRange(1, 12)
        self._eig_nmodes.setValue(3)
        self._method_hint = QLabel("")
        self._method_hint.setStyleSheet("color: #888;")

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Eigen method:"))
        row1.addWidget(self.method_combo)
        row1.addSpacing(18)
        row1.addWidget(QLabel("Freq band (Hz):"))
        row1.addWidget(self._eig_fmin)
        row1.addWidget(QLabel("–"))
        row1.addWidget(self._eig_fmax)
        row1.addSpacing(14)
        row1.addWidget(QLabel("Modes:"))
        row1.addWidget(self._eig_nmodes)
        row1.addSpacing(18)
        row1.addWidget(self._method_hint)
        row1.addStretch()
        controls_v.addLayout(row1)

        # Row 2 — spectrum display params (Window-1 grid; method-independent).
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(MIN_FFT_WINDOW_S, MAX_FFT_WINDOW_S)
        self.window_spin.setSingleStep(0.5)
        self.window_spin.setValue(self._plot_perf_config.normalized_time_window_s())
        self.detrend_check = QCheckBox("Detrend")
        self.detrend_check.setChecked(True)
        self.lowpass_check = QCheckBox("Low-pass")
        self.lowpass_cutoff = QDoubleSpinBox()
        self.lowpass_cutoff.setRange(0.1, 5000.0)
        self.lowpass_cutoff.setSingleStep(1.0)
        self.lowpass_cutoff.setValue(100.0)
        self.fft_interval_spin = QDoubleSpinBox()
        self.fft_interval_spin.setRange(MIN_FFT_UPDATE_MS, MAX_FFT_UPDATE_MS)
        self.fft_interval_spin.setSingleStep(50.0)
        self.fft_interval_spin.setDecimals(0)
        self.fft_interval_spin.setValue(float(self._refresh_interval_ms))
        self.fft_interval_spin.setSuffix(" ms")
        self.fft_interval_spin.valueChanged.connect(
            lambda ms: self.set_refresh_interval_ms(int(ms))
        )

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Window (s):"))
        row2.addWidget(self.window_spin)
        row2.addSpacing(14)
        row2.addWidget(self.detrend_check)
        row2.addSpacing(14)
        row2.addWidget(self.lowpass_check)
        row2.addWidget(QLabel("Cutoff (Hz):"))
        row2.addWidget(self.lowpass_cutoff)
        row2.addSpacing(14)
        row2.addWidget(QLabel("FFT refresh:"))
        row2.addWidget(self.fft_interval_spin)
        row2.addStretch()
        controls_v.addLayout(row2)

        # Status labels
        self._status_label = QLabel("Waiting for data...")
        self._eig_status = QLabel("Eigen-frequencies: waiting for 10 s of data…")

        # Two-panel layout (T11.6): Window 1 (PSD grid) | Window 2 (eigen-freqs), 50/50.
        windows = QSplitter(Qt.Horizontal)
        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.addWidget(QLabel("Per-sensor spectra (ax, ay)"))
        left_v.addWidget(self._glw)
        right_panel = QWidget()
        right_v = QVBoxLayout(right_panel)
        right_v.setContentsMargins(0, 0, 0, 0)
        # Right-panel view toggle: natural frequencies (default) vs mode shapes.
        view_row = QHBoxLayout()
        self._right_title = QLabel("Identified natural frequencies")
        self._show_shapes_check = QCheckBox("Show mode shapes")
        self._show_shapes_check.setToolTip(
            "Toggle the right panel between the identification spectrum + natural\n"
            "frequencies and the per-floor mode shapes (needs the sensor→floor map\n"
            "below; FDD gives signed shapes, FFT magnitude-only).")
        self._show_shapes_check.toggled.connect(self._on_view_toggled)
        view_row.addWidget(self._right_title)
        view_row.addStretch()
        view_row.addWidget(self._show_shapes_check)
        right_v.addLayout(view_row)
        # Sensor→floor placement (only shown/used in mode-shape view).
        self._shape_placement_widget = self._build_shape_placement()
        self._shape_placement_widget.setVisible(False)
        right_v.addWidget(self._shape_placement_widget)
        right_v.addWidget(self._eig_glw)         # frequencies view (default)
        right_v.addWidget(self._shape_glw)        # mode-shape view (toggled)
        self._shape_glw.setVisible(False)
        right_v.addWidget(self._eig_status)
        windows.addWidget(left_panel)
        windows.addWidget(right_panel)
        windows.setSizes([500, 500])

        # Layout ---------------------------------------------------------------
        layout = QVBoxLayout(self)
        layout.addWidget(controls_group)
        layout.addWidget(windows, stretch=1)
        layout.addWidget(self._status_label)

        # NOTE: This timer drives the legacy FFT refresh cadence. Align with the
        # new acquisition config in the upcoming GUI refactor before changing it.
        # Timer to recompute FFT periodically
        self._timer = QTimer(self)
        self._timer.setInterval(self._refresh_interval_ms)
        self._timer.timeout.connect(self._on_fft_timer)
        self._debug_fft_ema_ms: float = 0.0
        self._debug_fft_last_log: float = time.perf_counter()

        # After basic fields are initialized, wire FFT refresh interval from SignalsTab
        if self._signals_tab is not None:
            try:
                self._signals_tab.fft_refresh_interval_changed.connect(
                    self.set_refresh_interval_ms
                )
                logger.debug(
                    "FftTab: connected fft_refresh_interval_changed from SignalsTab"
                )
            except Exception:
                logger.exception(
                    "FftTab: failed to connect fft_refresh_interval_changed signal"
                )

        # Wiring
        self.window_spin.valueChanged.connect(self._on_controls_changed)
        self.window_spin.valueChanged.connect(self._update_fft_timer_interval)
        self.detrend_check.toggled.connect(self._on_controls_changed)
        self.lowpass_check.toggled.connect(self._on_controls_changed)
        self.lowpass_cutoff.valueChanged.connect(self._on_controls_changed)
        # T11.7: method selector swaps the method-specific group (Window 2 only).
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self._on_method_changed()  # set initial visibility
        self._update_fft_timer_interval()
        self._draw_waiting()

    @Slot()
    def _on_method_changed(self, *_: object) -> None:
        """T11.7: update the inline method hint when FDD/FFT changes (pure GUI, G1).

        FDD and FFT share every input, so there are no controls to swap — only the
        one-line description updates. Governs Window 2 (eigen identification); the
        PSD grid (Window 1) is unaffected (U1) and picks up the method on its next
        10 s batch.
        """
        if self.method_combo.currentText() == "FDD":
            self._method_hint.setText(
                "SVD of the cross-spectral-density matrix — signed mode shapes.")
        else:
            self._method_hint.setText(
                "Sensor-averaged Welch PSD peak-picking — magnitude-only shapes.")
        # Recompute immediately so the spectrum curve + peaks reflect the new method
        # without waiting for the next rolling tick (the worker guards re-entrancy).
        self._launch_eigen_compute()

    def apply_gui_acquisition_config(self, cfg: GuiAcquisitionConfig) -> None:
        """Backward-compatible alias for :meth:`update_acquisition_config`."""

        self.update_acquisition_config(cfg)

    def update_acquisition_config(self, config: GuiAcquisitionConfig) -> None:
        """
        Receive the latest GUI acquisition configuration (sampling rate, stream rate,
        record-only flag, calibration, etc.).
        """

        self._gui_acq_config = config
        try:
            self._device_rate_hz = float(config.sampling.device_rate_hz)
        except Exception:
            self._device_rate_hz = 0.0
        self._ensure_fft_frequency_axis(self._device_rate_hz)
        if config.record_only:
            if self._timer.isActive():
                self._timer.stop()
            self._status_label.setText("Record-only mode: live FFT disabled.")
        else:
            if self._stream_active and not self._timer.isActive():
                self._timer.start(self._refresh_interval_ms)
            self._request_full_refresh()
        if getattr(config, "calibration", None) is not None:
            try:
                self.set_calibration_offsets(config.calibration)
            except Exception:
                # Defensive: avoid breaking updates if calibration payload is missing.
                pass

    def set_sensor_selection(self, cfg: SensorSelectionConfig) -> None:
        """Backward-compatible alias for :meth:`update_sensor_selection`."""

        self.update_sensor_selection(cfg)

    def update_sensor_selection(self, selection: SensorSelectionConfig) -> None:
        """
        Receive the latest sensor/channel selection so FFT can restrict what it plots.
        """

        self._sensor_selection = selection
        self._request_full_refresh()

    def set_calibration_offsets(self, offsets: CalibrationOffsets | None) -> None:
        """
        Update calibration offsets used when preparing the FFT input.
        """

        self._calibration_offsets = offsets
        self._request_full_refresh()

    def set_record_only_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled:
            self._timer.stop()
            self._status_label.setText("Record-only mode: live FFT disabled.")
        else:
            if self._refresh_interval_ms > 0 and self._stream_active:
                self._timer.start(self._refresh_interval_ms)

    def _is_record_only(self) -> bool:
        cfg = getattr(self, "_gui_acq_config", None)
        return bool(getattr(cfg, "record_only", False))

    def _make_key(self, sensor_id: int, channel: str) -> SampleKey:
        return int(sensor_id), str(channel)

    def _clamp_fft_interval(self, interval_ms: float | int) -> int:
        try:
            interval = int(round(float(interval_ms)))
        except (TypeError, ValueError):
            interval = DEFAULT_FFT_UPDATE_MS
        return max(MIN_FFT_UPDATE_MS, min(interval, MAX_FFT_UPDATE_MS))

    def _active_stream_buffer(self) -> StreamingDataBuffer:
        """Return the current streaming buffer (prepping for shared LiveDataStore)."""
        return self._recorder_tab.data_buffer()

    def _get_buffer_window(
        self,
        key: SampleKey,
        *,
        window_s: float,
        data_buffer: StreamingDataBuffer | None = None,
    ) -> tuple[Sequence[float], Sequence[float]]:
        # Prefer reusing the time-windowed data held by SignalsTab; if that
        # is unavailable, fall back to querying the shared StreamingDataBuffer.
        sensor_id, channel = key
        window = self._window_from_signals_tab(sensor_id, channel, window_s)
        if window is not None:
            return window

        buffer = data_buffer or self._active_stream_buffer()
        return buffer.get_axis_series(sensor_id, channel, seconds=window_s)

    def _on_controls_changed(self, *args: object) -> None:
        """Trigger an FFT refresh when the user changes view/filter controls."""
        self._request_full_refresh()
        self._update_fft()

    def _request_full_refresh(self) -> None:
        self._force_next_update = True

    def _update_fft_timer_interval(self, *_: object) -> None:
        """Adjust the FFT refresh cadence based on the selected window length."""
        if not hasattr(self, "_timer"):
            return
        try:
            window_s = float(self.window_spin.value())
        except (TypeError, ValueError):
            window_s = DEFAULT_FFT_WINDOW_S
        window_s = max(MIN_FFT_WINDOW_S, min(window_s, MAX_FFT_WINDOW_S))
        desired_period_s = window_s / 2.0
        interval_ms = int(desired_period_s * 1000.0)
        self.set_refresh_interval_ms(interval_ms)

    def set_refresh_interval_ms(self, interval_ms: int) -> None:
        """Public setter so other tabs can tune how often the FFT updates."""
        clamped = self._clamp_fft_interval(interval_ms)
        self._refresh_interval_ms = clamped
        self._timer.setInterval(clamped)
        spin = getattr(self, "fft_interval_spin", None)
        if spin is not None:
            try:
                from PySide6.QtCore import QSignalBlocker

                blocker = QSignalBlocker(spin)
            except Exception:
                blocker = None
            spin.setValue(float(clamped))
            if blocker is not None:
                del blocker

    def set_max_fft_samples(self, n: int) -> None:
        """Public setter mainly for tests / tuning of the FFT sample cap."""
        self._max_fft_samples = max(256, int(n))

    def set_signals_tab(self, signals_tab: "SignalsTab | None") -> None:
        """Inject the SignalsTab reference so we can reuse its ring buffers."""
        self._signals_tab = signals_tab

    @Slot(str, float)
    def update_stream_rate(self, sensor_type: str, hz: float) -> None:
        """Receive stream-rate updates so FFT windows know how much data to expect."""
        if sensor_type != "mpu6050":
            return
        self._measured_rate_hz = float(hz) if hz > 0.0 else 0.0
        logger.info("FftTab: update_stream_rate %.2f Hz", self._measured_rate_hz)
        self._request_full_refresh()

    def set_sampling_rate_hz(self, hz: float) -> None:
        """
        Manually set the nominal sampling/stream rate used by the FFT timer.

        This is used when we already know the target stream/plot rate from
        the GUI (GuiAcquisitionConfig) before the RecorderTab has measured
        and reported a real rate.

        The device sampling rate is authoritative for the FFT axis.
        """
        try:
            value = float(hz)
        except (TypeError, ValueError):
            # Ignore invalid values; keep existing rate
            return

        if value <= 0.0:
            return

        self._device_rate_hz = value
        self._ensure_fft_frequency_axis(self._device_rate_hz)
        self._request_full_refresh()

    @Slot()
    def on_stream_started(self) -> None:
        logger.info("FftTab: on_stream_started")
        self._clear_layout()
        self._draw_waiting()
        self._status_label.setText("Streaming...")
        self._last_rendered_latest_ts = None
        self._request_full_refresh()
        self._stream_active = True
        if not self._timer.isActive() and not self._is_record_only():
            logger.debug(
                "FftTab: starting FFT timer at %d ms", self._refresh_interval_ms
            )
            self._timer.start(self._refresh_interval_ms)
        if not self._eig_timer.isActive() and not self._is_record_only():
            self._eig_timer.start()
            self._eig_status.setText(
                f"Eigen-frequencies: collecting first {EIGEN_BATCH_S:.0f} s of data…")

    @Slot()
    def on_stream_stopped(self) -> None:
        logger.info("FftTab: on_stream_stopped")
        # Keep last spectrum visible but update status.
        self._stream_active = False
        if self._timer.isActive():
            logger.debug("FftTab: stopping FFT timer")
            self._timer.stop()
        if self._eig_timer.isActive():
            self._eig_timer.stop()
        self._status_label.setText("Stopped")
        self._last_rendered_latest_ts = None

    # ----------------------------------------------------- Window 2 (eigen-freq)
    def _launch_eigen_compute(self) -> None:
        """Identify the structure's natural frequencies in a worker (Window 2).

        Captures the last ``EIGEN_BATCH_S`` from the controller's 120 s modal
        buffer (``snapshot_modal_capture`` — the same source Model Updating uses),
        not the 6 s display buffer. Skips if a batch is still running.
        """
        if self._eig_thread is not None:
            return  # previous batch still running; skip this tick
        if self._is_record_only() or not self._stream_active:
            return
        capture_fn = getattr(self._recorder_tab, "snapshot_modal_capture", None)
        if capture_fn is None:
            return
        fs = self._device_rate_hz or self._measured_rate_hz or None
        method = "fdd" if self.method_combo.currentText() == "FDD" else "fft"
        worker = _EigenFreqWorker(
            capture_fn, fs, method,
            float(self._eig_fmin.value()),
            float(self._eig_fmax.value()),
            int(self._eig_nmodes.value()),
            EIGEN_BATCH_S,
            placement=self._shape_sensor_story_map(),
            n_story=int(self._shape_floors.value()),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result.connect(self._on_eigen_result)
        worker.collecting.connect(self._on_eigen_collecting)
        worker.failed.connect(self._on_eigen_failed)
        worker.finished.connect(self._on_eigen_finished)
        self._eig_worker = worker
        self._eig_thread = thread
        thread.start()

    @Slot(object)
    def _on_eigen_result(self, payload) -> None:
        flist = [float(f) for f in payload.get("freqs", [])]
        self._last_eigen_freqs = flist
        self._render_eigen_frequencies(
            flist, payload.get("spec_f"), payload.get("spec_v"))
        self._render_grid_eigen_markers(flist)  # mirror onto the per-sensor grid
        if "mode_shapes_ux" in payload:
            self._last_shape_data = payload
        method = self.method_combo.currentText()
        if self._show_shapes_check.isChecked():
            # Mode-shape view active: render shapes; status set inside the renderer.
            self._render_mode_shapes(self._last_shape_data or payload)
        elif flist:
            txt = ", ".join(f"f{i + 1}={f:.2f} Hz" for i, f in enumerate(flist))
            self._eig_status.setText(f"Eigen-frequencies ({method}): {txt}")
        else:
            self._eig_status.setText(f"Eigen-frequencies ({method}): none identified")

    @Slot(float)
    def _on_eigen_collecting(self, available_s: float) -> None:
        need = modal_id.MIN_DURATION_S
        self._eig_status.setText(
            f"Eigen-frequencies: collecting {available_s:.0f}/{need:.0f} s of data…")

    @Slot(str)
    def _on_eigen_failed(self, message: str) -> None:
        self._eig_status.setText(f"Eigen-frequencies: {message}")

    @Slot()
    def _on_eigen_finished(self) -> None:
        thread = self._eig_thread
        worker = self._eig_worker
        self._eig_worker = None
        self._eig_thread = None
        if thread is not None:
            thread.quit()
            thread.wait()
            thread.deleteLater()      # don't accumulate a QThread per 10 s cycle
        if worker is not None:
            worker.deleteLater()

    def _render_eigen_frequencies(self, freqs: Sequence[float],
                                  spec_f=None, spec_v=None) -> None:
        """Draw the identification-spectrum curve (FDD 1st singular value / FFT
        averaged PSD) plus the identified natural frequencies as distinct-coloured
        vertical lines. The curve is what visibly changes between FDD and FFT."""
        # Spectrum response curve, band-limited to [f_min, f_max] for a clean view.
        if spec_f is not None and spec_v is not None and len(spec_f) and len(spec_v):
            sf = np.asarray(spec_f, dtype=float)
            sv = np.asarray(spec_v, dtype=float)
            fmin = float(self._eig_fmin.value())
            fmax = float(self._eig_fmax.value())
            band = (sf >= fmin) & (sf <= fmax)
            if band.any():
                sf, sv = sf[band], sv[band]
            self._eig_curve.setData(sf, sv)
            x_hi = float(sf[-1]) if sf.size else (max(freqs) * 1.3 + 1.0 if freqs else 1.0)
        else:
            self._eig_curve.setData([], [])
            x_hi = max(freqs) * 1.3 + 1.0 if freqs else 1.0
        for item in self._eig_items:
            self._eig_plot.removeItem(item)
        self._eig_items.clear()
        for i, f in enumerate(freqs):
            color = _EIGEN_COLORS[i % len(_EIGEN_COLORS)]
            line = pg.InfiniteLine(
                pos=float(f), angle=90, pen=pg.mkPen(color, width=2.0),
                label=f"f{i + 1}={f:.2f} Hz",
                labelOpts={"position": 0.92, "color": color})
            self._eig_plot.addItem(line)
            self._eig_items.append(line)
        self._eig_plot.setXRange(0.0, x_hi, padding=0.02)

    def _render_grid_eigen_markers(self, freqs: Sequence[float]) -> None:
        """Overlay the identified mode frequencies onto every per-sensor cell of the
        Window-1 grid as thin coloured vertical lines. Because these come from the
        selected FDD/FFT identification, switching the method visibly moves the peaks
        marked on the individual sensors (the per-sensor spectrum itself is the same
        magnitude curve; only the marked structural modes change)."""
        # Remove previous markers (guard against cells destroyed by a layout rebuild).
        for line in self._grid_eig_lines:
            try:
                line.getViewBox().removeItem(line)
            except Exception:
                pass
        self._grid_eig_lines.clear()
        plots = list(getattr(self, "_psd_plots", {}).values())
        if not plots:
            return
        for plot in plots:
            for i, f in enumerate(freqs):
                color = _EIGEN_COLORS[i % len(_EIGEN_COLORS)]
                line = pg.InfiniteLine(
                    pos=float(f), angle=90,
                    pen=pg.mkPen(color, width=1.0, style=Qt.DashLine))
                plot.addItem(line)
                self._grid_eig_lines.append(line)

    # ---------------------------------------------- mode shapes (optional view)
    def _build_shape_placement(self) -> QWidget:
        """Compact sensor→floor placement: a floor count + one combo per sensor.

        Mode shapes are spatial, so they need to know which floor each sensor sits
        on (frequencies don't — see §7). Mirrors the Model Updating tab's mapping
        but local + lightweight.
        """
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Floors:"))
        self._shape_floors = QSpinBox()
        self._shape_floors.setRange(1, 20)
        self._shape_floors.setValue(3)
        self._shape_floors.valueChanged.connect(self._refresh_shape_floor_combos)
        row.addWidget(self._shape_floors)
        row.addSpacing(12)
        self._shape_combos: dict[int, QComboBox] = {}
        for sid in self._shape_sensor_ids:
            row.addWidget(QLabel(f"S{sid}→"))
            combo = QComboBox()
            combo.currentTextChanged.connect(lambda *_: self._launch_eigen_compute())
            self._shape_combos[sid] = combo
            row.addWidget(combo)
        row.addStretch()
        self._refresh_shape_floor_combos()
        return w

    def _refresh_shape_floor_combos(self) -> None:
        n = int(self._shape_floors.value())
        for i, (sid, combo) in enumerate(self._shape_combos.items()):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([str(s) for s in range(1, n + 1)])
            options = [str(s) for s in range(1, n + 1)]
            combo.setCurrentText(prev if prev in options else str(min(i + 1, n)))
            combo.blockSignals(False)

    def _shape_sensor_story_map(self) -> dict[int, int]:
        """{sensor_id: 1-based floor} from the placement combos."""
        out: dict[int, int] = {}
        for sid, combo in getattr(self, "_shape_combos", {}).items():
            if combo.count():
                out[sid] = int(combo.currentText())
        return out

    @Slot(bool)
    def _on_view_toggled(self, show_shapes: bool) -> None:
        """Swap the right panel between the frequencies view and the mode-shape
        view (pure GUI, G1)."""
        self._shape_placement_widget.setVisible(show_shapes)
        self._shape_glw.setVisible(show_shapes)
        self._eig_glw.setVisible(not show_shapes)
        self._right_title.setText(
            "Identified mode shapes" if show_shapes else "Identified natural frequencies")
        if show_shapes and self._last_shape_data is not None:
            self._render_mode_shapes(self._last_shape_data)

    def _render_mode_shapes(self, shape_data: dict) -> None:
        """Draw per-floor mode shapes: amplitude (x) vs floor (y), one coloured
        profile per mode, anchored at the ground (floor 0, amplitude 0)."""
        for item in self._shape_items:
            self._shape_plot.removeItem(item)
        self._shape_items.clear()
        shapes = shape_data.get("mode_shapes_ux") or {}
        n_story = int(shape_data.get("n_story", 0))
        signed = bool(shape_data.get("signed", True))
        full = bool(shape_data.get("full_coverage", False))
        # Floors that actually have a sensor (sorted); shape values map to these.
        coverage = [int(s) for s in shape_data.get("coverage_stories", [])]
        if not shapes or not coverage:
            self._eig_status.setText(
                "Mode shapes: need ≥1 sensor per measured floor — set the sensor→floor map.")
            return
        for k in sorted(shapes.keys(), key=lambda s: int(s)):
            vals = [float(v) for v in shapes[k]]
            if len(vals) != len(coverage):
                continue
            color = _EIGEN_COLORS[(int(k) - 1) % len(_EIGEN_COLORS)]
            # Ground anchor at floor 0 (fixed base); plot each value at its floor.
            xs = [0.0] + vals
            ys = [0] + coverage
            curve = self._shape_plot.plot(
                xs, ys, pen=pg.mkPen(color, width=2.0),
                symbol="o", symbolBrush=color, symbolSize=8, name=f"f{k}")
            self._shape_items.append(curve)
        self._shape_plot.setYRange(0, max(1, n_story), padding=0.1)
        kind = "signed (FDD)" if signed else "magnitude (FFT)"
        cov = "full coverage" if full else f"partial ({len(coverage)}/{n_story} floors)"
        self._eig_status.setText(f"Mode shapes — {kind}, {cov}, |max|=1 per mode")

    # --------------------------------------------------------------- internals
    @staticmethod
    def _channel_units(channel: str) -> str:
        ch = channel.lower()
        if ch in {"ax", "ay", "az"}:
            return "m/s²"
        if ch in {"gx", "gy", "gz"}:
            return "deg/s"
        return ""

    def _format_overlay_label(self, sensor_id: int, channel: str) -> str:
        units = self._channel_units(channel)
        base = f"S{sensor_id} {channel.upper()}"
        return f"{base} [{units}]" if units else base

    def _min_samples_required(self, window_s: float) -> int:
        """
        Return the minimum number of samples required to attempt an FFT.
        For debugging we are a bit more permissive: ~25% of the expected
        samples, but at least 4.
        """
        expected_rate = self._measured_rate_hz or self._device_rate_hz
        if expected_rate > 0.0:
            expected = expected_rate * window_s
            min_samples = max(4, int(expected * 0.25))
            logger.debug(
                "FftTab: min_samples_required window_s=%.3f expected_rate=%.2f -> %d",
                window_s,
                expected_rate,
                min_samples,
            )
            return min_samples
        logger.debug("FftTab: min_samples_required with unknown stream_rate; using 4")
        return 4

    def _decimate_signal_for_fft(
        self,
        times: np.ndarray,
        values: np.ndarray,
        target_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Downsample arrays to ~target_points to cap FFT cost."""
        if target_points <= 0 or values.size <= target_points:
            return times, values

        step = max(2, values.size // target_points)
        indices = np.arange(0, values.size, step, dtype=int)
        if indices[-1] != values.size - 1:
            indices = np.append(indices, values.size - 1)
        return times[indices], values[indices]

    def _window_signal(
        self,
        buf: Sequence[Tuple[float, float]] | RingBuffer[Tuple[float, float]],
        window_s: float,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        if buf is None:
            return None
        points = list(buf)
        if len(points) < 4:
            return None
        t_latest = points[-1][0]
        t_min = t_latest - window_s

        times = [t for (t, _v) in points if t >= t_min]
        values = [v for (t, v) in points if t >= t_min]
        if len(values) < 4 or times[-1] <= times[0]:
            return None

        times_arr = np.asarray(times, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        if values_arr.size > self._max_fft_samples:
            times_arr = times_arr[-self._max_fft_samples :]
            values_arr = values_arr[-self._max_fft_samples :]

        if values_arr.size > self._fft_decimation_target:
            times_arr, values_arr = self._decimate_signal_for_fft(
                times_arr, values_arr, self._fft_decimation_target
            )
            if times_arr.size < 2 or values_arr.size < 2:
                return None

        dt = times_arr[-1] - times_arr[0]
        if dt <= 0.0:
            return None
        sample_rate_hz = (len(times_arr) - 1) / dt if dt > 0 else 1.0
        if sample_rate_hz <= 0.0:
            fallback_rate = self._measured_rate_hz or self._device_rate_hz
            sample_rate_hz = fallback_rate if fallback_rate > 0.0 else 1.0
        return times_arr, values_arr, sample_rate_hz

    def _preprocess_signal(
        self,
        values: np.ndarray,
        sample_rate_hz: float,
    ) -> np.ndarray:
        signal = values.copy()
        if self.detrend_check.isChecked():
            signal = filters.detrend(signal)
        if self.lowpass_check.isChecked():
            cutoff = float(self.lowpass_cutoff.value())
            nyquist = 0.5 * sample_rate_hz
            if 0.0 < cutoff < nyquist:
                signal = filters.butter_lowpass(
                    signal,
                    cutoff_hz=cutoff,
                    sample_rate_hz=sample_rate_hz,
                )
        return signal

    def _ensure_fft_frequency_axis(self, sample_rate_hz: float | None = None) -> None:
        """Ensure the cached frequency axis matches the latest sampling rate."""
        if sample_rate_hz is None or sample_rate_hz <= 0.0:
            sample_rate_hz = self._device_rate_hz
        sample_rate_hz = float(sample_rate_hz) if sample_rate_hz and sample_rate_hz > 0 else 1.0
        if np.isclose(sample_rate_hz, self._fft_sample_rate_hz, rtol=1e-3):
            return
        self._fft_sample_rate_hz = sample_rate_hz
        self._fft_freqs = np.fft.rfftfreq(self._fft_size, 1.0 / self._fft_sample_rate_hz)
        self._fft_window = np.hanning(self._fft_size)
        for plot in self._psd_plots.values():
            self._apply_frequency_limits(plot)

    def _apply_frequency_limits(self, plot: "pg.PlotItem") -> None:
        max_freq = float(DEFAULT_MAX_FREQUENCY_HZ)
        if self._fft_freqs.size > 0:
            max_freq = min(max_freq, float(self._fft_freqs[-1]))
        if max_freq <= 0.0 or not np.isfinite(max_freq):
            max_freq = 1.0
        plot.setXRange(0.0, max_freq, padding=0.02)

    def _compute_fft_magnitude(self, signal: np.ndarray) -> np.ndarray:
        """Return FFT magnitudes for the most recent fft_size samples."""
        if signal.size == 0:
            return np.zeros_like(self._fft_freqs)
        window = signal[-self._fft_size :]
        if window.size < self._fft_size:
            padded = np.zeros(self._fft_size, dtype=float)
            if window.size > 0:
                padded[-window.size :] = window
            window = padded
        windowed = window * self._fft_window
        fft_vals = np.fft.rfft(windowed)
        return np.abs(fft_vals)

    def estimate_modal_frequencies(self, count: int = 3) -> list[float]:
        """Estimate the dominant modal frequencies from the current FFT input window."""
        if count <= 0 or self._is_record_only():
            return []

        data_buffer = self._active_stream_buffer()
        sensor_ids = self._resolve_sensor_ids(data_buffer)
        if self._sensor_selection is not None and self._sensor_selection.active_sensors:
            sensor_ids = [sid for sid in sensor_ids if sid in self._sensor_selection.active_sensors]
        if not sensor_ids:
            return []

        selection = self._sensor_selection
        if selection is not None and selection.active_channels:
            channels = list(selection.active_channels)
        else:
            channels = ["ax", "ay"]
        channels = [ch for ch in channels if ch in ("ax", "ay")]  # T11.1: structural axes only
        if not channels:
            return []

        window_s = float(self.window_spin.value())
        min_samples = self._min_samples_required(window_s)
        aggregate: np.ndarray | None = None
        used = 0

        for sensor_id in sensor_ids:
            for ch in channels:
                key = self._make_key(sensor_id, ch)
                timestamps, values = self._get_buffer_window(key, window_s=window_s, data_buffer=data_buffer)
                if self._sequence_length(values) < min_samples or self._sequence_length(timestamps) < 2:
                    continue

                points = list(zip(timestamps, values))
                prepared = self._window_signal(points, window_s)
                if prepared is None:
                    continue

                _times_arr, values_arr, sample_rate_hz = prepared
                signal = self._preprocess_signal(values_arr, sample_rate_hz)
                axis_sample_rate = self._device_rate_hz if self._device_rate_hz > 0.0 else sample_rate_hz
                self._ensure_fft_frequency_axis(axis_sample_rate)
                magnitude = self._compute_fft_magnitude(signal)
                if magnitude.size != self._fft_freqs.size or magnitude.size == 0:
                    continue

                if aggregate is None:
                    aggregate = np.zeros_like(magnitude, dtype=float)
                aggregate += np.asarray(magnitude, dtype=float)
                used += 1

        if aggregate is None or used <= 0:
            return []

        aggregate /= float(used)
        freqs = np.asarray(self._fft_freqs, dtype=float)
        if freqs.size < 3:
            return []

        min_freq_hz = 0.2
        max_freq_hz = min(float(DEFAULT_MAX_FREQUENCY_HZ), float(freqs[-1]))
        mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
        idx = np.where(mask)[0]
        if idx.size < 3:
            return []

        candidate_peaks: list[tuple[float, float]] = []
        min_sep_hz = max(0.5, freqs[1] - freqs[0])
        for i in idx[1:-1]:
            center = aggregate[i]
            if not np.isfinite(center) or center <= 0.0:
                continue
            if center >= aggregate[i - 1] and center >= aggregate[i + 1]:
                candidate_peaks.append((float(freqs[i]), float(center)))

        if not candidate_peaks:
            return []

        candidate_peaks.sort(key=lambda item: item[1], reverse=True)
        selected: list[float] = []
        for freq, _amp in candidate_peaks:
            if all(abs(freq - existing) >= min_sep_hz for existing in selected):
                selected.append(freq)
            if len(selected) >= count:
                break

        selected.sort()
        return selected

    def _on_fft_timer(self) -> None:
        if not debug_enabled():
            self._update_fft()
            return

        logger.debug("FftTab: _on_fft_timer tick")
        start = time.perf_counter()
        self._update_fft()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        alpha = 0.2
        if self._debug_fft_ema_ms <= 0.0:
            self._debug_fft_ema_ms = elapsed_ms
        else:
            self._debug_fft_ema_ms = (
                alpha * elapsed_ms + (1.0 - alpha) * self._debug_fft_ema_ms
            )
        now_perf = time.perf_counter()
        if now_perf - self._debug_fft_last_log >= 5.0:
            interval = self._timer.interval() if hasattr(self, "_timer") else 0
            logger.debug(
                "FftTab: redraw interval=%d ms ema≈%.2f ms",
                interval,
                self._debug_fft_ema_ms,
            )
            self._debug_fft_last_log = now_perf

    def _update_fft(self) -> None:
        self._update_mpu6050_fft()

    def _update_mpu6050_fft(self) -> None:
        if self._is_record_only():
            self._status_label.setText("Record-only mode: live FFT disabled.")
            return

        data_buffer = self._active_stream_buffer()

        sensor_ids = self._resolve_sensor_ids(data_buffer)
        if self._sensor_selection is not None and self._sensor_selection.active_sensors:
            sensor_ids = [
                sid for sid in sensor_ids if sid in self._sensor_selection.active_sensors
            ]
        if not sensor_ids:
            self._draw_waiting()
            return

        latest_ts = data_buffer.latest_timestamp()
        if latest_ts is None:
            self._draw_waiting()
            return

        if (
            not self._force_next_update
            and self._last_rendered_latest_ts is not None
            and latest_ts <= self._last_rendered_latest_ts
        ):
            return

        selection = self._sensor_selection
        if selection is not None and selection.active_channels:
            channels = list(selection.active_channels)
        else:
            channels = ["ax", "ay"]

        # Spectrum shows only the structural horizontal axes (T11.1): drop gyro/other
        # channels even if a global selection includes them, so the grid is 3x2 (ax, ay).
        channels = [ch for ch in channels if ch in ("ax", "ay")]
        if not channels:
            logger.warning("FftTab: channels list empty; falling back to ['ax', 'ay']")
            channels = ["ax", "ay"]

        window_s = float(self.window_spin.value())
        min_samples = self._min_samples_required(window_s)

        if not self._ensure_fft_layout(sensor_ids, channels):
            logger.debug("FftTab: _ensure_fft_layout returned False; skipping update")
            self._status_label.setText("Waiting for layout.")
            return

        stats_samples = None
        stats_fs = None
        have_data = False

        for sensor_id in sensor_ids:
            for ch in channels:
                key = self._make_key(sensor_id, ch)
                timestamps, values = self._get_buffer_window(
                    key,
                    window_s=window_s,
                    data_buffer=data_buffer,
                )
                if (
                    self._sequence_length(values) < min_samples
                    or self._sequence_length(timestamps) < 2
                ):
                    self._clear_line(sensor_id, ch)
                    continue

                if self._calibration_offsets is not None and self._sequence_length(values):
                    offset = self._calibration_offsets.offset_for(sensor_id, ch)
                    if offset != 0.0:
                        try:
                            values = np.asarray(values, dtype=float) - float(offset)
                        except Exception:
                            values = [float(v) - float(offset) for v in values]

                points = list(zip(timestamps, values))
                prepared = self._window_signal(points, window_s)
                if prepared is None:
                    self._clear_line(sensor_id, ch)
                    continue

                _times_arr, values_arr, sample_rate_hz = prepared
                signal = self._preprocess_signal(values_arr, sample_rate_hz)

                axis_sample_rate = (
                    self._device_rate_hz if self._device_rate_hz > 0.0 else sample_rate_hz
                )
                self._ensure_fft_frequency_axis(axis_sample_rate)
                magnitude = self._compute_fft_magnitude(signal)
                if magnitude.size == 0:
                    self._clear_line(sensor_id, ch)
                    continue

                self._update_fft_line(key, magnitude)
                have_data = True
                if stats_samples is None:
                    stats_samples = self._fft_size
                    stats_fs = axis_sample_rate

        if not have_data:
            logger.debug(
                "FftTab: no usable FFT data sensor_ids=%s channels=%s min_samples=%d stream_rate=%.2f",
                sensor_ids,
                channels,
                min_samples,
                self._measured_rate_hz or self._device_rate_hz,
            )
            self._status_label.setText("Waiting for data...")
            self._last_rendered_latest_ts = latest_ts
            self._force_next_update = False
            return

        self._last_rendered_latest_ts = latest_ts
        self._force_next_update = False
        if stats_samples is not None and stats_fs is not None:
            self._status_label.setText(
                f"Window: {window_s:.1f} s, FFT samples: {stats_samples}, fs≈{stats_fs:.1f} Hz"
            )

    def _apply_subplot_limits(
        self,
        sensor_ids: Sequence[int],
        channels: Sequence[str],
    ) -> tuple[list[int], list[str], bool, bool]:
        limited_sensors = list(sensor_ids)
        limited_channels = list(channels)
        if not limited_sensors or not limited_channels:
            return limited_sensors, limited_channels, False, False
        limit = self._max_subplots
        if not limit or limit <= 0:
            return limited_sensors, limited_channels, False, False
        total = len(limited_sensors) * len(limited_channels)
        if total <= limit:
            return limited_sensors, limited_channels, False, False

        trimmed_channels = False
        max_channels = max(1, limit // len(limited_sensors))
        if len(limited_channels) > max_channels:
            limited_channels = limited_channels[:max_channels]
            trimmed_channels = True
        trimmed_sensors = False
        max_sensors = max(1, limit // len(limited_channels))
        if len(limited_sensors) > max_sensors:
            limited_sensors = limited_sensors[:max_sensors]
            trimmed_sensors = True
        return limited_sensors, limited_channels, trimmed_channels, trimmed_sensors

    def _ensure_fft_layout(self, sensor_ids: Sequence[int], channels: Sequence[str]) -> bool:
        sensor_list = [int(s) for s in sensor_ids]
        channel_list = [str(ch) for ch in channels]
        if not sensor_list or not channel_list:
            return False

        original_sensor_count = len(sensor_list)
        original_channel_count = len(channel_list)
        (
            sensor_list,
            channel_list,
            trimmed_channels,
            trimmed_sensors,
        ) = self._apply_subplot_limits(sensor_list, channel_list)
        if not sensor_list or not channel_list:
            return False

        signature: tuple = ("grid", tuple(sensor_list), tuple(channel_list))
        should_log_limits = (
            (trimmed_channels or trimmed_sensors)
            and signature != self._current_layout
        )
        if should_log_limits:
            limit = self._max_subplots
            if trimmed_channels and original_channel_count > len(channel_list):
                logger.warning(
                    "FFT tab: reducing visible channels from %d to %d to honor max subplot limit (%s).",
                    original_channel_count,
                    len(channel_list),
                    limit,
                )
            if trimmed_sensors and original_sensor_count > len(sensor_list):
                logger.warning(
                    "FFT tab: reducing visible sensors from %d to %d to honor max subplot limit (%s).",
                    original_sensor_count,
                    len(sensor_list),
                    limit,
                )

        if signature == self._current_layout:
            return True

        self._current_layout = signature
        self._psd_plots.clear()
        self._psd_curves.clear()
        self._glw.clear()
        self._ensure_fft_frequency_axis()

        nrows = len(sensor_list)
        for row_idx, sensor_id in enumerate(sensor_list):
            for col_idx, ch in enumerate(channel_list):
                plot = self._glw.addPlot(row=row_idx, col=col_idx)
                plot.showGrid(x=True, y=True, alpha=0.3)
                units = self._channel_units(ch)
                title = f"S{sensor_id} {ch.upper()}"
                if units:
                    title = f"{title} [{units}]"
                plot.setTitle(title)
                if row_idx == nrows - 1:
                    plot.setLabel("bottom", "Frequency", units="Hz")
                if col_idx == 0:
                    plot.setLabel("left", "Magnitude")
                curve = plot.plot(self._fft_freqs, np.zeros_like(self._fft_freqs),
                                  pen=self._psd_pen)
                key = self._make_key(sensor_id, ch)
                self._psd_plots[key] = plot
                self._psd_curves[key] = curve
                self._apply_frequency_limits(plot)
        # The previous markers were destroyed by _glw.clear(); re-apply the last
        # identified modes onto the fresh cells so they survive a layout rebuild.
        self._grid_eig_lines.clear()
        if self._last_eigen_freqs:
            self._render_grid_eigen_markers(self._last_eigen_freqs)
        return True

    def _update_fft_line(self, key: SampleKey, magnitude: np.ndarray) -> None:
        curve = self._psd_curves.get(key)
        if curve is None:
            return

        if magnitude.size != self._fft_freqs.size:
            padded = np.zeros_like(self._fft_freqs)
            count = min(len(padded), magnitude.size)
            if count > 0:
                padded[:count] = magnitude[:count]
            magnitude = padded

        # pyqtgraph auto-ranges Y; setData updates both axes in one call.
        curve.setData(self._fft_freqs, magnitude)

    def _clear_line(self, sensor_id: int, channel: str) -> None:
        key = self._make_key(sensor_id, channel)
        curve = self._psd_curves.get(key)
        if curve is not None:
            curve.setData(self._fft_freqs, np.zeros_like(self._fft_freqs))

    def _clear_layout(self) -> None:
        self._psd_plots.clear()
        self._psd_curves.clear()
        self._grid_eig_lines.clear()  # destroyed by _glw.clear(); drop stale refs
        self._current_layout = None
        self._glw.clear()

    def _draw_waiting(self) -> None:
        self._clear_layout()
        self._status_label.setText("Waiting for data...")

    def _window_from_signals_tab(
        self,
        sensor_id: int,
        channel: str,
        window_s: float,
    ) -> tuple[Sequence[float], Sequence[float]] | None:
        signals_tab = self._signals_tab
        if signals_tab is None:
            return None
        getter = getattr(signals_tab, "get_time_series_window", None)
        if getter is None:
            return None
        try:
            times, values = getter(sensor_id, channel, window_s)
        except Exception:
            return None
        if times is None or values is None:
            return None
        if self._sequence_length(times) < 2 or self._sequence_length(values) < 2:
            return None
        return times, values

    def _sequence_length(self, seq: object) -> int:
        """Return len(seq) while tolerating numpy arrays."""
        if seq is None:
            return 0
        try:
            return len(seq)  # type: ignore[arg-type]
        except TypeError:
            size = getattr(seq, "size", None)
            if size is None:
                return 0
            try:
                return int(size)
            except (TypeError, ValueError):
                return 0

    def _sensor_ids_from_signals_tab(self) -> list[int]:
        signals_tab = self._signals_tab
        if signals_tab is None:
            return []
        getter = getattr(signals_tab, "live_sensor_ids", None)
        if getter is None:
            return []
        try:
            ids = list(getter())
        except Exception:
            return []
        normalized: list[int] = []
        for sensor_id in ids:
            try:
                normalized.append(int(sensor_id))
            except (TypeError, ValueError):
                continue

        deduped: list[int] = []
        for sensor_id in normalized:
            if sensor_id not in deduped:
                deduped.append(sensor_id)
        return sorted(deduped)

    def _resolve_sensor_ids(self, data_buffer: StreamingDataBuffer) -> list[int]:
        sensor_ids = self._sensor_ids_from_signals_tab()
        if sensor_ids:
            return sensor_ids
        raw_ids = data_buffer.get_sensor_ids()
        return sorted(raw_ids, key=str)
