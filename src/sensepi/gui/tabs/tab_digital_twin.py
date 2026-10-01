"""Synchronized physical/numerical validation experiment.

The tab intentionally reuses three existing sources of truth:

* calibrated structural parameters from :class:`ModelUpdatingTab`;
* live MPU data from :class:`RecorderController`;
* the currently selected sonification model/settings from :class:`SonificationTab`.

No shaker-base accelerometer is assumed.  Synchronization therefore uses an
experiment start timestamp plus an optional response-onset correction.  The
estimated lag is always displayed and saved rather than hidden.
"""
from __future__ import annotations

import copy
import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...dataio import modal_session_loader as msl
from ...analysis import sensor_layout as slayout
from ...digital_twin import decisions as twin_decisions
from ..widgets.decision_panel import DecisionPanel
from ..widgets.wireframe import LiveStructureView
from ...digital_twin.comparison import compute_comparison_metrics, fft_amplitude


class _Twin3DCanvas(FigureCanvas):
    """3D undeformed/deformed OpenSees model view."""

    _STRUCTURE_BLUE = "#123B6D"
    _REFERENCE_GREY = "#A9B4C0"
    _BACKGROUND = "#F7F9FC"

    def __init__(self, parent: QWidget | None = None) -> None:
        self.fig = Figure(figsize=(6.0, 6.0))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        self.fig.patch.set_facecolor(self._BACKGROUND)
        self._defo_lines: list[Any] = []
        self._empty()

    def _empty(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor(self._BACKGROUND)
        self.ax.set_title("Calibrated numerical model", fontweight="semibold")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.grid(False)
        self.ax.view_init(elev=25, azim=-70)
        self.fig.tight_layout(pad=0.6)
        self.draw_idle()

    def initialize_model(self, modal_data: dict[str, Any] | None) -> None:
        self.ax.clear()
        self.ax.set_facecolor(self._BACKGROUND)
        self.ax.set_title("Calibrated numerical model — live response", fontweight="semibold")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.grid(False)
        self.ax.view_init(elev=25, azim=-70)
        self._defo_lines = []
        if not modal_data:
            self.draw_idle()
            return

        ctx = modal_data.get("ctx", {})
        vis_elems = list(ctx.get("vis_elems", []))
        vis_nodes = list(ctx.get("vis_nodes", []))
        node_xyz = dict(ctx.get("node_xyz", {}))
        if not vis_elems or not vis_nodes or not node_xyz:
            self.draw_idle()
            return

        xs = [float(node_xyz[n][0]) for n in vis_nodes]
        ys = [float(node_xyz[n][1]) for n in vis_nodes]
        zs = [float(node_xyz[n][2]) for n in vis_nodes]
        xmid = 0.5 * (min(xs) + max(xs))
        ymid = 0.5 * (min(ys) + max(ys))
        zmid = 0.5 * (min(zs) + max(zs))
        half = 0.55 * max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-9)
        self.ax.set_xlim(xmid - half, xmid + half)
        self.ax.set_ylim(ymid - half, ymid + half)
        self.ax.set_zlim(zmid - half, zmid + half)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        for n1, n2 in vis_elems:
            x1, y1, z1 = node_xyz[n1]
            x2, y2, z2 = node_xyz[n2]
            self.ax.plot(
                [x1, x2], [y1, y2], [z1, z2],
                color=self._REFERENCE_GREY, linestyle="--", linewidth=0.9, alpha=0.55,
            )
        for _ in vis_elems:
            line, = self.ax.plot(
                [], [], [], color=self._STRUCTURE_BLUE, linewidth=3.0, alpha=0.98,
            )
            self._defo_lines.append(line)
        self.fig.tight_layout(pad=0.6)
        self.draw_idle()

    def update_frame(self, frame: dict[str, Any]) -> None:
        for line, segment in zip(self._defo_lines, frame.get("deformed_segments", [])):
            (x1, y1, z1), (x2, y2, z2) = segment
            line.set_data([x1, x2], [y1, y2])
            line.set_3d_properties([z1, z2])
        self.draw_idle()


class _TwinPlotsCanvas(FigureCanvas):
    """Selected-story time-history overlay plus physical/numerical FFT comparison.

    The response artists are created once and then updated in-place.  Clearing
    and rebuilding Matplotlib axes on every sample is intentionally avoided;
    that was the main source of flicker and GUI lag in the first version.
    """

    _MAX_PLOT_POINTS = 3000

    def __init__(self, parent: QWidget | None = None) -> None:
        self.fig = Figure(figsize=(7.2, 6.0))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax_input = self.fig.add_subplot(2, 1, 1)
        self.ax_response = self.fig.add_subplot(2, 1, 2)
        self._ground_t = np.array([], dtype=float)
        self._ground_a = np.array([], dtype=float)
        self._input_full_line = None
        self._physical_line = None
        self._numerical_line = None
        self._physical_fft_line = None
        self._numerical_fft_line = None
        self._response_story: int | None = None
        # Blitting state: a cached picture of the axes furniture, reused while
        # only the curves change.  See _blit_draw.
        self._backgrounds: dict | None = None
        self._bg_stale = True
        self._blit_ok = True
        self._last_limits: tuple | None = None
        self._duration = 1.0
        self._time_ymax = 1.0e-6
        self._fft_ymax = 1.0e-6
        self.initialize()

    @staticmethod
    def _thin(t, y, max_points: int):
        t = np.asarray(t, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = min(t.size, y.size)
        t, y = t[:n], y[:n]
        if n <= max_points:
            return t, y
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        return t[idx], y[idx]

    def initialize(self, ground_t=None, ground_a=None) -> None:
        self._ground_t = np.asarray(ground_t if ground_t is not None else [], dtype=float)
        self._ground_a = np.asarray(ground_a if ground_a is not None else [], dtype=float)
        self._duration = float(self._ground_t[-1]) if self._ground_t.size else 1.0
        base_peak = (
            float(np.nanmax(np.abs(self._ground_a)))
            if self._ground_a.size else 0.0
        )
        self._time_ymax = max(2.0 * base_peak, 1.0e-6)
        self._fft_ymax = 1.0e-6
        self._response_story = None
        self._bg_stale = True
        self._configure_response(story=1)
        self.draw_idle()

    def _configure_response(self, *, story: int) -> None:
        self.ax_input.clear()
        self.ax_response.clear()
        self._response_story = int(story)

        self._input_full_line, = self.ax_input.plot(
            [], [], color="black", linestyle="--", linewidth=0.9,
            label="Base excitation", zorder=1
        )
        if self._ground_t.size:
            gt, ga = self._thin(
                self._ground_t, self._ground_a, self._MAX_PLOT_POINTS
            )
            self._input_full_line.set_data(gt, ga)
        # The four curves below are redrawn every frame, so they are marked
        # animated and kept OUT of the cached background.  The base-excitation
        # line is not: it never changes, and leaving it in the background is
        # what keeps it on screen after Arm, before any frame has arrived.
        self._physical_line, = self.ax_input.plot(
            [], [], linewidth=1.4, label="Physical model", zorder=3, animated=True
        )
        self._numerical_line, = self.ax_input.plot(
            [], [], linewidth=1.5, color="red", label="Numerical model (OpenSees)",
            zorder=4, animated=True
        )
        self.ax_input.set_title(
            f"Story {story} — Base excitation and model response"
        )
        self.ax_input.set_xlabel("Time (s)")
        self.ax_input.set_ylabel("Acceleration (m/s²)")
        self.ax_input.set_xlim(0.0, max(self._duration, 1.0e-9))
        self.ax_input.set_ylim(-self._time_ymax, self._time_ymax)
        self.ax_input.grid(True, alpha=0.25)
        self.ax_input.legend(loc="upper right", fontsize=9)

        self._physical_fft_line, = self.ax_response.plot(
            [], [], linewidth=1.4, label="Physical model", animated=True
        )
        self._numerical_fft_line, = self.ax_response.plot(
            [], [], linewidth=1.5, color="red", label="Numerical model (OpenSees)",
            animated=True
        )
        self.ax_response.set_title(f"Story {story} — FFT comparison")
        self.ax_response.set_xlabel("Frequency (Hz)")
        self.ax_response.set_ylabel("Amplitude (m/s²)")
        self.ax_response.set_xlim(0.0, 25.0)
        self.ax_response.set_ylim(0.0, self._fft_ymax)
        self.ax_response.grid(True, alpha=0.25)
        self.ax_response.legend(loc="upper right", fontsize=9)

        self.fig.tight_layout(pad=1.5)
        self._bg_stale = True

    def update_comparison(
        self,
        *,
        current_time: float,
        story: int,
        measured_t,
        measured_y,
        numerical_t,
        numerical_y,
        refresh_fft: bool = True,
    ) -> None:
        if self._response_story != int(story):
            self._configure_response(story=int(story))

        assert self._physical_line is not None
        assert self._numerical_line is not None
        assert self._physical_fft_line is not None
        assert self._numerical_fft_line is not None

        mt, my = self._thin(measured_t, measured_y, self._MAX_PLOT_POINTS)
        nt, ny = self._thin(numerical_t, numerical_y, self._MAX_PLOT_POINTS)
        self._physical_line.set_data(mt, my)
        self._numerical_line.set_data(nt, ny)
        if self._input_full_line is not None and self._ground_t.size:
            gt, ga = self._thin(
                self._ground_t, self._ground_a, self._MAX_PLOT_POINTS
            )
            self._input_full_line.set_data(gt, ga)
        self.ax_input.set_xlim(0.0, max(self._duration, current_time, 1.0e-9))

        ymax = float(np.nanmax(np.abs(self._ground_a))) if self._ground_a.size else 0.0
        if my.size:
            ymax = max(ymax, float(np.nanmax(np.abs(my))))
        if ny.size:
            ymax = max(ymax, float(np.nanmax(np.abs(ny))))
        if np.isfinite(ymax) and ymax > self._time_ymax:
            self._time_ymax = max(ymax * 1.15, 1.0e-6)
        self.ax_input.set_ylim(-self._time_ymax, self._time_ymax)

        if refresh_fft:
            fm, am = fft_amplitude(measured_t, measured_y)
            fn, an = fft_amplitude(numerical_t, numerical_y)
            fm, am = self._thin(fm, am, self._MAX_PLOT_POINTS)
            fn, an = self._thin(fn, an, self._MAX_PLOT_POINTS)
            self._physical_fft_line.set_data(fm, am)
            self._numerical_fft_line.set_data(fn, an)

            max_f = max(float(fm[-1]) if fm.size else 0.0, float(fn[-1]) if fn.size else 0.0)
            self.ax_response.set_xlim(0.0, min(25.0, max_f) if max_f > 0 else 25.0)
            fft_ymax = max(
                float(np.max(am)) if am.size else 0.0,
                float(np.max(an)) if an.size else 0.0,
            )
            if fft_ymax > self._fft_ymax:
                self._fft_ymax = max(fft_ymax * 1.15, 1.0e-6)
            self.ax_response.set_ylim(0.0, self._fft_ymax)

        # A limit change invalidates the cached picture; nothing else does.
        limits = (self.ax_input.get_xlim(), self.ax_input.get_ylim(),
                  self.ax_response.get_xlim(), self.ax_response.get_ylim())
        if limits != self._last_limits:
            self._last_limits = limits
            self._bg_stale = True

        # No clear(), no tight_layout(), and no artist recreation here.
        self._blit_draw()

    def resizeEvent(self, event):  # noqa: N802  (Qt naming)
        self._bg_stale = True
        super().resizeEvent(event)

    def _blit_draw(self) -> None:
        """Redraw only the curves, over a cached picture of the axes.

        A full redraw of this figure costs about 85 ms -- more than the 20 fps
        frame budget on its own, which is what made the traces judder and stole
        the time the 3D model needed to animate smoothly.  The grid, ticks,
        labels and legend do not change between frames, so they are rendered
        once and reused; only the four data curves are redrawn, at about 5 ms.

        Any backend that refuses to blit falls back to a normal redraw, so the
        picture is always correct even if it is slow.
        """
        artists = [a for a in (self._physical_line, self._numerical_line,
                               self._physical_fft_line, self._numerical_fft_line)
                   if a is not None]
        if not self._blit_ok or not artists:
            self.draw_idle()
            return
        axes = (self.ax_input, self.ax_response)
        try:
            if self._bg_stale or self._backgrounds is None:
                # draw() skips animated artists, so the cached background holds
                # the furniture and the static base-excitation line only.
                self.draw()
                self._backgrounds = {ax: self.copy_from_bbox(ax.bbox) for ax in axes}
                self._bg_stale = False
            for ax in axes:
                self.restore_region(self._backgrounds[ax])
            for artist in artists:
                artist.axes.draw_artist(artist)
            for ax in axes:
                self.blit(ax.bbox)
        except Exception:
            self._blit_ok = False
            self._backgrounds = None
            self.draw_idle()


class _CalibrationWorker(QObject):
    """Identify the structure from sensor data, then calibrate the model.

    Runs the whole chain off the GUI thread (G1/G4): identification, story
    mapping, and the least-squares fit. Deliberately calls the *pure*
    ``run_calibration`` rather than going through the Model Updating tab — that
    is what lets this tab own its own calibration instead of borrowing one.
    """

    log = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, params: dict[str, Any], session) -> None:
        super().__init__()
        self._params = copy.deepcopy(params)
        self._session = session

    @Slot()
    def run(self) -> None:
        try:
            from ..tabs.tab_model_updating import _build_sensor_exp_dict
            params = self._params
            self.log.emit("Identifying modes from the live sensors…\n")
            exp_dict, result, story_data = _build_sensor_exp_dict(self._session, params)
            if exp_dict is None:
                self.error.emit(
                    f"Identification failed: {getattr(result, 'message', 'no modes')}")
                return
            freqs = list(exp_dict.get("frequencies_hz", []))
            self.log.emit("  frequencies: "
                          + ", ".join(f"{f:.3f}" for f in freqs) + " Hz\n")
            coverage = list(getattr(story_data, "coverage_stories", []))
            self.log.emit(f"  measured storeys: {coverage or 'none'}\n")

            out_base = Path(params["project_dir"]) / "output"
            out_base.mkdir(parents=True, exist_ok=True)
            from opensees_model_updating.calibration.calibrator import (
                run_calibration)

            exp_data = {
                "freqs": freqs,
                "modes": exp_dict.get("mode_shapes_ux") or {},
                "use_mode_shapes": bool(params.get("use_mode_shapes", True))
                and bool(exp_dict.get("mode_shapes_ux")),
                "mode_shapes_available": bool(exp_dict.get("mode_shapes_ux")),
                "source_file": "live sensors (Digital Twin tab)",
                "raw_data": exp_dict,
                "n_modes_used": int(params.get("nCalibModes", len(freqs) or 1)),
            }
            self.log.emit("Calibrating…\n")
            calib_result, calibrated = run_calibration(
                params, exp_data, show_info=False)
            self.log.emit(
                f"  {'converged' if calib_result.success else 'did not converge'} "
                f"after {calib_result.nfev} evaluations\n")
            self.finished.emit({
                "designed": copy.deepcopy(params),
                "calibrated": calibrated,
                "exp_data": exp_data,
                "success": bool(calib_result.success),
                "message": str(calib_result.message),
            })
        except Exception as exc:
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")


class _DigitalTwinWorker(QObject):
    """Build the calibrated model, wait armed, then run in wall-clock time."""

    prepared = Signal(object)
    frame = Signal(object)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, setup: dict[str, Any]) -> None:
        super().__init__()
        self._setup = copy.deepcopy(setup)
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._realtime_offset_s = 0.0

    def request_start(self, realtime_offset_s: float = 0.0) -> None:
        self._realtime_offset_s = max(0.0, float(realtime_offset_s))
        self._start_event.set()

    def request_stop(self) -> None:
        self._stop_event.set()
        self._start_event.set()

    @Slot()
    def run(self) -> None:
        try:
            # No os.chdir. It is process-global: for as long as an experiment
            # ran it moved the working directory for the live acquisition thread
            # and the chorus worker too.
            out_base = Path(self._setup["project_dir"]) / "output"
            out_base.mkdir(parents=True, exist_ok=True)
            from opensees_model_updating.analysis.modal import extract_modal_results
            from ...digital_twin.opensees_runner import (
                load_ground_motion,
                run_digital_twin_stream,
            )

            params = copy.deepcopy(self._setup["params"])
            modal_data = extract_modal_results(
                params, normalize_modes=True, show_info=bool(params.get("show_info", False))
            )
            ground_t, ground_a = load_ground_motion(params)
            self.prepared.emit({
                "modal_data": modal_data,
                "ground_t": ground_t,
                "ground_a": ground_a,
                "duration_s": float(ground_t[-1]) if ground_t.size else 0.0,
            })

            while not self._start_event.wait(0.05):
                if self._stop_event.is_set():
                    self.finished.emit({"stopped": True, "armed_only": True})
                    return
            if self._stop_event.is_set():
                self.finished.emit({"stopped": True, "armed_only": True})
                return

            result = run_digital_twin_stream(
                params,
                modal_data,
                show_info=bool(params.get("show_info", False)),
                frame_callback=self.frame.emit,
                target_fps=20.0,
                realtime=True,
                realtime_offset_s=self._realtime_offset_s,
                sfac_anim=20.0,
                stop_requested=self._stop_event.is_set,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")


class DigitalTwinExperimentTab(QWidget):
    """Run a calibrated OpenSees model beside the live shaking-table test."""

    def __init__(
        self,
        *,
        recorder_controller,
        model_updating_tab,
        sonification_tab,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = recorder_controller
        self._model_tab = model_updating_tab
        self._sonification_tab = sonification_tab
        self._thread: QThread | None = None
        self._worker: _DigitalTwinWorker | None = None
        self._setup: dict[str, Any] | None = None
        self._prepared = False
        self._running = False
        self._sensor_t0: float | None = None
        self._sensor_start_click_t: float | None = None
        self._baseline: dict[int, float] = {}
        self._baseline_noise: dict[int, float] = {}
        self._auto_lag: float | None = None
        self._waiting_for_onset = False
        self._trigger_latency_s = 0.0
        self._last_frame: dict[str, Any] | None = None
        self._last_saved_dir: Path | None = None
        self._num_t_live: list[float] = []
        self._num_a_abs_live: list[list[float]] = []
        self._frame_dirty = False
        self._last_analysis_refresh_wall = 0.0
        # Self-containment: the model definition is snapshotted ONCE, when
        # Calibrate is pressed, so an experiment under way cannot be changed by
        # later edits on another tab.
        self._model_snapshot: dict[str, Any] | None = None
        self._onset_source = ""
        self._calibrated_params: dict[str, Any] | None = None
        self._calibration_source = ""
        self._decisions = None
        self._sensor_mapping: dict | None = None
        self._calib_thread: QThread | None = None
        self._calib_worker = None
        self._build_ui()

        # Render at a fixed GUI rate.  Worker frames are cheap to receive; if
        # several arrive while Matplotlib is busy, only the latest 3D state is
        # drawn.  Numerical integration remains independent of GUI speed.
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(50)  # about 20 frames/s
        self._render_timer.timeout.connect(self._render_latest_frame)
        self._render_timer.start()
        self._live_view.start()
        self._update_controls()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        controls = QGroupBox("Experiment control")
        row = QHBoxLayout(controls)
        self._calibrate_btn = QPushButton("1. Calibrate")
        self._calibrate_btn.setToolTip(
            "Identify the structure from the live sensors and calibrate the model "
            "here, in this tab. Takes a one-time snapshot of the model definition, "
            "so the experiment is not disturbed by later edits elsewhere.")
        self._arm_btn = QPushButton("2. Arm numerical model")
        self._start_btn = QPushButton("3. Start experiment")
        self._stop_btn = QPushButton("Stop")
        self._sonify_btn = QPushButton("▶ Sonify")
        self._sonify_btn.setCheckable(True)
        self._story_combo = QComboBox()
        self._auto_align = QCheckBox("Auto-sync shaker onset")
        self._auto_align.setChecked(True)
        self._auto_align.setToolTip(
            "When enabled, Start waits for sustained sensor motion clearly above the "
            "pre-start baseline, then starts the numerical model automatically. Small "
            "background movement is treated as baseline rather than as shaker onset."
        )
        self._lag_spin = QDoubleSpinBox()
        self._lag_spin.setRange(-3.0, 3.0)
        self._lag_spin.setDecimals(3)
        self._lag_spin.setSingleStep(0.01)
        self._lag_spin.setSuffix(" s")
        self._lag_spin.setToolTip(
            "Manual mode only. Positive = physical response started later; the displayed "
            "physical time is shifted left by this amount."
        )
        row.addWidget(self._calibrate_btn)
        row.addWidget(self._arm_btn)
        row.addWidget(self._start_btn)
        row.addWidget(self._stop_btn)
        row.addSpacing(10)
        row.addWidget(QLabel("Story:"))
        row.addWidget(self._story_combo)
        row.addWidget(self._auto_align)
        row.addWidget(QLabel("Lag:"))
        row.addWidget(self._lag_spin)
        row.addStretch(1)
        row.addWidget(self._sonify_btn)
        root.addWidget(controls)

        self._setup_label = QLabel(
            "Calibrate the model, or send the latest Continuous Update to Model, then "
            "start the live sensor stream and Arm. Experiment input source: Model "
            "Updating → Analysis (ground-motion file/preset, dt, factor and damping)."
        )
        self._setup_label.setWordWrap(True)
        root.addWidget(self._setup_label)

        # 2x2: the two comparison plots on the left (one canvas, two subplots),
        # the numerical model top-right, and the PHYSICAL structure bottom-right
        # with the decision panel beside it. The two right-hand panels are the
        # two halves of the twin — model and reality — so they sit in a column.
        splitter = QSplitter(Qt.Horizontal)
        self._plots = _TwinPlotsCanvas()
        self._model_3d = _Twin3DCanvas()
        self._live_view = LiveStructureView(self)
        self._live_view.set_controller(self._controller)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self._model_3d)

        lower = QWidget()
        lower_row = QHBoxLayout(lower)
        lower_row.setContentsMargins(0, 0, 0, 0)
        lower_row.addWidget(self._live_view, stretch=3)
        self._decision_panel = DecisionPanel(parent=self)
        lower_row.addWidget(self._decision_panel, stretch=2)
        right.addWidget(lower)
        right.setSizes([420, 420])

        splitter.addWidget(self._plots)
        splitter.addWidget(right)
        splitter.setSizes([620, 730])
        root.addWidget(splitter, stretch=1)

        metrics = QGroupBox("Live comparison")
        mrow = QHBoxLayout(metrics)
        self._metric_rms = QLabel("RMS error: —")
        self._metric_nrmse = QLabel("NRMSE: —")
        self._metric_corr = QLabel("Correlation: —")
        self._metric_peak = QLabel("Peak ratio: —")
        self._metric_lag = QLabel("Sync lag: —")
        self._sensor_label = QLabel("Sensor: —")
        for widget in (
            self._metric_rms,
            self._metric_nrmse,
            self._metric_corr,
            self._metric_peak,
            self._metric_lag,
            self._sensor_label,
        ):
            mrow.addWidget(widget)
        mrow.addStretch(1)
        root.addWidget(metrics)

        self._status = QLabel(
            "Ready. Recommended workflow: stream while structure is still → Arm → press Start → start the shaker. "
            "With Auto-sync enabled, the numerical model waits for the detected shaker onset."
        )
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._calibrate_btn.clicked.connect(self._calibrate)
        self._arm_btn.clicked.connect(self._arm)
        self._start_btn.clicked.connect(self._start_experiment)
        self._stop_btn.clicked.connect(self._stop_experiment)
        self._sonify_btn.clicked.connect(self._toggle_sonification)
        self._story_combo.currentIndexChanged.connect(self._refresh_plot)
        self._auto_align.toggled.connect(self._on_auto_align_changed)
        self._lag_spin.valueChanged.connect(self._refresh_plot)

    def _snapshot_axis_series(self, axis: str) -> dict[int, list[tuple[float, float]]]:
        """Read the existing thread-safe modal buffer without changing RecorderController."""
        modal_buffer = getattr(self._controller, "_modal_buffer", None)
        snapshot = getattr(modal_buffer, "snapshot_series", None)
        if not callable(snapshot):
            raise RuntimeError("Live modal sensor buffer is not available in RecorderController.")
        return snapshot(axis)

    def _build_setup(self) -> dict[str, Any]:
        """Assemble the experiment from the model this tab calibrated.

        Order of preference is the whole point of the three-step workflow:
        the calibration made HERE (step 1) drives the experiment; Model
        Updating's last calibration is only a fallback for a user who chose to
        calibrate there instead. Before this fix the in-tab result fed only the
        decision lights and Arm silently required a Model Updating calibration,
        which made step 1 decorative.
        """
        if self._model_tab.is_busy():
            raise ValueError(
                "Model Updating is currently busy. Finish or stop it before arming "
                "the Digital Twin experiment.")

        if self._calibrated_params:
            calibrated = self._calibrated_params
            self._calibration_source = "this tab"
        else:
            calibrated = self._model_tab.calibration_snapshot()
            self._calibration_source = "Model Updating"
        if not calibrated:
            raise ValueError(
                "No calibrated model is available. Press 1. Calibrate here, or run "
                "Calibrate in the Model Updating tab first.")

        current = self._model_definition()
        params = copy.deepcopy(calibrated)
        # Only experiment-specific values are read from the current Model Updating UI.
        # The calibrated structural snapshot itself is not changed.
        for key in ("gmFile", "dtGM", "gmFactor", "zeta", "show_info"):
            if key in current:
                params[key] = copy.deepcopy(current[key])
        params["run_transient"] = True

        sensor = current
        n_story = int(params.get("nStory", sensor.get("nStory", 1)))
        story_map = {
            int(sid): int(story)
            for sid, story in dict(sensor["sensor_story_map"]).items()
            if 1 <= int(story) <= n_story
        }
        if not story_map:
            raise ValueError(
                "No sensor-to-story mapping is available. Configure it in Model Updating first."
            )

        project_dir = Path(str(current["project_dir"]))
        return {
            "params": params,
            "project_dir": str(project_dir),
            "sensor_axis": str(sensor["sensor_axis"]),
            "sensor_story_map": story_map,
            "base_sensor_id": slayout.layout_from_mapping(
                self._sensor_mapping).base_sensor_id,
            "nStory": n_story,
            "calibrated_E": float(params["E"]),
            "calibrated_floor_masses": [float(v) for v in params["floor_masses"]],
            "calibration_source": self._calibration_source,
        }

    def _model_definition(self) -> dict[str, Any]:
        """The model definition, from the snapshot if one was taken.

        Taking the snapshot at Calibrate is what makes this tab self-contained:
        once calibration has run, the experiment is driven by the model it was
        calibrated against, and later edits on the Model Updating tab cannot
        change an experiment that is already under way. Before the first
        calibration there is nothing to fall back on but the live tab, and this
        goes through its one public method rather than its private internals.
        """
        if self._model_snapshot is not None:
            return copy.deepcopy(self._model_snapshot)
        return self._model_tab.model_definition_snapshot()

    def _active_sonification_model(self):
        """Return the currently selected existing sonification sub-tab."""
        tabs = getattr(self._sonification_tab, "_tabs", None)
        current_widget = getattr(tabs, "currentWidget", None)
        if callable(current_widget):
            return current_widget()
        return getattr(self._sonification_tab, "chorus_tab", None)

    @Slot()
    def on_stream_started(self) -> None:
        self._update_controls()

    @Slot()
    def on_stream_stopped(self) -> None:
        if self._running or self._prepared:
            self._stop_experiment()
        self._update_controls()

    def _update_controls(self) -> None:
        streaming = False
        try:
            streaming = bool(self._controller.is_streaming())
        except Exception:
            pass
        busy = self._thread is not None
        self._arm_btn.setEnabled(streaming and not busy)
        self._start_btn.setEnabled(self._prepared and not self._running)
        self._stop_btn.setEnabled(busy)
        self._lag_spin.setEnabled(not self._auto_align.isChecked())

    def _populate_stories(self, n_story: int) -> None:
        # Mark the storeys that no sensor covers. With four sensors on up to six
        # floors most of this list can be unmeasured, and selecting one of those
        # used to give an empty physical trace with no explanation.
        measured = set()
        if self._setup:
            measured = {int(v) for v in self._setup["sensor_story_map"].values()}
        else:
            # Before Arm, the selector follows the Settings map so it is usable
            # alongside Calibrate and the live wireframe.
            measured = set(slayout.layout_from_mapping(self._sensor_mapping).story_map.values())
        current = self._story_combo.currentData()
        self._story_combo.blockSignals(True)
        self._story_combo.clear()
        first_measured = None
        for story in range(1, int(n_story) + 1):
            if story in measured:
                self._story_combo.addItem(str(story), story)
                if first_measured is None:
                    first_measured = self._story_combo.count() - 1
            else:
                self._story_combo.addItem(f"{story} — no sensor", story)
        if current is not None and current in measured:
            idx = self._story_combo.findData(current)
            self._story_combo.setCurrentIndex(idx if idx >= 0 else 0)
        elif measured:
            # Default to the highest MEASURED storey — the top of the structure
            # moves most, and it is guaranteed to have data to compare.
            top = max(measured)
            idx = self._story_combo.findData(top)
            self._story_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self._story_combo.setCurrentIndex(max(0, self._story_combo.count() - 1))
        self._story_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Step 1 — calibrate, here
    # ------------------------------------------------------------------
    @Slot()
    def _calibrate(self) -> None:
        if self._calib_thread is not None or self._thread is not None:
            return
        try:
            # ONE snapshot, taken now. From here on the experiment is driven by
            # the model it was calibrated against.
            params = self._model_tab.model_definition_snapshot()
            layout = slayout.layout_from_mapping(self._sensor_mapping)
            if not layout.is_valid:
                raise ValueError(
                    "No sensor placement is set. Open Settings → Sensor placement "
                    "map and place at least one sensor on a floor.")
            capture = getattr(self._controller, "snapshot_modal_capture", None)
            if capture is None:
                raise ValueError("No live capture source is available.")
            session = capture(axis=layout.channel,
                              last_seconds=float(params.get("sensor_window_s", 30.0)),
                              target_fs=params.get("sensor_target_fs"))
            if not getattr(session, "success", True) or session.data.size == 0:
                raise ValueError(
                    "No sensor data yet. Start the live stream and let it run for "
                    f"at least {params.get('sensor_window_s', 30.0):.0f} s.")
            # The floor-0 sensor measures the shaker INPUT, so it is not a
            # response and must not enter an output-only identification.
            ids = list(session.sensor_ids)
            rows = layout.response_rows(ids)
            if not rows:
                raise ValueError(
                    "None of the streaming sensors is placed on a floor.")
            if len(rows) < len(ids):
                session = msl.sliced_session(session, rows)
            params["sensor_n_modes"] = layout.max_modes(
                int(params.get("sensor_n_modes", 3)))
        except Exception as exc:
            QMessageBox.warning(self, "Cannot calibrate", str(exc))
            self._status.setText(str(exc))
            return

        self._model_snapshot = params
        self._status.setText("Calibrating from the live sensors…")
        self._calibrate_btn.setEnabled(False)

        worker = _CalibrationWorker(params, session)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._on_calibration_log)
        worker.finished.connect(self._on_calibrated)
        worker.error.connect(self._on_calibration_error)
        self._calib_worker = worker
        self._calib_thread = thread
        thread.start()

    @Slot(str)
    def _on_calibration_log(self, text: str) -> None:
        self._status.setText(text.strip() or self._status.text())

    @Slot(object)
    def _on_calibrated(self, payload) -> None:
        self._calibrated_params = payload.get("calibrated")
        designed = payload.get("designed") or {}
        self._decisions = twin_decisions.decide(designed, self._calibrated_params)
        self._decision_panel.set_decisions(self._decisions, source="this calibration")
        note = "" if payload.get("success") else " (did not converge)"
        self._status.setText(
            f"Calibrated{note}. {self._decisions.summary()} "
            f"This calibration will drive the experiment — now Arm the numerical model.")
        self._clear_calibration_worker()

    @Slot(str)
    def _on_calibration_error(self, message: str) -> None:
        QMessageBox.critical(self, "Calibration failed", message)
        self._status.setText(message.splitlines()[0] if message else "Calibration failed.")
        self._clear_calibration_worker()

    def _clear_calibration_worker(self) -> None:
        thread, self._calib_thread = self._calib_thread, None
        worker, self._calib_worker = self._calib_worker, None
        if thread is not None:
            thread.quit()
            thread.wait()
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()
        self._calibrate_btn.setEnabled(True)
        self._update_controls()

    # ---------------------------------------------- placement (from Settings)
    def apply_sensor_map(self, mapping) -> None:
        """Adopt the placement map from Settings. This tab owns no picker."""
        if hasattr(mapping, "to_mapping"):
            mapping = mapping.to_mapping()
        self._sensor_mapping = dict(mapping) if isinstance(mapping, dict) else None
        layout = slayout.layout_from_mapping(self._sensor_mapping)
        if hasattr(self, "_live_view"):
            self._live_view.apply_sensor_map(self._sensor_mapping)
        if hasattr(self, "_story_combo") and not self._setup and layout.is_valid:
            self._populate_stories(layout.n_floors)

    # ------------------------------------------------- live physical wireframe
    @Slot()
    def _arm(self) -> None:
        if self._thread is not None:
            return
        try:
            setup = self._build_setup()
            series = self._snapshot_axis_series(setup["sensor_axis"])
            mapped = [sid for sid in setup["sensor_story_map"] if sid in series and series[sid]]
            if not mapped:
                raise ValueError(
                    "No live samples are available for the sensors mapped to stories. "
                    "Start the live stream and check the sensor→story mapping in Model Updating."
                )
        except Exception as exc:
            QMessageBox.warning(self, "Cannot arm Digital Twin", str(exc))
            self._status.setText(str(exc))
            return

        self._setup = setup
        self._populate_stories(setup["nStory"])
        self._auto_lag = None
        self._sensor_t0 = None
        self._sensor_start_click_t = None
        self._baseline = {}
        self._baseline_noise = {}
        self._waiting_for_onset = False
        self._trigger_latency_s = 0.0
        self._last_frame = None
        self._num_t_live = []
        self._num_a_abs_live = []
        self._frame_dirty = False
        self._last_analysis_refresh_wall = 0.0
        self._prepared = False
        self._running = False

        worker = _DigitalTwinWorker(setup)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.prepared.connect(self._on_prepared)
        worker.frame.connect(self._on_frame)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self._worker, self._thread = worker, thread
        self._status.setText("Preparing the calibrated OpenSees model…")
        self._update_controls()
        thread.start()

    @Slot(object)
    def _on_prepared(self, payload: dict[str, Any]) -> None:
        self._prepared = True
        self._plots.initialize(payload.get("ground_t"), payload.get("ground_a"))
        self._model_3d.initialize_model(payload.get("modal_data"))
        params = (self._setup or {}).get("params", {})
        duration = float(payload.get("duration_s", 0.0))
        self._setup_label.setText(
            f"Calibrated model ready · input: {Path(str(params.get('gmFile', ''))).name} · "
            f"dt={float(params.get('dtGM', 0.0)):.4g} s · factor={float(params.get('gmFactor', 1.0)):.4g} · "
            f"duration≈{duration:.2f} s · source: Model Updating → Analysis. "
            "No base sensor is used: this input is assumed to be the shaker input."
        )
        self._status.setText(
            "ARMED. Keep the structure in its normal quiet condition, then press Start experiment and start the shaker. "
            "With Auto-sync enabled, the numerical model waits for clear sustained shaker motion before it starts."
        )
        self._update_controls()

    def _capture_sensor_origin(self) -> None:
        if not self._setup:
            raise ValueError("Experiment setup is missing.")
        axis = self._setup["sensor_axis"]
        series = self._snapshot_axis_series(axis)
        map_ = self._setup["sensor_story_map"]
        sids = [int(sid) for sid in map_ if int(sid) in series and series[int(sid)]]
        if not sids:
            raise ValueError("No mapped live sensor data is available at experiment start.")

        latest = [float(series[sid][-1][0]) for sid in sids]
        origin = min(latest)  # latest time shared by all mapped sensors
        self._sensor_start_click_t = origin
        self._sensor_t0 = origin
        self._baseline = {}
        self._baseline_noise = {}
        # Amplitude of the excitation being run, so the trigger level follows
        # the experiment instead of a fixed number.  Read from the canvas,
        # which already holds the ground motion from Arm.
        ground_a = getattr(self._plots, "_ground_a", None)
        ga = (np.asarray(ground_a, dtype=float).reshape(-1)
              if ground_a is not None else np.zeros(0, dtype=float))
        ga = ga[np.isfinite(ga)]
        self._input_rms = float(np.sqrt(np.mean(ga * ga))) if ga.size else 0.0

        # Use a longer pre-start window than before and retain its robust noise
        # level.  Auto-sync compares post-Start motion against this baseline, so
        # normal low-level table/structure movement is not treated as shaker onset.
        for sid in sids:
            pairs = series[sid]
            vals = np.asarray(
                [float(v) for t, v in pairs if origin - 1.5 <= float(t) <= origin],
                dtype=float,
            )
            if vals.size == 0:
                vals = np.asarray([float(pairs[-1][1])], dtype=float)
            center = float(np.median(vals))
            # Keep the quiet-window RMS: that is what the onset test compares
            # against.  The old 4-sigma spike test assumed Gaussian background,
            # but this rig's background carries short bursts far above 4 sigma
            # and those were firing the trigger on a still structure.
            resid = vals - center
            rms = float(np.sqrt(np.mean(resid * resid))) if resid.size else 0.0
            if not np.isfinite(rms):
                rms = 0.0
            self._baseline[sid] = center
            self._baseline_noise[sid] = max(rms, 1.0e-9)

    def _detect_shaker_onset(self) -> tuple[float, float] | None:
        """Return ``(sensor_onset_timestamp, detection_delay_s)`` when motion is clear.

        Detection is referenced to the pre-Start baseline and requires sustained
        activity.  With multiple mapped sensors, at least two must agree.  This
        intentionally rejects isolated bumps and the small ambient motion that
        previously caused premature onset alignment.
        """
        if not self._setup or self._sensor_start_click_t is None:
            return None

        series = self._snapshot_axis_series(self._setup["sensor_axis"])
        # Prefer the BASE sensor when the placement provides one. It sits on the
        # shaker, so it measures the excitation itself; every other sensor
        # measures the structure's *response*, which by construction starts later
        # than the input — and that delay is part of what this experiment exists
        # to measure. Detecting the start from a response therefore builds the
        # very lag it is trying to observe into the synchronisation.
        base_sid = self._setup.get("base_sensor_id")
        if base_sid is not None and int(base_sid) in series and series[int(base_sid)]:
            mapped = [int(base_sid)]
            self._onset_source = f"base sensor S{int(base_sid)} (measures the input)"
        else:
            mapped = [
                int(sid) for sid in self._setup["sensor_story_map"]
                if int(sid) in series and series[int(sid)]
            ]
            self._onset_source = (
                f"{len(mapped)} structural sensor(s) — no base sensor is placed, "
                f"so the start is inferred from the response")
        if not mapped:
            return None

        candidates: list[float] = []
        latest_times: list[float] = []
        for sid in mapped:
            pairs = [
                (float(t), float(v)) for t, v in series[sid]
                if float(t) >= self._sensor_start_click_t
            ]
            if len(pairs) < 8:
                continue
            t = np.asarray([p[0] for p in pairs], dtype=float)
            y = np.asarray(
                [p[1] - self._baseline.get(sid, 0.0) for p in pairs], dtype=float
            )
            dt_values = np.diff(t)
            dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0.0)]
            if dt_values.size == 0:
                continue
            dt = float(np.median(dt_values))
            # Tuned on 170 recorded runs from this rig: with the structure still
            # the 0.5 s moving RMS never reached 0.23 m/s2, while real shaking
            # never fell below 0.88, so the trigger sits between the two.
            window_s, hold_s, background_factor, input_fraction = 0.5, 0.25, 6.0, 0.45
            win = max(4, int(round(window_s / max(dt, 1.0e-6))))
            hold = max(2, int(round(hold_s / max(dt, 1.0e-6))))
            if y.size < win + hold:
                continue
            # Sustained ENERGY, not one excursion: a shaker keeps feeding the
            # structure, while a knock or a footstep is gone within the window.
            energy = np.convolve(y * y, np.ones(win) / win, mode="valid")
            rms = np.sqrt(np.maximum(energy, 0.0))
            threshold = max(
                background_factor * float(self._baseline_noise.get(sid, 0.0)),
                input_fraction * float(getattr(self, "_input_rms", 0.0)),
            )
            if not np.isfinite(threshold) or threshold <= 0.0:
                continue
            active = (rms >= threshold).astype(int)
            if active.size < hold:
                continue
            sustained = np.convolve(active, np.ones(hold, dtype=int), mode="valid") >= hold
            idx = np.flatnonzero(sustained)
            if idx.size:
                candidates.append(float(t[int(idx[0])]))
            latest_times.append(float(t[-1]))

        needed = 1 if len(mapped) == 1 else min(2, len(mapped))
        if len(candidates) < needed or not latest_times:
            return None

        onset = float(np.median(sorted(candidates)[:needed]))
        latest_common = float(min(latest_times))
        delay = max(0.0, latest_common - onset)
        return onset, delay

    def _poll_shaker_trigger(self) -> None:
        if not self._waiting_for_onset or self._worker is None:
            return
        detected = self._detect_shaker_onset()
        if detected is None:
            return

        onset, delay = detected
        self._waiting_for_onset = False
        self._sensor_t0 = onset
        self._auto_lag = 0.0
        self._trigger_latency_s = float(delay)
        self._lag_spin.blockSignals(True)
        self._lag_spin.setValue(0.0)
        self._lag_spin.blockSignals(False)
        self._worker.request_start(realtime_offset_s=self._trigger_latency_s)
        self._status.setText(
            f"SHAKER ONSET DETECTED — numerical model started automatically. "
            f"Initial {self._trigger_latency_s:.3f} s detection delay is being caught up so "
            "the physical and numerical clocks remain synchronized."
        )

    @Slot()
    def _start_experiment(self) -> None:
        if not self._prepared or self._worker is None or self._running:
            return
        try:
            self._capture_sensor_origin()
        except Exception as exc:
            QMessageBox.warning(self, "Cannot start experiment", str(exc))
            return

        self._auto_lag = None
        self._trigger_latency_s = 0.0
        self._running = True
        if self._auto_align.isChecked():
            self._lag_spin.blockSignals(True)
            self._lag_spin.setValue(0.0)
            self._lag_spin.blockSignals(False)
            self._waiting_for_onset = True
            self._status.setText(
                "WAITING FOR SHAKER — start the shaker now. The numerical model will start "
                "automatically only after sustained sensor motion rises clearly above the "
                "pre-start baseline."
            )
        else:
            self._waiting_for_onset = False
            self._worker.request_start()
            self._status.setText(
                "RUNNING — manual synchronization mode. The numerical model started at the "
                "button press; use Lag if a manual time shift is needed."
            )
        self._update_controls()

    def _selected_story(self) -> int:
        data = self._story_combo.currentData()
        return int(data) if data is not None else 1

    def _story_sensor_ids(self, story: int) -> list[int]:
        if not self._setup:
            return []
        return sorted(
            int(sid) for sid, mapped_story in self._setup["sensor_story_map"].items()
            if int(mapped_story) == int(story)
        )

    def _physical_story_signal(self, story: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
        if not self._setup or self._sensor_t0 is None:
            return np.array([]), np.array([]), []
        series = self._snapshot_axis_series(self._setup["sensor_axis"])
        sids = [sid for sid in self._story_sensor_ids(story) if sid in series and series[sid]]
        arrays: list[tuple[np.ndarray, np.ndarray]] = []
        for sid in sids:
            pairs = [(float(t), float(v)) for t, v in series[sid] if float(t) >= self._sensor_t0]
            if len(pairs) < 2:
                continue
            t = np.asarray([p[0] - self._sensor_t0 for p in pairs], dtype=float)
            y = np.asarray([p[1] - self._baseline.get(sid, 0.0) for p in pairs], dtype=float)
            arrays.append((t, y))
        if not arrays:
            return np.array([]), np.array([]), sids
        if len(arrays) == 1:
            return arrays[0][0], arrays[0][1], sids

        lo = max(float(t[0]) for t, _ in arrays)
        hi = min(float(t[-1]) for t, _ in arrays)
        if hi <= lo:
            return arrays[0][0], arrays[0][1], sids
        ref_t = max(arrays, key=lambda item: item[0].size)[0]
        grid = ref_t[(ref_t >= lo) & (ref_t <= hi)]
        vals = [np.interp(grid, t, y) for t, y in arrays]
        return grid, np.mean(np.vstack(vals), axis=0), sids

    def _numerical_story_signal(self, frame: dict[str, Any], story: int) -> tuple[np.ndarray, np.ndarray]:
        # A finished result contains the exact full-resolution history.  Live
        # frames are intentionally compact, so their history is accumulated on
        # the GUI side instead of being retransmitted on every animation frame.
        if "t_hist" in frame and "floor_a_abs_hist" in frame:
            t = np.asarray(frame.get("t_hist", []), dtype=float)
            all_a = np.asarray(frame.get("floor_a_abs_hist", []), dtype=float)
        else:
            t = np.asarray(self._num_t_live, dtype=float)
            all_a = np.asarray(self._num_a_abs_live, dtype=float)
        idx = int(story) - 1
        if all_a.ndim != 2 or idx < 0 or idx >= all_a.shape[1]:
            return t[:0], np.array([])
        n = min(t.size, all_a.shape[0])
        return t[:n], all_a[:n, idx]

    def _alignment_lag(self, mt, my, nt, ny) -> float:
        # Auto-sync now synchronizes the *experiment start* itself instead of
        # estimating a response lag later from a short, possibly noisy record.
        # That preserves genuine physical-vs-model phase differences.
        if self._auto_align.isChecked():
            return 0.0
        return float(self._lag_spin.value())

    @Slot(object)
    def _on_frame(self, frame: dict[str, Any]) -> None:
        # This slot intentionally does no Matplotlib drawing.  It only receives
        # compact data from the worker; the QTimer below performs drawing at a
        # fixed rate so queued GUI work cannot accumulate.
        self._last_frame = frame
        t_chunk = np.asarray(frame.get("t_chunk", []), dtype=float).reshape(-1)
        a_chunk = np.asarray(frame.get("floor_a_abs_chunk", []), dtype=float)
        if t_chunk.size and a_chunk.ndim == 2:
            n = min(t_chunk.size, a_chunk.shape[0])
            self._num_t_live.extend(t_chunk[:n].tolist())
            self._num_a_abs_live.extend(a_chunk[:n].tolist())
        self._frame_dirty = True

    @Slot()
    def _render_latest_frame(self) -> None:
        self._poll_shaker_trigger()
        frame = self._last_frame
        if not frame or not self._frame_dirty:
            return
        self._frame_dirty = False
        self._model_3d.update_frame(frame)
        self._refresh_plot()

    @Slot()
    def _refresh_plot(self, *_) -> None:
        frame = self._last_frame
        if not frame:
            return
        story = self._selected_story()
        mt, my, sids = self._physical_story_signal(story)
        nt, ny = self._numerical_story_signal(frame, story)
        lag = self._alignment_lag(mt, my, nt, ny)
        mt_aligned = mt - lag
        now = time.perf_counter()
        story_changed = getattr(self._plots, "_response_story", None) != int(story)
        refresh_analysis = story_changed or (now - self._last_analysis_refresh_wall >= 0.25)
        self._plots.update_comparison(
            current_time=float(frame.get("time", 0.0)),
            story=story,
            measured_t=mt_aligned,
            measured_y=my,
            numerical_t=nt,
            numerical_y=ny,
            refresh_fft=refresh_analysis,
        )
        if refresh_analysis:
            self._last_analysis_refresh_wall = now
            metrics = compute_comparison_metrics(mt_aligned, my, nt, ny)
            self._metric_rms.setText(
                f"RMS error: {metrics.rms_error:.3g} m/s²" if np.isfinite(metrics.rms_error) else "RMS error: —"
            )
            self._metric_nrmse.setText(
                f"NRMSE: {metrics.nrmse_percent:.1f}%" if np.isfinite(metrics.nrmse_percent) else "NRMSE: —"
            )
            self._metric_corr.setText(
                f"Correlation: {metrics.correlation:.3f}" if np.isfinite(metrics.correlation) else "Correlation: —"
            )
            self._metric_peak.setText(
                f"Peak ratio P/N: {metrics.peak_ratio:.3f}" if np.isfinite(metrics.peak_ratio) else "Peak ratio P/N: —"
            )
            if self._auto_align.isChecked():
                self._metric_lag.setText(
                    f"Sync: sensor-triggered · catch-up {self._trigger_latency_s:.3f} s"
                )
            else:
                self._metric_lag.setText(f"Manual lag: {lag:+.3f} s")
            axis = (self._setup or {}).get("sensor_axis", "")
            self._sensor_label.setText(
                f"Sensor: {','.join(map(str, sids)) if sids else '—'} ({axis})"
            )

    @Slot(bool)
    def _on_auto_align_changed(self, enabled: bool) -> None:
        self._auto_lag = None
        self._lag_spin.setEnabled(not enabled)
        self._refresh_plot()

    @Slot()
    def _stop_experiment(self) -> None:
        self._waiting_for_onset = False
        if self._worker is not None:
            self._worker.request_stop()
        self._status.setText("Stopping Digital Twin experiment…")

    @Slot(object)
    def _on_finished(self, result: dict[str, Any]) -> None:
        self._waiting_for_onset = False
        self._running = False
        self._prepared = False
        if self._last_frame and not result.get("armed_only"):
            try:
                self._last_saved_dir = self._save_experiment(result)
                self._status.setText(f"Experiment finished. Results saved to {self._last_saved_dir}")
            except Exception as exc:
                self._status.setText(f"Experiment finished, but saving failed: {exc}")
        else:
            self._status.setText("Digital Twin experiment stopped.")
        self._update_controls()

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._waiting_for_onset = False
        self._running = False
        self._prepared = False
        self._status.setText("Digital Twin experiment failed.")
        QMessageBox.critical(self, "Digital Twin experiment failed", message)
        self._update_controls()

    def _clear_worker(self) -> None:
        self._thread = None
        self._worker = None
        self._update_controls()

    def _save_experiment(self, result: dict[str, Any]) -> Path:
        if not self._setup:
            raise RuntimeError("Missing setup")
        # Results of THIS tab go to the twin category of the output root, not
        # under the model workspace (which is where OpenSees' own recorders
        # write). AppPaths.twin_output existed and nothing wrote to it.
        from ...config.app_config import AppPaths
        root = AppPaths().twin_output
        run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            run_dir / "numerical_response.npz",
            t_hist=np.asarray(result.get("t_hist", []), dtype=float),
            floor_u_hist=np.asarray(result.get("floor_u_hist", []), dtype=float),
            floor_a_rel_hist=np.asarray(result.get("floor_a_rel_hist", []), dtype=float),
            floor_a_abs_hist=np.asarray(result.get("floor_a_abs_hist", []), dtype=float),
            ground_accel_hist=np.asarray(result.get("ground_accel_hist", []), dtype=float),
        )
        story_metrics = {}
        for story in range(1, int(self._setup["nStory"]) + 1):
            mt, my, sids = self._physical_story_signal(story)
            nt, ny = self._numerical_story_signal(result, story)
            lag = float(self._auto_lag or self._lag_spin.value())
            mt_aligned = mt - lag
            if mt.size:
                np.savetxt(
                    run_dir / f"physical_story_{story}.csv",
                    np.column_stack([mt, mt_aligned, my]),
                    delimiter=",",
                    header="time_raw_s,time_aligned_s,accel_m_s2",
                    comments="",
                )
            m = compute_comparison_metrics(mt_aligned, my, nt, ny)
            story_metrics[str(story)] = {
                "sensor_ids": sids,
                "rms_error_m_s2": m.rms_error,
                "nrmse_percent": m.nrmse_percent,
                "correlation": m.correlation,
                "peak_ratio_physical_over_numerical": m.peak_ratio,
            }
        params = self._setup["params"]
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "assumption": "No base accelerometer: shaker input is assumed equal to the OpenSees ground-motion command.",
            "alignment_method": "sensor-triggered shaker onset" if self._auto_align.isChecked() else "manual lag",
            "sync_lag_s": 0.0 if self._auto_align.isChecked() else float(self._lag_spin.value()),
            "trigger_detection_latency_s": float(self._trigger_latency_s),
            "sensor_axis": self._setup["sensor_axis"],
            "sensor_story_map": self._setup["sensor_story_map"],
            "gmFile": str(params.get("gmFile", "")),
            "dtGM": float(params.get("dtGM", 0.0)),
            "gmFactor": float(params.get("gmFactor", 1.0)),
            "zeta": float(params.get("zeta", 0.0)),
            "calibrated_E": float(self._setup["calibrated_E"]),
            "calibrated_floor_masses": self._setup["calibrated_floor_masses"],
            "story_metrics": story_metrics,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
        return run_dir

    def _set_sonification_button(self, *, checked: bool, text: str) -> None:
        self._sonify_btn.blockSignals(True)
        self._sonify_btn.setChecked(bool(checked))
        self._sonify_btn.setText(text)
        self._sonify_btn.blockSignals(False)

    def _verify_sonification_audio(self, model) -> None:
        """Report the original Chorus tab's silent-audio mode clearly.

        Bioacoustic Chorus intentionally continues running without an audio
        backend, which is useful for its visual panels but confusing from the
        Digital Twin tab because the Play button then appears successful while
        no sound can be heard.  This check does not modify the sonification
        implementation; it only exposes its existing status.
        """
        if model is None or not self._sonify_btn.isChecked():
            return
        worker = getattr(model, "_worker", None)
        if worker is None:
            self._set_sonification_button(checked=False, text="▶ Sonify")
            QMessageBox.information(
                self,
                "Sonification",
                "Sonification did not start. Make sure the live stream is running, "
                "then check the Sonification tab for its status.",
            )
            return

        status = getattr(model, "_status_extra", {}) or {}
        silent = bool(status.get("silent", getattr(worker, "_silent", False)))
        if silent:
            self._sonify_btn.setText("■ Stop sonify (SILENT)")
            QMessageBox.warning(
                self,
                "Sonification audio unavailable",
                "Bioacoustic Chorus is running in SILENT mode, so its analysis/visuals "
                "can run but Windows receives no audio.\n\n"
                "The project treats the audio backend as optional. In the SAME Python "
                "environment used to launch SensePi, install it with:\n\n"
                "    python -m pip install sounddevice\n\n"
                "Then restart SensePi. If sounddevice is already installed, open the "
                "Sonification tab and check its status; Windows may not be able to open "
                "the selected/default output device.",
            )

    @Slot(bool)
    def _toggle_sonification(self, checked: bool) -> None:
        model = self._active_sonification_model()
        if checked:
            if model is None:
                self._set_sonification_button(checked=False, text="▶ Sonify")
                return

            # Prefer the public wrapper when present; fall back to the original
            # Bioacoustic Chorus Play action so this remains compatible with the
            # shared project's unmodified sonification files.
            start_public = getattr(model, "start_playback", None)
            if callable(start_public):
                started = bool(start_public())
            else:
                start = getattr(model, "_on_start", None)
                if not callable(start):
                    self._set_sonification_button(checked=False, text="▶ Sonify")
                    QMessageBox.information(
                        self,
                        "Sonification",
                        "The currently selected Sonification model has no Play action. "
                        "Select Bioacoustic Chorus in the Sonification tab.",
                    )
                    return
                start()
                started = getattr(model, "_worker", None) is not None

            if not started:
                self._set_sonification_button(checked=False, text="▶ Sonify")
                QMessageBox.information(
                    self,
                    "Sonification",
                    "Sonification could not be started. Check that the live stream is "
                    "running and review the Sonification tab.",
                )
                return

            self._sonify_btn.setText("■ Stop sonify")
            # The audio device is opened on the Chorus worker thread, so wait a
            # moment before checking whether it fell back to silent mode.
            QTimer.singleShot(900, lambda m=model: self._verify_sonification_audio(m))
        else:
            if model is not None:
                stop_public = getattr(model, "stop_playback", None)
                if callable(stop_public):
                    stop_public()
                else:
                    stop = getattr(model, "_on_stop", None)
                    if callable(stop):
                        stop()
            self._set_sonification_button(checked=False, text="▶ Sonify")

    def shutdown(self) -> None:
        # Stop the display timers first: they poll the controller, and polling a
        # controller that is being torn down is how a clean exit turns into a
        # traceback on the way out.
        if getattr(self, "_live_view", None) is not None:
            self._live_view.stop()
        timer = getattr(self, "_render_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        calib = getattr(self, "_calib_thread", None)
        if calib is not None and calib.isRunning():
            calib.quit()
            calib.wait(2500)
        if self._worker is not None:
            self._worker.request_stop()
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.wait(2500)
