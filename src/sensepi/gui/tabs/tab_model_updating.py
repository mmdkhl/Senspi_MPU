from __future__ import annotations

import copy
import contextlib
import importlib.util
import io
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QPointF, QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPen, QTextCursor
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


REQUIRED_MODULES = ("openseespy", "opsvis")

# Input files bundled with the opensees_model_updating package
_OPENSEES_INPUT_DIR = Path(__file__).resolve().parents[3] / "opensees_model_updating" / "input"

# Default experimental modal data — mirrors experimental_modal_data.json.
# These are used to pre-populate the manual-input fields on first open so
# the user always sees sensible starting values rather than zeros.
_DEFAULT_EXP_FREQUENCIES: list[float] = [2.32, 6.5, 9.1]
_DEFAULT_EXP_MODE_SHAPES: list[list[float]] = [
    [ 0.30,  0.75,  1.00],   # mode 1 (story 1, 2, 3)
    [-1.00,  0.10,  0.95],   # mode 2
    [ 1.00, -0.85,  0.30],   # mode 3
]


def _default_workspace_dir() -> Path:
    return Path(__file__).resolve().parents[4]  # repo root


@dataclass
class _CalibrationState:
    available: bool = False
    input_signature: str | None = None
    calibrated_params: dict[str, Any] | None = None
    uncalibrated_response: dict[str, Any] | None = None
    prior_freqs: list[float] | None = None
    prior_periods: list[float] | None = None
    prior_mode_shapes: list[list[float]] | None = None


@dataclass
class _WorkerRequest:
    action: str
    project_dir: Path
    params: dict[str, Any]
    calibration_state: _CalibrationState


class _StdoutBuffer(io.StringIO):
    def __init__(self, emit) -> None:
        super().__init__()
        self._emit = emit

    def write(self, text: str) -> int:
        if text:
            self._emit(text)
        return len(text)

    def flush(self) -> None:
        return None


class _StructurePreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 400)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._n_story = 3
        self._lx = 0.245
        self._ly = 0.230
        self._columns: dict[int, list[bool]] = {}
        self._orientations: dict[int, list[str]] = {}

    def set_model(
        self,
        *,
        n_story: int,
        lx: float,
        ly: float,
        columns: dict[int, list[bool]],
        orientations: dict[int, list[str]],
    ) -> None:
        self._n_story = max(1, int(n_story))
        self._lx = max(0.001, float(lx))
        self._ly = max(0.001, float(ly))
        self._columns = copy.deepcopy(columns)
        self._orientations = copy.deepcopy(orientations)
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        w = float(self.width())
        h = float(self.height())
        n_show = max(1, min(20, self._n_story))
        ratio = max(0.85, min(1.15, self._lx / self._ly))
        top_margin = 34.0
        bottom_margin = 58.0
        story_h = max(8.0, min(42.0, (h - top_margin - bottom_margin) / n_show))
        base_w = max(84.0, min(w * 0.42, 220.0 * ratio))
        depth_x = min(62.0, max(38.0, base_w * 0.45))
        depth_y = min(44.0, max(28.0, depth_x * 0.70))
        x0 = max(28.0, (w - base_w - depth_x) / 2.0 - 10.0)
        y0 = h - bottom_margin

        corners = [
            ("C1", 0, QPointF(x0, y0)),
            ("C2", 1, QPointF(x0 + base_w, y0)),
            ("C3", 2, QPointF(x0 + base_w + depth_x, y0 - depth_y)),
            ("C4", 3, QPointF(x0 + depth_x, y0 - depth_y)),
        ]
        colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

        def up(point: QPointF, level: float) -> QPointF:
            return QPointF(point.x(), point.y() - story_h * level)

        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.setBrush(QColor("#f8fafc"))
        for level in range(n_show + 1):
            pts = [up(corners[i][2], level) for i in [0, 1, 2, 3]]
            painter.drawPolygon(pts)

        for label, index, base in corners:
            color = QColor(colors[index])
            for story in range(1, n_show + 1):
                present = self._columns.get(story, [True, True, True, True])[index]
                orient = self._orientations.get(story, ["Weak axis"] * 4)[index]
                pen = QPen(color if present else QColor("#cbd5e1"), 7 if orient == "Strong axis" else 4)
                if not present:
                    pen.setStyle(Qt.DashLine)
                    pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(up(base, story - 1), up(base, story))
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor("#ffffff"))
            for level in range(n_show + 1):
                p = up(base, level)
                painter.drawEllipse(p, 4.0, 4.0)
            p = up(base, max(0.45, n_show * 0.15))
            painter.setPen(QPen(color, 1))
            painter.drawText(int(p.x() - 18), int(max(18.0, p.y() - 8)), label)

        painter.setPen(QPen(QColor("#1f2d3d"), 2))
        ax0 = QPointF(w - 86.0, h - 22.0)
        painter.drawLine(ax0, QPointF(ax0.x() + 42.0, ax0.y()))
        painter.drawText(int(ax0.x() + 48), int(ax0.y() + 4), "X")
        painter.drawLine(ax0, QPointF(ax0.x() + 30.0, ax0.y() - 28.0))
        painter.drawText(int(ax0.x() + 36), int(ax0.y() - 28), "Y")
        painter.drawLine(ax0, QPointF(ax0.x(), ax0.y() - 42.0))
        painter.drawText(int(ax0.x() - 4), int(ax0.y() - 48), "Z")


class _MassLegend(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._lx = 0.245
        self._ly = 0.230

    def set_dimensions(self, lx: float, ly: float) -> None:
        self._lx = max(0.001, float(lx))
        self._ly = max(0.001, float(ly))
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        w = float(self.width())
        h = float(self.height())
        ratio = self._lx / self._ly
        plan_w = min(w * 0.55, 220.0)
        plan_h = min(plan_w / ratio, h * 0.48, 92.0)
        plan_w = plan_h * ratio
        left = (w - plan_w) / 2.0
        top = 44.0
        right = left + plan_w
        bottom = top + plan_h
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0

        painter.setPen(QPen(QColor("#1f2d3d"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(int(left), int(top), int(plan_w), int(plan_h))
        painter.setBrush(QColor("#1f2d3d"))
        painter.drawEllipse(QPointF(cx, cy), 4.0, 4.0)
        painter.drawText(int(cx - 48), int(cy - 10), "Center mass")

        corners = [
            (left, bottom, "C1", -24, 22),
            (right, bottom, "C2", 10, 22),
            (right, top, "C3", 10, -10),
            (left, top, "C4", -24, -10),
        ]
        for x, y, label, dx, dy in corners:
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QPointF(x, y), 7.0, 7.0)
            painter.drawText(int(x + dx), int(y + dy), label)

        painter.setPen(QPen(QColor("#1f2d3d"), 2))
        axis_y = min(h - 18.0, bottom + 38.0)
        painter.drawLine(QPointF(left - 48.0, axis_y), QPointF(left, axis_y))
        painter.drawText(int(left + 8), int(axis_y + 4), "X / shaking")
        painter.drawLine(QPointF(left - 48.0, axis_y), QPointF(left - 48.0, axis_y - 44.0))
        painter.drawText(int(left - 54), int(axis_y - 50), "Y")
        painter.drawText(int(w / 2.0 - 114), 22, "Additional mass placement labels")


class _ScaledImageLabel(QLabel):
    """QLabel that scales its pixmap to fill the available area while keeping aspect ratio."""

    def __init__(self, placeholder: str = "", parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self._raw: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)

    def set_figure(self, png_bytes: bytes | None) -> None:
        if not png_bytes:
            self._raw = None
            self.setPixmap(QPixmap())
            self.setText("No figure available.")
            return
        pm = QPixmap()
        pm.loadFromData(png_bytes)
        self._raw = pm
        self._rescale()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._raw is None or self._raw.isNull():
            return
        dpr = self.devicePixelRatio()
        target = self.size() * dpr
        scaled = self._raw.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled.setDevicePixelRatio(dpr)
        super().setPixmap(scaled)


class _LiveResponseCanvas(FigureCanvas):
    """Embedded Matplotlib canvas for live roof displacement/acceleration."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.fig = Figure(figsize=(6.6, 4.8))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumSize(320, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax_disp = self.fig.add_subplot(2, 1, 1)
        self.ax_acc = self.fig.add_subplot(2, 1, 2)
        self.line_disp, = self.ax_disp.plot([], [], color="red", linewidth=1.7, label="Calibrated")
        self.line_acc, = self.ax_acc.plot([], [], color="red", linewidth=1.7, label="Calibrated")
        self.overlay_disp = None
        self.overlay_acc = None
        self._tmax = 1.0
        self._configure_axes()

    def _configure_axes(self) -> None:
        self.ax_disp.set_title("Roof Displacement")
        self.ax_disp.set_xlabel("Time (s)")
        self.ax_disp.set_ylabel("Displacement (m)")
        self.ax_disp.grid(True, alpha=0.25)

        self.ax_acc.set_title("Roof Acceleration")
        self.ax_acc.set_xlabel("Time (s)")
        self.ax_acc.set_ylabel("Acceleration (m/s²)")
        self.ax_acc.grid(True, alpha=0.25)
        self.fig.tight_layout(pad=1.6)

    def initialize(self, overlay_response: dict[str, Any] | None = None, tmax: float | None = None) -> None:
        self.ax_disp.clear()
        self.ax_acc.clear()
        self.line_disp, = self.ax_disp.plot([], [], color="red", linewidth=1.7, label="Calibrated")
        self.line_acc, = self.ax_acc.plot([], [], color="red", linewidth=1.7, label="Calibrated")
        self._tmax = float(tmax) if tmax and tmax > 0 else 1.0

        if overlay_response is not None:
            self.ax_disp.plot(
                overlay_response["t_hist"], overlay_response["u_hist"],
                color="black", linestyle="--", linewidth=0.9, label="Uncalibrated",
            )
            self.ax_acc.plot(
                overlay_response["t_hist"], overlay_response["a_hist"],
                color="black", linestyle="--", linewidth=0.9, label="Uncalibrated",
            )
            self.ax_disp.legend(loc="upper right", fontsize=9)
            self.ax_acc.legend(loc="upper right", fontsize=9)

        self._configure_axes()
        self.ax_disp.set_xlim(0.0, self._tmax)
        self.ax_acc.set_xlim(0.0, self._tmax)
        self.draw_idle()

    def update_frame(self, frame: dict[str, Any]) -> None:
        t = np.asarray(frame.get("t_hist", []), dtype=float)
        u = np.asarray(frame.get("u_hist", []), dtype=float)
        a = np.asarray(frame.get("a_hist", []), dtype=float)
        tmax = float(frame.get("tmax", self._tmax) or self._tmax)
        if tmax > 0:
            self._tmax = tmax

        self.line_disp.set_data(t, u)
        self.line_acc.set_data(t, a)
        self.ax_disp.set_xlim(0.0, self._tmax)
        self.ax_acc.set_xlim(0.0, self._tmax)

        self.ax_disp.relim()
        self.ax_disp.autoscale_view(scalex=False, scaley=True)
        self.ax_acc.relim()
        self.ax_acc.autoscale_view(scalex=False, scaley=True)
        self.draw_idle()


class _Live3DCanvas(FigureCanvas):
    """Embedded Matplotlib 3D canvas for live frame deformation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.fig = Figure(figsize=(5.2, 3.4))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        self._defo_lines: list[Any] = []
        self._configure_empty()

    def _configure_empty(self) -> None:
        self.ax.clear()
        self.ax.set_title("3D Transient Response")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.grid(False)
        self.ax.view_init(elev=25, azim=-70)
        self.fig.tight_layout(pad=0.5)
        self.draw_idle()

    def initialize_model(self, modal_data: dict[str, Any] | None, title: str = "3D Transient Response") -> None:
        self.ax.clear()
        self.ax.set_title(title)
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
        half = 0.55 * max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0e-9)
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
            self.ax.plot([x1, x2], [y1, y2], [z1, z2], linestyle="--", linewidth=1.0, color="0.6")

        for _ in vis_elems:
            line, = self.ax.plot([], [], [], color="navy", linewidth=2.6)
            self._defo_lines.append(line)

        self.fig.tight_layout(pad=0.5)
        self.draw_idle()

    def update_frame(self, frame: dict[str, Any]) -> None:
        segments = frame.get("deformed_segments", [])
        for line, segment in zip(self._defo_lines, segments):
            (x1, y1, z1), (x2, y2, z2) = segment
            line.set_data([x1, x2], [y1, y2])
            line.set_3d_properties([z1, z2])
        self.draw_idle()


class _HalfWidthContainer(QWidget):
    def __init__(self, target: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._target = target

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._target.setMaximumWidth(max(360, self.width() // 2))


class _ModelUpdatingWorker(QObject):
    log = Signal(str)
    frame = Signal(object)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, request: _WorkerRequest) -> None:
        super().__init__()
        self._request = request

    @Slot()
    def run(self) -> None:
        try:
            with contextlib.redirect_stdout(_StdoutBuffer(self.log.emit)):
                result = self._run_impl()
        except Exception as exc:
            self.log.emit("\nModel updating failed:\n")
            self.log.emit(traceback.format_exc())
            self.error.emit(str(exc))
        else:
            self.finished.emit(result)

    def _run_impl(self) -> dict[str, Any]:
        project_dir = self._request.project_dir
        params = copy.deepcopy(self._request.params)
        previous_cwd = Path.cwd()

        from opensees_model_updating.analysis.modal import (  # type: ignore
            export_modal_files,
            extract_modal_results,
            render_mode_shapes_individually_to_png,
            render_mode_shapes_to_png,
        )
        from opensees_model_updating.analysis.transient import (  # type: ignore
            render_transient_to_png,
            run_transient_analysis_collect_data,
            run_transient_analysis_stream,
        )
        from opensees_model_updating.calibration.calibrator import (  # type: ignore
            prepare_experimental_modal_data,
            run_calibration,
        )
        from opensees_model_updating.io.loaders import write_json, write_text  # type: ignore
        from opensees_model_updating.reporting.calibration_report import (  # type: ignore
            generate_calibration_summary_png,
            make_modal_comparison_report,
            report_to_text,
            save_calibration_summary_figure,
        )

        try:
            os.chdir(project_dir)
            self.log.emit(f"Output workspace: {project_dir}\n")

            os.makedirs("input", exist_ok=True)
            os.makedirs("output", exist_ok=True)

            if self._request.action == "calibrate":
                original_params = copy.deepcopy(params)
                write_json("output/original_inputs.json", original_params)

                if params.get("experimental_data_source") == "manual":
                    self.log.emit("\nUsing manually entered experimental modal data.\n")
                    exp_data = _build_exp_data_from_gui_values(params)
                else:
                    json_path = params.get("experimental_json_path", "")
                    if json_path:
                        self.log.emit(f"\nLoading experimental data from: {json_path}\n")
                        exp_data = _load_exp_data_from_json_path(json_path, params)
                    else:
                        exp_data = prepare_experimental_modal_data(params)
                write_json("output/experimental_modal_data_loaded.json", exp_data["raw_data"])

                modal_before = extract_modal_results(
                    original_params, normalize_modes=True, show_info=params["show_info"]
                )
                export_modal_files(modal_before, "original")

                _sep = "─" * 60
                n_show = params["numModes"]
                orig_heights = original_params.get("story_heights", [])
                orig_masses = original_params["floor_masses"]

                self.log.emit(f"\n{_sep}\n")
                self.log.emit("PRIOR MODEL\n")
                self.log.emit(f"{_sep}\n")
                self.log.emit(f"  E = {float(original_params['E']):.4e} Pa\n")
                for i, m in enumerate(orig_masses):
                    h_str = f"  h = {orig_heights[i]:.4f} m," if i < len(orig_heights) else ""
                    self.log.emit(f"  Story {i+1}:{h_str}  self-weight mass = {float(m):.4f} kg\n")
                add_masses_orig = original_params.get("additional_masses", {})
                if any(any(float(v) > 0 for v in vals) for vals in add_masses_orig.values()):
                    self.log.emit("  Additional masses [center, C1, C2, C3, C4] kg:\n")
                    for sk in sorted(add_masses_orig, key=lambda k: int(k) if str(k).isdigit() else 0):
                        vals = add_masses_orig[sk]
                        if any(float(v) > 0 for v in vals):
                            self.log.emit(f"    Story {sk}: " + "  ".join(f"{float(v):.4f}" for v in vals) + "\n")
                self.log.emit("\n  Modal frequencies:\n")
                for i, (f, p) in enumerate(zip(modal_before["freqs"][:n_show], modal_before["periods"][:n_show])):
                    self.log.emit(f"    Mode {i+1}: f = {float(f):.4f} Hz   T = {float(p):.4f} s\n")
                self.log.emit("\n  Mode shapes UX (normalized, |max| = 1):\n")
                for i, phi in enumerate(modal_before["mode_shapes_ux_master"][:n_show]):
                    self.log.emit(f"    Mode {i+1}: " + "  ".join(f"Story {j+1}: {float(v):+.4f}" for j, v in enumerate(phi)) + "\n")
                self.log.emit("\n  Experimental target frequencies:\n")
                for i, f_tgt in enumerate(exp_data["freqs"][:params["nCalibModes"]]):
                    self.log.emit(f"    Mode {i+1}: {float(f_tgt):.4f} Hz\n")

                # Render prior model mode shapes while the model is still in memory
                mode_shapes_png = render_mode_shapes_to_png(modal_before, title_prefix="Prior model")

                if params["enable_calibration"]:
                    self.log.emit(f"\n{_sep}\n")
                    self.log.emit("CALIBRATION\n")
                    self.log.emit(f"{_sep}\n")
                    self.log.emit(f"  Modes used:          {params['nCalibModes']}\n")
                    self.log.emit(f"  Use mode shapes:     {params['use_mode_shapes']}\n")
                    self.log.emit(f"  Frequency weight:    {params['w_freq']:.4f}\n")
                    self.log.emit(f"  Mode-shape weight:   {params['w_mode']:.4f}\n")
                    self.log.emit(f"  E scale bounds:      [{params['E_scale_lb']:.3f}, {params['E_scale_ub']:.3f}] × E_prior\n")
                    self.log.emit(f"  Mass scale bounds:   [{params['m_scale_lb']:.3f}, {params['m_scale_ub']:.3f}] × m_prior\n")
                    self.log.emit(f"  Frequency tolerance: {params['freq_tol_percent']:.2f}%\n")
                    self.log.emit(f"  Max evaluations:     {params['max_nfev']}\n")
                    self.log.emit("\nRunning calibration optimizer...\n")
                    calib_result, calibrated_params = run_calibration(
                        original_params, exp_data, show_info=params["show_info"]
                    )
                    self.log.emit(f"  Result:       {'Converged' if calib_result.success else 'Did not converge'}\n")
                    self.log.emit(f"  Evaluations:  {calib_result.nfev}\n")
                    self.log.emit(f"  Message:      {calib_result.message}\n")

                    modal_after = extract_modal_results(
                        calibrated_params,
                        normalize_modes=True,
                        show_info=params["show_info"],
                    )
                    export_modal_files(modal_after, "calibrated")

                    e_orig_val = float(original_params["E"])
                    e_cal_val = float(calibrated_params["E"])
                    cal_heights = calibrated_params.get("story_heights", orig_heights)
                    cal_masses = calibrated_params["floor_masses"]

                    self.log.emit(f"\n{_sep}\n")
                    self.log.emit("UPDATED MODEL\n")
                    self.log.emit(f"{_sep}\n")
                    self.log.emit(f"  E = {e_cal_val:.4e} Pa  ({(e_cal_val - e_orig_val) / e_orig_val * 100:+.2f}% from prior)\n")
                    for i, (m_orig, m_cal) in enumerate(zip(orig_masses, cal_masses)):
                        m_chg = (float(m_cal) - float(m_orig)) / float(m_orig) * 100 if float(m_orig) != 0 else 0.0
                        h_str = f"  h = {cal_heights[i]:.4f} m," if i < len(cal_heights) else ""
                        self.log.emit(f"  Story {i+1}:{h_str}  self-weight mass = {float(m_cal):.4f} kg  ({m_chg:+.2f}% from prior)\n")
                    tgt_freqs = list(exp_data["freqs"])
                    self.log.emit("\n  Modal frequencies vs target:\n")
                    for i, (f_cal, p_cal) in enumerate(zip(modal_after["freqs"][:n_show], modal_after["periods"][:n_show])):
                        line = f"    Mode {i+1}: f = {float(f_cal):.4f} Hz   T = {float(p_cal):.4f} s"
                        if i < params["nCalibModes"] and i < len(tgt_freqs):
                            f_t = float(tgt_freqs[i])
                            err = (float(f_cal) - f_t) / f_t * 100 if f_t != 0 else 0.0
                            line += f"   (target: {f_t:.4f} Hz, error: {err:+.2f}%)"
                        self.log.emit(line + "\n")
                    self.log.emit("\n  Mode shapes UX (normalized, |max| = 1):\n")
                    for i, phi in enumerate(modal_after["mode_shapes_ux_master"][:n_show]):
                        self.log.emit(f"    Mode {i+1}: " + "  ".join(f"Story {j+1}: {float(v):+.4f}" for j, v in enumerate(phi)) + "\n")
                else:
                    calib_result = None
                    calibrated_params = original_params
                    modal_after = modal_before

                write_json("output/calibrated_inputs.json", calibrated_params)

                report = make_modal_comparison_report(
                    exp_data,
                    modal_before,
                    modal_after,
                    original_params,
                    calibrated_params,
                    calib_result,
                )
                report_text = report_to_text(report)
                write_json("output/modal_comparison_report.json", report)
                write_text("output/modal_comparison_report.txt", report_text)
                save_calibration_summary_figure(
                    exp_data,
                    modal_before,
                    modal_after,
                    original_params,
                    calibrated_params,
                    save_path="output/calibration_summary.png",
                )
                calib_summary_png = generate_calibration_summary_png(
                    exp_data, modal_before, modal_after, original_params, calibrated_params
                )

                uncalibrated_response = None
                if params["show_uncalibrated_response"]:
                    self.log.emit("\nCalculating uncalibrated transient response...\n")
                    modal_original_dynamic = extract_modal_results(
                        original_params,
                        normalize_modes=True,
                        show_info=params["show_info"],
                    )
                    uncalibrated_response = run_transient_analysis_collect_data(
                        original_params,
                        modal_original_dynamic,
                        show_info=params["show_info"],
                        recorder_prefix="original_",
                    )
                    np.savez(
                        "output/original_transient_response_overlay.npz",
                        t_hist=uncalibrated_response["t_hist"],
                        u_hist=uncalibrated_response["u_hist"],
                        a_hist=uncalibrated_response["a_hist"],
                    )

                return {
                    "action": "calibrate",
                    "params": calibrated_params,
                    "signature": _make_calibration_signature(params),
                    "uncalibrated_response": uncalibrated_response,
                    "report_text": report_text,
                    "output_dir": str(project_dir / "output"),
                    # Prior modal data — stored in _CalibrationState so Run Analysis can show it
                    "prior_freqs": [float(f) for f in modal_before["freqs"][:n_show]],
                    "prior_periods": [float(p) for p in modal_before["periods"][:n_show]],
                    "prior_mode_shapes": [
                        [float(v) for v in phi]
                        for phi in modal_before["mode_shapes_ux_master"][:n_show]
                    ],
                    # PNG figures to embed in the Output tab
                    "fig1_png": calib_summary_png,   # calibration comparison (4 subplots)
                    "fig2_png": mode_shapes_png,      # original (pre-calibration) mode shapes
                }

            run_params = copy.deepcopy(params)
            use_precalibrated = False
            current_signature = _make_calibration_signature(run_params)
            state = self._request.calibration_state

            if state.available and state.input_signature == current_signature:
                use_precalibrated = True
                final_params = copy.deepcopy(state.calibrated_params)
            else:
                final_params = run_params

            write_json("output/current_run_inputs.json", run_params)

            _sep = "─" * 60

            def _log_model_params(p: dict, heights: list, label: str) -> None:
                self.log.emit(f"\n{_sep}\n")
                self.log.emit(f"{label}\n")
                self.log.emit(f"{_sep}\n")
                self.log.emit(f"  E = {float(p['E']):.4e} Pa\n")
                for i, m in enumerate(p["floor_masses"]):
                    h_str = f"  h = {heights[i]:.4f} m," if i < len(heights) else ""
                    self.log.emit(f"  Story {i+1}:{h_str}  self-weight mass = {float(m):.4f} kg\n")
                add = p.get("additional_masses", {})
                if any(any(float(v) > 0 for v in vals) for vals in add.values()):
                    self.log.emit("  Additional masses [center, C1, C2, C3, C4] kg:\n")
                    for sk in sorted(add, key=lambda k: int(k) if str(k).isdigit() else 0):
                        vals = add[sk]
                        if any(float(v) > 0 for v in vals):
                            self.log.emit(f"    Story {sk}: " + "  ".join(f"{float(v):.4f}" for v in vals) + "\n")

            if use_precalibrated:
                write_json("output/calibrated_inputs_used_for_run.json", final_params)
                prior_heights = run_params.get("story_heights", [])
                _log_model_params(run_params, prior_heights, "PRIOR MODEL (uncalibrated)")
                if state.prior_freqs:
                    self.log.emit("\n  Modal frequencies:\n")
                    for i, (f, p) in enumerate(zip(state.prior_freqs, state.prior_periods or [])):
                        self.log.emit(f"    Mode {i+1}: f = {float(f):.4f} Hz   T = {float(p):.4f} s\n")
                if state.prior_mode_shapes:
                    self.log.emit("\n  Mode shapes UX (normalized, |max| = 1):\n")
                    for i, phi in enumerate(state.prior_mode_shapes):
                        self.log.emit(f"    Mode {i+1}: " + "  ".join(f"Story {j+1}: {float(v):+.4f}" for j, v in enumerate(phi)) + "\n")
                upd_heights = final_params.get("story_heights", [])
                _log_model_params(final_params, upd_heights, "UPDATED MODEL (calibrated)")
            else:
                prior_heights = final_params.get("story_heights", [])
                _log_model_params(final_params, prior_heights, "RUN ANALYSIS — Prior model (uncalibrated)")

            self.log.emit("\nExtracting modal properties...\n")
            final_modal = extract_modal_results(
                final_params, normalize_modes=True, show_info=run_params["show_info"]
            )
            export_modal_files(final_modal, "run_model")
            modal_summary = _modal_summary(final_modal, run_params["numModes"])

            self.log.emit("\nModal frequencies:\n")
            for i, (f, p) in enumerate(zip(final_modal["freqs"], final_modal["periods"])):
                self.log.emit(f"  Mode {i+1}: f = {float(f):.4f} Hz   T = {float(p):.4f} s\n")

            self.log.emit("\nMode shapes UX (normalized, max abs = 1):\n")
            for i, phi in enumerate(final_modal["mode_shapes_ux_master"][:run_params["numModes"]]):
                vals = "  ".join(f"Story {j+1}: {float(v):+.4f}" for j, v in enumerate(phi))
                self.log.emit(f"  Mode {i+1}: {vals}\n")

            transient_png: bytes | None = None
            transient_response = None
            overlay_response = (
                state.uncalibrated_response
                if use_precalibrated and run_params["show_uncalibrated_response"]
                else None
            )

            if run_params["run_transient"]:
                self.log.emit("\nRunning transient analysis...\n")
                self.frame.emit({
                    "kind": "init",
                    "modal_data": final_modal,
                    "overlay_response": overlay_response,
                    "title": "Calibrated 3D Response" if use_precalibrated else "3D Response",
                })
                transient_response = run_transient_analysis_stream(
                    final_params,
                    final_modal,
                    show_info=run_params["show_info"],
                    recorder_prefix="run_",
                    frame_callback=self.frame.emit,
                    plot_every=10,
                    anim_every=10,
                    realtime=True,
                    sfac_anim=20.0,
                )
                np.savez(
                    "output/run_transient_response.npz",
                    t_hist=transient_response["t_hist"],
                    u_hist=transient_response["u_hist"],
                    a_hist=transient_response["a_hist"],
                )

                t = np.asarray(transient_response["t_hist"])
                u = np.asarray(transient_response["u_hist"])
                a = np.asarray(transient_response["a_hist"])
                if len(u) > 0:
                    self.log.emit("\nTransient response summary:\n")
                    self.log.emit(f"  Duration:              {float(t[-1]):.3f} s\n")
                    self.log.emit(f"  Peak roof displacement: {float(np.max(np.abs(u))):.6f} m\n")
                    self.log.emit(f"  Peak roof acceleration: {float(np.max(np.abs(a))):.4f} m/s²\n")
                    self.log.emit(f"  Final displacement:     {float(u[-1]):.6f} m\n")

                self.log.emit("\nRendering transient response figure...\n")
                transient_png = render_transient_to_png(
                    transient_response, final_modal, overlay_response=overlay_response
                )

                # Individual mode shapes for the 2×2 right grid.
                self.log.emit("\nRendering mode shapes figures...\n")
                mode_shapes_pngs = render_mode_shapes_individually_to_png(final_modal)
            else:
                # No transient — render a combined mode shapes PNG for the static label.
                mode_title = "Calibrated" if use_precalibrated else "Run Model"
                self.log.emit("\nRendering mode shapes figure...\n")
                mode_shapes_pngs = []
                transient_png = render_mode_shapes_to_png(final_modal, title_prefix=mode_title)

            return {
                "action": "run",
                "used_calibration": use_precalibrated,
                "output_dir": str(project_dir / "output"),
                "modal_summary": modal_summary,
                "transient_response": transient_response,
                "overlay_response": overlay_response,
                # For run with transient: individual mode PNGs for the 2×2 grid.
                # For run without transient: fig1_png holds the combined mode shapes PNG,
                #   fig_mode_pngs is empty.
                "fig1_png": transient_png,
                "fig2_png": None,
                "fig_mode_pngs": mode_shapes_pngs,
            }
        finally:
            os.chdir(previous_cwd)


def _build_exp_data_from_gui_values(params: dict[str, Any]) -> dict[str, Any]:
    """
    Build the exp_data dict from GUI-entered modal values.

    Returns the same structure as prepare_experimental_modal_data() so the
    calibrator receives identical input whether data comes from the GUI or
    the JSON file.  This is the pipeline hook for future sensor data: when
    sensor-derived frequencies/mode-shapes are available, populate
    params["experimental_modal_data"] with the same dict structure and call
    this function instead of the JSON path.
    """
    raw = params["experimental_modal_data"]
    n_calib = params["nCalibModes"]
    use_shapes = params["use_mode_shapes"]

    freqs_all = raw.get("frequencies_hz", [])
    if n_calib > len(freqs_all):
        raise ValueError(
            f"Only {len(freqs_all)} frequencies entered, but nCalibModes = {n_calib}. "
            "Add more frequencies or reduce Calibration modes."
        )
    exp_freqs = np.array(freqs_all[:n_calib], dtype=float)

    shapes_raw = raw.get("mode_shapes_ux", {})
    mode_shapes_available = bool(shapes_raw)

    exp_modes: list[Any] = []
    if use_shapes and mode_shapes_available:
        for i in range(n_calib):
            key = str(i + 1)
            if key not in shapes_raw:
                raise ValueError(f"Mode shape for mode {i + 1} not entered.")
            vals = np.array(shapes_raw[key], dtype=float)
            m = float(np.max(np.abs(vals)))
            exp_modes.append(vals / m if m > 0.0 else vals)

    return {
        "freqs": exp_freqs,
        "modes": exp_modes,
        "use_mode_shapes": use_shapes and mode_shapes_available,
        "mode_shapes_available": mode_shapes_available,
        "source_file": "Manual input (SensePi GUI)",
        "raw_data": raw,
        "n_modes_used": n_calib,
    }


def _load_exp_data_from_json_path(json_path: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Load and prepare experimental modal data from an arbitrary JSON file.

    Mirrors prepare_experimental_modal_data() but accepts a configurable
    path so users can browse to any JSON file rather than using the
    hard-coded EXPERIMENTAL_MODAL_JSON constant.
    """
    from opensees_model_updating.io.loaders import load_experimental_modal_data  # type: ignore
    from opensees_model_updating.utils.math_utils import normalize_mode_maxabs   # type: ignore

    loaded = load_experimental_modal_data(
        json_path, params["nStory"], require_mode_shapes=params["use_mode_shapes"]
    )
    n_calib = params["nCalibModes"]
    if n_calib > len(loaded["frequencies_hz"]):
        raise ValueError(
            f"JSON file contains only {len(loaded['frequencies_hz'])} frequencies, "
            f"but nCalibModes = {n_calib}."
        )

    exp_modes: list[Any] = []
    mode_shapes_available = bool(loaded.get("mode_shapes_available", False))
    if params["use_mode_shapes"]:
        exp_modes = [normalize_mode_maxabs(v) for v in loaded["mode_shapes_ux"][:n_calib]]

    return {
        "freqs": loaded["frequencies_hz"][:n_calib],
        "modes": exp_modes,
        "use_mode_shapes": params["use_mode_shapes"] and mode_shapes_available,
        "mode_shapes_available": mode_shapes_available,
        "source_file": json_path,
        "raw_data": loaded["raw_data"],
        "n_modes_used": n_calib,
    }


def _round_for_signature(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {str(k): _round_for_signature(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_for_signature(v) for v in value]
    return value


def _modal_summary(modal_data: dict[str, Any], num_modes: int) -> dict[str, Any]:
    n_modes = min(int(num_modes), len(modal_data.get("freqs", [])))
    ctx = modal_data["ctx"]
    return {
        "freqs": np.asarray(modal_data["freqs"][:n_modes], dtype=float).tolist(),
        "periods": np.asarray(modal_data["periods"][:n_modes], dtype=float).tolist(),
        "mode_shapes_ux_master": [
            np.asarray(modal_data["mode_shapes_ux_master"][i], dtype=float).tolist()
            for i in range(n_modes)
        ],
        "vis_elems": [[int(i), int(j)] for i, j in ctx["vis_elems"]],
        "node_xyz": {
            str(int(node)): [float(coord) for coord in xyz]
            for node, xyz in ctx["node_xyz"].items()
        },
        "story_node_tags": {
            str(int(story)): [int(node) for node in nodes]
            for story, nodes in ctx["story_node_tags"].items()
        },
    }


def _make_calibration_signature(params: dict[str, Any]) -> str:
    keys = [
        "Lx",
        "Ly",
        "story_heights",
        "t_column",
        "b_column",
        "b_beam",
        "h_beam",
        "E",
        "nu",
        "floor_masses",
        "numModes",
        "zeta",
        "gmFactor",
        "gmFile",
        "dtGM",
        "enable_calibration",
        "use_mode_shapes",
        "mass_calibration_scope",
        "nCalibModes",
        "freq_tol_percent",
        "w_freq",
        "w_mode",
        "E_scale_lb",
        "E_scale_ub",
        "m_scale_lb",
        "m_scale_ub",
        "max_nfev",
        "story_column_layout",
        "column_orientation_layout",
        "additional_masses",
    ]
    return json.dumps(
        {key: _round_for_signature(params.get(key)) for key in keys},
        sort_keys=True,
    )


class ModelUpdatingTab(QWidget):
    """Native PySide controls for the bundled OpenSees model-updating workflow."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _ModelUpdatingWorker | None = None
        self._calibration_state = _CalibrationState()
        self._build_ui()
        self._set_defaults()
        self._rebuild_story_table()
        self._load_exp_json_defaults()  # must run last — after story table is fully built

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self._tabs = QTabWidget(self)
        root.addWidget(self._tabs, stretch=1)

        self._model_tab = QWidget(self)
        self._mass_tab = QWidget(self)
        self._analysis_tab = QWidget(self)
        self._calibration_tab = QWidget(self)
        self._output_tab = QWidget(self)
        self._tabs.addTab(self._model_tab, "Model")
        self._tabs.addTab(self._mass_tab, "Additional Mass")
        self._tabs.addTab(self._analysis_tab, "Analysis")
        self._tabs.addTab(self._calibration_tab, "Calibration")
        self._tabs.addTab(self._output_tab, "Output")

        self._build_model_tab()
        self._build_mass_tab()
        self._build_analysis_tab()
        self._build_calibration_tab()
        self._build_output_tab()

        actions = QHBoxLayout()
        self._calibrate_btn = QPushButton("Calibrate", self)
        self._run_btn = QPushButton("Run Analysis", self)
        actions.addWidget(self._calibrate_btn)
        actions.addWidget(self._run_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        self._status = QLabel("Ready.", self)
        root.addWidget(self._status)

        self._calibrate_btn.clicked.connect(lambda: self._start_worker("calibrate"))
        self._run_btn.clicked.connect(lambda: self._start_worker("run"))

    def _build_model_tab(self) -> None:
        layout = QVBoxLayout(self._model_tab)

        project_group = QGroupBox("Project", self)
        project_form = QFormLayout(project_group)
        self._project_dir_edit = QLineEdit(self)
        project_row = QHBoxLayout()
        project_row.addWidget(self._project_dir_edit, stretch=1)
        self._browse_project_btn = QPushButton("Browse...", self)
        project_row.addWidget(self._browse_project_btn)
        project_widget = QWidget(self)
        project_widget.setLayout(project_row)
        project_form.addRow("Output workspace folder:", project_widget)
        layout.addWidget(project_group)

        geom_group = QGroupBox("Geometry and Sections", self)
        geom_form = QFormLayout(geom_group)
        self._lx = self._double_spin(0.001, 100.0, 0.245, 4)
        self._ly = self._double_spin(0.001, 100.0, 0.230, 4)
        self._story_count = QSpinBox(self)
        self._story_count.setRange(1, 20)
        self._story_count.setValue(3)
        self._t_column = self._double_spin(0.0001, 10.0, 0.001, 5)
        self._b_column = self._double_spin(0.0001, 10.0, 0.006, 5)
        self._b_beam = self._double_spin(0.0001, 10.0, 0.001, 5)
        self._h_beam = self._double_spin(0.0001, 10.0, 0.008, 5)
        self._youngs_modulus = self._double_spin(1.0, 1.0e13, 200e9, 3)
        self._poisson = self._double_spin(0.0, 0.49, 0.33, 4)

        geom_form.addRow("Length X (m):", self._lx)
        geom_form.addRow("Width Y (m):", self._ly)
        geom_form.addRow("Stories:", self._story_count)
        geom_form.addRow("Column thickness (m):", self._t_column)
        geom_form.addRow("Column width (m):", self._b_column)
        geom_form.addRow("Beam width (m):", self._b_beam)
        geom_form.addRow("Beam height (m):", self._h_beam)
        geom_form.addRow("Young's modulus E (Pa):", self._youngs_modulus)
        geom_form.addRow("Poisson ratio:", self._poisson)
        geom_group.setMaximumWidth(360)
        geom_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        top_row = QHBoxLayout()

        preview_group = QGroupBox("3D Column Numbering Preview", self)
        preview_group.setMaximumWidth(300)
        preview_group.setMinimumHeight(400)
        preview_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        preview_layout = QVBoxLayout(preview_group)
        self._structure_preview = _StructurePreview(self)
        preview_layout.addWidget(self._structure_preview)

        self._story_table = QTableWidget(self)
        self._story_table.setColumnCount(11)
        self._story_table.setHorizontalHeaderLabels(
            [
                "Story",
                "Height",
                "Self Mass",
                "C1",
                "C2",
                "C3",
                "C4",
                "Orient C1",
                "Orient C2",
                "Orient C3",
                "Orient C4",
            ]
        )
        self._story_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._story_table.horizontalHeader().setStretchLastSection(False)
        self._story_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._story_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        controls = QHBoxLayout()
        self._set_all_weak_btn = QPushButton("Set all Weak axis", self)
        self._set_all_strong_btn = QPushButton("Set all Strong axis", self)
        controls.addWidget(self._set_all_weak_btn)
        controls.addWidget(self._set_all_strong_btn)
        controls.addStretch(1)

        story_group = QGroupBox("Story Height / Column Orientation", self)
        story_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        story_layout = QVBoxLayout(story_group)
        story_layout.addWidget(self._story_table, stretch=1)
        story_layout.addLayout(controls)

        top_row.addWidget(geom_group, stretch=0)
        top_row.addWidget(story_group, stretch=1)
        top_row.addWidget(preview_group, stretch=0)
        layout.addLayout(top_row, stretch=1)

        self._browse_project_btn.clicked.connect(self._browse_project_dir)
        self._story_count.valueChanged.connect(self._rebuild_story_table)
        for widget in (self._lx, self._ly):
            widget.valueChanged.connect(self._refresh_previews)
        self._set_all_weak_btn.clicked.connect(lambda: self._set_all_orientations("Weak axis"))
        self._set_all_strong_btn.clicked.connect(lambda: self._set_all_orientations("Strong axis"))

    def _build_mass_tab(self) -> None:
        layout = QVBoxLayout(self._mass_tab)

        mass_group = QGroupBox("Additional Mass Placement", self)
        mass_layout = QVBoxLayout(mass_group)
        self._mass_legend = _MassLegend(self)
        mass_layout.addWidget(self._mass_legend)
        layout.addWidget(mass_group)

        self._mass_table = QTableWidget(self)
        self._mass_table.setColumnCount(6)
        self._mass_table.setHorizontalHeaderLabels(
            ["Story", "Center", "C1", "C2", "C3", "C4"]
        )
        self._mass_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._mass_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._mass_table, stretch=1)

    def _build_analysis_tab(self) -> None:
        layout = self._scrollable_tab(self._analysis_tab)
        group = QGroupBox("Modal and Transient Analysis", self)
        form = QFormLayout(group)

        self._num_modes = QSpinBox(self)
        self._num_modes.setRange(2, 20)
        self._num_modes.setValue(3)
        self._zeta = self._double_spin(0.0, 1.0, 0.005, 4)
        self._gm_factor = self._double_spin(-1.0e6, 1.0e6, 9.81, 4)
        self._dt_gm = self._double_spin(1.0e-8, 10.0, 0.01, 6)
        self._run_transient = QCheckBox(self)
        self._run_transient.setChecked(True)
        self._show_info = QCheckBox(self)

        # Ground-motion preset selector
        self._gm_preset = QComboBox(self)
        _GM_PRESETS = [
            ("1 Hz sine  (sine_1Hz_accel.txt)",  "sine_1Hz_accel.txt"),
            ("2 Hz sine  (sine_2Hz_accel.txt)",  "sine_2Hz_accel.txt"),
            ("3 Hz sine  (sine_3Hz_accel.txt)",  "sine_3Hz_accel.txt"),
            ("5 Hz sine  (sine_5Hz_accel.txt)",  "sine_5Hz_accel.txt"),
            ("Custom (browse below)",             ""),
        ]
        self._gm_preset_files = _GM_PRESETS
        for label, _ in _GM_PRESETS:
            self._gm_preset.addItem(label)

        # Ground-motion custom file row (shown for all presets; editable for Custom)
        self._gm_file_edit = QLineEdit(self)
        gm_row = QHBoxLayout()
        gm_row.addWidget(self._gm_file_edit, stretch=1)
        self._browse_gm_btn = QPushButton("Browse...", self)
        gm_row.addWidget(self._browse_gm_btn)
        gm_widget = QWidget(self)
        gm_widget.setLayout(gm_row)

        form.addRow("Number of modes:", self._num_modes)
        form.addRow("Damping ratio:", self._zeta)
        form.addRow("Ground-motion factor:", self._gm_factor)
        form.addRow("Ground-motion dt (s):", self._dt_gm)
        form.addRow("Ground-motion preset:", self._gm_preset)
        form.addRow("Ground-motion file:", gm_widget)
        form.addRow("Run transient analysis:", self._run_transient)
        form.addRow("Verbose OpenSees output:", self._show_info)
        layout.addWidget(group)
        layout.addStretch(1)

        self._browse_gm_btn.clicked.connect(self._browse_ground_motion)
        self._gm_preset.currentIndexChanged.connect(self._on_gm_preset_changed)

    @staticmethod
    def _scrollable_tab(tab_widget: QWidget) -> QVBoxLayout:
        """
        Replace the tab widget's layout with a QScrollArea and return the
        inner VBoxLayout so content can be added to it normally.
        """
        outer = QVBoxLayout(tab_widget)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        container = QWidget()
        inner = QVBoxLayout(container)
        inner.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(container)
        outer.addWidget(scroll)
        return inner

    def _build_calibration_tab(self) -> None:
        layout = self._scrollable_tab(self._calibration_tab)
        group = QGroupBox("Calibration Settings", self)
        form = QFormLayout(group)

        self._enable_calibration = QCheckBox(self)
        self._enable_calibration.setChecked(True)
        self._use_mode_shapes = QCheckBox(self)
        self._use_mode_shapes.setChecked(False)
        self._mass_scope = QComboBox(self)
        self._mass_scope.addItems(
            ["Self-weight mass only", "Total mass including additional masses"]
        )
        self._n_calib_modes = QSpinBox(self)
        self._n_calib_modes.setRange(1, 20)
        self._n_calib_modes.setValue(3)
        self._freq_tol = self._double_spin(0.0, 100.0, 5.0, 3)
        self._w_freq = self._double_spin(0.0, 1000.0, 1.0, 4)
        self._w_mode = self._double_spin(0.0, 1000.0, 0.35, 4)
        self._e_lb = self._double_spin(0.001, 1000.0, 0.70, 4)
        self._e_ub = self._double_spin(0.001, 1000.0, 1.30, 4)
        self._m_lb = self._double_spin(0.001, 1000.0, 0.70, 4)
        self._m_ub = self._double_spin(0.001, 1000.0, 1.30, 4)
        self._max_nfev = QSpinBox(self)
        self._max_nfev.setRange(1, 100000)
        self._max_nfev.setValue(200)
        self._show_uncalibrated_response = QCheckBox(self)
        self._show_uncalibrated_response.setChecked(True)

        form.addRow("Enable calibration:", self._enable_calibration)
        form.addRow("Use mode shapes:", self._use_mode_shapes)
        form.addRow("Mass calibration target:", self._mass_scope)
        form.addRow("Calibration modes:", self._n_calib_modes)
        form.addRow("Frequency tolerance (%):", self._freq_tol)
        form.addRow("Frequency weight:", self._w_freq)
        form.addRow("Mode-shape weight:", self._w_mode)
        form.addRow("E scale lower bound:", self._e_lb)
        form.addRow("E scale upper bound:", self._e_ub)
        form.addRow("Mass scale lower bound:", self._m_lb)
        form.addRow("Mass scale upper bound:", self._m_ub)
        form.addRow("Max optimizer evaluations:", self._max_nfev)
        form.addRow("Store uncalibrated response:", self._show_uncalibrated_response)
        layout.addWidget(group)

        self._build_exp_data_section(layout)
        layout.addStretch(1)

        self._n_calib_modes.valueChanged.connect(self._rebuild_exp_data_widgets)

    # ------------------------------------------------------------------
    # Experimental modal data section
    # ------------------------------------------------------------------

    def _build_exp_data_section(self, parent_layout: QVBoxLayout) -> None:
        """Add the Experimental Modal Data group to the Calibration tab."""
        group = QGroupBox("Experimental Modal Data (target frequencies and mode shapes)", self)
        outer = QVBoxLayout(group)

        # ── Source selector ───────────────────────────────────────────
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Data source:", self))
        self._exp_source = QComboBox(self)
        self._exp_source.addItems(["Manual input", "From JSON file"])
        source_row.addWidget(self._exp_source)
        source_row.addStretch(1)
        outer.addLayout(source_row)

        # ── Manual input panel ────────────────────────────────────────
        self._exp_manual_widget = QWidget(self)
        manual_vbox = QVBoxLayout(self._exp_manual_widget)
        manual_vbox.setContentsMargins(0, 4, 0, 0)

        # Frequencies row (rebuilt dynamically)
        freq_group = QGroupBox("Frequencies (Hz) — one per calibration mode", self)
        self._freq_row_layout = QHBoxLayout(freq_group)
        self._exp_freq_spins: list[QDoubleSpinBox] = []
        manual_vbox.addWidget(freq_group)

        # Mode shapes table (rebuilt dynamically)
        shapes_group = QGroupBox(
            "Mode Shapes UX — rows = stories, columns = modes (normalized, |max| = 1)", self
        )
        shapes_vbox = QVBoxLayout(shapes_group)
        self._exp_mode_table = QTableWidget(self)
        self._exp_mode_table.setMinimumHeight(120)
        shapes_vbox.addWidget(self._exp_mode_table)
        manual_vbox.addWidget(shapes_group)

        outer.addWidget(self._exp_manual_widget)

        # ── JSON file panel ───────────────────────────────────────────
        self._exp_json_widget = QWidget(self)
        json_row = QHBoxLayout(self._exp_json_widget)
        json_row.setContentsMargins(0, 4, 0, 0)
        json_row.addWidget(QLabel("JSON file:", self))
        self._exp_json_edit = QLineEdit(self)
        json_row.addWidget(self._exp_json_edit, stretch=1)
        self._browse_exp_json_btn = QPushButton("Browse...", self)
        json_row.addWidget(self._browse_exp_json_btn)

        outer.addWidget(self._exp_json_widget)
        parent_layout.addWidget(group)

        # Wire
        self._exp_source.currentIndexChanged.connect(self._on_exp_source_changed)
        self._browse_exp_json_btn.clicked.connect(self._browse_exp_json)

        # Build initial table (defaults loaded later in _set_defaults)
        self._rebuild_exp_data_widgets()
        self._on_exp_source_changed()

    @Slot()
    def _on_exp_source_changed(self) -> None:
        manual = self._exp_source.currentText() == "Manual input"
        self._exp_manual_widget.setVisible(manual)
        self._exp_json_widget.setVisible(not manual)

    @Slot()
    def _rebuild_exp_data_widgets(self) -> None:
        """Rebuild freq spinboxes + mode-shape table to match nCalibModes × nStory."""
        if not hasattr(self, "_exp_freq_spins"):
            return
        n_modes = int(self._n_calib_modes.value()) if hasattr(self, "_n_calib_modes") else 3
        n_story = int(self._story_count.value()) if hasattr(self, "_story_count") else 3

        # ── Save existing freq values ─────────────────────────────────
        old_freqs = [s.value() for s in self._exp_freq_spins]

        # Clear freq layout
        while self._freq_row_layout.count():
            item = self._freq_row_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._exp_freq_spins = []
        for i in range(n_modes):
            cell = QWidget(self)
            cell_vbox = QVBoxLayout(cell)
            cell_vbox.setContentsMargins(2, 0, 2, 0)
            cell_vbox.addWidget(QLabel(f"Mode {i + 1}", self))
            spin = QDoubleSpinBox(self)
            spin.setRange(0.001, 9999.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            if i < len(old_freqs):
                default = old_freqs[i]
            elif i < len(_DEFAULT_EXP_FREQUENCIES):
                default = _DEFAULT_EXP_FREQUENCIES[i]
            else:
                default = 1.0
            spin.setValue(default)
            cell_vbox.addWidget(spin)
            self._exp_freq_spins.append(spin)
            self._freq_row_layout.addWidget(cell)
        self._freq_row_layout.addStretch(1)

        # ── Save existing mode-shape values ───────────────────────────
        old_shapes: dict[tuple[int, int], float] = {}
        for r in range(self._exp_mode_table.rowCount()):
            for c in range(self._exp_mode_table.columnCount()):
                w = self._exp_mode_table.cellWidget(r, c)
                if isinstance(w, QDoubleSpinBox):
                    old_shapes[(r, c)] = w.value()

        self._exp_mode_table.setRowCount(n_story)
        self._exp_mode_table.setColumnCount(n_modes)
        self._exp_mode_table.setHorizontalHeaderLabels(
            [f"Mode {i + 1}" for i in range(n_modes)]
        )
        self._exp_mode_table.setVerticalHeaderLabels(
            [f"Story {i + 1}" for i in range(n_story)]
        )
        self._exp_mode_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._exp_mode_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        for r in range(n_story):
            for c in range(n_modes):
                spin = QDoubleSpinBox(self)
                spin.setRange(-999.0, 999.0)
                spin.setDecimals(4)
                spin.setSingleStep(0.01)
                if (r, c) in old_shapes:
                    default = old_shapes[(r, c)]
                elif c < len(_DEFAULT_EXP_MODE_SHAPES) and r < len(_DEFAULT_EXP_MODE_SHAPES[c]):
                    default = _DEFAULT_EXP_MODE_SHAPES[c][r]
                else:
                    default = 0.0
                spin.setValue(default)
                self._exp_mode_table.setCellWidget(r, c, spin)

    def _read_exp_data(self) -> dict[str, Any]:
        """Collect frequencies and mode shapes from the manual-input widgets."""
        n_modes = len(self._exp_freq_spins)
        n_story = self._exp_mode_table.rowCount()

        freqs = [float(s.value()) for s in self._exp_freq_spins]

        mode_shapes_ux: dict[str, list[float]] = {}
        for c in range(n_modes):
            vals = []
            for r in range(n_story):
                w = self._exp_mode_table.cellWidget(r, c)
                vals.append(float(w.value()) if isinstance(w, QDoubleSpinBox) else 0.0)
            mode_shapes_ux[str(c + 1)] = vals

        return {
            "frequencies_hz": freqs,
            "mode_shapes_ux": mode_shapes_ux,
            "notes": "Entered via SensePi GUI",
        }

    def _load_exp_json_defaults(self) -> None:
        """
        Pre-populate the manual-input widgets.

        Tries to read from the bundled JSON file first.  Falls back to the
        hardcoded _DEFAULT_EXP_* constants so the fields always open with
        the expected shaking-table values even if the file is missing.
        """
        try:
            import json as _json
            json_path = _OPENSEES_INPUT_DIR / "experimental_modal_data.json"
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            freqs: list[float] = [float(v) for v in data.get("frequencies_hz", [])]
            raw_shapes: dict[str, list[float]] = {
                k: [float(x) for x in v]
                for k, v in data.get("mode_shapes_ux", {}).items()
            }
        except Exception:
            # Fall back to hardcoded defaults
            freqs = list(_DEFAULT_EXP_FREQUENCIES)
            raw_shapes = {str(i + 1): list(col) for i, col in enumerate(_DEFAULT_EXP_MODE_SHAPES)}

        for i, spin in enumerate(self._exp_freq_spins):
            if i < len(freqs):
                spin.setValue(freqs[i])

        for c in range(self._exp_mode_table.columnCount()):
            key = str(c + 1)
            if key in raw_shapes:
                for r, val in enumerate(raw_shapes[key]):
                    if r < self._exp_mode_table.rowCount():
                        w = self._exp_mode_table.cellWidget(r, c)
                        if isinstance(w, QDoubleSpinBox):
                            w.setValue(val)

    @Slot()
    def _browse_exp_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose experimental modal data JSON",
            self._exp_json_edit.text().strip() or str(_OPENSEES_INPUT_DIR),
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self._exp_json_edit.setText(path)

    @Slot(int)
    def _on_gm_preset_changed(self, index: int) -> None:
        _, filename = self._gm_preset_files[index]
        if filename:
            self._gm_file_edit.setText(str(_OPENSEES_INPUT_DIR / filename))
            self._gm_file_edit.setReadOnly(True)
        else:
            self._gm_file_edit.setReadOnly(False)

    def _build_output_tab(self) -> None:
        outer = QVBoxLayout(self._output_tab)

        # Left side: static PNG (calibrate) or live roof response curves (run).
        self._fig1_label = _ScaledImageLabel(
            "Run Calibrate or Analysis\nto see results here.", self
        )
        self._live_response_canvas = _LiveResponseCanvas(self)
        self._live_response_canvas.hide()

        left_col = QWidget(self)
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._fig1_label, stretch=1)
        left_layout.addWidget(self._live_response_canvas, stretch=1)

        # Right side — two mutually exclusive panels:
        #   _fig2_label:       combined mode shapes PNG (calibrate / run without transient)
        #   _run_right_widget: 2×2 grid — Mode 1, Mode 2, Mode 3, live 3D canvas (run with transient)
        self._fig2_label = _ScaledImageLabel("", self)

        self._mode_shape_labels: list[_ScaledImageLabel] = [
            _ScaledImageLabel("", self),
            _ScaledImageLabel("", self),
            _ScaledImageLabel("", self),
        ]
        self._live_3d_canvas = _Live3DCanvas(self)
        # Override the class minimum so the 3D cell doesn't force its column wider than the mode cells.
        self._live_3d_canvas.setMinimumSize(10, 10)

        self._run_right_widget = QWidget(self)
        run_grid = QGridLayout(self._run_right_widget)
        run_grid.setContentsMargins(0, 0, 0, 0)
        run_grid.setSpacing(4)
        run_grid.addWidget(self._mode_shape_labels[0], 0, 0)
        run_grid.addWidget(self._mode_shape_labels[1], 0, 1)
        run_grid.addWidget(self._mode_shape_labels[2], 1, 0)
        run_grid.addWidget(self._live_3d_canvas, 1, 1)
        # Equal column and row weights so all four cells share space evenly.
        run_grid.setColumnStretch(0, 1)
        run_grid.setColumnStretch(1, 1)
        run_grid.setRowStretch(0, 1)
        run_grid.setRowStretch(1, 1)
        self._run_right_widget.hide()

        right_col = QWidget(self)
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._fig2_label, stretch=1)
        right_layout.addWidget(self._run_right_widget, stretch=1)

        fig_row = QWidget(self)
        fig_layout = QHBoxLayout(fig_row)
        fig_layout.setContentsMargins(0, 0, 0, 0)
        fig_layout.addWidget(left_col, stretch=1)
        fig_layout.addWidget(right_col, stretch=1)

        # Log panel.
        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(80)

        # Vertical splitter: figures top, log bottom.
        splitter = QSplitter(Qt.Vertical, self)
        splitter.addWidget(fig_row)
        splitter.addWidget(self._log)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        outer.addWidget(splitter)

    def _set_defaults(self) -> None:
        self._project_dir_edit.setText(str(_default_workspace_dir()))
        self._gm_file_edit.setText(str(_OPENSEES_INPUT_DIR / "sine_1Hz_accel.txt"))
        self._exp_json_edit.setText(str(_OPENSEES_INPUT_DIR / "experimental_modal_data.json"))

    def _double_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSingleStep(10 ** -min(decimals, 4))
        return spin

    def _rebuild_story_table(self) -> None:
        existing = self._read_story_table(allow_empty=True)
        existing_masses = self._read_mass_table(allow_empty=True)
        n_story = int(self._story_count.value())
        self._story_table.setRowCount(n_story)

        for row in range(n_story):
            story = row + 1
            old = existing.get(story, {})
            self._story_table.setItem(row, 0, QTableWidgetItem(str(story)))
            self._story_table.item(row, 0).setFlags(self._story_table.item(row, 0).flags())
            self._story_table.setCellWidget(
                row,
                1,
                self._table_double(old.get("height", 0.24), 0.001, 100.0, 4),
            )
            self._story_table.setCellWidget(
                row,
                2,
                self._table_double(old.get("mass", 0.27), 0.0001, 1.0e9, 4),
            )
            for col in range(4):
                check = QCheckBox(self)
                check.setChecked(old.get("columns", [True, True, True, True])[col])
                check.stateChanged.connect(self._refresh_previews)
                self._story_table.setCellWidget(row, 3 + col, check)

            old_orient = old.get("orientation", ["Weak axis"] * 4)
            if isinstance(old_orient, str):
                old_orient = [old_orient] * 4
            for col in range(4):
                orient = QComboBox(self)
                orient.addItems(["Weak axis", "Strong axis"])
                orient.setCurrentText(old_orient[col] if col < len(old_orient) else "Weak axis")
                orient.currentTextChanged.connect(self._refresh_previews)
                self._story_table.setCellWidget(row, 7 + col, orient)

        self._rebuild_mass_table(existing_masses)
        self._sync_mode_counts_to_story_count()
        self._refresh_previews()

    def _rebuild_mass_table(self, existing: dict[int, list[float]] | None = None) -> None:
        existing = existing or {}
        n_story = int(self._story_count.value())
        self._mass_table.setRowCount(n_story)
        for row in range(n_story):
            story = row + 1
            self._mass_table.setItem(row, 0, QTableWidgetItem(str(story)))
            self._mass_table.item(row, 0).setFlags(self._mass_table.item(row, 0).flags())
            values = existing.get(story, [0.0, 0.0, 0.0, 0.0, 0.0])
            for col, value in enumerate(values[:5]):
                self._mass_table.setCellWidget(
                    row,
                    col + 1,
                    self._table_double(value, 0.0, 1.0e9, 4),
                )

    def _table_double(
        self,
        value: float,
        minimum: float,
        maximum: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(float(value))
        spin.setSingleStep(0.01)
        return spin

    def _read_story_table(self, *, allow_empty: bool = False) -> dict[int, dict[str, Any]]:
        rows = self._story_table.rowCount()
        if rows == 0 and allow_empty:
            return {}

        out: dict[int, dict[str, Any]] = {}
        for row in range(rows):
            story = row + 1
            height = self._story_table.cellWidget(row, 1)
            mass = self._story_table.cellWidget(row, 2)
            if height is None or mass is None:
                if allow_empty:
                    continue
                raise ValueError("Story table is incomplete.")

            columns = []
            for col in range(4):
                widget = self._story_table.cellWidget(row, 3 + col)
                columns.append(bool(widget.isChecked()) if isinstance(widget, QCheckBox) else False)

            orientation = []
            for col in range(4):
                widget = self._story_table.cellWidget(row, 7 + col)
                if isinstance(widget, QComboBox):
                    orientation.append(widget.currentText())
                else:
                    orientation.append("Weak axis")

            out[story] = {
                "height": float(height.value()),
                "mass": float(mass.value()),
                "columns": columns,
                "orientation": orientation,
            }
        return out

    def _read_mass_table(self, *, allow_empty: bool = False) -> dict[int, list[float]]:
        if not hasattr(self, "_mass_table"):
            return {}
        rows = self._mass_table.rowCount()
        if rows == 0 and allow_empty:
            return {}

        out: dict[int, list[float]] = {}
        for row in range(rows):
            story = row + 1
            values = []
            for col in range(5):
                widget = self._mass_table.cellWidget(row, col + 1)
                if widget is None and allow_empty:
                    values.append(0.0)
                elif isinstance(widget, QDoubleSpinBox):
                    values.append(float(widget.value()))
                else:
                    raise ValueError("Additional mass table is incomplete.")
            out[story] = values
        return out

    def _sync_mode_counts_to_story_count(self) -> None:
        n_story = int(self._story_count.value())
        target_modes = max(2, min(4, n_story))
        if hasattr(self, "_num_modes"):
            self._num_modes.setValue(target_modes)
        if hasattr(self, "_n_calib_modes"):
            self._n_calib_modes.setValue(max(1, min(3, target_modes, n_story)))
        self._rebuild_exp_data_widgets()

    def _set_all_orientations(self, value: str) -> None:
        for row in range(self._story_table.rowCount()):
            for col in range(4):
                widget = self._story_table.cellWidget(row, 7 + col)
                if isinstance(widget, QComboBox):
                    widget.setCurrentText(value)
        self._refresh_previews()

    @Slot()
    def _refresh_previews(self, *_: object) -> None:
        if not hasattr(self, "_structure_preview"):
            return
        story_data = self._read_story_table(allow_empty=True)
        columns = {
            story: list(data.get("columns", [True, True, True, True]))
            for story, data in story_data.items()
        }
        orientations = {
            story: list(data.get("orientation", ["Weak axis"] * 4))
            for story, data in story_data.items()
        }
        self._structure_preview.set_model(
            n_story=int(self._story_count.value()),
            lx=float(self._lx.value()),
            ly=float(self._ly.value()),
            columns=columns,
            orientations=orientations,
        )
        if hasattr(self, "_mass_legend"):
            self._mass_legend.set_dimensions(float(self._lx.value()), float(self._ly.value()))

    @Slot()
    def _browse_project_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Output Workspace Folder",
            self._project_dir_edit.text().strip(),
        )
        if path:
            self._project_dir_edit.setText(path)

    @Slot()
    def _browse_ground_motion(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose acceleration input",
            str(Path(self._gm_file_edit.text().strip()).parent),
            "Text Files (*.txt);;All Files (*)",
        )
        if path:
            self._gm_file_edit.setText(path)

    def _collect_params(self) -> dict[str, Any]:
        project_dir = Path(self._project_dir_edit.text().strip())
        if not project_dir.is_dir():
            raise ValueError(f"Output workspace folder does not exist: {project_dir}")

        story_data = self._read_story_table()
        mass_data = self._read_mass_table()
        n_story = int(self._story_count.value())

        story_column_layout = {}
        column_orientation_layout = {}
        additional_masses = {}
        for story in range(1, n_story + 1):
            row = story_data[story]
            cols = [index + 1 for index, present in enumerate(row["columns"]) if present]
            if not cols:
                raise ValueError(f"Story {story} must have at least one column present.")
            story_column_layout[story] = cols
            orientations = row["orientation"]
            column_orientation_layout[story] = {
                col: "strong" if orientations[col - 1] == "Strong axis" else "weak"
                for col in (1, 2, 3, 4)
            }
            additional_masses[story] = list(mass_data.get(story, [0.0, 0.0, 0.0, 0.0, 0.0]))

        gm_file = Path(self._gm_file_edit.text().strip())
        if not gm_file.exists():
            raise ValueError(f"Ground-motion file not found: {gm_file}")

        params = {
            "Lx": float(self._lx.value()),
            "Ly": float(self._ly.value()),
            "nStory": n_story,
            "story_heights": [story_data[story]["height"] for story in range(1, n_story + 1)],
            "floor_masses": [story_data[story]["mass"] for story in range(1, n_story + 1)],
            "t_column": float(self._t_column.value()),
            "b_column": float(self._b_column.value()),
            "b_beam": float(self._b_beam.value()),
            "h_beam": float(self._h_beam.value()),
            "E": float(self._youngs_modulus.value()),
            "nu": float(self._poisson.value()),
            "numModes": int(self._num_modes.value()),
            "zeta": float(self._zeta.value()),
            "gmFactor": float(self._gm_factor.value()),
            "gmFile": str(gm_file),
            "dtGM": float(self._dt_gm.value()),
            "show_info": bool(self._show_info.isChecked()),
            "run_transient": bool(self._run_transient.isChecked()),
            "enable_calibration": bool(self._enable_calibration.isChecked()),
            "use_mode_shapes": bool(self._use_mode_shapes.isChecked()),
            "mass_calibration_scope": (
                "total_mass"
                if self._mass_scope.currentText() == "Total mass including additional masses"
                else "self_weight_only"
            ),
            "nCalibModes": int(self._n_calib_modes.value()),
            "freq_tol_percent": float(self._freq_tol.value()),
            "w_freq": float(self._w_freq.value()),
            "w_mode": float(self._w_mode.value()),
            "E_scale_lb": float(self._e_lb.value()),
            "E_scale_ub": float(self._e_ub.value()),
            "m_scale_lb": float(self._m_lb.value()),
            "m_scale_ub": float(self._m_ub.value()),
            "max_nfev": int(self._max_nfev.value()),
            "show_uncalibrated_response": bool(self._show_uncalibrated_response.isChecked()),
            "story_column_layout": story_column_layout,
            "column_orientation_layout": column_orientation_layout,
            "additional_masses": additional_masses,
        }

        # Experimental modal data source
        if self._exp_source.currentText() == "Manual input":
            params["experimental_data_source"] = "manual"
            params["experimental_modal_data"] = self._read_exp_data()
        else:
            params["experimental_data_source"] = "json"
            params["experimental_json_path"] = self._exp_json_edit.text().strip()

        self._validate_params(params)
        return params

    def _validate_params(self, params: dict[str, Any]) -> None:
        missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError(
                "Missing Python package(s): "
                + ", ".join(missing)
                + "\n\nInstall them in the same Python environment used to launch SensePi:\n"
                + 'pip install -e ".[model-updating]"'
            )
        if params["nCalibModes"] > params["numModes"]:
            raise ValueError("Calibration modes cannot exceed total extracted modes.")
        if params["nCalibModes"] > params["nStory"]:
            raise ValueError("Calibration modes should not exceed the number of stories.")
        if params["E_scale_lb"] >= params["E_scale_ub"]:
            raise ValueError("E scale lower bound must be smaller than upper bound.")
        if params["m_scale_lb"] >= params["m_scale_ub"]:
            raise ValueError("Mass scale lower bound must be smaller than upper bound.")

    def _display_png(self, label: _ScaledImageLabel, png_bytes: bytes | None) -> None:
        label.set_figure(png_bytes)

    def _prepare_live_view(self, action: str, run_transient: bool) -> None:
        """Choose whether the Output tab shows static PNGs or the live 2×2 grid."""
        if action == "run" and run_transient:
            self._fig1_label.hide()
            self._fig2_label.hide()
            self._live_response_canvas.show()
            self._run_right_widget.show()
            self._live_response_canvas.initialize(None)
            self._live_3d_canvas.initialize_model(None)
        else:
            self._live_response_canvas.hide()
            self._run_right_widget.hide()
            self._fig1_label.show()
            self._fig2_label.show()

    @Slot(object)
    def _on_animation_frame(self, frame: dict[str, Any]) -> None:
        kind = frame.get("kind")
        if kind == "init":
            self._fig1_label.hide()
            self._fig2_label.hide()
            self._live_response_canvas.show()
            self._run_right_widget.show()
            self._live_response_canvas.initialize(frame.get("overlay_response"))
            self._live_3d_canvas.initialize_model(
                frame.get("modal_data"),
                title=str(frame.get("title") or "3D Transient Response"),
            )
            return

        self._live_response_canvas.update_frame(frame)
        self._live_3d_canvas.update_frame(frame)

    def _start_worker(self, action: str) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "Model updating is already running.")
            return

        try:
            params = self._collect_params()
        except Exception as exc:
            self._tabs.setCurrentWidget(self._output_tab)
            self._log.clear()
            self._append_log(f"Invalid model-updating input:\n{exc}\n")
            self._status.setText("Model updating input is invalid.")
            QMessageBox.critical(self, "Invalid model-updating input", str(exc))
            return

        request = _WorkerRequest(
            action=action,
            project_dir=Path(self._project_dir_edit.text().strip()),
            params=params,
            calibration_state=copy.deepcopy(self._calibration_state),
        )
        worker = _ModelUpdatingWorker(request)
        thread = QThread(self)
        worker.moveToThread(thread)

        worker.log.connect(self._append_log)
        worker.frame.connect(self._on_animation_frame)
        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)

        self._worker = worker
        self._thread = thread
        action_label = "analysis" if action == "run" else action
        self._set_busy(True, f"Running {action_label}...")
        self._log.clear()
        self._tabs.setCurrentWidget(self._output_tab)
        self._prepare_live_view(action, params.get("run_transient", False))
        thread.start()

    @Slot(str)
    def _append_log(self, text: str) -> None:
        if text:
            self._log.moveCursor(QTextCursor.End)
            self._log.insertPlainText(text)
            self._log.moveCursor(QTextCursor.End)

    @Slot(object)
    def _on_worker_finished(self, result: dict[str, Any]) -> None:
        action = result.get("action")
        self._tabs.setCurrentWidget(self._output_tab)

        if action == "run" and result.get("transient_response") is not None:
            # Keep the live canvases; populate the individual mode shape cells.
            self._fig1_label.hide()
            self._fig2_label.hide()
            self._live_response_canvas.show()
            self._run_right_widget.show()
            mode_pngs = result.get("fig_mode_pngs") or []
            for i, lbl in enumerate(self._mode_shape_labels):
                lbl.set_figure(mode_pngs[i] if i < len(mode_pngs) else None)
        else:
            self._live_response_canvas.hide()
            self._run_right_widget.hide()
            self._fig1_label.show()
            self._fig2_label.show()
            self._display_png(self._fig1_label, result.get("fig1_png"))
            self._display_png(self._fig2_label, result.get("fig2_png"))

        if action == "calibrate":
            self._calibration_state = _CalibrationState(
                available=True,
                input_signature=result["signature"],
                calibrated_params=copy.deepcopy(result["params"]),
                uncalibrated_response=result.get("uncalibrated_response"),
                prior_freqs=result.get("prior_freqs"),
                prior_periods=result.get("prior_periods"),
                prior_mode_shapes=result.get("prior_mode_shapes"),
            )
            if result.get("report_text"):
                self._append_log("\n\n" + result["report_text"])
            self._append_log("\n─── Calibration complete. Results shown above. ───\n")
            self._status.setText(f"Calibration complete. Output: {result['output_dir']}")
        else:
            suffix = " using calibrated parameters" if result.get("used_calibration") else ""
            self._append_log("\n─── Analysis complete. ───\n")
            self._status.setText(f"Analysis complete{suffix}. Output: {result['output_dir']}")
        self._set_busy(False, self._status.text())

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self._set_busy(False, "Model updating failed.")
        QMessageBox.critical(self, "Model updating failed", message)

    def _clear_worker(self) -> None:
        self._thread = None
        self._worker = None

    def _set_busy(self, busy: bool, status: str) -> None:
        self._calibrate_btn.setEnabled(not busy)
        self._run_btn.setEnabled(not busy)
        self._status.setText(status)
