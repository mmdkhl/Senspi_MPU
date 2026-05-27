from __future__ import annotations

import copy
import contextlib
import importlib.util
import io
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QPointF, QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPen, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


REQUIRED_MODULES = ("openseespy", "opsvis")


def _default_project_dir() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "opensees-model-updating"
        / "opensees-model-updating"
    )


@dataclass
class _CalibrationState:
    available: bool = False
    input_signature: str | None = None
    calibrated_params: dict[str, Any] | None = None
    uncalibrated_response: dict[str, Any] | None = None


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


class _HalfWidthContainer(QWidget):
    def __init__(self, target: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._target = target

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._target.setMaximumWidth(max(360, self.width() // 2))


class _ModelUpdatingWorker(QObject):
    log = Signal(str)
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
        inserted_path = False

        if str(project_dir) not in sys.path:
            sys.path.insert(0, str(project_dir))
            inserted_path = True

        try:
            os.chdir(project_dir)
            self.log.emit(f"Using OpenSees project: {project_dir}\n")

            from src.analysis.modal import (  # type: ignore
                export_modal_files,
                extract_modal_results,
            )
            from src.analysis.transient import (  # type: ignore
                run_transient_analysis_collect_data,
            )
            from src.calibration.calibrator import (  # type: ignore
                prepare_experimental_modal_data,
                run_calibration,
            )
            from src.io.loaders import write_json, write_text  # type: ignore
            from src.reporting.calibration_report import (  # type: ignore
                make_modal_comparison_report,
                report_to_text,
                save_calibration_summary_figure,
            )

            os.makedirs("input", exist_ok=True)
            os.makedirs("output", exist_ok=True)

            if self._request.action == "calibrate":
                original_params = copy.deepcopy(params)
                write_json("output/original_inputs.json", original_params)

                exp_data = prepare_experimental_modal_data(params)
                write_json("output/experimental_modal_data_loaded.json", exp_data["raw_data"])

                self.log.emit("\nRunning original modal analysis...\n")
                modal_before = extract_modal_results(
                    original_params, normalize_modes=True, show_info=params["show_info"]
                )
                export_modal_files(modal_before, "original")

                if params["enable_calibration"]:
                    self.log.emit("\nRunning automatic calibration...\n")
                    calib_result, calibrated_params = run_calibration(
                        original_params, exp_data, show_info=params["show_info"]
                    )
                    self.log.emit("\nCalibration finished.\n")
                    self.log.emit(f"Success: {calib_result.success}\n")
                    self.log.emit(f"Message: {calib_result.message}\n")

                    self.log.emit("\nRunning calibrated modal analysis...\n")
                    modal_after = extract_modal_results(
                        calibrated_params,
                        normalize_modes=True,
                        show_info=params["show_info"],
                    )
                    export_modal_files(modal_after, "calibrated")
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
            if use_precalibrated:
                self.log.emit("\nUsing calibrated parameters from this tab.\n")
                write_json("output/calibrated_inputs_used_for_run.json", final_params)
            else:
                self.log.emit("\nRunning current input model without stored calibration.\n")

            self.log.emit("\nPreparing model for analysis...\n")
            final_modal = extract_modal_results(
                final_params, normalize_modes=True, show_info=run_params["show_info"]
            )
            export_modal_files(final_modal, "run_model")
            modal_summary = _modal_summary(final_modal, run_params["numModes"])

            transient_response = None
            if run_params["run_transient"]:
                self.log.emit("\nRunning transient analysis...\n")
                transient_response = run_transient_analysis_collect_data(
                    final_params,
                    final_modal,
                    show_info=run_params["show_info"],
                    recorder_prefix="run_",
                )
                np.savez(
                    "output/run_transient_response.npz",
                    t_hist=transient_response["t_hist"],
                    u_hist=transient_response["u_hist"],
                    a_hist=transient_response["a_hist"],
                )
                self.log.emit("Saved transient response to output/run_transient_response.npz\n")

            return {
                "action": "run",
                "used_calibration": use_precalibrated,
                "output_dir": str(project_dir / "output"),
                "modal_summary": modal_summary,
                "transient_response": transient_response,
                "overlay_response": state.uncalibrated_response
                if use_precalibrated and run_params["show_uncalibrated_response"]
                else None,
            }
        finally:
            os.chdir(previous_cwd)
            if inserted_path:
                try:
                    sys.path.remove(str(project_dir))
                except ValueError:
                    pass


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
        project_form.addRow("OpenSees project folder:", project_widget)
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
        self._youngs_modulus = self._double_spin(1.0, 1.0e13, 6.9e10, 3)
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
        layout = QVBoxLayout(self._analysis_tab)
        group = QGroupBox("Modal and Transient Analysis", self)
        form = QFormLayout(group)

        self._num_modes = QSpinBox(self)
        self._num_modes.setRange(2, 20)
        self._num_modes.setValue(3)
        self._zeta = self._double_spin(0.0, 1.0, 0.02, 4)
        self._gm_factor = self._double_spin(-1.0e6, 1.0e6, 1.0, 4)
        self._dt_gm = self._double_spin(1.0e-8, 10.0, 0.01, 6)
        self._run_transient = QCheckBox(self)
        self._run_transient.setChecked(True)
        self._show_info = QCheckBox(self)

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
        form.addRow("Ground-motion file:", gm_widget)
        form.addRow("Run transient analysis:", self._run_transient)
        form.addRow("Verbose OpenSees output:", self._show_info)
        layout.addWidget(group)
        layout.addStretch(1)

        self._browse_gm_btn.clicked.connect(self._browse_ground_motion)

    def _build_calibration_tab(self) -> None:
        layout = QVBoxLayout(self._calibration_tab)
        group = QGroupBox("Calibration Settings", self)
        form = QFormLayout(group)

        self._enable_calibration = QCheckBox(self)
        self._enable_calibration.setChecked(True)
        self._use_mode_shapes = QCheckBox(self)
        self._use_mode_shapes.setChecked(True)
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
        self._max_nfev.setValue(80)
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
        layout.addStretch(1)

    def _build_output_tab(self) -> None:
        layout = QVBoxLayout(self._output_tab)
        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        layout.addWidget(self._log, stretch=1)

    def _set_defaults(self) -> None:
        project_dir = _default_project_dir()
        self._project_dir_edit.setText(str(project_dir))
        self._gm_file_edit.setText(str(project_dir / "input" / "sine_1Hz_accel.txt"))

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
            "Choose OpenSees Model Updating Project",
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
        if not (project_dir / "src").exists():
            raise ValueError(f"Invalid OpenSees project folder: {project_dir}")

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

    def _plot_results(self, result: dict[str, Any]) -> None:
        modal = result.get("modal_summary") or {}
        transient = result.get("transient_response")
        overlay = result.get("overlay_response")

        response_fig, (ax_u, ax_a) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        response_fig.canvas.manager.set_window_title("SensePi Run Analysis - Transient Response")
        if transient is not None:
            t = np.asarray(transient["t_hist"], dtype=float)
            u = np.asarray(transient["u_hist"], dtype=float)
            a = np.asarray(transient["a_hist"], dtype=float)
            ax_u.plot(t, u, color="#dc2626", linewidth=1.4, label="Run")
            ax_a.plot(t, a, color="#dc2626", linewidth=1.4, label="Run")
            if overlay is not None:
                ot = np.asarray(overlay["t_hist"], dtype=float)
                ax_u.plot(ot, np.asarray(overlay["u_hist"], dtype=float), color="#111827", linestyle="--", linewidth=1.0, label="Uncalibrated")
                ax_a.plot(ot, np.asarray(overlay["a_hist"], dtype=float), color="#111827", linestyle="--", linewidth=1.0, label="Uncalibrated")
            ax_u.legend(loc="best")
            ax_a.legend(loc="best")
        else:
            ax_u.text(0.5, 0.5, "Transient analysis was disabled.", ha="center", va="center")
            ax_a.set_axis_off()
        ax_u.set_title("Roof Displacement")
        ax_u.set_ylabel("Displacement (m)")
        ax_a.set_title("Roof Acceleration")
        ax_a.set_xlabel("Time (s)")
        ax_a.set_ylabel("Acceleration (m/s^2)")
        ax_u.grid(True, alpha=0.25)
        ax_a.grid(True, alpha=0.25)
        response_fig.tight_layout()

        shapes = modal.get("mode_shapes_ux_master") or []
        freqs = modal.get("freqs") or []
        if shapes:
            self._plot_3d_mode_shapes(modal, shapes, freqs)
        else:
            fig = plt.figure(figsize=(8, 5))
            fig.canvas.manager.set_window_title("SensePi Run Analysis - Mode Shapes")
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No modal result data returned.", ha="center", va="center")
            ax.set_axis_off()

        summary_fig = plt.figure(figsize=(7, 5))
        summary_fig.canvas.manager.set_window_title("SensePi Run Analysis - Modal Summary")
        ax = summary_fig.add_subplot(111)
        periods = modal.get("periods") or []
        if freqs:
            modes = np.arange(1, len(freqs) + 1)
            ax.plot(modes, freqs, marker="o", linewidth=1.6, color="#2563eb")
            ax.set_xticks(modes)
            ax.set_title("Run Model Frequencies")
            ax.set_xlabel("Mode")
            ax.set_ylabel("Frequency (Hz)")
            if periods:
                for mode, freq, period in zip(modes, freqs, periods):
                    ax.annotate(f"T={float(period):.3g}s", (mode, freq), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
            ax.grid(True, alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No frequency data returned.", ha="center", va="center")
            ax.set_axis_off()
        summary_fig.tight_layout()
        plt.show(block=False)

    def _plot_3d_mode_shapes(
        self,
        modal: dict[str, Any],
        shapes: list[Any],
        freqs: list[Any],
    ) -> None:
        node_xyz = {
            int(node): np.asarray(xyz, dtype=float)
            for node, xyz in (modal.get("node_xyz") or {}).items()
        }
        vis_elems = [(int(i), int(j)) for i, j in (modal.get("vis_elems") or [])]
        story_node_tags = {
            int(story): [int(node) for node in nodes]
            for story, nodes in (modal.get("story_node_tags") or {}).items()
        }
        if not node_xyz or not vis_elems or not story_node_tags:
            fig = plt.figure(figsize=(8, 5))
            fig.canvas.manager.set_window_title("SensePi Run Analysis - Mode Shapes")
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "3D geometry was not returned.", ha="center", va="center")
            ax.set_axis_off()
            return

        all_xyz = np.vstack(list(node_xyz.values()))
        xmin, ymin, zmin = np.min(all_xyz, axis=0)
        xmax, ymax, zmax = np.max(all_xyz, axis=0)
        span = max(float(xmax - xmin), float(ymax - ymin), float(zmax - zmin), 1.0e-6)
        xpad = ypad = zpad = span * 0.18
        deform_scale = span * 0.22

        n_modes = len(shapes)
        fig = plt.figure(figsize=(4.5 * n_modes, 5.5))
        fig.canvas.manager.set_window_title("SensePi Run Analysis - 3D Mode Shapes")
        for index, shape in enumerate(shapes):
            ax = fig.add_subplot(1, n_modes, index + 1, projection="3d")
            phi_by_node: dict[int, float] = {}
            phi = np.asarray(shape, dtype=float)
            for story, nodes in story_node_tags.items():
                if 1 <= story <= len(phi):
                    for node in nodes:
                        phi_by_node[node] = float(phi[story - 1])

            def deformed(node: int) -> np.ndarray:
                xyz = node_xyz[node].copy()
                xyz[0] += deform_scale * phi_by_node.get(node, 0.0)
                return xyz

            for n1, n2 in vis_elems:
                p1 = node_xyz[n1]
                p2 = node_xyz[n2]
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color="#9ca3af",
                    linestyle="--",
                    linewidth=0.8,
                )
                d1 = deformed(n1)
                d2 = deformed(n2)
                ax.plot(
                    [d1[0], d2[0]],
                    [d1[1], d2[1]],
                    [d1[2], d2[2]],
                    color="#1d4ed8",
                    linewidth=2.0,
                )

            title = f"Mode {index + 1}"
            if index < len(freqs):
                title += f"\nf={float(freqs[index]):.3g} Hz"
            ax.set_title(title)
            ax.set_xlim(float(xmin - xpad), float(xmax + xpad + deform_scale))
            ax.set_ylim(float(ymin - ypad), float(ymax + ypad))
            ax.set_zlim(float(zmin - zpad), float(zmax + zpad))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=25, azim=-70)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        fig.tight_layout()

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
        if action == "calibrate":
            self._calibration_state = _CalibrationState(
                available=True,
                input_signature=result["signature"],
                calibrated_params=copy.deepcopy(result["params"]),
                uncalibrated_response=result.get("uncalibrated_response"),
            )
            if result.get("report_text"):
                self._append_log("\n\n" + result["report_text"])
            self._status.setText(f"Calibration complete. Output: {result['output_dir']}")
            QMessageBox.information(self, "Calibration complete", "Calibration finished.")
        else:
            suffix = " using calibrated parameters" if result.get("used_calibration") else ""
            self._plot_results(result)
            self._status.setText(f"Analysis complete{suffix}. Output: {result['output_dir']}")
            QMessageBox.information(self, "Analysis complete", "Model analysis finished.")
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
