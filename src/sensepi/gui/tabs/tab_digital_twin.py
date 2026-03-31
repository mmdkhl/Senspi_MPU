from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ...opensees import (
    DigitalTwinAnalysisParams,
    DigitalTwinAnalysisResult,
    run_digital_twin_analysis,
)


@dataclass
class ModalFrequencyComparison:
    simulated_hz: list[float]
    measured_hz: list[float]
    abs_error_hz: list[float]



def _default_input_dir() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "DigitalTwin OpenSees"
        / "DigitalTwin OpenSees"
        / "input"
    )



def _default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "DigitalTwin OpenSees"
        / "DigitalTwin OpenSees"
        / "output"
    )



def _parse_float_list(text: str, *, field_name: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"{field_name} must contain at least one number.")
    return values


@dataclass
class _RunRequest:
    params: DigitalTwinAnalysisParams


class _DigitalTwinWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, request: _RunRequest) -> None:
        super().__init__()
        self._request = request

    @Slot()
    def run(self) -> None:
        try:
            result = run_digital_twin_analysis(self._request.params)
        except Exception as exc:  # pragma: no cover - UI surface
            self.error.emit(str(exc))
        else:
            self.finished.emit(result)


class DigitalTwinTab(QWidget):
    """In-app controls for running the OpenSees-based digital twin model."""

    def __init__(self, fft_tab=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fft_tab = fft_tab
        self._worker_thread: QThread | None = None
        self._worker: _DigitalTwinWorker | None = None
        self._last_result: DigitalTwinAnalysisResult | None = None
        self._build_ui()
        self._set_default_paths()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        files_group = QGroupBox("Files", self)
        files_form = QFormLayout(files_group)
        self._gm_file_edit = QLineEdit(self)
        gm_row = QHBoxLayout()
        gm_row.addWidget(self._gm_file_edit, stretch=1)
        self._browse_gm_button = QPushButton("Browse...", self)
        gm_row.addWidget(self._browse_gm_button)
        gm_row_widget = QWidget(self)
        gm_row_widget.setLayout(gm_row)
        files_form.addRow("Ground motion input:", gm_row_widget)

        self._output_dir_edit = QLineEdit(self)
        out_row = QHBoxLayout()
        out_row.addWidget(self._output_dir_edit, stretch=1)
        self._browse_out_button = QPushButton("Browse...", self)
        out_row.addWidget(self._browse_out_button)
        out_row_widget = QWidget(self)
        out_row_widget.setLayout(out_row)
        files_form.addRow("Output directory:", out_row_widget)
        root.addWidget(files_group)

        model_group = QGroupBox("Model Parameters", self)
        model_form = QFormLayout(model_group)

        self._story_heights_edit = QLineEdit("0.24, 0.24, 0.24", self)
        self._floor_masses_edit = QLineEdit("0.3, 0.5, 0.3", self)
        model_form.addRow("Story heights (m):", self._story_heights_edit)
        model_form.addRow("Floor masses (kg):", self._floor_masses_edit)

        self._num_modes_spin = QSpinBox(self)
        self._num_modes_spin.setRange(1, 12)
        self._num_modes_spin.setValue(4)
        model_form.addRow("Modes:", self._num_modes_spin)

        self._dt_spin = QDoubleSpinBox(self)
        self._dt_spin.setRange(0.0001, 1.0)
        self._dt_spin.setDecimals(4)
        self._dt_spin.setValue(0.01)
        model_form.addRow("Time step (s):", self._dt_spin)

        self._damping_spin = QDoubleSpinBox(self)
        self._damping_spin.setRange(0.0, 0.5)
        self._damping_spin.setDecimals(4)
        self._damping_spin.setValue(0.005)
        model_form.addRow("Damping ratio:", self._damping_spin)
        root.addWidget(model_group)

        compare_group = QGroupBox("FFT Comparison", self)
        compare_form = QFormLayout(compare_group)
        self._fft_source_label = QLabel(
            "The first 3 measured eigenfrequencies are taken from the current Spectrum/FFT analysis.",
            self,
        )
        self._fft_source_label.setWordWrap(True)
        compare_form.addRow("Measured source:", self._fft_source_label)
        root.addWidget(compare_group)

        button_row = QHBoxLayout()
        self._run_button = QPushButton("Run Analysis", self)
        button_row.addWidget(self._run_button)
        button_row.addStretch()
        root.addLayout(button_row)

        self._status_label = QLabel(
            "Run the Digital Twin model, then compare its first 3 modal frequencies against the Spectrum tab.",
            self,
        )
        root.addWidget(self._status_label)

        self._summary = QPlainTextEdit(self)
        self._summary.setReadOnly(True)
        self._summary.setPlaceholderText("Digital Twin summary and modal-frequency comparison will appear here.")
        root.addWidget(self._summary, stretch=1)

        self._figure = Figure(figsize=(8, 7))
        self._canvas = FigureCanvasQTAgg(self._figure)
        root.addWidget(self._canvas, stretch=2)

        self._browse_gm_button.clicked.connect(self._browse_ground_motion)
        self._browse_out_button.clicked.connect(self._browse_output_dir)
        self._run_button.clicked.connect(self._start_run)

    def _set_default_paths(self) -> None:
        input_dir = _default_input_dir()
        output_dir = _default_output_dir()
        default_file = input_dir / "sine_1Hz_accel.txt"
        self._gm_file_edit.setText(str(default_file))
        self._output_dir_edit.setText(str(output_dir))

    @Slot()
    def _browse_ground_motion(self) -> None:
        current_dir = str(Path(self._gm_file_edit.text().strip()).parent or _default_input_dir())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose acceleration input",
            current_dir,
            "Text Files (*.txt);;All Files (*)",
        )
        if path:
            self._gm_file_edit.setText(path)

    @Slot()
    def _browse_output_dir(self) -> None:
        current_dir = self._output_dir_edit.text().strip() or str(_default_output_dir())
        path = QFileDialog.getExistingDirectory(self, "Choose output directory", current_dir)
        if path:
            self._output_dir_edit.setText(path)

    def _collect_params(self) -> DigitalTwinAnalysisParams:
        story_heights = _parse_float_list(
            self._story_heights_edit.text(),
            field_name="Story heights",
        )
        floor_masses = _parse_float_list(
            self._floor_masses_edit.text(),
            field_name="Floor masses",
        )
        return DigitalTwinAnalysisParams(
            gm_file=Path(self._gm_file_edit.text().strip()),
            output_dir=Path(self._output_dir_edit.text().strip()),
            story_heights=story_heights,
            floor_masses=floor_masses,
            num_modes=int(self._num_modes_spin.value()),
            dt_seconds=float(self._dt_spin.value()),
            damping_ratio=float(self._damping_spin.value()),
        )

    @Slot()
    def _start_run(self) -> None:
        try:
            params = self._collect_params()
            params.validate()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Digital Twin input", str(exc))
            return

        self._run_button.setEnabled(False)
        self._status_label.setText("Running Digital Twin analysis...")
        self._summary.setPlainText("OpenSees run in progress...")

        worker = _DigitalTwinWorker(_RunRequest(params=params))
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_run_finished)
        worker.error.connect(self._on_run_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self._worker = worker
        self._worker_thread = thread
        thread.start()

    @Slot(object)
    def _on_run_finished(self, result: DigitalTwinAnalysisResult) -> None:
        self._last_result = result
        self._run_button.setEnabled(True)
        self._status_label.setText(f"Digital Twin run complete. Output: {result.output_dir}")
        self._summary.setPlainText(self._format_summary(result))
        self._render_plots(result)

    @Slot(str)
    def _on_run_error(self, message: str) -> None:
        self._run_button.setEnabled(True)
        self._status_label.setText("Digital Twin run failed.")
        self._summary.setPlainText(message)
        if "OpenSeesPy is not installed" in message:
            QMessageBox.warning(self, "Missing dependency", message)
        else:
            QMessageBox.critical(self, "Digital Twin failed", message)

    def _clear_worker(self) -> None:
        self._worker = None
        self._worker_thread = None

    def _measured_fft_frequencies(self, count: int = 3) -> list[float]:
        fft_tab = self._fft_tab
        if fft_tab is None:
            return []
        getter = getattr(fft_tab, "estimate_modal_frequencies", None)
        if getter is None:
            return []
        try:
            freqs = list(getter(count=count))
        except Exception:
            return []
        cleaned: list[float] = []
        for value in freqs:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0.0 and np.isfinite(numeric):
                cleaned.append(numeric)
        return cleaned[:count]

    def _build_modal_comparison(self, result: DigitalTwinAnalysisResult) -> ModalFrequencyComparison | None:
        simulated = [float(value) for value in result.modal.frequencies_hz[:3]]
        measured = self._measured_fft_frequencies(count=3)
        if len(simulated) < 3 or len(measured) < 3:
            return None
        abs_error = [abs(sim - meas) for sim, meas in zip(simulated, measured)]
        return ModalFrequencyComparison(
            simulated_hz=simulated,
            measured_hz=measured,
            abs_error_hz=abs_error,
        )

    def _format_summary(self, result: DigitalTwinAnalysisResult) -> str:
        lines = [
            f"Ground motion: {result.params.gm_file}",
            f"Output directory: {result.output_dir}",
            f"Stories: {len(result.params.story_heights)}",
            f"Story heights (m): {', '.join(f'{value:.3f}' for value in result.params.story_heights)}",
            f"Floor masses (kg): {', '.join(f'{value:.3f}' for value in result.params.floor_masses)}",
            "",
            "OpenSees eigenfrequencies:",
        ]
        for index, freq in enumerate(result.modal.frequencies_hz[:3], start=1):
            lines.append(f"Mode {index}: {float(freq):.3f} Hz")

        comparison = self._build_modal_comparison(result)
        if comparison is None:
            lines.extend(
                [
                    "",
                    "Measured FFT eigenfrequencies:",
                    "Not available yet. Start acquisition and open the Spectrum tab so the GUI has FFT data to compare.",
                ]
            )
        else:
            lines.extend(["", "Measured FFT eigenfrequencies:"])
            for index, freq in enumerate(comparison.measured_hz, start=1):
                lines.append(f"Mode {index}: {freq:.3f} Hz")
            lines.extend(["", "Mode-by-mode comparison:"])
            for index, (sim, meas, err) in enumerate(
                zip(comparison.simulated_hz, comparison.measured_hz, comparison.abs_error_hz),
                start=1,
            ):
                lines.append(
                    f"Mode {index}: OpenSees={sim:.3f} Hz, FFT={meas:.3f} Hz, |error|={err:.3f} Hz"
                )

        if result.transient.time_s:
            max_disp = max(abs(value) for value in result.transient.roof_disp_x_m)
            max_accel = max(abs(value) for value in result.transient.roof_accel_x_ms2)
            lines.extend(
                [
                    "",
                    "Transient results:",
                    f"Samples: {len(result.transient.time_s)}",
                    f"Max roof displacement X: {max_disp:.6e} m",
                    f"Max roof acceleration X: {max_accel:.6e} m/s^2",
                ]
            )
        return "\n".join(lines)

    def _render_plots(self, result: DigitalTwinAnalysisResult) -> None:
        self._figure.clear()
        comparison = self._build_modal_comparison(result)
        has_comparison = comparison is not None
        if has_comparison:
            ax_disp = self._figure.add_subplot(3, 1, 1)
            ax_accel = self._figure.add_subplot(3, 1, 2)
            ax_modes = self._figure.add_subplot(3, 1, 3)
        else:
            ax_disp = self._figure.add_subplot(2, 1, 1)
            ax_accel = self._figure.add_subplot(2, 1, 2)
            ax_modes = None

        time_s = np.asarray(result.transient.time_s, dtype=np.float64)
        roof_disp = np.asarray(result.transient.roof_disp_x_m, dtype=np.float64)
        roof_accel = np.asarray(result.transient.roof_accel_x_ms2, dtype=np.float64)

        ax_disp.plot(time_s, roof_disp, color="navy")
        ax_disp.set_title("Roof Displacement X")
        ax_disp.set_ylabel("m")
        ax_disp.grid(True, alpha=0.3)

        ax_accel.plot(time_s, roof_accel, color="darkred")
        ax_accel.set_title("Roof Acceleration X")
        ax_accel.set_xlabel("Time (s)")
        ax_accel.set_ylabel("m/s^2")
        ax_accel.grid(True, alpha=0.3)

        if ax_modes is not None and comparison is not None:
            modes = np.array([1, 2, 3], dtype=float)
            width = 0.35
            ax_modes.bar(modes - width / 2.0, comparison.simulated_hz, width=width, color="steelblue", label="OpenSees")
            ax_modes.bar(modes + width / 2.0, comparison.measured_hz, width=width, color="darkorange", label="FFT")
            ax_modes.set_xticks(modes)
            ax_modes.set_xticklabels(["Mode 1", "Mode 2", "Mode 3"])
            ax_modes.set_ylabel("Hz")
            ax_modes.set_title("First 3 Eigenfrequency Comparison")
            ax_modes.grid(True, axis="y", alpha=0.3)
            ax_modes.legend(loc="upper right", fontsize=8)

        self._figure.tight_layout()
        self._canvas.draw_idle()
