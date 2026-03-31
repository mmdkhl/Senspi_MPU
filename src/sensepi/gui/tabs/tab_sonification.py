from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...sonification.structural import load_structural_csv, sonify_file


class SonificationTab(QWidget):
    """GUI controls for converting structural CSV responses into WAV audio."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset_measurements: list[str] = []
        self._build_ui()
        self._set_mode_widgets()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Input / output file controls
        io_group = QGroupBox("Files")
        io_form = QFormLayout(io_group)
        self._input_edit = QLineEdit(self)
        input_row = QHBoxLayout()
        input_row.addWidget(self._input_edit, stretch=1)
        self._browse_input_btn = QPushButton("Browse CSV...", self)
        input_row.addWidget(self._browse_input_btn)
        self._inspect_btn = QPushButton("Load CSV Info", self)
        input_row.addWidget(self._inspect_btn)
        input_row_widget = QWidget(self)
        input_row_widget.setLayout(input_row)
        io_form.addRow("Input CSV:", input_row_widget)

        self._output_edit = QLineEdit(self)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_edit, stretch=1)
        self._browse_output_btn = QPushButton("Browse WAV...", self)
        output_row.addWidget(self._browse_output_btn)
        output_row_widget = QWidget(self)
        output_row_widget.setLayout(output_row)
        io_form.addRow("Output WAV:", output_row_widget)
        layout.addWidget(io_group)

        # Shared parameters
        common_group = QGroupBox("Common")
        common_form = QFormLayout(common_group)
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItems(["melody", "harmonic"])
        common_form.addRow("Mode:", self._mode_combo)

        self._joint_spin = QSpinBox(self)
        self._joint_spin.setRange(0, 10000)
        self._joint_spin.setValue(28)
        common_form.addRow("Joint:", self._joint_spin)

        self._sample_rate_spin = QSpinBox(self)
        self._sample_rate_spin.setRange(8000, 192000)
        self._sample_rate_spin.setValue(44100)
        common_form.addRow("Sample rate (Hz):", self._sample_rate_spin)
        layout.addWidget(common_group)

        # Melody parameters
        self._melody_group = QGroupBox("Melody Parameters")
        melody_form = QFormLayout(self._melody_group)
        self._measurement_combo = QComboBox(self)
        self._measurement_combo.setEditable(True)
        self._measurement_combo.addItem("U1")
        melody_form.addRow("Measurement:", self._measurement_combo)

        self._root_note_spin = QSpinBox(self)
        self._root_note_spin.setRange(48, 84)
        self._root_note_spin.setValue(60)
        melody_form.addRow("Root MIDI note:", self._root_note_spin)

        self._note_duration_spin = QDoubleSpinBox(self)
        self._note_duration_spin.setRange(0.01, 2.0)
        self._note_duration_spin.setDecimals(3)
        self._note_duration_spin.setSingleStep(0.01)
        self._note_duration_spin.setValue(0.15)
        melody_form.addRow("Note duration (s):", self._note_duration_spin)
        layout.addWidget(self._melody_group)

        # Harmonic parameters
        self._harmonic_group = QGroupBox("Harmonic Parameters")
        harmonic_form = QFormLayout(self._harmonic_group)

        self._r1_edit = QLineEdit("R1", self)
        harmonic_form.addRow("R1 measurement:", self._r1_edit)

        self._u1_edit = QLineEdit("U1", self)
        harmonic_form.addRow("U1 measurement:", self._u1_edit)

        self._base_freq_spin = QDoubleSpinBox(self)
        self._base_freq_spin.setRange(20.0, 5000.0)
        self._base_freq_spin.setDecimals(2)
        self._base_freq_spin.setValue(220.0)
        harmonic_form.addRow("Base frequency (Hz):", self._base_freq_spin)

        self._num_harmonics_spin = QSpinBox(self)
        self._num_harmonics_spin.setRange(1, 200)
        self._num_harmonics_spin.setValue(40)
        harmonic_form.addRow("Number of harmonics:", self._num_harmonics_spin)

        self._harmonic_depth_spin = QDoubleSpinBox(self)
        self._harmonic_depth_spin.setRange(0.0, 1.0)
        self._harmonic_depth_spin.setDecimals(4)
        self._harmonic_depth_spin.setSingleStep(0.01)
        self._harmonic_depth_spin.setValue(0.05)
        harmonic_form.addRow("Harmonic mod depth:", self._harmonic_depth_spin)

        self._freq_depth_spin = QDoubleSpinBox(self)
        self._freq_depth_spin.setRange(0.0, 1000.0)
        self._freq_depth_spin.setDecimals(3)
        self._freq_depth_spin.setSingleStep(0.5)
        self._freq_depth_spin.setValue(5.0)
        harmonic_form.addRow("Freq mod depth:", self._freq_depth_spin)
        layout.addWidget(self._harmonic_group)

        self._generate_btn = QPushButton("Generate WAV", self)
        layout.addWidget(self._generate_btn)

        self._status = QLabel("Choose input CSV and output WAV, then click Generate WAV.", self)
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._browse_input_btn.clicked.connect(self._browse_input)
        self._browse_output_btn.clicked.connect(self._browse_output)
        self._inspect_btn.clicked.connect(self._load_csv_info)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._generate_btn.clicked.connect(self._on_generate)

    @Slot(str)
    def _on_mode_changed(self, _mode: str) -> None:
        self._set_mode_widgets()

    def _set_mode_widgets(self) -> None:
        mode = self._mode_combo.currentText().strip().lower()
        is_melody = mode == "melody"
        self._melody_group.setEnabled(is_melody)
        self._harmonic_group.setEnabled(not is_melody)
        self._sample_rate_spin.setValue(44100 if is_melody else 48000)

    @Slot()
    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose Structural CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            self._input_edit.setText(path)

    @Slot()
    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Choose Output WAV", "", "WAV Files (*.wav)")
        if path:
            if not path.lower().endswith(".wav"):
                path = f"{path}.wav"
            self._output_edit.setText(path)

    @Slot()
    def _load_csv_info(self) -> None:
        input_path = self._input_edit.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Missing input", "Choose an input CSV first.")
            return

        try:
            dataset = load_structural_csv(input_path)
        except Exception as exc:
            QMessageBox.critical(self, "CSV parse failed", str(exc))
            return

        joints = dataset.joints()
        if joints:
            self._joint_spin.setValue(joints[0])
        self._dataset_measurements = list(dataset.measurements)

        self._measurement_combo.clear()
        for name in self._dataset_measurements:
            self._measurement_combo.addItem(name)

        if "U1" in self._dataset_measurements:
            self._measurement_combo.setCurrentText("U1")
        if "R1" in self._dataset_measurements:
            self._r1_edit.setText("R1")
        if "U1" in self._dataset_measurements:
            self._u1_edit.setText("U1")

        self._status.setText(
            f"Loaded {Path(input_path).name}: joints={joints[:10]} measurements={', '.join(self._dataset_measurements[:8])}"
        )

    @Slot()
    def _on_generate(self) -> None:
        input_path = self._input_edit.text().strip()
        output_path = self._output_edit.text().strip()
        if not input_path or not output_path:
            QMessageBox.warning(self, "Missing files", "Set both input CSV and output WAV paths.")
            return

        mode = self._mode_combo.currentText().strip().lower()
        kwargs = {
            "measurement": self._measurement_combo.currentText().strip() or "U1",
            "root_note": int(self._root_note_spin.value()),
            "note_duration": float(self._note_duration_spin.value()),
            "r1_measurement": self._r1_edit.text().strip() or "R1",
            "u1_measurement": self._u1_edit.text().strip() or "U1",
            "base_freq": float(self._base_freq_spin.value()),
            "num_harmonics": int(self._num_harmonics_spin.value()),
            "harmonic_mod_depth": float(self._harmonic_depth_spin.value()),
            "freq_mod_depth": float(self._freq_depth_spin.value()),
            "sample_rate": int(self._sample_rate_spin.value()),
        }

        try:
            self._generate_btn.setEnabled(False)
            self._status.setText("Generating WAV...")
            out = sonify_file(
                csv_path=input_path,
                output_path=output_path,
                mode=mode,
                joint=int(self._joint_spin.value()),
                kwargs=kwargs,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Sonification failed", str(exc))
            self._status.setText("Generation failed.")
        else:
            self._status.setText(f"Wrote: {out}")
            QMessageBox.information(self, "Success", f"WAV generated:\n{out}")
        finally:
            self._generate_btn.setEnabled(True)
