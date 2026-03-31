from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Slot, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...config.app_config import AppPaths


class LogsTab(QWidget):
    """
    Application log viewer for troubleshooting the desktop GUI.

    Responsibilities:
    - Enumerate log files under :class:`AppPaths.logs` and present them for quick
      inspection.
    - Provide a tail/follow mode to watch runtime output while other tabs are
      driving live streams or syncing recordings.
    - Focused purely on diagnostics; it does not participate in data flow
      between recorder, signals, or FFT tabs.
    """

    _MAX_READ_BYTES = 250_000

    def __init__(
        self, app_paths: AppPaths | None = None, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._paths = app_paths or AppPaths()
        self._paths.ensure()
        self._log_files: list[Path] = []

        # Timer used when “Follow tail” is enabled
        self._timer = QTimer(self)
        self._timer.setInterval(1000)  # ms; adjust if needed
        self._timer.timeout.connect(self._on_timer_tick)

        layout = QVBoxLayout(self)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Log file:"))
        self._file_combo = QComboBox(self)
        control_row.addWidget(self._file_combo, stretch=1)
        self._refresh_button = QPushButton("Refresh", self)
        self._follow_check = QCheckBox("Follow tail", self)
        self._follow_check.setChecked(True)
        control_row.addWidget(self._follow_check)
        control_row.addWidget(self._refresh_button)
        layout.addLayout(control_row)

        self._view = QPlainTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self._view, stretch=1)

        self._status_label = QLabel("Select a log file to view.", self)
        layout.addWidget(self._status_label)

        self._refresh_button.clicked.connect(self._refresh_log_list)
        self._file_combo.currentIndexChanged.connect(
            self._on_selection_changed
        )
        self._follow_check.toggled.connect(self._on_follow_toggled)

        self._refresh_log_list()
        # Start/stop timer based on initial checkbox state
        self._on_follow_toggled(self._follow_check.isChecked())

    def _log_dir(self) -> Path:
        return self._paths.logs

    def _collect_log_files(self) -> List[Path]:
        log_dir = self._log_dir()
        if not log_dir.exists():
            return []
        patterns = ("*.log", "*.txt")
        files: dict[Path, float] = {}
        for pattern in patterns:
            for path in log_dir.glob(pattern):
                try:
                    files[path] = path.stat().st_mtime
                except FileNotFoundError:
                    continue
        return sorted(files, key=files.get, reverse=True)

    @Slot()
    def _refresh_log_list(self, _checked: bool = False) -> None:
        files = self._collect_log_files()
        self._log_files = files
        current_text = self._file_combo.currentText()
        was_blocked = self._file_combo.blockSignals(True)
        self._file_combo.clear()
        for path in files:
            self._file_combo.addItem(path.name, userData=str(path))
        if files:
            index = 0
            if current_text:
                idx = self._file_combo.findText(current_text)
                if idx >= 0:
                    index = idx
            self._file_combo.setCurrentIndex(index)
            self._file_combo.blockSignals(was_blocked)
            self._load_log_file(files[index])
        else:
            self._file_combo.blockSignals(was_blocked)
            self._view.clear()
            self._status_label.setText("No log files found.")

    @Slot(int)
    def _on_selection_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._log_files):
            self._view.clear()
            return
        self._load_log_file(self._log_files[index])

    @Slot(bool)
    def _on_follow_toggled(self, enabled: bool) -> None:
        """
        Start or stop periodic refresh of the current log file based on the
        'Follow tail' checkbox.
        """
        if enabled and self._file_combo.currentIndex() >= 0:
            self._timer.start()
        else:
            self._timer.stop()

    @Slot()
    def _on_timer_tick(self) -> None:
        """
        Timer callback: reload the currently selected file when following.

        This reuses _load_log_file and keeps all truncation / cursor behavior.
        """
        if not self._follow_check.isChecked():
            return
        index = self._file_combo.currentIndex()
        if index < 0 or index >= len(self._log_files):
            return
        self._load_log_file(self._log_files[index])

    def _load_log_file(self, path: Path) -> None:
        if not path.exists():
            self._status_label.setText(f"File not found: {path.name}")
            return
        try:
            text = self._read_tail(path)
        except Exception as exc:
            self._view.setPlainText("")
            self._status_label.setText(f"Failed to read {path.name}: {exc}")
            return

        self._view.setPlainText(text)
        if self._follow_check.isChecked():
            self._view.moveCursor(QTextCursor.End)
        self._status_label.setText(str(path))

    def _read_tail(self, path: Path) -> str:
        size = path.stat().st_size
        prefix = ""
        read_bytes = self._MAX_READ_BYTES
        if size > read_bytes:
            prefix = (
                f"... showing last {read_bytes // 1024} KB "
                "- file truncated for display ...\n"
            )
        with path.open("rb") as handle:
            if size > read_bytes:
                handle.seek(size - read_bytes)
            data = handle.read()
        text = data.decode("utf-8", errors="replace")
        return prefix + text
