"""Record *where* the GUI thread is stuck when Windows reports "not responding".

A Windows ``AppHangB1`` event says only that python.exe stopped pumping
messages; it carries no Python stack. This watchdog closes that gap.

A QTimer on the GUI thread re-arms ``faulthandler.dump_traceback_later`` every
second. If the event loop stalls for ``timeout_s``, faulthandler's own C
watchdog thread (which does not need the GIL) writes the Python stack of every
thread to the hang log. ``faulthandler.enable`` on the same file also captures
hard crashes (access violations, aborts inside Qt).
"""

from __future__ import annotations

import faulthandler
import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QTimer

logger = logging.getLogger(__name__)


class HangWatchdog(QObject):
    def __init__(self, log_path: Path, timeout_s: float = 5.0,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timeout_s = float(timeout_s)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Kept open for the whole process: faulthandler writes to the raw fd.
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)
        self._file.write(
            f"\n===== SensePi session started {datetime.now().isoformat(timespec='seconds')} "
            f"(hang threshold {self._timeout_s:.0f} s) =====\n")
        self._file.flush()
        faulthandler.enable(file=self._file, all_threads=True)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._pet)
        self.path = log_path

    def start(self) -> None:
        self._pet()
        self._timer.start()
        logger.info("Hang watchdog writing to %s", self.path)

    def _pet(self) -> None:
        # Cancelling and re-arming is cheap; a dump only happens if this slot
        # has not run for timeout_s, i.e. the GUI event loop is blocked.
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(
            self._timeout_s, repeat=True, file=self._file, exit=False)

    def stop(self) -> None:
        self._timer.stop()
        faulthandler.cancel_dump_traceback_later()
