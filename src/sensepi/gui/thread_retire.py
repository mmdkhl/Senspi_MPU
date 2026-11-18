"""End worker threads without blocking the GUI thread.

The old pattern, in a worker's ``finished`` slot on the GUI thread::

    thread.quit(); thread.wait(); thread.deleteLater()

blocks the event loop until the thread has really exited. ``wait()`` has no
timeout there, so a worker that is slow to unwind (or never does) freezes the
whole window until Windows reports "Python not responding".

:class:`ThreadRetirer` asks the thread to quit and keeps the thread and worker
referenced until ``QThread.finished`` arrives, so neither is destroyed while
still running (which aborts Qt). Call :meth:`wait_all` on shutdown only.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QThread, Slot


class ThreadRetirer(QObject):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pending: list[tuple[QThread, QObject | None]] = []

    def retire(self, thread: QThread | None, worker: QObject | None) -> None:
        if thread is None:
            if worker is not None:
                worker.deleteLater()
            return
        self._pending.append((thread, worker))
        thread.finished.connect(self._release_finished)
        thread.quit()  # thread-safe
        if thread.isFinished():
            self._release_finished()

    @staticmethod
    def _is_running(thread: QThread) -> bool:
        try:
            return thread.isRunning()
        except RuntimeError:  # C++ object already gone: it finished
            return False

    @Slot()
    def _release_finished(self) -> None:
        alive = []
        for thread, worker in self._pending:
            if self._is_running(thread):
                alive.append((thread, worker))
                continue
            for obj in (worker, thread):
                if obj is None:
                    continue
                try:
                    obj.deleteLater()
                except RuntimeError:
                    pass
        self._pending = alive

    def wait_all(self, timeout_ms: int = 3000) -> bool:
        """Block (shutdown only) until retired threads end. True if all did."""
        deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
        ok = True
        for thread, _worker in list(self._pending):
            if not self._is_running(thread):
                continue
            remaining = int(max(0.0, deadline - time.monotonic()) * 1000)
            try:
                if not thread.wait(remaining):
                    ok = False
            except RuntimeError:
                pass
        self._release_finished()
        return ok
