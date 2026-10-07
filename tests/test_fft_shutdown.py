"""The Spectrum tab must stop its timers and threads when the app closes.

FftTab had no ``shutdown()`` at all: closing the window while an eigen-frequency
identification was in flight destroyed a running QThread, which aborts the
process (0xc0000409 on Windows). Its two retire paths also called
``thread.quit(); thread.wait()`` on the GUI thread with no timeout —
``_on_eigen_finished`` runs once per ~10 s cycle, so that was a recurring stall.
"""

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop, QObject, QThread, Slot
from PySide6.QtWidgets import QApplication


class _SlowWorker(QObject):
    def __init__(self, hold_s: float = 0.4) -> None:
        super().__init__()
        self._hold_s = float(hold_s)

    @Slot()
    def run(self) -> None:
        time.sleep(self._hold_s)


class TestFftShutdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        from sensepi.gui.main_window import MainWindow

        self.win = MainWindow()
        self.addCleanup(self.win.close)
        self.tab = self.win.fft_tab

    def _pump(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.app.processEvents(QEventLoop.AllEvents, 10)

    def _busy_pair(self, hold_s: float = 0.4):
        thread = QThread()
        worker = _SlowWorker(hold_s)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        thread.start()
        self._pump(0.05)
        return thread, worker

    def test_the_tab_exposes_a_shutdown_hook(self):
        # MainWindow.closeEvent calls this; without it the tab's threads
        # outlived the window.
        self.assertTrue(callable(getattr(self.tab, "shutdown", None)))

    def test_shutdown_stops_every_timer(self):
        for name in ("_timer", "_eig_timer", "_final_countdown_timer"):
            timer = getattr(self.tab, name, None)
            if timer is not None:
                timer.start(50)

        self.tab.shutdown(wait_ms=1000)

        for name in ("_timer", "_eig_timer", "_final_countdown_timer"):
            timer = getattr(self.tab, name, None)
            if timer is not None:
                self.assertFalse(timer.isActive(), f"{name} still running after shutdown")

    def test_shutdown_waits_for_an_identification_thread_in_flight(self):
        thread, worker = self._busy_pair(0.3)
        self.tab._eig_thread = thread
        self.tab._eig_worker = worker

        self.tab.shutdown(wait_ms=3000)

        # Shutdown is the one place a wait is correct: the thread must be done,
        # because destroying a running QThread aborts the process.
        self.assertIsNone(self.tab._eig_thread)
        self.assertIsNone(self.tab._eig_worker)
        self.assertEqual(self.tab._thread_retirer._pending, [])

    def test_the_per_cycle_retire_does_not_block_the_gui_thread(self):
        thread, worker = self._busy_pair(0.5)
        self.tab._eig_thread = thread
        self.tab._eig_worker = worker
        self.addCleanup(self.tab._thread_retirer.wait_all, 3000)

        t0 = time.perf_counter()
        self.tab._on_eigen_finished()
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 0.1,
                        f"_on_eigen_finished blocked for {elapsed * 1000:.0f} ms; "
                        "it runs once per identification cycle")
        self.assertIsNone(self.tab._eig_thread)

    def test_closing_the_window_shuts_the_tab_down(self):
        calls = []
        self.tab.shutdown = lambda *a, **k: calls.append(True)

        self.win.close()

        self.assertEqual(calls, [True], "closeEvent did not shut the Spectrum tab down")


if __name__ == "__main__":
    unittest.main()
