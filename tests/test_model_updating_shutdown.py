"""Closing the app while Continuous Update runs must stop its thread first.

MainWindow.closeEvent stopped the Digital Twin, the sonification and the stream,
but never the Model Updating workers, so closing mid-run destroyed a running
QThread ("QThread: Destroyed while thread is still running") and crashed the
process. The continuous thread's ``finished -> quit`` hop was also queued to the
GUI thread, so a GUI-thread wait for it could only time out.
"""

import os
import time
import types
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop
from PySide6.QtWidgets import QApplication

try:
    import openseespy.opensees  # noqa: F401  (the worker imports the calibrator)
    _HAS_OPENSEES = True
except Exception:  # pragma: no cover - env-dependent
    _HAS_OPENSEES = False


def _params():
    n = 3
    return {
        "nStory": n, "nCalibModes": 3, "E": 1.0, "floor_masses": [1.0] * n,
        "E_scale_lb": 0.5, "E_scale_ub": 1.5, "m_scale_lb": 0.5, "m_scale_ub": 1.5,
        "w_freq": 1.0, "w_mode": 0.35, "max_nfev": 60,
        "sensor_axis": "ax", "sensor_n_modes": 3, "use_mode_shapes": False,
        "sensor_f_min": 0.5, "sensor_f_max": 12.0, "sensor_target_fs": None,
    }


def _no_data_yet(**_kwargs):
    # The loop keeps waiting for a full window -- the state it is in for most
    # of every cycle.
    return types.SimpleNamespace(success=False, duration_s=0.0, message="collecting")


@unittest.skipUnless(_HAS_OPENSEES, "needs openseespy")
class TestModelUpdatingShutdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def _pump(self, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.app.processEvents(QEventLoop.AllEvents, 20)

    def _launch(self, tab):
        from sensepi.gui.tabs import tab_model_updating as mu
        worker = mu._ContinuousUpdateWorker(
            _params(), {"duration_s": 30.0, "calibration_method": "bayesian"}, _no_data_yet)
        tab._launch_continuous(worker)
        self._pump(0.3)
        thread = tab._continuous_thread
        self.assertTrue(thread.isRunning())
        return thread

    def test_shutdown_stops_a_running_continuous_update(self):
        from sensepi.gui.tabs.tab_model_updating import ModelUpdatingTab
        tab = ModelUpdatingTab()
        thread = self._launch(tab)

        t0 = time.perf_counter()
        tab.shutdown()
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 2.0)
        self.assertTrue(thread.isFinished())
        self._pump(0.2)
        tab.deleteLater()
        self._pump(0.1)

    def test_closing_the_main_window_stops_continuous_update(self):
        from sensepi.gui.main_window import MainWindow
        w = MainWindow()
        thread = self._launch(w.model_updating_tab)

        w.close()

        self.assertTrue(thread.isFinished())
        self._pump(0.2)


class TestCloseWaitsForAOneShotJob(unittest.TestCase):
    """Calibrate / Run Analysis cannot be interrupted mid-OpenSees. A close that
    outlasts the wait must be postponed, not destroy the running thread."""

    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def _pump_until(self, predicate, timeout):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            self.app.processEvents(QEventLoop.AllEvents, 20)
            if predicate():
                return True
        return False

    def _start_slow_job(self, tab, seconds):
        from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

        class _SlowJob(QObject):
            finished = Signal()

            @Slot()
            def run(self):
                time.sleep(seconds)          # stands in for an OpenSees run
                self.finished.emit()

        worker = _SlowJob()
        thread = QThread(tab)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit, Qt.DirectConnection)
        thread.finished.connect(tab._clear_worker)
        tab._worker, tab._thread = worker, thread
        thread.start()
        def _wait():
            try:
                thread.wait(5000)
            except RuntimeError:  # deleted along with the closed window
                pass
        self.addCleanup(_wait)
        return thread

    def test_shutdown_reports_a_job_still_running(self):
        from sensepi.gui.tabs.tab_model_updating import ModelUpdatingTab
        tab = ModelUpdatingTab()
        thread = self._start_slow_job(tab, 1.0)

        self.assertFalse(tab.shutdown(wait_ms=100))
        self.assertTrue(thread.wait(5000))
        self.assertTrue(self._pump_until(lambda: not tab.is_busy(), 2.0))
        self.assertTrue(tab.shutdown(wait_ms=100))

    def test_close_is_postponed_until_the_job_finishes(self):
        from sensepi.gui.main_window import MainWindow
        w = MainWindow()
        w._shutdown_wait_ms = 100
        w.show()
        thread = self._start_slow_job(w.model_updating_tab, 1.0)

        w.close()

        self.assertTrue(w.isVisible(), "closed while a job was still running")
        self.assertTrue(thread.isRunning())
        self.assertTrue(self._pump_until(lambda: not w.isVisible(), 5.0),
                        "window did not close after the job finished")
        self.assertTrue(thread.isFinished())


if __name__ == "__main__":
    unittest.main()
