"""Stopping the stream must not block the GUI thread, and a late finish must not
tear down the stream that replaced it.

``stop_live_stream(wait=True)`` was called from the Stop button and the auto-stop
timer, freezing the window for up to 5 s while the ingest thread drained. Making
Stop non-blocking creates a second problem: the old worker's ``finished`` can
arrive *after* a new stream has started, and ``_on_ingest_finished`` would then
clear the new stream's worker and emit stream_stopped for it. Hence the sender
check, and ``wait_for_ingest_threads()`` for the two places a wait is correct
(app close, and just before opening a new stream).
"""

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QObject, Signal
from PySide6.QtWidgets import QApplication


class _Sender(QObject):
    """Stands in for an ingest worker so that sender() is a real Qt sender."""

    finished = Signal()


class TestIngestThreadHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        from sensepi.gui.recorder_controller import RecorderController

        self.ctrl = RecorderController()
        self.addCleanup(self.ctrl.deleteLater)

    def test_wait_for_ingest_threads_exists_and_succeeds_when_idle(self):
        self.assertTrue(self.ctrl.wait_for_ingest_threads(1000))

    def test_wait_for_ingest_threads_returns_quickly_when_idle(self):
        t0 = time.perf_counter()
        self.ctrl.wait_for_ingest_threads(5000)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 0.1,
                        "waiting with no threads burned the whole timeout")

    def test_a_late_finish_from_the_previous_worker_is_ignored(self):
        current = _Sender()
        stale = _Sender()
        self.ctrl._ingest_worker = current
        self.ctrl._ingest_thread = None

        stopped = []
        self.ctrl.stream_stopped.connect(lambda: stopped.append(True))
        errors = []
        self.ctrl.error_reported.connect(errors.append)

        # The previous stream's worker finishes after the new one started.
        stale.finished.connect(self.ctrl._on_ingest_finished)
        stale.finished.emit()

        self.assertIs(self.ctrl._ingest_worker, current,
                      "a late finish from the old worker cleared the new stream")
        self.assertEqual(stopped, [],
                         "a late finish emitted stream_stopped for the live stream")
        self.assertEqual(errors, [],
                         "a late finish reported the live stream as stopped unexpectedly")

    def test_the_current_workers_finish_is_still_handled(self):
        current = _Sender()
        self.ctrl._ingest_worker = current
        self.ctrl._stop_requested = True       # a real, requested stop

        stopped = []
        self.ctrl.stream_stopped.connect(lambda: stopped.append(True))

        current.finished.connect(self.ctrl._on_ingest_finished)
        current.finished.emit()

        self.assertIsNone(self.ctrl._ingest_worker)
        self.assertEqual(stopped, [True], "the live worker's finish was swallowed")

    def test_stop_live_stream_does_not_wait_by_default(self):
        import inspect

        sig = inspect.signature(self.ctrl.stop_live_stream)
        self.assertIs(sig.parameters["wait"].default, False,
                      "stop_live_stream must default to not blocking the GUI thread")


class TestMainWindowStopIsNonBlocking(unittest.TestCase):
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

    def test_pressing_stop_does_not_wait_for_the_ingest_thread(self):
        seen = []
        self.win.recorder_tab.stop_live_stream = lambda **kw: seen.append(kw)

        self.win._on_stop_stream_requested()

        self.assertEqual(seen, [{"wait": False}],
                         "Stop still blocks the GUI thread waiting for the thread to drain")

    def test_the_auto_stop_timeout_does_not_wait_either(self):
        seen = []
        self.win.recorder_tab.stop_live_stream = lambda **kw: seen.append(kw)

        self.win._on_auto_stop_timeout()

        self.assertEqual(seen, [{"wait": False}])

    def test_closing_the_window_does_wait(self):
        # Close is the one place a wait is correct: destroying a running
        # QThread aborts the process.
        seen = []
        self.win.recorder_tab.stop_live_stream = lambda **kw: seen.append(("stop", kw))
        self.win.recorder_tab.wait_for_ingest_threads = lambda ms: seen.append(("wait", ms))

        self.win.close()

        self.assertIn(("stop", {"wait": True}), seen)
        self.assertTrue(any(s[0] == "wait" for s in seen),
                        "closeEvent did not wait for draining ingest threads")


if __name__ == "__main__":
    unittest.main()
