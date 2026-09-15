"""Start/stop lifecycle tests for the Sonification tab.

These exercise the thread boundary that static reading is worst at: worker
creation, queued shutdown, repeated cycles and application close. Skipped when
Qt cannot open an offscreen platform.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception:                                    # pragma: no cover
    HAVE_QT = False

import numpy as np

from sensepi.sonification.chorus.catalog import catalog_available

HAVE_DATA = catalog_available()
_APP = None


def _ensure_app():
    """Get (or create) the QApplication, or return None if that is impossible.

    Qt allows exactly one application object per process. ``test_continuous_mode_b``
    builds a *QCoreApplication*, which cannot host widgets; if it ran first, a
    QApplication can never be created here. Returning None lets the GUI tests
    skip with an explanation instead of aborting the whole process.
    """
    global _APP
    if not HAVE_QT:
        return None
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    elif not isinstance(app, QApplication):
        return None
    _APP = app
    return app


_NO_WIDGETS = ("a QCoreApplication created by another test module owns this "
               "process, so widgets cannot be created; run this module on its own")


def setUpModule():
    _ensure_app()


class _FakeSnapshot:
    def __init__(self, data, fs):
        self.data = data
        self.fs = fs


if HAVE_QT:
    class _FakeController(QObject):
        """Stands in for RecorderController: same surface, no SSH, no hardware."""

        stream_started = Signal()
        stream_stopped = Signal()
        stream_rate_updated = Signal(str, float)

        def __init__(self, streaming=True):
            super().__init__()
            self._streaming = streaming
            self.window_s = 30.0
            self.shrink_attempts = 0
            rng = np.random.default_rng(5)
            t = np.arange(0, 60, 1 / 41.0)
            self._data = np.vstack([
                np.sin(2 * np.pi * 8.37 * t) * 0.05 + rng.standard_normal(t.size) * 0.002
                for _ in range(3)])

        def is_streaming(self):
            return self._streaming

        def set_modal_window_seconds(self, s):
            if s < self.window_s:
                self.shrink_attempts += 1
            self.window_s = float(s)

        def modal_window_seconds(self):
            return self.window_s

        def require_modal_window_seconds(self, s):
            # the safe API: grow only, asking for less is not a shrink
            if float(s) > self.window_s:
                self.window_s = float(s)

        def snapshot_modal_capture(self, *, axis="ax", last_seconds=None, target_fs=None):
            n = int((last_seconds or 6.0) * 41.0)
            return _FakeSnapshot(self._data[:, -n:], 41.0)


@unittest.skipUnless(HAVE_QT, "PySide6 not importable")
@unittest.skipUnless(HAVE_DATA, "species catalog / grain bank not installed")
class TestTabLifecycle(unittest.TestCase):
    def setUp(self):
        self.app = _ensure_app()
        if self.app is None:
            self.skipTest(_NO_WIDGETS)

    def _tab(self, streaming=True):
        from sensepi.gui.tabs.tab_bioacoustic_chorus import BioacousticChorusTab
        controller = _FakeController(streaming)
        tab = BioacousticChorusTab(recorder_controller=controller)
        return tab, controller

    def _pump(self, ms=600):
        from PySide6.QtCore import QDeadlineTimer, QThread
        deadline = QDeadlineTimer(ms)
        while not deadline.hasExpired():
            self.app.processEvents()
            QThread.msleep(10)
        self.app.processEvents()

    def test_start_is_gated_on_the_live_stream(self):
        tab, _ = self._tab(streaming=False)
        self.assertFalse(tab._btn_start.isEnabled())
        tab.deleteLater()

    def test_start_then_stop_leaves_no_thread_running(self):
        tab, _ = self._tab()
        tab._on_start()
        self._pump(700)
        self.assertIsNotNone(tab._thread)
        tab._on_stop()
        self._pump(300)
        self.assertIsNone(tab._worker)
        self.assertIsNone(tab._thread)

    def test_repeated_cycles_do_not_accumulate_threads(self):
        tab, _ = self._tab()
        for _ in range(3):
            tab._on_start()
            self._pump(400)
            tab._on_stop()
            self._pump(200)
            self.assertIsNone(tab._thread)
        tab.deleteLater()

    def test_stop_is_idempotent(self):
        tab, _ = self._tab()
        tab._on_start()
        self._pump(400)
        tab._on_stop()
        tab._on_stop()               # must not raise or double-tear-down
        self._pump(200)
        self.assertIsNone(tab._thread)

    def test_shutdown_stops_a_running_model(self):
        tab, _ = self._tab()
        tab._on_start()
        self._pump(500)
        tab.shutdown()
        self._pump(200)
        self.assertIsNone(tab._thread)

    def test_worker_never_shrinks_the_shared_modal_window(self):
        """Mode B shares this buffer; the chorus may only raise the floor."""
        tab, controller = self._tab()
        controller.window_s = 400.0          # a long Mode B run is in progress
        tab._on_start()
        self._pump(600)
        tab._on_stop()
        self._pump(200)
        self.assertEqual(controller.shrink_attempts, 0)
        self.assertGreaterEqual(controller.window_s, 400.0)

    def test_knob_changes_are_safe_before_and_during_a_run(self):
        tab, _ = self._tab()
        tab._on_knob("master", 0.5)          # no worker yet
        self.assertAlmostEqual(tab._cfg.master, 0.5)
        tab._on_start()
        self._pump(400)
        for v in np.linspace(0.2, 3.0, 15):
            tab._on_knob("f_lo", float(v))   # simulates a slider drag
        self._pump(300)
        tab._on_stop()
        self._pump(200)
        self.assertIsNone(tab._thread)

    def test_frames_reach_the_panels(self):
        tab, _ = self._tab()
        tab._on_start()
        self._pump(1200)
        tab._drain_queue()
        self.assertNotEqual(tab._status.text(), "idle")
        tab._on_stop()
        self._pump(200)


@unittest.skipUnless(HAVE_QT, "PySide6 not importable")
class TestMainWindowIntegration(unittest.TestCase):
    def setUp(self):
        self.app = _ensure_app()
        if self.app is None:
            self.skipTest(_NO_WIDGETS)

    def test_window_builds_and_closes_cleanly(self):
        from sensepi.gui.main_window import MainWindow
        w = MainWindow()
        titles = [w._tabs.tabText(i) for i in range(w._tabs.count())]
        # both sonification workflows are present and independent
        # ONE top-level Sonification tab, hosting the models as sub-tabs
        self.assertIn("Sonification", titles)
        self.assertNotIn("Bioacoustic Chorus", titles)
        son = w.sonification_tab
        subs = [son._tabs.tabText(i) for i in range(son._tabs.count())]
        self.assertEqual(subs, ["Bioacoustic Chorus", "Team Model"])
        self.assertIsNotNone(son.chorus_tab._controller)
        self.assertTrue(hasattr(son, "shutdown"))
        self.assertTrue(hasattr(son.team_tab, "shutdown"))
        w.close()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
