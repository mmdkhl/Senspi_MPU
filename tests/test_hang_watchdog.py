"""A GUI freeze must leave a Python stack behind.

A Windows ``AppHangB1`` event says only that python.exe stopped pumping
messages; it carries no Python traceback, so the freezes reported from the rig
were undiagnosable. HangWatchdog re-arms ``faulthandler.dump_traceback_later``
from a 1 s QTimer on the GUI thread: if the event loop stalls past the timeout,
faulthandler's own C watchdog — which does not need the GIL — writes every
thread's stack to the log.

Note the standing constraint: because the re-arm happens once per second, a
``timeout_s`` below 1 s reports a perfectly healthy loop as hung. The app uses
5 s.
"""

import faulthandler
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop
from PySide6.QtWidgets import QApplication


class TestHangWatchdog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "nested" / "hang_traces.log"
        self._watchdogs = []
        # Cleanups run last-added-first, so these are registered in reverse of
        # the order we need: stop the watchdogs, put process-wide faulthandler
        # state back, and only then close the log the watchdog held open.
        self.addCleanup(self._close_logs)
        self.addCleanup(faulthandler.enable)
        self.addCleanup(faulthandler.cancel_dump_traceback_later)

    def _close_logs(self):
        for wd in self._watchdogs:
            try:
                wd._file.close()
            except Exception:
                pass

    def _watchdog(self, timeout_s: float = 5.0):
        from sensepi.gui.hang_watchdog import HangWatchdog

        wd = HangWatchdog(self.path, timeout_s=timeout_s)
        self._watchdogs.append(wd)
        self.addCleanup(wd.stop)
        return wd

    def _pump(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.app.processEvents(QEventLoop.AllEvents, 10)
            time.sleep(0.01)

    def test_it_creates_its_log_including_missing_parents(self):
        wd = self._watchdog()

        self.assertTrue(self.path.exists())
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("SensePi session started", text)
        self.assertIn("hang threshold", text,
                      "the log does not record the threshold it is using")
        self.assertEqual(wd.path, self.path)

    def test_start_arms_the_timer_and_stop_disarms_it(self):
        wd = self._watchdog()
        self.assertFalse(wd._timer.isActive())

        wd.start()
        self.assertTrue(wd._timer.isActive())

        wd.stop()
        self.assertFalse(wd._timer.isActive())

    def test_a_stalled_event_loop_is_recorded(self):
        # Block the GUI thread the way a bad slot would; the dump must name the
        # code that was blocking, which is the whole point of the watchdog.
        wd = self._watchdog(timeout_s=0.3)
        wd.start()
        self.app.processEvents(QEventLoop.AllEvents, 10)

        time.sleep(0.9)                       # the stall being detected
        wd.stop()

        text = self.path.read_text(encoding="utf-8")
        self.assertIn("Timeout", text,
                      "a stalled GUI thread produced no faulthandler dump")
        self.assertIn("test_a_stalled_event_loop_is_recorded", text,
                      "the dump does not point at the code that was blocking")

    def test_a_responsive_loop_is_not_reported(self):
        # Above the 1 s re-arm interval, as the app configures it.
        wd = self._watchdog(timeout_s=2.0)
        wd.start()

        self._pump(1.5)
        wd.stop()

        self.assertNotIn("Timeout", self.path.read_text(encoding="utf-8"),
                         "a healthy event loop was reported as hung")

    def test_stop_disarms_faulthandler_so_a_later_stall_is_not_reported(self):
        wd = self._watchdog(timeout_s=0.3)
        wd.start()
        wd.stop()

        time.sleep(0.6)                       # would have fired had stop not disarmed

        self.assertNotIn("Timeout", self.path.read_text(encoding="utf-8"),
                         "stop() left faulthandler armed")


if __name__ == "__main__":
    unittest.main()
