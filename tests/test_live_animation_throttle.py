"""Run Analysis's live plots redraw at a bounded rate, not once per frame.

The transient worker emits ~10 frames/s, each carrying the whole response
history so far. The GUI redrew both live matplotlib canvases for every one,
which kept the GUI thread busy for most of the run and, contending with the
worker for the GIL, froze it for 0.6-1.6 s at a time. Every frame contains all
earlier data, so only the latest pending frame needs drawing.
"""

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop
from PySide6.QtWidgets import QApplication


def _frame(i, kind="frame"):
    return {"kind": kind, "i": i, "t_hist": [], "u_hist": [], "a_hist": [],
            "tmax": 10.0, "deformed_segments": []}


class TestLiveAnimationThrottle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        from sensepi.gui.tabs.tab_model_updating import ModelUpdatingTab
        self.tab = ModelUpdatingTab()
        self.addCleanup(self.tab.deleteLater)
        self.drawn = []
        self.tab._live_response_canvas.update_frame = lambda f: self.drawn.append(f["i"])
        self.tab._live_3d_canvas.update_frame = lambda f: None

    def _pump(self, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.app.processEvents(QEventLoop.AllEvents, 10)

    def test_a_burst_of_frames_is_drawn_once_with_the_latest(self):
        self.tab._on_animation_frame(_frame(0))       # first frame: drawn at once
        for i in range(1, 30):
            self.tab._on_animation_frame(_frame(i))
        self.assertEqual(self.drawn, [0])

        self._pump(0.5)

        self.assertEqual(self.drawn, [0, 29])

    def test_the_final_frame_is_drawn_immediately(self):
        self.tab._on_animation_frame(_frame(0))
        self.tab._on_animation_frame(_frame(1))
        self.tab._on_animation_frame(_frame(2, kind="final"))

        self.assertEqual(self.drawn, [0, 2])
        self._pump(0.5)
        self.assertEqual(self.drawn, [0, 2], "a stale frame was drawn after the final one")


if __name__ == "__main__":
    unittest.main()
