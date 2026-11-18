"""The Digital Twin tab must not render while no stream is running.

Its 20 fps matplotlib render timer and the 2.5 fps wireframe view were started in
``__init__`` and never stopped, so they kept waking the GUI thread even with the
tab idle and no data arriving — reported as "Python not responding" after two to
three minutes. They are now started by ``on_stream_started`` and stopped by
``on_stream_stopped``.

The static structure is unaffected: the wireframe is drawn by
``apply_sensor_map`` via ``canvas.set_layout``, not by its timer, and
``refresh()`` already returns early with no controller.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


class TestTwinIdleTimers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        from sensepi.gui.main_window import MainWindow

        # The tab needs its recorder / model-updating / sonification
        # collaborators, so build it the way the app does.
        self.win = MainWindow()
        self.addCleanup(self.win.close)
        self.tab = self.win.digital_twin_tab

    def test_nothing_renders_before_a_stream_starts(self):
        self.assertFalse(self.tab._render_timer.isActive(),
                         "the 20 fps render timer runs with no stream")
        self.assertFalse(self.tab._live_view._timer.isActive(),
                         "the wireframe view polls with no stream")

    def test_a_stream_starts_the_render_timers(self):
        self.tab.on_stream_started()

        self.assertTrue(self.tab._render_timer.isActive())
        self.assertTrue(self.tab._live_view._timer.isActive())

    def test_stopping_the_stream_stops_them_again(self):
        self.tab.on_stream_started()
        self.tab.on_stream_stopped()

        self.assertFalse(self.tab._render_timer.isActive(),
                         "the render timer kept running after the stream stopped")
        self.assertFalse(self.tab._live_view._timer.isActive())

    def test_the_render_slot_does_nothing_while_idle(self):
        polled = []
        self.tab._poll_shaker_trigger = lambda: polled.append(True)

        self.tab._render_latest_frame()

        self.assertEqual(polled, [],
                         "the render slot did work with no stream and no experiment")

    def test_the_render_slot_works_once_a_stream_is_running(self):
        polled = []
        self.tab._poll_shaker_trigger = lambda: polled.append(True)
        self.tab.on_stream_started()

        self.tab._render_latest_frame()

        self.assertEqual(polled, [True])

    def test_an_experiment_keeps_rendering_after_the_stream_stops(self):
        # A run in progress still needs its frames drawn; only the idle case is
        # throttled off.
        self.tab.on_stream_started()
        self.tab._running = True

        self.tab.on_stream_stopped()

        polled = []
        self.tab._poll_shaker_trigger = lambda: polled.append(True)
        self.tab._render_latest_frame()
        self.assertEqual(polled, [True],
                         "a running experiment stopped being rendered")

    def test_the_static_wireframe_is_drawn_without_any_timer(self):
        drawn = []
        self.tab._live_view.canvas.set_layout = lambda layout: drawn.append(layout)

        self.tab._live_view.apply_sensor_map(
            {"n_floors": 3, "axis": "x",
             "placements": [{"sensor_id": 1, "floor": 1, "cell": "B2"}]})

        self.assertEqual(len(drawn), 1,
                         "the structure drawing depended on the idle timer")
        self.assertFalse(self.tab._live_view._timer.isActive())


if __name__ == "__main__":
    unittest.main()
