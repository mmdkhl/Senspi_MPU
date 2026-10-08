"""Structure Pulse live mode: replays per capture cycle.

A live cycle captures a fixed window, 10 s by default, and sweeps it once. On a
slow structure that is a slow figure, and by the time it has been heard the
window it describes is already old.

``Replay`` compresses the sweep so several fit inside one capture window: at x5
a 10 s window is swept in 2 s and repeats five times, until the next cycle is
ready to take over. The capture length does not change, only how fast it is
read back.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


class TestReplayPerCycle(unittest.TestCase):
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
        self.tab = self.win.sonification_tab.pulse_tab
        self.tab._spin_live.setValue(10)

    def _live_config(self, replay):
        """The config the next live cycle would be rendered with."""
        self.tab._spin_replay.setValue(replay)
        self.tab._live_worker = object()          # take the live branch
        try:
            return self.tab._current_config()
        finally:
            self.tab._live_worker = None

    def test_it_defaults_to_one_sweep_per_cycle(self):
        self.assertEqual(self.tab._spin_replay.value(), 1)

    def test_the_sweep_is_compressed_to_fit_the_replays(self):
        for replay in (2, 5, 10):
            with self.subTest(replay=replay):
                cfg = self._live_config(replay)
                self.assertAlmostEqual(cfg.duration_s, 10.0 / replay, places=3)

    def test_a_longer_window_scales_with_it(self):
        self.tab._spin_live.setValue(30)

        self.assertAlmostEqual(self._live_config(5).duration_s, 6.0, places=3)

    def test_the_capture_window_itself_is_untouched(self):
        # Replay changes how fast the window is read back, never how much of
        # the structure is captured.
        self._live_config(5)

        self.assertEqual(self.tab._spin_live.value(), 10)

    def test_at_one_the_sweep_box_still_governs(self):
        self.tab._spin_dur.setValue(14.0)

        self.assertAlmostEqual(self._live_config(1).duration_s, 14.0, places=3)

    def test_replay_is_ignored_when_not_live(self):
        # Playing an opened recording is a single sweep of the whole thing.
        self.tab._spin_dur.setValue(14.0)
        self.tab._spin_replay.setValue(5)

        self.assertAlmostEqual(self.tab._current_config().duration_s, 14.0, places=3)

    def test_the_sweep_never_collapses_to_nothing(self):
        self.tab._spin_live.setValue(3)

        self.assertGreaterEqual(self._live_config(20).duration_s, 1.0)

    def test_changing_it_does_not_disturb_the_cycle_already_playing(self):
        # Re-rendering under the playhead would cut the sound, for a setting
        # whose whole purpose is how the sound is paced.
        self.tab._spin_replay.setValue(4)        # no live worker: must not raise

        self.assertEqual(self.tab._spin_replay.value(), 4)



class TestAudifyInLiveMode(unittest.TestCase):
    """Audify is paced by the data, not by the Sweep box.

    Its length is the record divided by the speed-up, so one pass can be under
    a second. In live mode that played once and then left silence until the
    next cycle, which read as the voice being broken. It now repeats until the
    next cycle arrives.

    Its speed is deliberately NOT retimed to the replay slot: the automatic
    factor is what carries a 2 Hz mode up to roughly 220 Hz, and stretching a
    pass to fill window/replay lands it near 16 Hz, where there is nothing to
    hear.
    """

    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        import numpy as np

        from sensepi.gui.main_window import MainWindow
        from sensepi.sonification.structure_pulse.analysis import build_dataset_from_arrays

        self.win = MainWindow()
        self.addCleanup(self.win.close)
        self.tab = self.win.sonification_tab.pulse_tab

        fs = 100.0
        t = np.arange(int(40 * fs)) / fs
        sig = np.sin(2 * np.pi * 2.1 * t) + 0.4 * np.sin(2 * np.pi * 6.3 * t)
        data = np.vstack([sig * (1 + 0.1 * k) for k in range(3)])
        self.tab._dataset = build_dataset_from_arrays({"ax": data}, fs=fs, source="t")
        self.tab._populate_views(self.tab._dataset.view_names())
        self.tab._combo_view.setCurrentText("Time history")
        self.tab._combo_voice.setCurrentIndex(self.tab._combo_voice.findData("audify"))

    def test_audify_produces_audio_on_a_time_view(self):
        self.tab._rerender()

        self.assertIsNotNone(self.tab._render)
        self.assertGreater(self.tab._render.audio.size, 0)

    def test_replay_does_not_change_the_audify_speed(self):
        # Changing it would trade audibility for an exact repeat count.
        speeds = []
        for replay in (1, 5, 10):
            self.tab._spin_replay.setValue(replay)
            self.tab._live_worker = object()
            try:
                speeds.append(self.tab._current_config().audify_speed)
            finally:
                self.tab._live_worker = None

        self.assertEqual(speeds, [0.0, 0.0, 0.0], "replay retimed audify")

    def test_an_explicit_speed_is_honoured_when_asked_for(self):
        # The config field existed but nothing read it.
        from sensepi.sonification.structure_pulse.render import render_view
        from sensepi.sonification.structure_pulse.types import PulseConfig

        ds = self.tab._dataset
        view = ds.view("Time history")

        cfg_auto = PulseConfig(); cfg_auto.voice = "audify"
        cfg_fast = PulseConfig(); cfg_fast.voice = "audify"; cfg_fast.audify_speed = 400.0
        r_auto = render_view(view, cfg_auto, modal_frequencies=ds.modal.frequencies_hz,
                             data_fs=ds.fs)
        r_fast = render_view(view, cfg_fast, modal_frequencies=ds.modal.frequencies_hz,
                             data_fs=ds.fs)

        self.assertIn("400", r_fast.message)
        self.assertLess(r_fast.duration_s, r_auto.duration_s)

    def test_audify_still_refuses_a_view_that_is_not_a_signal(self):
        self.tab._combo_view.setCurrentText("Spectrum")
        self.tab._rerender()

        self.assertEqual(self.tab._render.audio.size, 0)
        self.assertIn("time-domain", self.tab._render.message)

if __name__ == "__main__":
    unittest.main()
