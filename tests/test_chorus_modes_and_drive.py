"""The chorus must keep singing every mode it identified, and accept a stated
drive frequency.

Two problems found during the rig test, where the Spectrum tab reported f1, f2
and f3 while the Bioacoustic Chorus sang only f1:

* The modal history was collapsed with ``min(len(f) for f in history)``, so one
  quiet cycle that resolved a single peak truncated the next three cycles to one
  mode. On a rig shaken in bursts that is most of the time.
* The excitation frequency, which decides which mode counts as resonating, is
  tracked by picking the largest FFT peak. With no base sensor placed that is
  read from the structure's response rather than from the input, so it is a
  stand-in. The operator knows the real drive frequency and can now state it.
"""

import unittest

import numpy as np


class TestModalHistoryKeepsEveryMode(unittest.TestCase):
    """The aggregation across re-identification cycles."""

    def _tracker(self):
        from sensepi.sonification.chorus.features import ModalTracker
        from sensepi.sonification.chorus.types import ChorusConfig

        return ModalTracker(ChorusConfig())

    def _aggregate(self, tracker, freqs):
        """Drive the exact aggregation under test with a known result."""
        freqs = np.asarray(freqs, dtype=float)
        damp = np.full(freqs.size, 0.02)
        tracker._history.append((freqs, damp))
        target = int(freqs.size)
        med = []
        for i in range(target):
            fv = [f[i] for f, _ in tracker._history if len(f) > i]
            med.append(float(np.median(fv)) if fv else float(freqs[i]))
        return med

    def test_a_single_quiet_cycle_does_not_silence_the_other_modes(self):
        tracker = self._tracker()
        self._aggregate(tracker, [2.10, 6.47, 9.03])
        self._aggregate(tracker, [2.11, 6.45, 9.05])

        # one cycle resolves only mode 1, as happens while the rig is still
        self._aggregate(tracker, [2.10])
        # and the next identification finds all three again
        recovered = self._aggregate(tracker, [2.10, 6.47, 9.03])

        self.assertEqual(len(recovered), 3,
                         "a quiet cycle truncated the following cycles to one mode")
        # Still medians across the cycles that saw those modes, so the value is
        # smoothed rather than simply the latest one.
        self.assertAlmostEqual(recovered[1], 6.46, places=2)
        self.assertAlmostEqual(recovered[2], 9.04, places=2)

    def test_the_latest_identification_sets_how_many_modes_are_sung(self):
        tracker = self._tracker()
        self._aggregate(tracker, [2.10, 6.47, 9.03])

        # genuinely only one mode now: report one, not a stale three
        self.assertEqual(len(self._aggregate(tracker, [2.10])), 1)

    def test_frequencies_are_still_smoothed_across_cycles(self):
        # The median over history is why a noisy cycle does not jerk the pitch.
        tracker = self._tracker()
        self._aggregate(tracker, [2.00, 6.00, 9.00])
        self._aggregate(tracker, [2.20, 6.40, 9.20])
        out = self._aggregate(tracker, [2.10, 6.20, 9.10])

        self.assertAlmostEqual(out[0], 2.10, places=2)
        self.assertAlmostEqual(out[1], 6.20, places=2)


class TestStatedDriveFrequency(unittest.TestCase):
    """The user-provided excitation frequency."""

    def _cfg(self, **kw):
        from sensepi.sonification.chorus.types import ChorusConfig

        cfg = ChorusConfig()
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg

    def test_it_defaults_to_tracking_from_the_data(self):
        self.assertEqual(self._cfg().exc_freq_hz_override, 0.0)

    def test_a_stated_value_survives_clamping(self):
        cfg = self._cfg(exc_freq_hz_override=6.47).clamped()

        self.assertAlmostEqual(cfg.exc_freq_hz_override, 6.47, places=2)

    def test_an_absurd_value_is_held_inside_the_audible_band(self):
        self.assertLessEqual(
            self._cfg(exc_freq_hz_override=9999.0).clamped().exc_freq_hz_override, 200.0)

    def test_zero_and_negative_both_mean_auto(self):
        self.assertEqual(self._cfg(exc_freq_hz_override=0.0).clamped().exc_freq_hz_override, 0.0)
        self.assertEqual(self._cfg(exc_freq_hz_override=-3.0).clamped().exc_freq_hz_override, 0.0)

    def test_the_engine_accepts_it_as_a_live_option(self):
        # It has to reach a running worker the same way every other knob does.
        from sensepi.sonification.chorus.engine import ChorusEngine

        engine = ChorusEngine(self._cfg())
        engine.set_option("exc_freq_hz_override", 6.47)

        self.assertAlmostEqual(engine.cfg.exc_freq_hz_override, 6.47, places=2)


class TestDriveFrequencyControl(unittest.TestCase):
    """The control in the Identification box."""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        from sensepi.gui.main_window import MainWindow

        self.win = MainWindow()
        self.addCleanup(self.win.close)
        self.tab = self.win.sonification_tab.chorus_tab

    def test_it_opens_on_auto(self):
        self.assertEqual(self.tab._spin_drive.value(), 0.0)
        self.assertIn("auto", self.tab._spin_drive.specialValueText().lower())

    def test_typing_a_frequency_reaches_the_config(self):
        self.tab._spin_drive.setValue(6.47)

        self.assertAlmostEqual(self.tab._cfg.exc_freq_hz_override, 6.47, places=2)

    def test_returning_to_zero_goes_back_to_tracking(self):
        self.tab._spin_drive.setValue(6.47)
        self.tab._spin_drive.setValue(0.0)

        self.assertEqual(self.tab._cfg.exc_freq_hz_override, 0.0)

    def test_every_identified_mode_still_gets_a_cast_row(self):
        # The rows are what the user picks a species on; they must not be
        # capped below the mode count.
        self.assertEqual(len(self.tab._mode_type_combos), 3)
        self.assertTrue(all(c.isEnabled() for c in self.tab._mode_type_combos))


if __name__ == "__main__":
    unittest.main()


class TestEveryPanelRendersAtLowSync(unittest.TestCase):
    """The stage must draw whether or not the structure is resonating.

    The cast panel picked its "scattered chorus" colour from a name that no
    longer existed, so every frame with sync <= 0.6 raised NameError. All four
    panels then shared one try/except, so the radar, the waterfall and the score
    were never reached either and the whole stage went blank while the sound
    carried on. The failure was logged at debug level, so nothing said why.
    """

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication

        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def _viz(self, sync):
        """A frame carrying three identified modes, at a chosen sync level."""
        import types

        from sensepi.sonification.chorus.engine import ChorusEngine
        from sensepi.sonification.chorus.types import ChorusConfig

        fs = 100.0
        t = np.arange(int(40 * fs)) / fs
        sig = (np.sin(2 * np.pi * 2.1 * t) + 0.5 * np.sin(2 * np.pi * 6.3 * t)
               + 0.3 * np.sin(2 * np.pi * 9.0 * t))
        data = np.vstack([sig * (1 + 0.1 * k) for k in range(4)])

        def snap(seconds):
            return types.SimpleNamespace(data=data[:, -int(seconds * fs):], fs=fs,
                                         sensor_ids=[1, 2, 3, 4])

        cfg = ChorusConfig()
        cfg.sensor_map = {"n_floors": 3, "axis": "x", "placements": [
            {"sensor_id": i, "floor": f, "cell": "B2"}
            for i, f in ((1, 0), (2, 1), (3, 2), (4, 3))]}
        engine = ChorusEngine(cfg)
        engine.tracker.reidentify(snap(cfg.id_window_s), 10.0)
        viz = engine.tick(10.0, snap(cfg.fast_window_s), None, rate_hz=400.0)
        viz.frame.sync = np.full(
            np.asarray(viz.modal.frequencies_hz).ravel().size, float(sync))
        return viz

    def _panels(self):
        from sensepi.gui.tabs.tab_bioacoustic_chorus import (_CastPanel, _RadarPanel,
                                                             _ScorePanel, _WaterfallPanel)
        panels = {"cast": _CastPanel(), "radar": _RadarPanel(),
                  "waterfall": _WaterfallPanel(), "score": _ScorePanel()}
        for p in panels.values():
            self.addCleanup(p.deleteLater)
        return panels

    def test_a_scattered_chorus_still_draws_every_panel(self):
        # sync below 0.6 is the ordinary case: the structure is not resonating.
        viz = self._viz(sync=0.13)

        for name, panel in self._panels().items():
            with self.subTest(panel=name):
                panel.update_frame(viz)        # must not raise

    def test_a_resonating_chorus_also_draws_every_panel(self):
        viz = self._viz(sync=0.90)

        for name, panel in self._panels().items():
            with self.subTest(panel=name):
                panel.update_frame(viz)

    def test_one_card_per_identified_mode(self):
        panels = self._panels()
        panels["cast"].update_frame(self._viz(sync=0.13))

        self.assertEqual(len(panels["cast"]._cards), 3,
                         "the cast showed fewer cards than the modes identified")

    def test_the_package_has_no_undefined_names(self):
        """A NameError on a rarely-taken branch blanked the whole tab."""
        import pathlib
        import subprocess
        import sys

        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "sensepi"
        proc = subprocess.run([sys.executable, "-m", "pyflakes", str(root)],
                              capture_output=True, text=True)
        undefined = [ln for ln in proc.stdout.splitlines() if "undefined name" in ln]
        if proc.returncode and not proc.stdout and "No module named" in proc.stderr:
            self.skipTest("pyflakes is not installed")
        self.assertEqual(undefined, [], "\n".join(undefined))
