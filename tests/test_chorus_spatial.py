"""The four sensors must be audible as a place, not just as more numbers.

Three claims are checked here, because each is a thing the tab could not do
before the placement map reached it:

* individuals are distributed by the **mode shape**, so a top-heavy mode puts
  its animals at the top and a mode with a sign change splits around its node;
* stereo position comes from the **plan cell**, so the rig has a left and right;
* torsion is located on the **floor that twists**, rather than averaged flat.
"""
import dataclasses
import unittest

import numpy as np

from sensepi.sonification.chorus.features import (FeatureExtractor, _layout_of,
                                                  _row_geometry)
from sensepi.sonification.chorus.types import ChorusConfig, ModalState

FS = 100.0
WIN = 600

#: A minimal SpeciesInfo. Only pulse_rate and unit_s reach the code under test;
#: the rest just have to be present and of the right kind.
_FIELD_STUB = {"species": "test", "group": "cricket", "carrier": 2000.0,
               "echeme_rate": 3.0, "pulse_rate": 60.0, "unit_s": 0.010,
               "snr": 12.0, "flat": 0.2, "license": "CC0",
               "observer": "test", "observation": "test", "common": "Test",
               "type": "crickets"}

MAP = {"n_floors": 3, "axis": "x", "placements": [
    {"sensor_id": 1, "floor": 0, "cell": "B2"},      # base / shaker
    {"sensor_id": 2, "floor": 1, "cell": "A1"},      # left
    {"sensor_id": 3, "floor": 2, "cell": "B2"},      # centre
    {"sensor_id": 4, "floor": 3, "cell": "C3"}]}     # right


class _Snap:
    def __init__(self, data, fs, ids=None):
        self.data, self.fs = data, fs
        self.sensor_ids = ids or list(range(1, len(data) + 1))


class TestRowGeometry(unittest.TestCase):
    def test_plan_column_becomes_stereo_position(self):
        cfg = ChorusConfig()
        cfg.sensor_map = MAP
        floors, pans = _row_geometry(_layout_of(cfg), [1, 2, 3, 4])
        self.assertEqual(floors, (0, 1, 2, 3))
        self.assertLess(pans[1], -0.5, "column A should sit left")
        self.assertAlmostEqual(pans[2], 0.0, msg="column B should sit centre")
        self.assertGreater(pans[3], 0.5, "column C should sit right")

    def test_missing_map_puts_everything_centre_without_failing(self):
        floors, pans = _row_geometry(_layout_of(ChorusConfig()), [1, 2, 3, 4])
        self.assertEqual(floors, (0, 0, 0, 0))
        self.assertEqual(pans, (0.0, 0.0, 0.0, 0.0))


class TestModeShapePlacement(unittest.TestCase):
    """Population follows |mode shape|, so the shape is audible in space."""

    @staticmethod
    def _entry():
        from sensepi.sonification.chorus.types import CastEntry, SpeciesInfo
        info = SpeciesInfo(**{f.name: _FIELD_STUB[f.name]
                              for f in dataclasses.fields(SpeciesInfo)})
        return CastEntry(info=info, mode=0, role="lead",
                         target_carrier=2000.0, mode_freq=3.0)

    def _voice(self, shape_col, n=400):
        from sensepi.sonification.chorus.renderer import _Voice
        cfg = ChorusConfig()
        cfg.chorus_size = n / 5.0          # _MAX_IND["lead"] is 5
        shapes = np.asarray(shape_col, dtype=float)[:, None]
        return _Voice(self._entry(), [np.zeros(256, dtype=float)], cfg,
                      np.random.default_rng(3), shapes=shapes)

    def test_top_heavy_mode_sings_mostly_at_the_top(self):
        v = self._voice([0.1, 0.4, 0.8, 1.0])
        counts = np.bincount(np.asarray(v.row_i, dtype=int), minlength=4)
        self.assertGreater(counts[3], counts[0] * 2,
                           f"top floor should dominate, got {counts}")

    def test_a_node_is_quiet_but_not_silent(self):
        # A second mode with a sign change: row 1 is the node.
        v = self._voice([1.0, 0.02, -0.7, -1.0])
        counts = np.bincount(np.asarray(v.row_i, dtype=int), minlength=4)
        self.assertLess(counts[1], counts[0] / 2, f"node should be quiet: {counts}")
        self.assertGreater(counts[1], 0,
                           "a node that is absolutely silent reads as a dead sensor")

    def test_a_flat_shape_spreads_evenly(self):
        v = self._voice([1.0, 1.0, 1.0, 1.0])
        counts = np.bincount(np.asarray(v.row_i, dtype=int), minlength=4)
        self.assertLess(float(np.std(counts)) / max(float(np.mean(counts)), 1e-9), 0.35,
                        f"a flat shape should not clump: {counts}")

    def test_no_shape_still_produces_a_population(self):
        from sensepi.sonification.chorus.renderer import _Voice
        v = _Voice(self._entry(), [np.zeros(256)], ChorusConfig(),
                   np.random.default_rng(3), shapes=None)
        self.assertEqual(np.size(v.row_i), v.n)


class TestTorsionIsLocated(unittest.TestCase):
    def _run(self, twisting_row, seconds=45.0):
        t = np.arange(int(seconds * FS)) / FS
        rng = np.random.default_rng(1)
        ax = np.vstack([np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(t.size)
                        for f in (2.1, 6.3, 9.0, 11.0)])
        swell = np.clip((t - 15.0) / 5.0, 0, 1)
        rows = [0.01 * rng.standard_normal(t.size) for _ in range(4)]
        rows[twisting_row] = (0.02 + 0.6 * swell) * np.sin(2 * np.pi * 1.6 * t)
        gz = np.vstack(rows)
        cfg = ChorusConfig()
        cfg.sensor_map = MAP
        fx = FeatureExtractor(cfg)
        state = ModalState(frequencies_hz=np.array([2.1, 6.3, 9.0]),
                           damping=np.full(3, 0.02), fs=FS, ok=True, message="ok")
        out = None
        for k in range(10, int(seconds * 10) - 5):
            e = int(k * 0.1 * FS)
            out = fx.update(_Snap(ax[:, max(0, e - WIN):e], FS),
                            _Snap(gz[:, max(0, e - WIN):e], FS),
                            state, t=k * 0.1, rate_hz=FS)
            if k == 250:
                break
        return out

    def test_only_the_twisting_floor_reads_high(self):
        frame = self._run(twisting_row=3)
        tf = np.asarray(frame.torsion_floor)
        self.assertEqual(tf.size, 4)
        self.assertGreater(tf[3], 0.5, f"the twisting floor should be loud: {tf}")
        self.assertLess(float(np.max(tf[:3])), 0.2,
                        f"still floors should stay quiet: {tf}")

    def test_a_different_floor_moves_the_reading(self):
        frame = self._run(twisting_row=1)
        tf = np.asarray(frame.torsion_floor)
        self.assertEqual(int(np.argmax(tf)), 1, f"got {tf}")

    def test_pan_follows_the_live_rotation(self):
        """It used to average a detrended window, i.e. ~0 — pinned centre."""
        t = np.arange(3000) / FS
        rng = np.random.default_rng(2)
        ax = np.vstack([np.sin(2 * np.pi * f * t) for f in (2.1, 6.3, 9.0, 11.0)])
        gz = np.vstack([0.01 * rng.standard_normal(t.size)] * 3
                       + [0.5 * np.sin(2 * np.pi * 1.6 * t)])
        cfg = ChorusConfig()
        cfg.sensor_map = MAP
        fx = FeatureExtractor(cfg)
        state = ModalState(frequencies_hz=np.array([2.1, 6.3, 9.0]),
                           damping=np.full(3, 0.02), fs=FS, ok=True, message="ok")
        pans = []
        for k in range(10, 250):
            e = int(k * 0.1 * FS)
            pans.append(fx.update(_Snap(ax[:, max(0, e - WIN):e], FS),
                                  _Snap(gz[:, max(0, e - WIN):e], FS),
                                  state, t=k * 0.1, rate_hz=FS).torsion_pan)
        self.assertGreater(float(np.ptp(pans)), 0.8,
                           "the torsion voice does not move across the stereo field")
        self.assertGreater(int(np.sum(np.diff(np.sign(pans)) != 0)), 10,
                           "it should cross the centre as the structure twists back")


class TestIdentificationUsesTheMap(unittest.TestCase):
    def test_base_row_is_excluded_and_modes_are_capped(self):
        from sensepi.sonification.chorus.features import ModalTracker
        t = np.arange(3000) / FS
        rng = np.random.default_rng(4)
        # Row 0 (the base) carries a frequency no structural row has.
        data = np.vstack([np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(t.size)
                          for f in (3.7, 6.3, 9.0, 11.0)])
        cfg = ChorusConfig()
        cfg.sensor_map = MAP
        cfg.n_modes = 3
        state = ModalTracker(cfg).reidentify(_Snap(data, FS), 0.0)
        self.assertTrue(state.ok, state.message)
        self.assertLessEqual(state.frequencies_hz.size, 3)
        for f in state.frequencies_hz:
            self.assertGreater(abs(f - 3.7), 0.4,
                               "the shaker's own frequency was reported as a mode")


if __name__ == "__main__":
    unittest.main()
