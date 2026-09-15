"""Every channel of the sensor feeds the chorus.

driven axis -> modes · other horizontal axis -> cross-axis chorus · az ->
ambient bed · gx/gy -> rocking (drift voice) · gz -> torsion · the shaker's
own sensor -> excitation reference. No Qt, no audio device.
"""
from __future__ import annotations

import unittest

import numpy as np

from sensepi.sonification.chorus import ChorusConfig, ChorusEngine, run_offline
from sensepi.sonification.chorus.features import FeatureExtractor
from sensepi.sonification.chorus.renderer import ChorusRenderer
from sensepi.sonification.chorus.types import ControlFrame, ModalState

FS = 41.0
MAP = {"n_floors": 3, "axis": "x", "placements": [
    {"sensor_id": 1, "floor": 0, "cell": "B2"},      # base / shaker
    {"sensor_id": 2, "floor": 1, "cell": "A1"},
    {"sensor_id": 3, "floor": 2, "cell": "A3"},
    {"sensor_id": 4, "floor": 3, "cell": "C1"},
]}


class _Snap:
    def __init__(self, data, fs=FS, ids=(1, 2, 3, 4)):
        self.data = np.asarray(data, dtype=float)
        self.fs = float(fs)
        self.sensor_ids = list(ids)[: self.data.shape[0]]


def _sine(f, n, amp, rows=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return np.vstack([amp * np.sin(2 * np.pi * f * t + 0.3 * r)
                      + rng.standard_normal(n) * 1e-4 for r in range(rows)])


def _run(fx, modal, main, channels, ticks=40, n=246):
    """Feed the same windows repeatedly so the rolling normalisers settle."""
    frame = None
    for i in range(ticks):
        frame = fx.update(_Snap(main[:, :n]), None, modal, t=i * 0.05, rate_hz=FS,
                          channels={k: _Snap(v[:, :n]) for k, v in channels.items()})
    return frame


class TestExtraChannels(unittest.TestCase):
    def setUp(self):
        self.cfg = ChorusConfig()
        self.cfg.sensor_map = MAP
        self.modal = ModalState(frequencies_hz=np.array([2.0, 6.0]), ok=True)

    def test_absent_channels_leave_the_frame_neutral(self):
        fx = FeatureExtractor(self.cfg)
        f = _run(fx, self.modal, _sine(2.0, 400, 0.05), {})
        self.assertEqual(f.channels, ())
        self.assertEqual(f.cross_energy.size, 0)
        self.assertEqual(f.cross_ratio, 0.0)
        self.assertEqual(f.vertical, 0.0)
        self.assertEqual(f.rock_floor.size, 0)

    def test_unstreamed_channel_reads_as_zero_not_noise(self):
        fx = FeatureExtractor(self.cfg)
        zeros = np.zeros((4, 400))
        f = _run(fx, self.modal, _sine(2.0, 400, 0.05),
                 {"ay": zeros, "az": zeros, "gx": zeros, "gy": zeros})
        self.assertEqual(f.channels, ())
        self.assertEqual(f.vertical, 0.0)

    def test_cross_axis_motion_is_seen_and_grows_with_its_amplitude(self):
        fx = FeatureExtractor(self.cfg)
        main = _sine(2.0, 400, 0.05)
        quiet = _run(fx, self.modal, main, {"ay": _sine(2.0, 400, 0.002, seed=1)})
        loud = _run(fx, self.modal, main, {"ay": _sine(2.0, 400, 0.05, seed=2)})
        self.assertIn("ay", loud.channels)
        self.assertEqual(loud.cross_energy.size, 2)
        self.assertGreater(loud.cross_ratio, quiet.cross_ratio)
        self.assertGreater(loud.cross_ratio, 0.5)

    def test_vertical_motion_raises_vertical(self):
        fx = FeatureExtractor(self.cfg)
        main = _sine(2.0, 400, 0.05)
        _run(fx, self.modal, main, {"az": _sine(7.0, 400, 0.001, seed=3)}, ticks=30)
        f = _run(fx, self.modal, main, {"az": _sine(7.0, 400, 0.08, seed=4)}, ticks=3)
        self.assertIn("az", f.channels)
        self.assertGreater(f.vertical, 0.5)

    def test_rocking_is_located_on_the_rocking_floor(self):
        fx = FeatureExtractor(self.cfg)
        main = _sine(2.0, 400, 0.05)
        still = np.zeros((4, 400)) + 1e-3 * np.random.default_rng(5).standard_normal((4, 400))
        _run(fx, self.modal, main, {"gx": still, "gy": still}, ticks=30)
        rock = still.copy()
        rock[3] += 2.0 * np.sin(2 * np.pi * 3.0 * np.arange(400) / FS)   # top floor rocks
        f = _run(fx, self.modal, main, {"gx": rock, "gy": still}, ticks=3)
        self.assertEqual(f.rock_floor.size, 4)
        self.assertEqual(int(np.argmax(f.rock_floor)), 3)
        self.assertGreater(f.rock_floor[3], 0.5)

    def test_base_sensor_is_the_excitation_reference(self):
        """The shaker row reports the drive frequency; the floors need not."""
        fx = FeatureExtractor(self.cfg)
        n = 400
        t = np.arange(n) / FS
        data = np.zeros((4, n))
        data[0] = 0.05 * np.sin(2 * np.pi * 5.5 * t)          # the table, at 5.5 Hz
        data[1:] = 0.05 * np.sin(2 * np.pi * 2.0 * t)         # floors ringing at mode 1
        f = fx.update(_Snap(data), None, self.modal, t=0.0, rate_hz=FS)
        self.assertAlmostEqual(f.exc_freq_hz, 5.5, delta=0.15)

    def test_without_a_map_the_floors_are_the_reference(self):
        cfg = ChorusConfig()          # no placement
        fx = FeatureExtractor(cfg)
        n = 400
        t = np.arange(n) / FS
        data = np.zeros((3, n))
        data[0] = 0.05 * np.sin(2 * np.pi * 5.5 * t)
        data[1:] = 0.05 * np.sin(2 * np.pi * 2.0 * t)
        f = fx.update(_Snap(data, ids=(1, 2, 3)), None, self.modal, t=0.0, rate_hz=FS)
        self.assertAlmostEqual(f.exc_freq_hz, 2.0, delta=0.15)


class TestRendererUsesTheChannels(unittest.TestCase):
    def _renderer_with(self, frame):
        r = ChorusRenderer(ChorusConfig())
        r.update_frame(frame)
        return r

    def test_vertical_thickens_the_ambient_bed(self):
        from sensepi.sonification.chorus.renderer import _Voice
        from sensepi.sonification.chorus.types import CastEntry, SpeciesInfo
        info = SpeciesInfo("amb", "cicadidae", 5000.0, 3.0, 60.0, 0.01, 30.0, 0.2,
                           type="cicadas")
        entry = CastEntry(info, -1, "ambient", 5000.0, 2.6)
        bank = [np.ones(256, dtype=np.float32) * 0.1]
        louds = []
        for vert in (0.0, 1.0):
            r = ChorusRenderer(ChorusConfig())
            fr = ControlFrame(env_global=0.1, vertical=vert)
            r.update_frame(fr)
            v = _Voice(entry, bank, r.cfg, r.rng)
            v.prime(0.0)
            r._voices = [v]
            r._grain_budget = 1e9
            r._sched_pos = 0
            r._schedule_voice(v, 0.0, 2.0)
            louds.append(sum(float(np.abs(t).sum()) for t in r._tiers))
        self.assertGreater(louds[1], louds[0] * 1.3)

    def test_rocking_drives_the_drift_voice(self):
        from sensepi.sonification.chorus.renderer import _Voice
        from sensepi.sonification.chorus.types import CastEntry, SpeciesInfo
        info = SpeciesInfo("rasp", "acrididae", 6000.0, 8.0, 100.0, 0.002, 30.0, 0.5,
                           type="grasshoppers")
        entry = CastEntry(info, -3, "drift", 6000.0, 4.0)
        bank = [np.ones(128, dtype=np.float32) * 0.1]
        out = []
        for rock in (np.zeros(4), np.array([0.0, 0.0, 0.0, 1.0])):
            r = ChorusRenderer(ChorusConfig())
            r.update_frame(ControlFrame(drift=np.zeros(3), rock_floor=rock,
                                        pan_of=(0.0, -0.85, -0.85, 0.85),
                                        floor_of=(0, 1, 2, 3)))
            v = _Voice(entry, bank, r.cfg, r.rng)
            v.prime(0.0)
            r._voices = [v]
            r._grain_budget = 1e9
            r._schedule_voice(v, 0.0, 2.0)
            out.append(sum(float(np.abs(t).sum()) for t in r._tiers))
        self.assertEqual(out[0], 0.0)
        self.assertGreater(out[1], 0.0)


class TestOfflineAndEngine(unittest.TestCase):
    def test_run_offline_accepts_channels(self):
        n = int(FS * 12)
        main = _sine(2.0, n, 0.05, rows=3)
        cfg = ChorusConfig()
        audio, frames, _ = run_offline(_Snap(main, ids=(1, 2, 3)), None, cfg,
                                       duration_s=10.0,
                                       channels={"ay": _Snap(_sine(2.0, n, 0.03, rows=3, seed=7),
                                                             ids=(1, 2, 3)),
                                                 "az": _Snap(_sine(9.0, n, 0.03, rows=3, seed=8),
                                                             ids=(1, 2, 3))})
        self.assertTrue(np.isfinite(audio).all())
        self.assertTrue(any("ay" in f.frame.channels for f in frames[-20:]))
        self.assertTrue(any("az" in f.frame.channels for f in frames[-20:]))

    def test_tick_without_channels_still_works(self):
        engine = ChorusEngine(ChorusConfig())
        viz = engine.tick(0.1, _Snap(_sine(2.0, 246, 0.05, rows=3), ids=(1, 2, 3)), None)
        self.assertEqual(viz.frame.channels, ())


if __name__ == "__main__":
    unittest.main()
