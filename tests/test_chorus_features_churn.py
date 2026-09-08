"""The chorus must survive the mode and sensor counts changing mid-sound.

Both counts move at runtime: identification returns two modes instead of three
when a peak drops below the prominence threshold, and a sensor can stop
streaming. Every rolling history in the feature extractor is per-mode or
per-sensor wide, so a width change made the deque unstackable and raised inside
the audio worker — which meant silence, not a visible error.
"""
import unittest

import numpy as np

from sensepi.sonification.chorus.features import FeatureExtractor
from sensepi.sonification.chorus.types import ChorusConfig, ModalState

FS = 100.0
N = 600


class _Snap:
    """Minimal stand-in for the ModalSession the live worker passes."""

    def __init__(self, data, fs):
        self.data, self.fs = data, fs


def _signal(n_sensors=4):
    t = np.arange(N) / FS
    rng = np.random.default_rng(0)
    return np.vstack([np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(N)
                      for f in (2.1, 6.3, 9.0, 11.0)[:n_sensors]])


def _drive(fx, data, freqs, k0, k1, vary=True):
    """Feed frames. ``vary`` swells the amplitude frame to frame.

    The features are rolling *percentiles*, so a perfectly stationary input has
    no spread and normalises to 0 by definition. Varying the level is what makes
    "did the feature recover" a meaningful question rather than an artefact.
    """
    last = None
    for k in range(k0, k1):
        gain = (1.0 + 0.8 * np.sin(k * 0.37)) if vary else 1.0
        state = ModalState(frequencies_hz=np.array(freqs, dtype=float),
                           damping=np.full(len(freqs), 0.02),
                           fs=FS, ok=True, message="ok")
        last = fx.update(_Snap(data * gain, FS), _Snap(data * gain, FS), state,
                         t=k * 0.1, rate_hz=FS)
    return last


class TestCountChurn(unittest.TestCase):
    def test_mode_count_may_drop_and_recover(self):
        fx = FeatureExtractor(ChorusConfig())
        data = _signal()
        _drive(fx, data, [2.1, 6.3, 9.0], 0, 40)     # fill the histories
        _drive(fx, data, [2.1, 6.3], 40, 80)         # a peak drops out
        frame = _drive(fx, data, [2.1, 6.3, 9.0], 80, 120)   # and comes back
        self.assertEqual(frame.band_energy.size, 3)

    def test_sensor_count_may_drop(self):
        """Independent of the mode count — a streaming sensor can simply stop."""
        fx = FeatureExtractor(ChorusConfig())
        _drive(fx, _signal(4), [2.1, 6.3, 9.0], 0, 40)
        frame = _drive(fx, _signal(3), [2.1, 6.3, 9.0], 40, 80)
        self.assertEqual(frame.motion.size, 3)

    def test_both_counts_changing_together(self):
        fx = FeatureExtractor(ChorusConfig())
        _drive(fx, _signal(4), [2.1, 6.3, 9.0], 0, 40)
        _drive(fx, _signal(3), [2.1, 6.3], 40, 80)
        _drive(fx, _signal(2), [2.1], 80, 120)
        frame = _drive(fx, _signal(4), [2.1, 6.3, 9.0], 120, 200)
        self.assertEqual(frame.band_energy.size, 3)
        self.assertEqual(frame.motion.size, 4)

    def test_features_recover_rather_than_staying_dead(self):
        """After the churn the normalised features must come back to life.

        Clearing a history costs the percentile its history, so the guard is only
        acceptable if the feature refills afterwards instead of reading 0 forever.
        """
        fx = FeatureExtractor(ChorusConfig())
        data = _signal()
        _drive(fx, data, [2.1, 6.3, 9.0], 0, 40)
        _drive(fx, data, [2.1, 6.3], 40, 60)
        _drive(fx, data, [2.1, 6.3, 9.0], 60, 180)
        # Peak over a stretch, not one frame: band energy is a rolling
        # percentile of a swelling signal, so a single frame can sit at the
        # bottom of the swell and read 0 quite correctly.
        peak = max(float(np.max(_drive(fx, data, [2.1, 6.3, 9.0], k, k + 1).band_energy))
                   for k in range(180, 210))
        self.assertGreater(peak, 0.1,
                           "band energy never recovered after the mode count changed")


class TestNormaliseIsDefensive(unittest.TestCase):
    def test_ragged_history_returns_zeros_instead_of_raising(self):
        from collections import deque
        hist = deque([np.zeros(3)] * 5 + [np.zeros(2)])
        out = FeatureExtractor._normalise(hist, np.zeros(2))
        self.assertEqual(out.shape, (2,))
        self.assertTrue(np.all(out == 0.0))

    def test_push_clears_on_a_width_change(self):
        from collections import deque
        hist = deque(maxlen=10)
        for _ in range(5):
            FeatureExtractor._push(hist, np.ones(3))
        self.assertEqual(len(hist), 5)
        FeatureExtractor._push(hist, np.ones(2))
        self.assertEqual(len(hist), 1)
        self.assertEqual(np.shape(hist[-1]), (2,))


if __name__ == "__main__":
    unittest.main()
