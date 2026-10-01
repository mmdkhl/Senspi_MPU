"""The log-decrement damping estimate must say when it cannot be trusted.

The estimator fits A0*exp(-beta*t) to successive response peaks, which is only
meaningful on a free decay. On steady shaking or ambient motion the peaks do
not decay, yet the fit still returned a plausible-looking zeta (R^2 0.4-0.8 on
this rig's data) that the Spectrum tab showed without comment and copied into
Model Updating. Free decays fit with R^2 >= 0.99.
"""

import unittest

import numpy as np
from scipy import signal

from sensepi.analysis.modal import estimate_damping_first_mode_real_response as estimate

FS = 100.0
T = np.arange(0.0, 20.0, 1.0 / FS)


def _free_decay(zeta, f=2.8, noise=0.005, seed=0):
    wn = 2.0 * np.pi * f
    wd = wn * np.sqrt(1.0 - zeta ** 2)
    rng = np.random.default_rng(seed)
    return np.exp(-zeta * wn * T) * np.sin(wd * T) + rng.normal(0.0, noise, T.size)


def _steady_random_response(f=2.8, seed=0):
    # A lightly damped mode driven by broadband noise: stationary, no decay.
    b, a = signal.iirpeak(f, f / 0.04, FS)
    return signal.lfilter(b, a, np.random.default_rng(seed).normal(0.0, 1.0, T.size))


class TestDampingReliability(unittest.TestCase):
    def test_free_decay_is_reliable_and_accurate(self):
        for zeta in (0.01, 0.02, 0.05):
            result = estimate(T, _free_decay(zeta), FS, 2.8)
            self.assertTrue(result["reliable"], f"zeta={zeta}, R2={result['fit_R2']}")
            self.assertAlmostEqual(result["zeta"], zeta, delta=0.1 * zeta)

    def test_steady_random_response_is_flagged_unreliable(self):
        for seed in range(5):
            result = estimate(T, _steady_random_response(seed=seed), FS, 2.8)
            self.assertFalse(result["reliable"], f"seed={seed}, R2={result['fit_R2']}")


class TestOnlyReliableDampingReachesModelUpdating(unittest.TestCase):
    def test_unreliable_damping_is_not_sent(self):
        from sensepi.gui.tabs.tab_fft import damping_for_model_updating

        self.assertEqual(damping_for_model_updating({"zeta": 0.02, "reliable": True}), 0.02)
        self.assertIsNone(damping_for_model_updating({"zeta": 0.005, "reliable": False}))
        self.assertIsNone(damping_for_model_updating({"zeta": 0.005}))
        self.assertIsNone(damping_for_model_updating({}))


if __name__ == "__main__":
    unittest.main()
