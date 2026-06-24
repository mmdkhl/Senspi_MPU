# -*- coding: utf-8 -*-
"""Unit tests for the Stage-1 ModalStateTracker (CU-2).

Pure numpy, no Qt / OpenSees -> runs in any env. These lock down the four
behaviours the continuous-update redesign depends on:
  * a cluster of consistent readings shrinks sigma (band tightens on agreement);
  * a lone outlier barely moves the mean and *raises* sigma (never breaks the loop);
  * a sustained step migrates the mean (real change tracked, not smoothed away);
  * association rejects a far peak (it is logged, not folded);
  * with < 3 measured DOFs the MAC gate is NOT applied (frequency-NN fallback);
  * tracks are seeded from the lowest-frequency peaks (fixes the D2 mislabelling).
"""

import unittest

import numpy as np

from sensepi.analysis.modal_tracker import (
    ModalStateTracker,
    ReadingDiagnostic,
    _mac,
)


class TestRobustEWMA(unittest.TestCase):
    def test_cluster_shrinks_sigma(self):
        # First inflate sigma with a couple of spread readings, then feed a tight
        # cluster: sigma must shrink back down (band tightens on agreement).
        trk = ModalStateTracker(n_modes=1, var_floor_frac=0.005)
        trk.update([8.30])
        trk.update([8.50])           # spread -> inflates v
        inflated = trk.update([8.10])
        sigma_high = inflated.freq_sigma[0]
        rng = np.random.default_rng(0)
        out = inflated
        for _ in range(60):
            out = trk.update([8.30 + rng.normal(0, 0.01)])
        self.assertLess(out.freq_sigma[0], sigma_high)
        # Converged near the true value with a small, honest scatter sigma.
        self.assertAlmostEqual(out.frequencies[0], 8.30, delta=0.05)
        self.assertLess(out.freq_sigma[0], 0.1)

    def test_lone_outlier_barely_moves_mean_and_raises_sigma(self):
        trk = ModalStateTracker(n_modes=1)
        # Settle on 8.30 with a few clean reads.
        out = None
        for _ in range(20):
            out = trk.update([8.30])
        m_before = out.frequencies[0]
        s_before = out.freq_sigma[0]
        # One wild outlier far outside the gate-> still associated? No: it is far,
        # so it is gated out entirely. Use a moderate outlier inside the gate.
        out = trk.update([8.30 + 3.0 * s_before])  # ~3 sigma: inside gate, treated as outlier
        # Mean moves only a small fraction of the 3-sigma jump.
        self.assertLess(abs(out.frequencies[0] - m_before), 0.5 * 3.0 * s_before)
        # Uncertainty went UP (an outlier inflates the scatter).
        self.assertGreater(out.freq_sigma[0], s_before)
        # The reading is logged as an outlier, folded (not dropped).
        folded = [d for d in out.diagnostics if d.track == 0]
        self.assertTrue(folded and folded[0].note == "outlier")
        self.assertTrue(folded[0].folded)
        self.assertGreater(folded[0].weight, 0.0)  # never zero

    def test_persistent_shift_migrates_mean(self):
        trk = ModalStateTracker(n_modes=1)
        for _ in range(25):
            trk.update([8.30])
        out = None
        # A real structural change: the mode is now consistently at 7.50 Hz.
        for _ in range(40):
            out = trk.update([7.50])
        self.assertAlmostEqual(out.frequencies[0], 7.50, delta=0.1)
        # And the band re-shrinks once the shift is the new consensus.
        self.assertLess(out.freq_sigma[0], 0.3)

    def test_far_peak_is_not_folded(self):
        trk = ModalStateTracker(n_modes=1, assoc_gate=4.0, var_floor_frac=0.02)
        for _ in range(20):
            trk.update([8.30])
        m_before = trk.update([8.30]).frequencies[0]
        # A spurious 16 Hz peak is many sigma away -> unassociated, mean unchanged.
        out = trk.update([16.0])
        self.assertAlmostEqual(out.frequencies[0], m_before, delta=1e-6)
        unassoc = [d for d in out.diagnostics if d.note == "unassociated"]
        self.assertTrue(unassoc and abs(unassoc[0].freq - 16.0) < 1e-9)


class TestSeeding(unittest.TestCase):
    def test_seeds_from_lowest_frequencies(self):
        # First cycle returns a strong spurious 16 Hz peak among the real low modes.
        # The tracker must seed the THREE LOWEST (1.9, 6.2, 8.3), not the 16 Hz noise.
        trk = ModalStateTracker(n_modes=3)
        out = trk.update([16.0, 1.9, 8.3, 6.2])
        self.assertEqual(len(out.frequencies), 3)
        np.testing.assert_allclose(sorted(out.frequencies), [1.9, 6.2, 8.3], atol=1e-9)
        # The 16 Hz peak is logged as unassociated (not a track).
        self.assertTrue(any(d.note == "unassociated" and abs(d.freq - 16.0) < 1e-9
                            for d in out.diagnostics))

    def test_three_modes_associate_independently(self):
        trk = ModalStateTracker(n_modes=3)
        trk.update([1.90, 6.25, 8.30])
        out = None
        for _ in range(15):
            out = trk.update([1.91, 6.24, 8.33])
        self.assertEqual(len(out.frequencies), 3)
        np.testing.assert_allclose(out.frequencies, [1.91, 6.24, 8.33], atol=0.05)


class TestShapesAndMac(unittest.TestCase):
    def test_mac_helper_properties(self):
        a = np.array([1.0, 0.5, -0.3])
        self.assertAlmostEqual(_mac(a, a), 1.0)
        self.assertAlmostEqual(_mac(a, -a), 1.0)            # sign-invariant
        self.assertAlmostEqual(_mac(a, 2.0 * a), 1.0)        # scale-invariant
        self.assertAlmostEqual(_mac([1.0, 0.0], [0.0, 1.0]), 0.0)  # orthogonal
        self.assertEqual(_mac([], [1.0]), 0.0)               # guard

    def test_shapes_tracked_and_consolidated(self):
        trk = ModalStateTracker(n_modes=2)
        cov = [1, 2, 3]
        shapes = [[0.3, 0.7, 1.0], [1.0, -0.2, -0.8]]
        out = trk.update([2.0, 6.0], shapes=shapes, coverage_stories=cov)
        for _ in range(10):
            out = trk.update([2.0, 6.0], shapes=shapes, coverage_stories=cov)
        self.assertTrue(out.shapes_available)
        self.assertEqual(out.coverage_stories, cov)
        self.assertEqual(len(out.shapes[0]), 3)
        # freq_sigma_rel is well-defined and small after a clean cluster.
        self.assertTrue(all(np.isfinite(r) for r in out.freq_sigma_rel))

    def test_under_three_dofs_no_mac_gate(self):
        # 2 measured DOFs: MAC is degenerate, so a frequency-close peak with an
        # opposite-looking 2-element shape must STILL associate (freq-NN fallback).
        trk = ModalStateTracker(n_modes=1, mac_gate=0.9)
        cov = [1, 2]
        trk.update([5.0], shapes=[[1.0, 0.9]], coverage_stories=cov)
        out = trk.update([5.0], shapes=[[1.0, -0.9]], coverage_stories=cov)  # shape "flipped"
        folded = [d for d in out.diagnostics if d.track == 0 and d.folded]
        self.assertTrue(folded, "with <3 DOFs the peak must associate by frequency despite MAC")


if __name__ == "__main__":
    unittest.main()
