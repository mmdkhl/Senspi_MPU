"""Tests for the MAC helper + MAC-based mode pairing (PHASE 5 / B2).

Pure numpy — these import only ``math_utils`` (no openseespy) and run everywhere.
"""

import unittest

import numpy as np

from opensees_model_updating.utils.math_utils import mac, pair_modes_by_mac


class TestMac(unittest.TestCase):
    def test_identical_scale_and_sign_invariant(self):
        a = np.array([0.3, 0.7, 1.0])
        self.assertAlmostEqual(mac(a, a), 1.0, places=12)
        self.assertAlmostEqual(mac(a, -a), 1.0, places=12)        # sign-invariant
        self.assertAlmostEqual(mac(a, 2.5 * a), 1.0, places=12)   # scale-invariant

    def test_orthogonal(self):
        a = np.array([1.0, 0.0, -1.0])
        b = np.array([1.0, 0.0, 1.0])      # a . b = 0
        self.assertAlmostEqual(mac(a, b), 0.0, places=12)

    def test_bad_input_returns_zero(self):
        self.assertEqual(mac(np.array([]), np.array([])), 0.0)
        self.assertEqual(mac(np.array([1.0, 2.0]), np.array([1.0])), 0.0)   # length mismatch
        self.assertEqual(mac(np.zeros(3), np.array([1.0, 2.0, 3.0])), 0.0)  # zero vector


class TestPairModesByMac(unittest.TestCase):
    def _model(self):
        return [np.array([1.0, 0.5, 0.2]),
                np.array([0.2, -1.0, 0.6]),
                np.array([0.3, 0.4, -1.0])]

    def test_identity_when_aligned(self):
        model = self._model()
        exp = [m.copy() for m in model]
        self.assertEqual(pair_modes_by_mac(model, exp), [0, 1, 2])

    def test_pairs_swapped_modes(self):
        m0, m1, m2 = self._model()
        # Experimental modes are a reordering: exp0=m1, exp1=m0, exp2=m2.
        exp = [m1.copy(), m0.copy(), m2.copy()]
        self.assertEqual(pair_modes_by_mac([m0, m1, m2], exp), [1, 0, 2])

    def test_few_dof_falls_back_to_frequency_order(self):
        # Only 2 spatial points -> MAC degenerate -> identity (NOT [1,0]).
        model = [np.array([1.0, 0.3]), np.array([0.3, -1.0])]
        exp = [np.array([0.3, -1.0]), np.array([1.0, 0.3])]   # swapped, but 2 DOFs
        self.assertEqual(pair_modes_by_mac(model, exp), [0, 1])

    def test_empty(self):
        self.assertEqual(pair_modes_by_mac([], []), [])


if __name__ == "__main__":
    unittest.main()
