"""Partial-coverage residual tests for the calibrator (B1 / BLOCKER-6/7 + RISK-MU-2).

These exercise ``modal_residuals`` directly. Importing the calibrator pulls in the
OpenSees forward model, so the whole class is skipped where ``openseespy`` is
unavailable (e.g. the ``senspi_env`` conda env). The canonical build/test env is
``senspi_mpu``, which has it. The heavy eigen-solve is monkeypatched out so the test
is fast and deterministic — it validates the residual *length/slicing* logic, which
is the part B1 changed and where ``least_squares`` crashes if it is wrong.
"""

import unittest

import numpy as np

try:  # calibrator import requires openseespy (transitively)
    import openseespy.opensees  # noqa: F401
    _HAS_OPENSEES = True
except Exception:  # pragma: no cover - env-dependent
    _HAS_OPENSEES = False


def _raise(*_args, **_kwargs):
    raise RuntimeError("forced forward-model failure")


@unittest.skipUnless(_HAS_OPENSEES, "calibrator import requires openseespy (use senspi_mpu)")
class TestModalResidualsPartialCoverage(unittest.TestCase):
    """Success-branch and failure-branch residual lengths must always match."""

    def setUp(self):
        from opensees_model_updating.calibration import calibrator

        self.calibrator = calibrator
        # A fake 3-story forward model: 3 freqs + full-length signed shapes.
        self._fake_modal = {
            "freqs": [2.30, 6.50, 9.10],
            "mode_shapes_ux_master": [
                [0.30, 0.70, 1.00],
                [-1.00, 0.20, 0.90],
                [0.80, -1.00, 0.50],
            ],
        }
        self._orig_apply = calibrator.apply_calibration_vector
        self._orig_extract = calibrator.extract_modal_results
        # Replace the heavy pieces so the residual logic runs without a real solve.
        calibrator.apply_calibration_vector = lambda bp, x: bp
        calibrator.extract_modal_results = lambda p, **k: self._fake_modal

    def tearDown(self):
        self.calibrator.apply_calibration_vector = self._orig_apply
        self.calibrator.extract_modal_results = self._orig_extract

    def _exp(self, use_shapes, modes, measured_dofs):
        return {
            "freqs": np.array([2.30, 6.50]),
            "modes": modes,
            "use_mode_shapes": use_shapes,
            "mode_shapes_available": use_shapes,
            "n_modes_used": 2,
            "measured_dof_indices": measured_dofs,
        }

    def _x(self):
        return np.array([1.0, 1.0, 1.0, 1.0])

    def _force_failure(self):
        self.calibrator.extract_modal_results = _raise

    def test_full_coverage_lengths_match(self):
        modes = [np.array([0.30, 0.70, 1.00]), np.array([-1.00, 0.20, 0.90])]
        exp = self._exp(True, modes, [0, 1, 2])
        ok = self.calibrator.modal_residuals(self._x(), {"nStory": 3}, exp)
        self._force_failure()
        bad = self.calibrator.modal_residuals(self._x(), {"nStory": 3}, exp)
        self.assertEqual(ok.shape, bad.shape)
        self.assertEqual(len(ok), 2 + 2 * 3)  # 2 freqs + 2 modes x 3 stories

    def test_partial_coverage_lengths_match(self):
        # Measured stories {1,3} -> 0-based DOF indices [0, 2]; partial shape len 2.
        modes = [np.array([0.30, 1.00]), np.array([-1.00, 0.90])]
        exp = self._exp(True, modes, [0, 2])
        ok = self.calibrator.modal_residuals(self._x(), {"nStory": 3}, exp)
        self._force_failure()
        bad = self.calibrator.modal_residuals(self._x(), {"nStory": 3}, exp)
        self.assertEqual(ok.shape, bad.shape)
        self.assertEqual(len(ok), 2 + 2 * 2)  # 2 freqs + 2 modes x 2 measured DOFs
        self.assertTrue(np.all(np.isfinite(ok)))  # sliced + re-normalized cleanly

    def test_frequency_only_length(self):
        exp = self._exp(False, [], None)
        ok = self.calibrator.modal_residuals(self._x(), {"nStory": 3}, exp)
        self.assertEqual(len(ok), 2)  # 2 freq residuals, no shape terms

    def test_temporal_anchor_length_consistency(self):
        # R2 (T6.3): the optional temporal-smoothness block adds (1+nStory) terms;
        # success and failure branches must stay the same length (off by default).
        modes = [np.array([0.30, 0.70, 1.00]), np.array([-1.00, 0.20, 0.90])]
        exp = self._exp(True, modes, [0, 1, 2])
        base = {"nStory": 3, "temporal_anchor": (np.array([1.0, 1.0, 1.0, 1.0]), 0.3)}
        ok = self.calibrator.modal_residuals(self._x(), base, exp)
        self._force_failure()
        bad = self.calibrator.modal_residuals(self._x(), base, exp)
        self.assertEqual(ok.shape, bad.shape)
        self.assertEqual(len(ok), 2 + 2 * 3 + 4)  # freqs + shapes + (1+nStory) anchor

    def test_mac_pairing_fixes_swapped_modes(self):
        # Model returns modes in one order; the experimental data is a REORDERING
        # (exp mode i corresponds to model mode [1,0,2][i] in both freq and shape).
        # With MAC pairing the residual collapses to ~0; with naive frequency-order
        # pairing it would be large -> proves B2 is active in the residual path.
        s0, s1, s2 = [1.0, 0.5, 0.2], [0.2, -1.0, 0.6], [0.3, 0.4, -1.0]
        self.calibrator.extract_modal_results = lambda p, **k: {
            "freqs": [2.0, 6.0, 9.0],
            "mode_shapes_ux_master": [s0, s1, s2],
        }
        exp = {
            "freqs": np.array([6.0, 2.0, 9.0]),               # swapped order
            "modes": [np.array(s1), np.array(s0), np.array(s2)],
            "use_mode_shapes": True,
            "mode_shapes_available": True,
            "n_modes_used": 3,
            "measured_dof_indices": [0, 1, 2],
        }
        res = self.calibrator.modal_residuals(self._x(), {"nStory": 3}, exp)
        self.assertTrue(np.allclose(res, 0.0, atol=1e-9))

    def test_partial_phi_num_renormalized_on_measured_support(self):
        # phi_num measured DOFs [0,2] = [0.30, 1.00] (already max-abs 1 here);
        # use a model shape whose max is OFF the measured support to prove the
        # re-normalization (RISK-MU-2): full = [0.30, 1.00(off), 0.50]; measured
        # DOFs [0,2] -> [0.30, 0.50] must renormalize to [0.60, 1.00].
        self.calibrator.extract_modal_results = lambda p, **k: {
            "freqs": [2.30, 6.50],
            "mode_shapes_ux_master": [[0.30, 1.00, 0.50], [0.30, 1.00, 0.50]],
        }
        # phi_exp on measured support already normalized to [0.60, 1.00] -> zero
        # shape residual if phi_num is correctly re-normalized on the same support.
        modes = [np.array([0.60, 1.00]), np.array([0.60, 1.00])]
        exp = self._exp(True, modes, [0, 2])
        res = self.calibrator.modal_residuals(self._x(), {"nStory": 3}, exp)
        shape_block = res[2:]  # drop the 2 frequency residuals
        self.assertTrue(np.allclose(shape_block, 0.0, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
