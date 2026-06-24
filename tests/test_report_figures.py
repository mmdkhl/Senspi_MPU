"""Partial-coverage mode-shape reporting (PHASE 8 / T8.3).

Pure (numpy + matplotlib-Agg); no openseespy. Verifies the measured-vs-model overlay
helpers and that the embedded calibration-summary figure renders under PARTIAL
coverage (which would previously crash on a length mismatch).
"""

import unittest

import numpy as np

from opensees_model_updating.reporting.calibration_report import (
    _align_norm_on_support,
    _measured_dofs,
    generate_calibration_summary_png,
)


class TestMeasuredDofs(unittest.TestCase):
    def test_full_coverage_defaults_to_all(self):
        self.assertEqual(list(_measured_dofs({}, 3)), [0, 1, 2])

    def test_partial_coverage(self):
        self.assertEqual(list(_measured_dofs({"measured_dof_indices": [0, 2]}, 3)), [0, 2])


class TestAlignNormOnSupport(unittest.TestCase):
    def test_returns_full_length_normed_on_support(self):
        model_full = np.array([0.3, 1.0, 0.5])     # peak is OFF the measured support
        phi_exp = np.array([0.6, 1.0])
        idx = np.array([0, 2])
        out = _align_norm_on_support(model_full, phi_exp, idx)
        self.assertEqual(out.size, 3)              # full-length line returned
        self.assertAlmostEqual(float(np.max(np.abs(out[idx]))), 1.0, places=9)

    def test_sign_aligned_to_measured(self):
        model_full = np.array([-0.3, -1.0, -0.5])
        phi_exp = np.array([0.3, 0.5])
        idx = np.array([0, 2])
        out = _align_norm_on_support(model_full, phi_exp, idx)
        self.assertGreater(float(np.dot(out[idx], phi_exp)), 0.0)


class TestSummaryFigurePartialCoverage(unittest.TestCase):
    def test_partial_coverage_renders_without_crash(self):
        exp_data = {
            "n_modes_used": 1,
            "use_mode_shapes": True,
            "freqs": [2.3],
            "modes": [np.array([0.6, 1.0])],      # 2 measured stories
            "measured_dof_indices": [0, 2],        # stories 1 & 3 of a 3-story
        }
        modal_before = {"freqs": [2.0], "mode_shapes_ux_master": [[0.3, 0.7, 1.0]]}
        modal_after = {"freqs": [2.3], "mode_shapes_ux_master": [[0.6, 0.8, 1.0]]}
        png = generate_calibration_summary_png(
            exp_data, modal_before, modal_after, {"nStory": 3}, {"nStory": 3})
        self.assertIsInstance(png, (bytes, bytearray))
        self.assertGreater(len(png), 0)


if __name__ == "__main__":
    unittest.main()
