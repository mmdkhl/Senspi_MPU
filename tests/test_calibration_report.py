"""The comparison report must accept every calibration result the GUI produces.

Calibrate defaults to the Bayesian engine, whose result carries the posterior
``mean`` (and ``sigma``) rather than scipy's ``x``. The report used to read
``calib_result.x`` unconditionally, so every default Calibrate run finished the
optimisation and then crashed while writing the report.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from opensees_model_updating.calibration.bayesian import BayesianResult
from opensees_model_updating.reporting.calibration_report import (
    make_modal_comparison_report,
    report_to_text,
)


def _inputs():
    exp_data = {"n_modes_used": 2, "freqs": [2.6, 8.0], "use_mode_shapes": False}
    modal_before = {"freqs": [2.4, 7.5]}
    modal_after = {"freqs": [2.59, 7.9]}
    base = {"freq_tol_percent": 5.0, "E": 2.0e11, "floor_masses": [0.3, 0.3], "nStory": 2}
    final = {"E": 1.9e11, "floor_masses": [0.31, 0.29]}
    return exp_data, modal_before, modal_after, base, final


class TestReportAcceptsBayesianResult(unittest.TestCase):
    def test_bayesian_result_reports_posterior_mean_and_sigma(self):
        mean = np.array([0.95, 1.02, 0.98])
        sigma = np.array([0.05, 0.03, 0.04])
        result = BayesianResult(
            mean=mean, cov=np.diag(sigma ** 2), sigma=sigma,
            bands_1sigma=(mean - sigma, mean + sigma),
            bands_95=(mean - 1.96 * sigma, mean + 1.96 * sigma),
            success=True, status=1, cost=0.12, n_data=4, nfev=17, message="ok",
        )

        report = make_modal_comparison_report(*_inputs(), calib_result=result)

        self.assertEqual(report["optimizer"]["x"], mean.tolist())
        self.assertEqual(report["optimizer"]["sigma"], sigma.tolist())
        self.assertIn("x       =", report_to_text(report))

    def test_scipy_result_still_reports_x(self):
        result = SimpleNamespace(success=True, status=1, message="ok", nfev=9,
                                 cost=0.2, x=np.array([1.0, 0.9]))

        report = make_modal_comparison_report(*_inputs(), calib_result=result)

        self.assertEqual(report["optimizer"]["x"], [1.0, 0.9])
        self.assertNotIn("sigma", report["optimizer"])


if __name__ == "__main__":
    unittest.main()
