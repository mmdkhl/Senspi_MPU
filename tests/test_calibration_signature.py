"""GAP-MU-2: the calibration cache signature must change when the method or any new
Bayesian knob changes, so switching engines never serves a stale cached result.

Imports the GUI tab module (PySide6); skipped where that import is unavailable.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from sensepi.gui.tabs.tab_model_updating import _make_calibration_signature
    _HAS_GUI = True
except Exception:  # pragma: no cover - env-dependent
    _HAS_GUI = False


@unittest.skipUnless(_HAS_GUI, "PySide6 / GUI module import unavailable")
class TestCalibrationSignature(unittest.TestCase):
    def _base(self):
        return {
            "calibration_method": "bayesian",
            "use_mode_shapes": True,
            "sigma_prior_E": 0.30,
            "sigma_prior_m": 0.15,
            "sigma_data_scale": 0.02,
            "mass_similarity_weight": 0.5,
            "w_freq": 1.0,
            "w_mode": 0.35,
        }

    def test_method_changes_signature(self):
        a = self._base()
        b = dict(a, calibration_method="least_squares")
        self.assertNotEqual(_make_calibration_signature(a), _make_calibration_signature(b))

    def test_bayesian_knob_changes_signature(self):
        a = self._base()
        for knob, val in [("sigma_prior_E", 0.5), ("sigma_prior_m", 0.25),
                          ("sigma_data_scale", 0.05), ("mass_similarity_weight", 0.9)]:
            b = dict(a, **{knob: val})
            self.assertNotEqual(_make_calibration_signature(a),
                                _make_calibration_signature(b),
                                msg=f"signature unchanged when {knob} changed")

    def test_same_params_same_signature(self):
        self.assertEqual(_make_calibration_signature(self._base()),
                         _make_calibration_signature(dict(self._base())))


if __name__ == "__main__":
    unittest.main()
