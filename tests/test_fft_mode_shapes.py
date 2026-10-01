"""FFT mode shapes are usable, and the Spectrum tab draws them.

The Spectrum tab refused to draw FFT mode shapes ("DEBT-7: known-wrong, MAC
0.002 vs truth"). That figure came from a stale test comparing the FFT path's
*signed* shapes against the *unsigned* truth. Measured sign-insensitively, the
FFT shapes match the true shapes as closely as FDD's; Model Updating already
used them and the Spectrum final values already sent them.
"""

import os
import unittest

import numpy as np
from scipy import signal

from sensepi.analysis import modal

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FREQS = [2.8, 8.1, 12.3]
SHAPES = np.array([[0.328, 0.591, 0.737],     # mode 1 over floors 1..3
                   [0.737, 0.328, -0.591],    # mode 2
                   [0.591, -0.737, 0.328]])   # mode 3


def _mac(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a @ b) ** 2 / ((a @ a) * (b @ b)))


def _randomly_excited_frame(seed, fs=100.0, seconds=60.0):
    """3-storey response to broadband shaking: 2 %-damped modes plus sensor noise."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    q = np.array([signal.lfilter(*signal.iirpeak(f, f / 0.04, fs), rng.normal(0, 1, n))
                  for f in FREQS])
    return SHAPES.T @ q + 0.05 * rng.normal(0, 1, (3, n)), fs


class TestFftShapeAccuracy(unittest.TestCase):
    def test_fft_shapes_match_the_true_shapes_on_a_randomly_excited_frame(self):
        for seed in range(5):
            x, fs = _randomly_excited_frame(seed)
            result = modal.identify_modes(x, fs, f_min=0.5, f_max=20.0, n_modes=3, method="fft")
            self.assertTrue(result.success, result.message)
            matched = 0
            for f_true, shape_true in zip(FREQS, SHAPES):
                k = int(np.argmin([abs(f - f_true) for f in result.frequencies_hz]))
                if abs(result.frequencies_hz[k] - f_true) > 0.1 * f_true:
                    continue    # peak-picking can miss a weak mode; shapes are the point here
                matched += 1
                self.assertGreater(_mac(result.mode_shapes_sensor[k], shape_true), 0.99,
                                   f"seed {seed}, mode at {f_true} Hz")
            self.assertGreaterEqual(matched, 2, f"seed {seed}")


class TestSpectrumTabDrawsFftShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def test_fft_shapes_are_drawn_like_fdd_shapes(self):
        from sensepi.gui.main_window import MainWindow
        w = MainWindow()
        self.addCleanup(w.close)
        tab = w.fft_tab
        shape_data = {
            "method": "fft", "n_story": 3, "coverage_stories": [1, 2, 3],
            "full_coverage": True, "freqs": FREQS,
            "mode_shapes_ux": {str(i + 1): list(SHAPES[i]) for i in range(3)},
        }

        tab._render_mode_shapes(shape_data)

        self.assertTrue(tab._shape_items, "no FFT mode-shape curves were drawn")
        self.assertIn("Live mode shapes (FFT)", tab._eig_status.text())

    def test_the_method_is_not_labelled_frequencies_only(self):
        from sensepi.gui.main_window import MainWindow
        w = MainWindow()
        self.addCleanup(w.close)
        tab = w.fft_tab
        tab.method_combo.setCurrentIndex(tab.method_combo.findData("fft"))

        self.assertNotIn("frequencies only", tab.method_combo.currentText().lower())
        self.assertNotIn("NOT available", tab._method_hint.text())


if __name__ == "__main__":
    unittest.main()
