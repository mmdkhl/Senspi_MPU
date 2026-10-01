"""The Spectrum tab's live FFT must not run while the tab is hidden.

Its timer recomputed and redrew every sensor's spectrum on the GUI thread even
while another tab was showing. During a Model Updating Run Analysis that work
competed with the OpenSees worker and contributed to ~1 s GUI freezes.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


class TestFftIdleWhenHidden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def test_fft_updates_only_while_its_tab_is_showing(self):
        from sensepi.gui.main_window import MainWindow
        w = MainWindow()
        self.addCleanup(w.close)
        w.show()
        calls = []
        w.fft_tab._update_mpu6050_fft = lambda: calls.append(1)

        w._tabs.setCurrentWidget(w.model_updating_tab)
        self.app.processEvents()
        w.fft_tab._on_fft_timer()
        self.assertEqual(calls, [], "FFT recomputed while the Spectrum tab was hidden")

        w._tabs.setCurrentWidget(w.fft_tab)
        self.app.processEvents()
        w.fft_tab._on_fft_timer()
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
