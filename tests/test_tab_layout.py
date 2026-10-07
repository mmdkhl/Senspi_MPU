"""Where the tabs are, and what the Calibration form opens on.

The Digital Twin Experiment tab was renamed **Digital Shadow** and moved inside
Model Updating, and Model Updating's **Output** sub-tab was renamed **Digital
Twin**. Nothing about what either tab does changed — only its name and its place.

Watch the crossover when reading this: the tab the UI calls *Digital Shadow* is
`tab_digital_twin.py` (and `MainWindow.digital_twin_tab`, and
`output/digital_twin/`), while the sub-tab the UI calls *Digital Twin* is Model
Updating's `_output_tab`. The internal names were left alone on purpose —
renaming the package or the results folder would move saved experiment data.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


class TestTabLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        from sensepi.gui.main_window import MainWindow

        self.win = MainWindow()
        self.addCleanup(self.win.close)
        self.mu = self.win.model_updating_tab

    def _titles(self, tabs):
        return [tabs.tabText(i) for i in range(tabs.count())]

    def test_the_top_level_tabs(self):
        self.assertEqual(
            self._titles(self.win._tabs),
            ["Live Signals", "Spectrum", "Model Updating", "Sonification", "Settings"])

    def test_the_old_top_level_twin_tab_is_gone(self):
        titles = self._titles(self.win._tabs)
        self.assertNotIn("Digital Twin Experiment", titles)
        self.assertNotIn("Digital Shadow", titles,
                         "Digital Shadow belongs inside Model Updating, not beside it")

    def test_the_model_updating_sub_tabs_in_order(self):
        self.assertEqual(
            self._titles(self.mu._tabs),
            ["Model", "Additional Mass", "Analysis", "Calibration",
             "Digital Twin", "Digital Shadow"])

    def test_digital_twin_is_the_renamed_output_sub_tab(self):
        # Seven call sites do setCurrentWidget(self._output_tab); the rename must
        # be a label change only, or every one of them silently stops working.
        i = self._titles(self.mu._tabs).index("Digital Twin")
        self.assertIs(self.mu._tabs.widget(i), self.mu._output_tab)
        self.assertNotIn("Output", self._titles(self.mu._tabs))

    def test_digital_shadow_is_the_experiment_tab_itself(self):
        from sensepi.gui.tabs.tab_digital_twin import DigitalTwinExperimentTab

        i = self._titles(self.mu._tabs).index("Digital Shadow")
        hosted = self.mu._tabs.widget(i)
        self.assertIsInstance(hosted, DigitalTwinExperimentTab)
        # Still owned and shut down from MainWindow, which is what closeEvent
        # relies on.
        self.assertIs(hosted, self.win.digital_twin_tab)
        self.assertTrue(callable(getattr(hosted, "shutdown", None)))

    def test_the_shadow_still_reaches_its_collaborator(self):
        # It was moved, not rewired: it still reads the calibrated model from
        # the Model Updating tab.
        self.assertIs(self.win.digital_twin_tab._model_tab, self.mu)

    def test_the_shadow_still_follows_the_stream(self):
        shadow = self.win.digital_twin_tab
        self.assertFalse(shadow._render_timer.isActive())

        shadow.on_stream_started()
        self.assertTrue(shadow._render_timer.isActive(),
                        "nesting the tab broke its stream wiring")

        shadow.on_stream_stopped()
        self.assertFalse(shadow._render_timer.isActive())


class TestCalibrationDefaults(unittest.TestCase):
    """What the Calibration form shows before anyone touches it."""

    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        from sensepi.gui.tabs.tab_model_updating import ModelUpdatingTab

        self.tab = ModelUpdatingTab()
        self.addCleanup(self.tab.deleteLater)

    def test_calibration_method_defaults_to_bayesian(self):
        self.assertEqual(self.tab._calibration_method.currentText(), "Bayesian (Gaussian)")

    def test_analysis_scope_defaults_to_frequency_plus_mode_shapes(self):
        self.assertEqual(self.tab._analysis_scope.currentText(), "Frequency + mode shapes")

    def test_mass_target_defaults_to_the_total_including_additional_masses(self):
        self.assertEqual(self.tab._mass_scope.currentText(),
                         "Total mass including additional masses")

    def test_loading_identified_data_still_overrides_the_scope(self):
        # The default must not fight the data: identified data with no shapes has
        # to pull the scope back to frequency only.
        self.tab._apply_identified_to_fields(
            {"frequencies_hz": [1.0, 4.0, 7.0], "mode_shapes_ux": {}}, False)

        self.assertEqual(self.tab._analysis_scope.currentText(), "Frequency only")

    def test_identified_data_with_shapes_keeps_mode_shapes_on(self):
        self.tab._apply_identified_to_fields(
            {"frequencies_hz": [1.0, 4.0], "mode_shapes_ux": {1: [1.0, 0.5]}}, True)

        self.assertEqual(self.tab._analysis_scope.currentText(), "Frequency + mode shapes")


if __name__ == "__main__":
    unittest.main()
