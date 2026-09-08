"""Live-test round 2: the twin view in Model Updating's output, Live Signals
normalisation, and a guard against the duplicated-method defect."""
import ast
import os
import unittest
from pathlib import Path

import numpy as np

from sensepi.digital_twin import decisions as D

DESIGN = {"floor_masses": [0.2, 0.2, 0.2], "E": 2.0e11, "m_scale_lb": 0.5,
          "m_scale_ub": 2.0, "E_scale_lb": 0.5, "E_scale_ub": 2.0, "use_mode_shapes": True}


class TestNoMethodIsDefinedTwice(unittest.TestCase):
    """A class that defines a method twice keeps the SECOND silently. That
    reinstated the GUI-thread wireframe once and hid a calibration path once."""

    def test_every_gui_module(self):
        root = Path(__file__).resolve().parents[1] / "src" / "sensepi" / "gui"
        offenders = []
        for py in root.rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    seen = {}
                    for item in node.body:
                        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        # @x.setter / @x.deleter legitimately re-use the name.
                        if any(isinstance(d, ast.Attribute) and d.attr in ("setter", "deleter")
                               for d in item.decorator_list):
                            continue
                        if item.name in seen:
                            offenders.append(f"{py.name}:{node.name}.{item.name}")
                        seen[item.name] = item.lineno
        self.assertEqual(offenders, [], f"methods defined twice: {offenders}")


class TestGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def test_decision_panel_shows_lights_and_one_line(self):
        from sensepi.gui.widgets.decision_panel import DecisionPanel
        panel = DecisionPanel()
        panel.set_decisions(D.decide(DESIGN, {"floor_masses": [0.36, 0.2, 0.2], "E": 1.7e11}),
                            source="cycle 3")
        self.assertEqual(panel._body_layout.count(), 4)          # F3 F2 F1 E
        self.assertTrue(panel.summary.text().startswith("cycle 3 ·"))
        self.assertIn("confounded", panel.summary.toolTip())     # reasoning in the tooltip
        panel.set_decisions(None)
        self.assertEqual(panel._body_layout.count(), 0)

    def test_output_twin_view_follows_cycle_and_rolling(self):
        from sensepi.gui.main_window import MainWindow
        w = MainWindow()
        mu = w.model_updating_tab
        payload = {
            "cycle": 7, "fig1_png": None, "fig2_fdd_png": None, "fig2_track_png": None,
            "twin_decisions_cycle": D.decide(DESIGN, {"floor_masses": [0.36, 0.2, 0.2], "E": 2.0e11}),
            "twin_decisions_rolling": D.decide(DESIGN, {"floor_masses": [0.24, 0.2, 0.2], "E": 1.7e11}),
            "twin_rolling_n": 5,
        }
        mu._cont_view_combo.setCurrentIndex(2)                   # Digital twin view
        mu._on_continuous_result(payload)
        self.app.processEvents()
        self.assertTrue(mu._twin_panel.summary.text().startswith("cycle 7 ·"))
        mu._twin_react_combo.setCurrentIndex(1)                  # rolling average
        self.app.processEvents()
        self.assertTrue(mu._twin_panel.summary.text().startswith("rolling average of 5"))
        self.app.processEvents()
        w.close()
        self.app.processEvents()

    def test_live_signals_normalise_is_symmetric_about_zero(self):
        from sensepi.gui.tabs.tab_signals import SignalsTab
        from sensepi.sensors.mpu6050 import MpuSample
        tab = SignalsTab(None)
        plot = tab._plot
        self.assertTrue(tab.normalise_check.isChecked())
        t0 = 1_000_000_000
        for i in range(600):
            plot.add_sample(MpuSample(timestamp_ns=t0 + i * 10_000_000, ax=2.0 + 0.5 * np.sin(i / 8),
                                      ay=0.0, az=0.0, gx=0.0, gy=0.0, gz=0.0, sensor_id=1, t_s=i / 100))
        plot.redraw()
        self.app.processEvents()
        key = next((k for k in plot._plots if k[1] == "ax"), None)
        self.assertIsNotNone(key)
        lo, hi = plot._plots[key].viewRange()[1]
        self.assertLess(abs(lo + hi), 0.05 * max(abs(hi), 1e-9), f"range [{lo}, {hi}] not centred")
        self.assertGreater(hi, 0.4)                                # follows the 0.5 amplitude
        tab.normalise_check.setChecked(False)
        self.assertFalse(plot._normalise_enabled)


if __name__ == "__main__":
    unittest.main()
