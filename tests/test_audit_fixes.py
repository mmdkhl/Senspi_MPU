"""Pins for the phase-5 audit fixes that are cheap to reproduce without hardware."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sensepi.dataio import modal_session_loader as msl


def _write_session(root: Path, stamp: str, n: int = 300, with_audit: bool = True) -> Path:
    d = root / "Pi_11" / "mpu" / stamp
    d.mkdir(parents=True)
    t = np.arange(n) / 100.0
    for sid in (1, 2):
        rows = ["timestamp_ns,t_s,sensor_id,ax,ay,gz"]
        rows += [f"{int(x*1e9)},{x:.3f},{sid},{np.sin(x):.4f},0,0" for x in t]
        (d / f"mpu_S{sid}_{stamp}.csv").write_text("\n".join(rows))
        if with_audit:
            a = d / f"audit_{stamp}"
            a.mkdir(exist_ok=True)
            (a / f"raw_S{sid}_{stamp}.csv").write_text("\n".join(rows))
    return d


class TestSessionsAreFolders(unittest.TestCase):
    """LS-7: a second recording must never merge with the first."""

    def test_each_recording_is_its_own_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write_session(root, "2026-01-01_10-00-00")
            b = _write_session(root, "2026-01-02_10-00-00")
            found = msl.list_sessions(root)
            self.assertEqual({p.name for p in found}, {a.name, b.name})

    def test_the_audit_copy_is_not_loaded_with_the_main_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _write_session(Path(tmp), "2026-01-01_10-00-00")
            files = msl.find_session_files(d)
            self.assertEqual(len(files), 2)
            self.assertFalse(any("raw_" in f.name for f in files))
            self.assertFalse(any("audit_" in str(f) for f in files))

    def test_a_host_folder_is_not_itself_a_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_session(root, "2026-01-01_10-00-00")
            names = [p.name for p in msl.list_sessions(root)]
            self.assertNotIn("Pi_11", names)
            self.assertNotIn("mpu", names)

    def test_sliced_session_keeps_ids_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _write_session(Path(tmp), "2026-01-01_10-00-00")
            s = msl.load_session(d, axis="ax")
            t = msl.sliced_session(s, [1])
            self.assertEqual(t.sensor_ids, [2])
            self.assertEqual(t.data.shape[0], 1)


class TestGuiFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def test_one_preset_change_emits_one_map(self):
        """S-4: seven emissions per preset change, coalesced to one."""
        from sensepi.gui.widgets.sensor_map import SensorMapWidget, PRESETS
        w = SensorMapWidget()
        self.app.processEvents()
        n = []
        w.mapChanged.connect(lambda m: n.append(1))
        w.combo_preset.setCurrentText(list(PRESETS)[2])
        self.app.processEvents()
        self.assertEqual(len(n), 1)

    def test_sensor_count_changes_rematch_the_preset_label(self):
        """S-6: one sensor cannot be a 'torsion pair'."""
        from sensepi.gui.widgets.sensor_map import SensorMapWidget, CUSTOM
        w = SensorMapWidget()
        w.combo_preset.setCurrentText("3 storeys — one per floor + torsion pair on top")
        w.set_sensor_count(1)
        self.app.processEvents()
        self.assertEqual(w.combo_preset.currentText(), CUSTOM)

    def test_partial_coverage_fills_the_right_storeys(self):
        """SP-1 / MU-6: shapes in coverage order land on their storeys."""
        from PySide6.QtWidgets import QDoubleSpinBox
        from sensepi.gui.tabs.tab_model_updating import ModelUpdatingTab
        t = ModelUpdatingTab()
        t._story_count.setValue(5)
        t._rebuild_exp_data_widgets()
        t._apply_identified_to_fields({
            "frequencies_hz": [2.0],
            "mode_shapes_ux": {"1": [0.1, 0.2, 0.8, 1.0]},
            "coverage_stories": [1, 2, 4, 5],
        }, True)
        col = 0
        got = {}
        for r in range(t._exp_mode_table.rowCount()):
            w = t._exp_mode_table.cellWidget(r, col)
            if isinstance(w, QDoubleSpinBox):
                got[r + 1] = round(w.value(), 3)
        self.assertEqual(got[4], 0.8)
        self.assertEqual(got[5], 1.0)
        self.assertNotEqual(got.get(3), 0.8, "storey 4's value landed on storey 3")

    def test_digital_twin_arms_on_its_own_calibration(self):
        """DT-1: the in-tab calibration drives the experiment."""
        from sensepi.gui.main_window import MainWindow
        w = MainWindow()
        dt = w.digital_twin_tab
        # nothing calibrated anywhere -> a message that names step 1
        with self.assertRaises(ValueError) as cm:
            dt._build_setup()
        self.assertIn("Calibrate here", str(cm.exception))
        # calibrate "here" -> Arm uses it
        snap = w.model_updating_tab.model_definition_snapshot()
        dt._model_snapshot = snap
        dt._calibrated_params = dict(snap)
        setup = dt._build_setup()
        self.assertEqual(setup["calibration_source"], "this tab")
        self.app.processEvents()
        w.close()
        self.app.processEvents()


class TestGuardrailG2(unittest.TestCase):
    """No tab file may touch SSH; RecorderController and the remote workers own it."""

    def test_no_tab_imports_ssh(self):
        root = Path(__file__).resolve().parents[1] / "src" / "sensepi" / "gui" / "tabs"
        offenders = []
        for py in root.glob("tab_*.py"):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if any(tok in stripped for tok in ("SSHClient", "paramiko", "PiRecorder")):
                    offenders.append(f"{py.name}:{i}")
        self.assertEqual(offenders, [], f"SSH reached a tab file: {offenders}")


class TestRendererKeepsState(unittest.TestCase):
    """SN-1: a knob change must not zero the reverb state."""

    def test_set_config_preserves_comb_state(self):
        try:
            from sensepi.sonification.chorus.engine import ChorusEngine
            from sensepi.sonification.chorus.types import ChorusConfig
            eng = ChorusEngine(ChorusConfig())
        except Exception as exc:                        # catalog missing etc.
            raise unittest.SkipTest(f"engine unavailable: {exc}")
        if not eng.available:
            raise unittest.SkipTest("engine data unavailable")
        r = eng.renderer
        state_before = r._combs[0][2]
        state_before[0, 0] = 0.123
        eng.set_option("master", 0.4)
        self.assertIs(r._combs[0][2], state_before)
        self.assertEqual(float(r._combs[0][2][0, 0]), 0.123)
        # brightness changes coefficients, still not state
        eng.set_option("brightness", 1.2)
        self.assertIs(r._combs[0][2], state_before)


if __name__ == "__main__":
    unittest.main()
