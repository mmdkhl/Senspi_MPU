"""Calibration must survive closing the app.

Two separate things were in-memory only and lost on exit:

* the per-sensor/channel **offsets** taken in Settings, which the Spectrum tab
  subtracts from live data; and
* the Model Updating tab's **calibrated parameters**, which cost an OpenSees run
  to produce.

Both are now written atomically (tmp + replace, so a crash mid-write cannot
leave a half-file) and read back at startup. The model state additionally keeps
its ``input_signature``, so a restored file that no longer matches the current
model inputs is simply not reused.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


class TestCalibrationOffsetsPersistence(unittest.TestCase):
    """MainWindow side: the Settings offsets."""

    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        from sensepi.gui.main_window import MainWindow

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "calibration_offsets.json"
        # Redirect off the real config dir BEFORE any window is built: the
        # constructor restores, and closeEvent saves.
        #
        # Take the descriptor out of __dict__, not off the class: class access
        # unwraps the staticmethod to a plain function, and restoring THAT would
        # rebind it as an instance method, so every later MainWindow in the run
        # would raise on the implicit self.
        name = "_calibration_offsets_path"
        self._orig = MainWindow.__dict__[name]
        setattr(MainWindow, name, staticmethod(lambda: self.path))
        self.addCleanup(setattr, MainWindow, name, self._orig)
        self.MainWindow = MainWindow

    def _offsets(self):
        from sensepi.gui.config.acquisition_state import CalibrationOffsets

        return CalibrationOffsets(
            per_sensor_channel_offset={(1, "ax"): 0.125, (3, "gz"): -2.5},
            description="unit test")

    def _window(self):
        win = self.MainWindow()
        self.addCleanup(win.close)
        return win

    def test_offsets_are_written_when_they_change(self):
        win = self._window()

        win._on_calibration_changed(self._offsets())

        self.assertTrue(self.path.exists(), "a new calibration was not persisted")
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["per_sensor_channel_offset"]["1:ax"], 0.125)
        self.assertEqual(saved["per_sensor_channel_offset"]["3:gz"], -2.5)

    def test_offsets_come_back_in_the_next_session(self):
        first = self._window()
        first._on_calibration_changed(self._offsets())
        first.close()

        second = self._window()

        restored = second._current_calibration_offsets
        self.assertIsNotNone(restored, "offsets were not restored at startup")
        self.assertEqual(restored.offset_for(1, "ax"), 0.125)
        self.assertEqual(restored.offset_for(3, "gz"), -2.5)

    def test_the_restored_offsets_reach_the_spectrum_tab(self):
        first = self._window()
        first._on_calibration_changed(self._offsets())
        first.close()

        second = self._window()

        # The tab is the consumer; restoring without telling it would silently
        # leave live data uncorrected.
        self.assertIsNotNone(second.fft_tab._calibration_offsets,
                             "the Spectrum tab was never told about the restored offsets")
        self.assertEqual(second.fft_tab._calibration_offsets.offset_for(1, "ax"), 0.125)

    def test_an_unreadable_file_is_ignored_rather_than_fatal(self):
        self.path.write_text("{ this is not json", encoding="utf-8")

        win = self._window()        # must not raise

        self.assertIsNone(win._current_calibration_offsets)

    def test_clearing_the_calibration_removes_the_file(self):
        from sensepi.gui.config.acquisition_state import CalibrationOffsets

        win = self._window()
        win._on_calibration_changed(self._offsets())
        self.assertTrue(self.path.exists())

        win._on_calibration_changed(CalibrationOffsets())

        self.assertFalse(self.path.exists(),
                         "an empty calibration left a stale file behind")

    def test_no_temporary_file_is_left_behind(self):
        win = self._window()

        win._on_calibration_changed(self._offsets())

        leftovers = list(Path(self._tmp.name).glob("*.tmp"))
        self.assertEqual(leftovers, [], f"atomic write left {leftovers}")


class TestModelCalibrationStatePersistence(unittest.TestCase):
    """Model Updating side: the calibrated parameters."""

    def setUp(self):
        from sensepi.gui.tabs import tab_model_updating as mu

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self._orig = mu._default_workspace_dir
        mu._default_workspace_dir = lambda: self.dir
        self.addCleanup(setattr, mu, "_default_workspace_dir", self._orig)
        self.mu = mu

    def _state(self):
        return self.mu._CalibrationState(
            available=True,
            input_signature="sig-abc",
            calibrated_params={"E": 2.1e10, "floor_masses": [1.0, 1.0, 1.0]},
            uncalibrated_response={"t_hist": np.array([0.0, 0.1, 0.2]),
                                   "u_hist": np.array([0.0, 1.5, -1.5]),
                                   "label": "uncalibrated"},
            prior_freqs=[1.2, 4.0, 7.3],
        )

    def test_a_saved_state_round_trips(self):
        self.mu._save_calibration_state(self._state())

        back = self.mu._load_calibration_state()

        self.assertTrue(back.available)
        self.assertEqual(back.input_signature, "sig-abc")
        self.assertEqual(back.calibrated_params["E"], 2.1e10)
        self.assertEqual(back.prior_freqs, [1.2, 4.0, 7.3])

    def test_response_histories_come_back_as_arrays(self):
        # The live response canvas indexes these as numpy arrays; JSON would
        # otherwise hand back plain lists.
        self.mu._save_calibration_state(self._state())

        back = self.mu._load_calibration_state()

        self.assertIsInstance(back.uncalibrated_response["t_hist"], np.ndarray)
        np.testing.assert_allclose(back.uncalibrated_response["u_hist"],
                                   [0.0, 1.5, -1.5])
        self.assertEqual(back.uncalibrated_response["label"], "uncalibrated")

    def test_numpy_scalars_and_arrays_in_params_are_serialisable(self):
        state = self._state()
        state.calibrated_params = {"E": np.float64(3.0), "k": np.array([1.0, 2.0])}

        self.mu._save_calibration_state(state)      # must not raise
        back = self.mu._load_calibration_state()

        self.assertEqual(back.calibrated_params["E"], 3.0)
        self.assertEqual(back.calibrated_params["k"], [1.0, 2.0])

    def test_an_unavailable_state_deletes_the_file(self):
        self.mu._save_calibration_state(self._state())
        path = self.dir / self.mu._CALIBRATION_STATE_FILE
        self.assertTrue(path.exists())

        self.mu._save_calibration_state(self.mu._CalibrationState())

        self.assertFalse(path.exists(),
                         "resetting the calibration left a stale file to be restored")

    def test_a_missing_file_loads_as_unavailable(self):
        back = self.mu._load_calibration_state()

        self.assertFalse(back.available)

    def test_a_corrupt_file_loads_as_unavailable(self):
        (self.dir / self.mu._CALIBRATION_STATE_FILE).write_text(
            "not json at all", encoding="utf-8")

        back = self.mu._load_calibration_state()

        self.assertFalse(back.available)

    def test_a_file_without_a_signature_is_refused(self):
        # Without a signature there is no way to tell whether the parameters
        # still match the current model inputs, so it must not be reused.
        (self.dir / self.mu._CALIBRATION_STATE_FILE).write_text(
            json.dumps({"available": True, "calibrated_params": {"E": 1.0}}),
            encoding="utf-8")

        back = self.mu._load_calibration_state()

        self.assertFalse(back.available)

    def test_no_temporary_file_is_left_behind(self):
        self.mu._save_calibration_state(self._state())

        self.assertEqual(list(self.dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
