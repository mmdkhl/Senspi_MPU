"""Model Updating must work on a fresh clone.

The default workspace (``output/model``) is not in the repository -- only
``output/.gitkeep`` is -- and nothing created it, so Run Analysis, Calibrate and
the Digital Twin all refused to start with "Output workspace folder does not
exist" until someone made the folder by hand. The default is now created on
first use; a folder the user typed that does not exist is still an error.
"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


class TestDefaultModelWorkspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_root = os.environ.get("SENSEPI_OUTPUT_ROOT")
        os.environ["SENSEPI_OUTPUT_ROOT"] = self._tmp.name

    def tearDown(self):
        if self._old_root is None:
            os.environ.pop("SENSEPI_OUTPUT_ROOT", None)
        else:
            os.environ["SENSEPI_OUTPUT_ROOT"] = self._old_root
        self._tmp.cleanup()

    def _tab(self):
        from sensepi.gui.tabs.tab_model_updating import ModelUpdatingTab
        tab = ModelUpdatingTab()
        self.addCleanup(tab.deleteLater)
        return tab

    def test_default_workspace_is_created_on_first_use(self):
        tab = self._tab()
        workspace = Path(self._tmp.name) / "model"
        self.assertFalse(workspace.exists())

        tab._collect_params()

        self.assertTrue(workspace.is_dir())

    def test_a_typed_folder_that_does_not_exist_is_still_rejected(self):
        tab = self._tab()
        missing = Path(self._tmp.name) / "typo" / "modle"
        tab._project_dir_edit.setText(str(missing))

        with self.assertRaises(ValueError) as cm:
            tab._collect_params()
        self.assertIn("does not exist", str(cm.exception))
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
