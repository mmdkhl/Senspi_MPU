"""One resolved output root, and no dependence on the working directory.

The bug these guard against was not a wrong path — it was a path that depended on
where the process happened to be standing, which produced files in the right
place most of the time and somewhere else the rest of the time.
"""
import os
import tempfile
import unittest
from pathlib import Path

from opensees_model_updating import paths as ops_paths
from sensepi.config.app_config import AppPaths


class TestOpenSeesPaths(unittest.TestCase):
    """The package keeps working standalone; the host can redirect it."""

    def test_the_standalone_default_is_unchanged(self):
        # The team runs this package from its own folder. If this ever stops
        # being a bare relative "output", their workflow has been changed.
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path.cwd()
            try:
                os.chdir(tmp)
                self.assertEqual(ops_paths.output_dir(create=False), Path("output"))
            finally:
                os.chdir(cwd)

    def test_an_explicit_base_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = ops_paths.output_dir(Path(tmp) / "here", create=False)
            self.assertEqual(got, Path(tmp) / "here")

    def test_the_environment_variable_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[ops_paths.ENV_VAR] = tmp + "/env"
            try:
                self.assertEqual(ops_paths.output_dir(create=False), Path(tmp) / "env")
            finally:
                del os.environ[ops_paths.ENV_VAR]

    def test_explicit_base_beats_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[ops_paths.ENV_VAR] = tmp + "/env"
            try:
                got = ops_paths.output_dir(Path(tmp) / "arg", create=False)
                self.assertEqual(got, Path(tmp) / "arg")
            finally:
                del os.environ[ops_paths.ENV_VAR]

    def test_recorder_strings_are_absolute(self):
        """These cross into the OpenSees C++ layer, where relative is silent."""
        with tempfile.TemporaryDirectory() as tmp:
            got = ops_paths.output_str("time_roof_accel_X.out", tmp)
            self.assertTrue(Path(got).is_absolute())

    def test_the_directory_is_created_on_demand(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "made" / "deep"
            self.assertFalse(target.exists())
            ops_paths.output_dir(target)
            self.assertTrue(target.is_dir())


class TestAppPaths(unittest.TestCase):
    def test_everything_written_lives_under_one_root(self):
        p = AppPaths()
        for child in (p.sensor_recordings, p.model_output, p.twin_output,
                      p.sonification_output):
            self.assertEqual(child.parent, p.output_root,
                             f"{child} is not under the output root")

    def test_logs_stay_separate(self):
        """Diagnostics are not results; clearing one must not clear the other."""
        p = AppPaths()
        self.assertNotEqual(p.logs, p.output_root)
        self.assertFalse(str(p.logs).startswith(str(p.output_root)))

    def test_recordings_have_their_own_folder(self):
        p = AppPaths()
        self.assertEqual(p.sensor_recordings.name, "sensor_recordings")

    def test_the_legacy_alias_points_at_the_new_place(self):
        """Any caller missed during the move must still land correctly."""
        p = AppPaths()
        self.assertEqual(p.raw_data, p.sensor_recordings)

    def test_recordings_can_be_moved_on_their_own(self):
        os.environ["SENSEPI_DATA_ROOT"] = "/tmp/sensepi_rec_test"
        try:
            p = AppPaths()
            self.assertEqual(p.sensor_recordings, Path("/tmp/sensepi_rec_test"))
            self.assertEqual(p.model_output, p.output_root / "model")
        finally:
            del os.environ["SENSEPI_DATA_ROOT"]

    def test_output_root_is_overridable(self):
        os.environ["SENSEPI_OUTPUT_ROOT"] = "/tmp/sensepi_out_test"
        try:
            self.assertEqual(AppPaths().output_root, Path("/tmp/sensepi_out_test"))
        finally:
            del os.environ["SENSEPI_OUTPUT_ROOT"]

    def test_every_path_is_absolute(self):
        p = AppPaths()
        for path in (p.sensor_recordings, p.logs, p.output_root,
                     p.model_output, p.twin_output, p.sonification_output):
            self.assertTrue(path.is_absolute(), f"{path} is not absolute")


class TestOneRecordingLocation(unittest.TestCase):
    """Writers and readers must not be able to drift apart again."""

    def test_the_writer_and_the_readers_agree(self):
        from sensepi.dataio import file_paths, modal_session_loader
        p = AppPaths()
        # The loader's default root and the recorder's destination are the same
        # folder, so a recording can never land somewhere nothing looks.
        self.assertEqual(modal_session_loader.list_sessions.__module__,
                         "sensepi.dataio.modal_session_loader")
        self.assertTrue(str(file_paths.__file__).endswith("file_paths.py"))
        self.assertEqual(p.sensor_recordings, p.raw_data)

    def test_nothing_still_points_at_the_old_data_tree(self):
        root = Path(__file__).resolve().parents[1] / "src" / "sensepi"
        offenders = []
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                if 'REPO_ROOT / "data"' in line or '"data" / "raw"' in line:
                    offenders.append(f"{py.relative_to(root)}:{i}")
        self.assertEqual(offenders, [], f"old data/ paths remain at {offenders}")


class TestNoWorkingDirectoryDependence(unittest.TestCase):
    def test_the_application_never_changes_the_process_directory(self):
        """os.chdir is process-global: it moves every thread, not the caller.

        The app runs SSH ingest, the chorus worker and a model worker at the same
        time. If this ever comes back, a relative path in any of them can resolve
        somewhere else for the duration of a calibration.
        """
        root = Path(__file__).resolve().parents[1] / "src" / "sensepi"
        offenders = []
        for py in root.rglob("*.py"):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "os.chdir" in stripped:
                    offenders.append(f"{py.relative_to(root)}:{i}")
        self.assertEqual(offenders, [], f"os.chdir reintroduced at {offenders}")

    def test_the_default_model_workspace_is_not_the_repo_root(self):
        """It was, and that is what grew a stray output/ at the top of the repo."""
        from sensepi.gui.tabs.tab_model_updating import _default_workspace_dir
        p = AppPaths()
        got = _default_workspace_dir()
        self.assertNotEqual(got.resolve(), p.repo_root.resolve())
        self.assertTrue(str(got).startswith(str(p.output_root)))


if __name__ == "__main__":
    unittest.main()
