import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sensepi.opensees.models import DigitalTwinAnalysisParams  # noqa: E402
from sensepi.opensees.runner import _load_ground_motion  # noqa: E402


class DigitalTwinModelTests(unittest.TestCase):
    def test_params_validate_requires_matching_story_and_mass_lengths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = pathlib.Path(tmpdir) / "gm.txt"
            gm.write_text("0.0\n0.1\n", encoding="utf-8")

            params = DigitalTwinAnalysisParams(
                gm_file=gm,
                output_dir=pathlib.Path(tmpdir) / "out",
                story_heights=[0.24, 0.24],
                floor_masses=[0.3],
            )
            with self.assertRaises(ValueError):
                params.validate()

    def test_load_ground_motion_accepts_decimal_commas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = pathlib.Path(tmpdir) / "gm.txt"
            gm.write_text("0,0\n0,5\n1.25\n", encoding="utf-8")

            values = _load_ground_motion(gm)
            self.assertEqual(values, [0.0, 0.5, 1.25])


if __name__ == "__main__":
    unittest.main()
