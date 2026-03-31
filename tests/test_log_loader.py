import pathlib
import sys
import tempfile
import unittest

import numpy as np

# Ensure src/ is on path for direct test execution
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sensepi.dataio.log_loader import chunk_array, load_csv  # noqa: E402


class LogLoaderTest(unittest.TestCase):
    def test_load_csv_without_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "no_header.csv"
            path.write_text("1,2,3\n4,5,6\n", encoding="utf-8")

            data = load_csv(path)

            np.testing.assert_array_equal(data, np.array([[1, 2, 3], [4, 5, 6]]))

    def test_load_csv_with_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "with_header.csv"
            path.write_text("time,x,y\n1,2,3\n4,5,6\n", encoding="utf-8")

            data = load_csv(path)

            np.testing.assert_array_equal(data, np.array([[1, 2, 3], [4, 5, 6]]))

    def test_chunk_array_requires_positive_size(self):
        array = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            list(chunk_array(array, 0))


if __name__ == "__main__":
    unittest.main()
