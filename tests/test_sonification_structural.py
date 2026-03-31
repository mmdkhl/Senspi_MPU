import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sensepi.sonification.structural import (  # noqa: E402
    build_harmonic_audio,
    build_melody_audio,
    load_structural_csv,
    write_wav,
)


CSV_TEXT = """Joint;Time;U1;R1
-;s;mm;deg
28;0,0;0,0;0,0
28;0,1;0,5;0,1
28;0,2;1,0;0,3
"""


class SonificationStructuralTests(unittest.TestCase):
    def test_load_structural_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "sample.csv"
            path.write_text(CSV_TEXT, encoding="utf-8")

            ds = load_structural_csv(path)
            self.assertEqual(ds.joints(), [28])
            np.testing.assert_allclose(ds.get_time_series(28), np.array([0.0, 0.1, 0.2]))
            np.testing.assert_allclose(ds.get_measurement(28, "U1"), np.array([0.0, 0.5, 1.0]))

    def test_build_and_write_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "sample.csv"
            out_melody = pathlib.Path(tmpdir) / "melody.wav"
            out_harmonic = pathlib.Path(tmpdir) / "harmonic.wav"
            path.write_text(CSV_TEXT, encoding="utf-8")

            ds = load_structural_csv(path)
            melody = build_melody_audio(ds, joint=28, measurement="U1", note_duration=0.01)
            harmonic = build_harmonic_audio(ds, joint=28, base_freq=220, num_harmonics=5, sample_rate=2000)

            self.assertGreater(melody.size, 0)
            self.assertGreater(harmonic.size, 0)

            write_wav(out_melody, melody, sample_rate=44100)
            write_wav(out_harmonic, harmonic, sample_rate=2000)
            self.assertTrue(out_melody.exists())
            self.assertTrue(out_harmonic.exists())


if __name__ == "__main__":
    unittest.main()

