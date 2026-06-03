"""Tests for the recorded-session loader (M3 disk path)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sensepi.dataio import modal_session_loader as msl


def _write_jsonl(path: Path, fs=200.0, duration=20.0, n_sensors=3):
    n = int(fs * duration)
    with path.open("w", encoding="utf-8") as fh:
        # Interleave sensors, each with its own ax frequency to tell them apart.
        for i in range(n):
            t = i / fs
            for sid in range(1, n_sensors + 1):
                ax = float(np.sin(2 * np.pi * (sid) * t))
                fh.write(json.dumps({
                    "timestamp_ns": int(t * 1e9),
                    "t_s": t,
                    "sensor_id": sid,
                    "ax": ax, "ay": 0.0, "az": 9.81,
                    "gx": 0.0, "gy": 0.0, "gz": 0.0,
                }) + "\n")


class TestLoadSession(unittest.TestCase):
    def test_loads_and_aligns(self):
        with tempfile.TemporaryDirectory() as d:
            session = Path(d) / "shake_20260101_120000"
            session.mkdir()
            _write_jsonl(session / "log.jsonl")
            out = msl.load_session(session)
            self.assertTrue(out.success, out.message)
            self.assertEqual(out.sensor_ids, [1, 2, 3])
            self.assertEqual(out.data.shape[0], 3)
            self.assertAlmostEqual(out.fs, 200.0, delta=1.0)
            self.assertGreater(out.data.shape[1], 100)

    def test_last_seconds_trim(self):
        with tempfile.TemporaryDirectory() as d:
            session = Path(d) / "sess"
            session.mkdir()
            _write_jsonl(session / "log.jsonl", duration=20.0)
            out = msl.load_session(session, last_seconds=5.0)
            self.assertTrue(out.success, out.message)
            self.assertAlmostEqual(out.duration_s, 5.0, delta=0.5)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            out = msl.load_session(Path(d))
            self.assertFalse(out.success)

    def test_feeds_identify_modes(self):
        # End-to-end: loader output should be directly consumable by M2.
        from sensepi.analysis import modal
        with tempfile.TemporaryDirectory() as d:
            session = Path(d) / "sess"
            session.mkdir()
            _write_jsonl(session / "log.jsonl", duration=30.0)
            out = msl.load_session(session)
            self.assertTrue(out.success, out.message)
            result = modal.identify_modes(out.data, out.fs, f_min=0.5, f_max=20.0, n_modes=3)
            self.assertTrue(result.success, result.message)


def _write_header_csv(path: Path, fs=42.5, duration=20.0, n_sensors=3):
    """Mimic the real Pi CSV: header row, columns timestamp_ns,t_s,sensor_id,ax,ay,gz."""
    import numpy as np
    n = int(fs * duration)
    for sid in range(1, n_sensors + 1):
        f = path / f"mpu_S{sid}_2025-12-18_02-35-53.csv"
        with f.open("w", encoding="utf-8") as fh:
            fh.write("timestamp_ns,t_s,sensor_id,ax,ay,gz\n")
            for i in range(n):
                t = i / fs
                ax = float(np.sin(2 * np.pi * 2.0 * t))
                fh.write(f"{int(t*1e9)},{t:.9f},{sid},{ax},0.0,0.0\n")


class TestHeaderCsv(unittest.TestCase):
    """The real Pi recording is header CSV with only ax/ay/gz (no az/gx/gy)."""

    def test_loads_header_csv(self):
        with tempfile.TemporaryDirectory() as d:
            session = Path(d) / "pi-5"
            mpu = session / "mpu"
            mpu.mkdir(parents=True)
            _write_header_csv(mpu)
            out = msl.load_session(session, axis="ax")
            self.assertTrue(out.success, out.message)
            self.assertEqual(out.sensor_ids, [1, 2, 3])
            self.assertEqual(out.data.shape[0], 3)
            self.assertAlmostEqual(out.fs, 42.5, delta=2.0)

    def test_header_csv_feeds_identify(self):
        from sensepi.analysis import modal
        with tempfile.TemporaryDirectory() as d:
            session = Path(d) / "pi-5"
            mpu = session / "mpu"
            mpu.mkdir(parents=True)
            _write_header_csv(mpu, duration=30.0)
            out = msl.load_session(session, axis="ax")
            result = modal.identify_modes(out.data, out.fs, f_min=0.5, f_max=15.0, n_modes=1)
            self.assertTrue(result.success, result.message)
            self.assertAlmostEqual(result.frequencies_hz[0], 2.0, delta=0.3)

    def test_missing_axis_column(self):
        # If the requested axis isn't a column, that file yields no samples.
        with tempfile.TemporaryDirectory() as d:
            session = Path(d) / "pi-5"
            mpu = session / "mpu"
            mpu.mkdir(parents=True)
            _write_header_csv(mpu)
            out = msl.load_session(session, axis="az")  # az not present
            self.assertFalse(out.success)


if __name__ == "__main__":
    unittest.main()
