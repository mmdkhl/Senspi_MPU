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


if __name__ == "__main__":
    unittest.main()
