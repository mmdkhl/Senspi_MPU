"""Streaming CSV writer for PC-side Smart Recording (PHASE 12 / SF-5).

Pure I/O. The key check is the FORMAT CONTRACT (live_signals_builder_plan §14): a
Smart-Recording file must parse identically through BOTH the Model Updating loader
(reads columns by NAME) and log_loader (reads columns by POSITION).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sensepi.dataio import modal_session_loader as msl
from sensepi.dataio.csv_writer import (
    SENSOR_CSV_HEADER,
    StreamingCsvWriter,
    write_recording_meta,
)
from sensepi.dataio.log_loader import load_csv


class TestStreamingCsvWriter(unittest.TestCase):
    def _write(self, path: Path, sid: int, n: int = 600, fs: float = 50.0) -> int:
        w = StreamingCsvWriter(path).open()
        for i in range(n):
            t = i / fs
            w.write_sample(
                int(t * 1e9), round(t, 6), sid,
                float(np.sin(2 * np.pi * 2.3 * t)),
                float(np.sin(2 * np.pi * 6.5 * t)), 0.0)
        w.close()
        return w.rows_written

    def test_header_and_row_count(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mpu_S1_2026-06-22_00-00-00.csv"
            rows = self._write(p, 1, n=120)
            self.assertEqual(rows, 120)
            with p.open() as fh:
                self.assertEqual(fh.readline().strip(), ",".join(SENSOR_CSV_HEADER))

    def test_round_trip_through_both_loaders(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mpu_S1_2026-06-22_00-00-00.csv"
            self._write(p, 1, n=600, fs=50.0)

            # 1) Model Updating "Load & Identify" path — reads columns by NAME.
            session = msl.load_session(p, axis="ax")
            self.assertTrue(session.success, session.message)
            self.assertIn(1, session.sensor_ids)

            # 2) log_loader / plotter / Sonification inspect — reads by POSITION.
            arr = load_csv(p)
            self.assertEqual(arr.shape[1], len(SENSOR_CSV_HEADER))  # 6 columns, order intact
            self.assertEqual(arr.shape[0], 600)

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mpu_S2_2026-06-22_00-00-00.csv"
            with StreamingCsvWriter(p) as w:
                w.write_sample(0, 0.0, 2, 0.1, 0.2, 0.0)
            self.assertTrue(p.exists())

    def test_meta_sidecar_extra_keys(self):
        with tempfile.TemporaryDirectory() as d:
            mp = Path(d) / "mpu_S1_x.csv.meta.json"
            write_recording_meta(
                mp, pc_start_utc="2026-06-22T00:00:00Z",
                actual_hz=43.0, requested_hz=100.0, probe_duration_s=5.0)
            data = json.loads(mp.read_text())
            self.assertEqual(data["actual_hz"], 43.0)
            self.assertEqual(data["requested_hz"], 100.0)


if __name__ == "__main__":
    unittest.main()
