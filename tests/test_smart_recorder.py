"""Tests for the PC-side Smart Recording writer (Front C / SF-2/3/5).

Pure + headless (no Qt, no SSH): feed fake MPU sample batches through the writer
thread and confirm the files round-trip through both loaders, decimation works, and
the meta sidecar carries the true rate + PC clock.
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from sensepi.dataio import modal_session_loader as msl
from sensepi.dataio.log_loader import load_csv
from sensepi.dataio.smart_recorder import SmartRecorder
from sensepi.sensors.mpu6050 import MpuSample


def _batch(sids, i, fs):
    t = i / fs
    return [
        MpuSample(
            timestamp_ns=int(t * 1e9),
            ax=float(np.sin(2 * np.pi * 2.3 * t)),
            ay=float(np.sin(2 * np.pi * 6.5 * t)),
            az=0.0, gx=0.0, gy=0.0, gz=0.0,
            sensor_id=sid, t_s=round(t, 6),
        )
        for sid in sids
    ]


class TestSmartRecorder(unittest.TestCase):
    def _record(self, decimate=1, n=600, fs=50.0, sids=(1, 2, 3)):
        d = Path(tempfile.mkdtemp()) / "mpu"
        rec = SmartRecorder(
            out_dir=d, session_name=None, sensor_ids=sids,
            start_dt=datetime(2026, 6, 22, 0, 0, 0),
            requested_hz=fs, actual_hz=fs, decimate=decimate)
        rec.start()
        for i in range(n):
            rec.submit(_batch(sids, i, fs))
        written = rec.stop()
        return rec, written

    def test_files_round_trip_both_loaders(self):
        rec, written = self._record(decimate=1, n=600)
        for sid, path in rec.data_paths.items():
            self.assertTrue(path.exists())
            session = msl.load_session(path, axis="ax")   # by NAME
            self.assertTrue(session.success, session.message)
            arr = load_csv(path)                            # by POSITION
            self.assertEqual(arr.shape[1], 6)
        self.assertEqual(written[1], 600)

    def test_decimation_on_writer_only(self):
        _rec, written = self._record(decimate=2, n=600)
        self.assertEqual(written[1], 300)   # every 2nd sample kept

    def test_meta_carries_true_rate_and_pc_clock(self):
        rec, _written = self._record(decimate=1, n=100)
        data_path = rec.data_paths[1]
        meta_path = Path(str(data_path) + ".meta.json")
        self.assertTrue(meta_path.exists())
        data = json.loads(meta_path.read_text())
        self.assertIn("actual_hz", data)
        self.assertIn("requested_hz", data)
        self.assertIn("pc_start_utc", data)

    def test_audit_block_measures_true_incoming_rate(self):
        # Stream 600 samples at a known 50 Hz with decimate=2 -> incoming ~50 Hz,
        # written ~25 Hz. The audit must report the TRUE incoming rate from timestamps.
        rec, written = self._record(decimate=2, n=600, fs=50.0)
        meta = json.loads((Path(str(rec.data_paths[1]) + ".meta.json")).read_text())
        self.assertIn("audit", meta)
        a = meta["audit"]
        self.assertEqual(a["samples_received"], 600)      # pre-decimation
        self.assertEqual(a["samples_written"], 300)       # post-decimation
        self.assertAlmostEqual(a["rate_incoming_hz"], 50.0, delta=0.5)
        self.assertAlmostEqual(a["rate_written_hz"], 25.0, delta=0.5)
        self.assertAlmostEqual(a["delivered_vs_requested"], 1.0, delta=0.05)
        # dt stats + histogram + gaps + per-channel stats are present.
        for key in ("mean", "median", "std", "min", "max", "p05", "p95", "p99"):
            self.assertIn(key, a["dt_ms"])
        self.assertAlmostEqual(a["dt_ms"]["median"], 20.0, delta=1.0)  # 50 Hz -> 20 ms
        self.assertIn("dt_histogram_ms", a)
        self.assertIn("gaps", a)
        self.assertIn("clock", a)
        self.assertEqual(set(a["channels"]), {"ax", "ay", "az", "gx", "gy", "gz"})
        self.assertEqual(a["channels"]["ax"]["n"], 600)   # all incoming counted

    def test_under_sampling_is_visible(self):
        # Requested 100 Hz but only 40 Hz delivered -> delivered_vs_requested ~0.4.
        d = Path(tempfile.mkdtemp()) / "mpu"
        rec = SmartRecorder(out_dir=d, session_name=None, sensor_ids=(1,),
                            start_dt=datetime(2026, 6, 22, 0, 0, 0),
                            requested_hz=100.0, actual_hz=40.0, decimate=1)
        rec.start()
        for i in range(400):
            rec.submit(_batch((1,), i, 40.0))   # true 40 Hz
        rec.stop()
        a = json.loads((Path(str(rec.data_paths[1]) + ".meta.json")).read_text())["audit"]
        self.assertAlmostEqual(a["rate_incoming_hz"], 40.0, delta=0.5)
        self.assertAlmostEqual(a["delivered_vs_requested"], 0.40, delta=0.03)

    def test_raw_audit_folder_has_all_channels_undecimated(self):
        rec, _written = self._record(decimate=2, n=600, fs=50.0)
        audit_dir = rec.audit_dir
        self.assertIsNotNone(audit_dir)
        self.assertTrue(audit_dir.exists())
        self.assertTrue((audit_dir / "audit_summary.json").exists())
        raw = sorted(audit_dir.glob("raw_S1_*.csv"))
        self.assertEqual(len(raw), 1)
        lines = raw[0].read_text().strip().splitlines()
        self.assertEqual(lines[0], "timestamp_ns,t_s,sensor_id,ax,ay,az,gx,gy,gz")  # superset
        self.assertEqual(len(lines) - 1, 600)             # un-decimated (all incoming)
        # The main CSV stays the locked 6-column schema and decimated.
        main_lines = rec.data_paths[1].read_text().strip().splitlines()
        self.assertEqual(main_lines[0], "timestamp_ns,t_s,sensor_id,ax,ay,gz")
        self.assertEqual(len(main_lines) - 1, 300)

    def test_undelivered_channel_is_null_not_naninf(self):
        # The Pi streams ax/ay/gz only; az/gx/gy arrive non-finite. The meta must be
        # VALID JSON (null stats + n_finite=0), never Infinity/NaN.
        d = Path(tempfile.mkdtemp()) / "mpu"
        rec = SmartRecorder(out_dir=d, session_name=None, sensor_ids=(1,),
                            start_dt=datetime(2026, 6, 22, 0, 0, 0),
                            requested_hz=50.0, actual_hz=50.0, decimate=1)
        rec.start()
        for i in range(50):
            t = i / 50.0
            rec.submit([MpuSample(timestamp_ns=int(t * 1e9), ax=0.1, ay=0.2,
                                  az=float("nan"), gx=float("nan"), gy=float("nan"),
                                  gz=0.3, sensor_id=1, t_s=round(t, 6))])
        rec.stop()
        text = (Path(str(rec.data_paths[1]) + ".meta.json")).read_text()
        self.assertNotIn("Infinity", text)
        self.assertNotIn("NaN", text)
        ch = json.loads(text)["audit"]["channels"]
        self.assertEqual(ch["az"]["n_finite"], 0)
        self.assertIsNone(ch["az"]["min"])
        self.assertEqual(ch["ax"]["n_finite"], 50)        # delivered channel intact
        self.assertAlmostEqual(ch["ax"]["mean"], 0.1, places=4)

    def test_audit_off_keeps_minimal_meta_and_no_raw(self):
        d = Path(tempfile.mkdtemp()) / "mpu"
        rec = SmartRecorder(out_dir=d, session_name=None, sensor_ids=(1,),
                            start_dt=datetime(2026, 6, 22, 0, 0, 0),
                            requested_hz=50.0, actual_hz=50.0, decimate=1,
                            audit=False, save_raw=False)
        rec.start()
        for i in range(50):
            rec.submit(_batch((1,), i, 50.0))
        rec.stop()
        meta = json.loads((Path(str(rec.data_paths[1]) + ".meta.json")).read_text())
        self.assertNotIn("audit", meta)                   # back to the minimal sidecar
        self.assertIsNone(rec.audit_dir)


if __name__ == "__main__":
    unittest.main()
