"""snapshot_modal_capture(last_seconds=N) copies only the window it needs.

The Sonification chorus snapshots the shared capture every 50 ms, several axes
per tick, asking for the last few seconds. Each call used to copy the whole
120 s buffer under its lock and then align all of it, which made one tick cost
~200 ms: the worker never caught up, starved the GUI thread and timed out on
Stop. The bounded copy must give exactly the same session as the full one.
"""

import math
import unittest

import numpy as np

from sensepi.dataio.modal_session_loader import align_per_sensor_series
from sensepi.gui.recorder_controller import ModalCaptureBuffer
from sensepi.sensors.mpu6050 import MpuSample


def _filled_buffer(seconds=120, fs=100.0):
    buf = ModalCaptureBuffer(window_seconds=120.0)
    batch = []
    for i in range(int(seconds * fs)):
        t = i / fs
        for sid in (1, 2, 3, 4):
            # sensors sample slightly apart, as the real multiplexed Pi does
            ts = t + sid * 0.0005
            batch.append(MpuSample(timestamp_ns=int(ts * 1e9),
                                   ax=math.sin(2 * math.pi * 2.8 * ts) * sid,
                                   ay=0.0, az=0.0, gx=0.0, gy=0.0, gz=0.0,
                                   sensor_id=sid, t_s=ts))
    buf.add_batch(batch)
    return buf


class TestBoundedSnapshot(unittest.TestCase):
    def test_bounded_series_copies_only_the_recent_window(self):
        buf = _filled_buffer()

        series = buf.snapshot_series("ax", last_seconds=5.0)

        for rows in series.values():
            span = rows[-1][0] - rows[0][0]
            self.assertLess(span, 10.0)
            self.assertGreaterEqual(span, 5.0)

    def test_bounded_snapshot_aligns_to_the_same_session_as_the_full_copy(self):
        buf = _filled_buffer()

        full = align_per_sensor_series(buf.snapshot_series("ax"), last_seconds=5.0)
        bounded = align_per_sensor_series(
            buf.snapshot_series("ax", last_seconds=5.0), last_seconds=5.0)

        self.assertEqual(bounded.sensor_ids, full.sensor_ids)
        self.assertAlmostEqual(bounded.fs, full.fs, places=6)
        self.assertEqual(bounded.data.shape, full.data.shape)
        np.testing.assert_allclose(bounded.data, full.data)


class TestControllerUsesTheBoundedCopy(unittest.TestCase):
    def test_snapshot_modal_capture_copies_only_its_window(self):
        from sensepi.gui.recorder_controller import RecorderController

        class _Recording(ModalCaptureBuffer):
            windows = []

            def snapshot_series(self, axis, last_seconds=None):
                self.windows.append(last_seconds)
                return super().snapshot_series(axis, last_seconds=last_seconds)

        rc = RecorderController()
        buf = _Recording(window_seconds=120.0)
        buf.add_batch(MpuSample(timestamp_ns=i * 10_000_000, ax=1.0, ay=0.0, az=0.0,
                                gx=0.0, gy=0.0, gz=0.0, sensor_id=1, t_s=i / 100.0)
                      for i in range(3000))
        rc._modal_buffer = buf

        session = rc.snapshot_modal_capture(axis="ax", last_seconds=5.0)

        self.assertTrue(session.success, session.message)
        self.assertEqual(buf.windows, [5.0])


if __name__ == "__main__":
    unittest.main()
