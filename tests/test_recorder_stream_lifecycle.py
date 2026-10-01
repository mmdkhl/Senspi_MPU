"""Stream start/stop lifecycle of RecorderController, against a fake Pi.

* A new stream must not inherit the previous stream's modal capture. The Pi
  restarts ``t_s`` at 0 on every run, so leftover samples share timestamps with
  the new ones, and "the last N seconds" -- anchored at the newest timestamp in
  the buffer -- kept returning the previous run for as long as it had lasted.
* ``stop_live_stream(wait=True)`` must not block the GUI thread until its
  timeout. The worker's ``finished -> thread.quit`` hop was queued to the GUI
  thread, which was the thread blocked in ``wait()``.
"""

import os
import threading
import time
import unittest
from pathlib import PurePosixPath

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop
from PySide6.QtWidgets import QApplication

import sensepi.gui.recorder_controller as rcm
from sensepi.config.app_config import HostConfig
from sensepi.config.sampling import SamplingConfig
from sensepi.gui.config.acquisition_state import GuiAcquisitionConfig, SensorSelectionConfig
from sensepi.sensors.mpu6050 import MpuSample


class _FakeStream:
    """Endless JSON lines at ~400 lines/s until closed, like the Pi's stdout."""

    def __init__(self):
        self._closed = threading.Event()
        self._n = 0

    def close(self):
        self._closed.set()

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed.is_set():
            raise StopIteration
        time.sleep(0.0025)
        self._n += 1
        sid = self._n % 4 + 1
        return ('{"timestamp_ns":%d,"t_s":%.4f,"sensor_id":%d,"ax":0.1,"ay":0.0,"gz":0.0}'
                % (self._n * 2_500_000, self._n / 400.0, sid))


class _FakePiRecorder:
    def __init__(self, *_args, **_kwargs):
        pass

    def connect(self):
        pass

    def close(self):
        pass

    def stop_remote_logger(self):
        pass

    def stream_mpu6050(self, **_kwargs):
        return _FakeStream()


_HOST = HostConfig(
    name="fake", host="127.0.0.1", user="u", port=22,
    base_path=PurePosixPath("/tmp"), data_dir=PurePosixPath("/tmp"),
    pi_config_path=PurePosixPath("/tmp/pi_config.yaml"), password="",
)
_CFG = GuiAcquisitionConfig(
    sampling=SamplingConfig(device_rate_hz=100.0, mode_key="raw"),
    stream_rate_hz=100.0,
    record_only=False,
    sensor_selection=SensorSelectionConfig(active_sensors=[1, 2, 3, 4], active_channels=[]),
)


class TestStreamLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def setUp(self):
        self._orig_recorder = rcm.PiRecorder
        rcm.PiRecorder = _FakePiRecorder
        self.rc = rcm.RecorderController()

    def tearDown(self):
        try:
            self.rc.stop_live_stream(wait=True, wait_timeout_ms=5000)
            self._pump(0.2)
        finally:
            rcm.PiRecorder = self._orig_recorder

    def _pump(self, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.app.processEvents(QEventLoop.AllEvents, 20)

    def _start(self):
        self.rc.start_live_stream(recording_enabled=False, gui_config=_CFG, host_cfg=_HOST)

    def test_new_stream_starts_with_an_empty_modal_capture(self):
        # A previous run: 60 s at 100 Hz, value 1.0, timestamps from t_s = 0.
        old = [MpuSample(timestamp_ns=i * 10_000_000, ax=1.0, ay=0.0, az=0.0, gx=0.0,
                         gy=0.0, gz=0.0, sensor_id=sid, t_s=i / 100.0)
               for i in range(6000) for sid in (1, 2)]
        self.rc._modal_buffer.add_batch(old)
        self.assertGreater(self.rc.modal_available_seconds(), 50.0)

        self._start()

        self.assertEqual(self.rc._modal_buffer.snapshot_series("ax"), {})

    def test_stop_with_wait_returns_promptly_and_the_thread_is_finished(self):
        self._start()
        self._pump(0.5)
        thread = self.rc._ingest_thread
        self.assertIsNotNone(thread)

        t0 = time.perf_counter()
        self.rc.stop_live_stream(wait=True, wait_timeout_ms=5000)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 1.0)
        self.assertTrue(thread.isFinished())

    def test_a_second_stop_does_not_report_an_unexpected_end(self):
        # Stop, then another stop before the queued "ingest finished" callback
        # has run (closing the window right after Stop does exactly this). The
        # second stop reset the stop-requested flag, so the callback reported
        # "Live stream stopped unexpectedly".
        errors = []
        self.rc.error_reported.connect(errors.append)
        self._start()
        self._pump(0.3)

        self.rc.stop_live_stream(wait=True, wait_timeout_ms=3000)
        self.rc.stop_live_stream(wait=True, wait_timeout_ms=3000)
        self._pump(0.3)

        self.assertEqual(errors, [])

    def test_repeated_start_stop_finishes_every_ingest_thread(self):
        # Stop dropped the last Python reference to the ingest worker, which has
        # no Qt parent and so is owned by Python: the GUI thread destroyed it.
        # Before the thread ran, started->start then had nothing to call, the
        # thread idled forever and Stop sat out its timeout; while it ran, the
        # object was deleted under it (an intermittent access violation).
        # Half the cycles stop straight after starting, before any event is
        # processed.
        for i in range(30):
            self._start()
            if i % 2:
                self._pump(0.05)
            thread = self.rc._ingest_thread
            t0 = time.perf_counter()
            self.rc.stop_live_stream(wait=True, wait_timeout_ms=3000)
            self.assertLess(time.perf_counter() - t0, 1.0, f"cycle {i}")
            self.assertTrue(thread.isFinished(), f"cycle {i}")
            self._pump(0.02)
        self.assertFalse(self.rc.is_streaming())


if __name__ == "__main__":
    unittest.main()
