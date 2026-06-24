"""Smart-Recording probe accuracy fix (audit 2026-06-24).

The probe used to read the COMBINED all-sensors rate (~N x per-sensor + startup burst),
mislabelling ~41 Hz as ~160 Hz and triggering spurious decimation that wrote *fewer*
samples than were available. These cover the two pure helpers behind the fix:
per-sensor median-dt rate, and the floor-decimation that never drops below the request.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from sensepi.gui.recorder_controller import _per_sensor_hz_from_times, _decimate_for
    _HAS_GUI = True
except Exception:  # pragma: no cover - env-dependent
    _HAS_GUI = False


def _times(hz, n, start=0.0):
    return [start + i / hz for i in range(n)]


@unittest.skipUnless(_HAS_GUI, "PySide6 / GUI module import unavailable")
class TestPerSensorRate(unittest.TestCase):
    def test_per_sensor_not_combined(self):
        # 3 sensors each at a true 41 Hz. The estimate must be ~41 (per-sensor), NOT
        # ~123 (combined) and certainly not the old ~160.
        probe = {sid: _times(41.0, 205) for sid in (1, 2, 3)}
        rep, rates = _per_sensor_hz_from_times(probe)
        self.assertAlmostEqual(rep, 41.0, delta=1.0)
        self.assertEqual(set(rates), {1, 2, 3})
        for hz in rates.values():
            self.assertAlmostEqual(hz, 41.0, delta=1.0)

    def test_median_is_robust_to_startup_burst(self):
        # A short fast burst then steady 41 Hz: median dt ignores the burst.
        burst = _times(160.0, 10)                          # ~6 ms spacing
        steady = _times(41.0, 200, start=burst[-1] + 0.024)
        rep, _ = _per_sensor_hz_from_times({1: burst + steady})
        self.assertAlmostEqual(rep, 41.0, delta=2.0)

    def test_sparse_sensor_skipped(self):
        rep, rates = _per_sensor_hz_from_times({1: _times(41.0, 100), 2: [0.0, 0.1]})
        self.assertIn(1, rates)
        self.assertNotIn(2, rates)   # < 5 samples -> skipped
        self.assertAlmostEqual(rep, 41.0, delta=1.0)

    def test_empty_returns_zero(self):
        self.assertEqual(_per_sensor_hz_from_times({}), (0.0, {}))


@unittest.skipUnless(_HAS_GUI, "PySide6 / GUI module import unavailable")
class TestDecimateFloor(unittest.TestCase):
    def test_no_decimation_when_under_delivering(self):
        # The real case: device delivers 41 Hz, user requested 100 / 50 -> record ALL.
        self.assertEqual(_decimate_for(41.0, 100.0), 1)
        self.assertEqual(_decimate_for(41.0, 50.0), 1)

    def test_floor_never_below_request(self):
        # Only decimate at >=2x, and written = measured/decimate stays >= requested.
        self.assertEqual(_decimate_for(150.0, 100.0), 1)   # 1.5x -> would drop below -> no
        self.assertEqual(_decimate_for(205.0, 100.0), 2)   # -> 102.5 Hz written (>=100)
        self.assertEqual(_decimate_for(300.0, 100.0), 3)
        self.assertEqual(_decimate_for(0.0, 100.0), 1)
        self.assertEqual(_decimate_for(50.0, 0.0), 1)


if __name__ == "__main__":
    unittest.main()
