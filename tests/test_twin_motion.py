"""Reconstructing a displacement-like signal from acceleration, for display.

The wireframe shows *relative* motion between floors, so that is what is pinned
here — not absolute amplitude, which is approximate and normalised away.
"""
import unittest

import numpy as np

from sensepi.digital_twin import motion

FS = 100.0
WIN = 12.0
T = np.arange(int(WIN * FS)) / FS


def _sine(f, amp=0.001):
    """(true displacement, the acceleration a sensor would see)."""
    u = amp * np.sin(2 * np.pi * f * T)
    return u, -((2 * np.pi * f) ** 2) * u


class TestRelativeMotion(unittest.TestCase):
    """The claim the wireframe actually rests on."""

    def _ratios(self, shape, f, noise=0.002, seed=0):
        rng = np.random.default_rng(seed)
        u, a = _sine(f)
        rows = np.outer(shape, a) + rng.normal(0, noise, (len(shape), T.size))
        rec = np.vstack([motion.integrate_twice(r, FS) for r in rows])
        i = int((WIN - motion.SETBACK_S) * FS) - 1
        seg = slice(i - int(2 * FS), i)
        amp = np.array([np.sqrt(np.mean(rec[k, seg] ** 2)) for k in range(len(shape))])
        return amp / amp.max()

    def test_a_mode_shape_survives_across_the_band(self):
        shape = np.array([0.35, 0.72, 1.00])
        for f in (1.0, 2.1, 5.0, 9.0):
            got = self._ratios(shape, f)
            self.assertLess(float(np.max(np.abs(got - shape))), 0.02,
                            f"{f} Hz: {got} vs {shape}")

    def test_a_sign_change_survives(self):
        shape = np.array([1.0, -0.4, -0.9])
        got = self._ratios(np.abs(shape), 3.0)
        self.assertLess(float(np.max(np.abs(got - np.abs(shape)))), 0.02)


class TestDriftImmunity(unittest.TestCase):
    """The reason this is not just cumsum twice."""

    def test_dc_and_ramp_do_not_run_away(self):
        u, a = _sine(2.0)
        polluted = a + 0.5 + 0.02 * T
        rec = motion.integrate_twice(polluted, FS)
        self.assertLess(float(np.max(np.abs(rec))), 0.01,
                        "the reconstruction ran away on a DC offset")

    def test_naive_integration_would_have_run_away(self):
        # Pins WHY the filtering exists; if this ever stops being true the
        # module's whole justification has changed.
        _u, a = _sine(2.0)
        naive = np.cumsum(np.cumsum(a + 0.5) / FS) / FS
        self.assertGreater(float(np.max(np.abs(naive))), 1.0)


class TestReadBack(unittest.TestCase):
    def test_it_does_not_read_the_last_sample(self):
        x = np.arange(1200, dtype=float)
        self.assertNotEqual(motion.read_back(x, FS), x[-1])
        self.assertAlmostEqual(motion.read_back(x, FS),
                               x[-1 - int(motion.SETBACK_S * FS)])

    def test_reading_back_beats_reading_the_edge(self):
        u, a = _sine(2.0)
        rec = motion.integrate_twice(a, FS)
        i = rec.size - 1 - int(motion.SETBACK_S * FS)
        self.assertLess(abs(rec[i] - u[i]), abs(rec[-1] - u[-1]))

    def test_short_input_is_safe(self):
        self.assertEqual(motion.read_back(np.array([]), FS), 0.0)
        self.assertEqual(motion.displacement_at(np.zeros(4), FS), 0.0)


class TestYaw(unittest.TestCase):
    def test_a_rate_integrates_to_an_angle(self):
        f, rate_amp = 2.0, 0.2
        rate = rate_amp * np.sin(2 * np.pi * f * T)
        ang = motion.integrate_once(rate, FS)
        expected = rate_amp / (2 * np.pi * f)
        mid = slice(int(4 * FS), int(8 * FS))
        got = float(np.sqrt(2) * np.sqrt(np.mean(ang[mid] ** 2)))
        self.assertAlmostEqual(got, expected, delta=0.25 * expected)

    def test_gyro_rate_is_degrees_per_second(self):
        """The MPU6050 reports deg/s; the angle must come back in radians."""
        rate_dps = 2.5 * np.sin(2 * np.pi * 1.6 * T)          # like the rig
        as_deg = motion.angle_at(rate_dps, FS, degrees_per_second=False)
        as_rad = motion.angle_at(rate_dps, FS)
        self.assertAlmostEqual(as_rad, np.deg2rad(as_deg), places=12)
        # 2.5 deg/s at 1.6 Hz is ~0.25 degrees of angle: milliradians, not radians
        self.assertLess(abs(as_rad), 0.02)

    def test_a_still_gyro_gives_no_rotation(self):
        self.assertAlmostEqual(motion.angle_at(np.zeros(1200), FS), 0.0, places=9)


class TestScale(unittest.TestCase):
    def test_scale_fills_the_target(self):
        s = motion.normalising_scale([0.002, -0.004, 0.001], target=0.9)
        self.assertAlmostEqual(0.004 * s, 0.9, places=9)

    def test_a_dead_input_does_not_divide_by_zero(self):
        self.assertEqual(motion.normalising_scale([0.0, 0.0]), 1.0)
        self.assertEqual(motion.normalising_scale([]), 1.0)


if __name__ == "__main__":
    unittest.main()
