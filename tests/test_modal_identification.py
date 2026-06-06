"""Tests for the operational modal identification module (M2).

Uses synthetic modal-superposition signals with known frequencies and mode
shapes — no Pi or OpenSees required.
"""

import unittest

import numpy as np

from sensepi.analysis import modal


# Known "truth" for the synthetic structure (3 sensors / 3 stories).
TRUE_FREQS = [2.3, 6.5, 9.1]
TRUE_SHAPES = [
    [0.30, 0.75, 1.00],   # mode 1
    [-1.00, 0.10, 0.95],  # mode 2
    [1.00, -0.85, 0.30],  # mode 3
]


def _mac(a, b):
    """Modal Assurance Criterion between two real shape vectors (sign-invariant)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = (a @ b) ** 2
    den = (a @ a) * (b @ b)
    return float(num / den) if den > 0 else 0.0


def _synthetic(fs=200.0, duration=60.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    # Independent modal coordinates (different freqs + random phase).
    q = []
    for f in TRUE_FREQS:
        q.append(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)))
    q = np.array(q)  # (n_modes, n)
    shapes = np.array(TRUE_SHAPES)  # (n_modes, n_sensors)
    # x_sensor = sum_m shape[m, sensor] * q[m]  -> (n_sensors, n)
    x = shapes.T @ q
    x += 0.02 * rng.standard_normal(x.shape)  # measurement noise
    return x, fs


class TestIdentifyModes(unittest.TestCase):
    def test_recovers_frequencies(self):
        x, fs = _synthetic()
        result = modal.identify_modes(x, fs, f_min=0.5, f_max=20.0, n_modes=3)
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.n_modes_found, 3)
        for got, expected in zip(result.frequencies_hz, TRUE_FREQS):
            self.assertAlmostEqual(got, expected, delta=0.2)

    def test_recovers_mode_shapes(self):
        x, fs = _synthetic()
        result = modal.identify_modes(x, fs, n_modes=3)
        for got, expected in zip(result.mode_shapes_sensor, TRUE_SHAPES):
            self.assertGreater(_mac(got, expected), 0.95)

    def test_shapes_normalized_maxabs_one(self):
        x, fs = _synthetic()
        result = modal.identify_modes(x, fs, n_modes=3)
        for shape in result.mode_shapes_sensor:
            self.assertAlmostEqual(max(abs(v) for v in shape), 1.0, places=6)

    def test_rejects_short_recording(self):
        x, fs = _synthetic(duration=5.0)
        result = modal.identify_modes(x, fs, n_modes=3)
        self.assertFalse(result.success)
        self.assertIn("too short", result.message.lower())

    def test_invalid_fs(self):
        x, _ = _synthetic()
        result = modal.identify_modes(x, 0.0)
        self.assertFalse(result.success)


class TestIdentifyModesFFT(unittest.TestCase):
    def test_recovers_frequencies(self):
        x, fs = _synthetic()
        result = modal.identify_modes(x, fs, f_min=0.5, f_max=20.0, n_modes=3, method="fft")
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.method, "fft")
        self.assertEqual(result.n_modes_found, 3)
        for got, expected in zip(result.frequencies_hz, TRUE_FREQS):
            self.assertAlmostEqual(got, expected, delta=0.2)

    def test_shapes_are_magnitude(self):
        # FFT recovers magnitude only (sign unknown), so compare against |truth|.
        x, fs = _synthetic()
        result = modal.identify_modes(x, fs, n_modes=3, method="fft")
        for got, expected in zip(result.mode_shapes_sensor, TRUE_SHAPES):
            self.assertGreater(_mac(got, np.abs(expected)), 0.9)
            self.assertTrue(all(v >= 0 for v in got))

    def test_shapes_normalized_maxabs_one(self):
        x, fs = _synthetic()
        result = modal.identify_modes(x, fs, n_modes=3, method="fft")
        for shape in result.mode_shapes_sensor:
            self.assertAlmostEqual(max(abs(v) for v in shape), 1.0, places=6)

    def test_default_method_is_fdd(self):
        x, fs = _synthetic()
        self.assertEqual(modal.identify_modes(x, fs).method, "fdd")

    def test_unknown_method_fails(self):
        x, fs = _synthetic()
        result = modal.identify_modes(x, fs, method="bogus")
        self.assertFalse(result.success)


class TestEstimateFs(unittest.TestCase):
    def test_uniform(self):
        t = np.arange(1000) / 200.0
        self.assertAlmostEqual(modal.estimate_fs(t), 200.0, places=3)

    def test_too_few_samples(self):
        self.assertTrue(np.isnan(modal.estimate_fs(np.array([0.0]))))


class TestMapToStories(unittest.TestCase):
    def _result(self):
        return modal.ExperimentalModalResult(
            frequencies_hz=[2.3, 6.5],
            mode_shapes_sensor=[[0.3, 0.75, 1.0], [-1.0, 0.1, 0.95]],
            n_modes_found=2,
            success=True,
        )

    def test_full_coverage_builds_shapes(self):
        out = modal.map_to_stories(self._result(), sensor_story_map=[1, 2, 3], n_story=3)
        self.assertTrue(out.mode_shapes_available)
        self.assertEqual(set(out.mode_shapes_ux.keys()), {"1", "2"})
        self.assertEqual(len(out.mode_shapes_ux["1"]), 3)
        # Story-2 mode-1 value should equal sensor-2 value (1:1 mapping), normalized.
        self.assertAlmostEqual(max(abs(v) for v in out.mode_shapes_ux["1"]), 1.0, places=6)

    def test_partial_coverage_frequency_only(self):
        # 3 sensors but two share the top story -> only stories {1,3} covered.
        out = modal.map_to_stories(self._result(), sensor_story_map=[1, 3, 3], n_story=3)
        self.assertFalse(out.mode_shapes_available)
        self.assertEqual(out.mode_shapes_ux, {})
        self.assertEqual(out.coverage_stories, [1, 3])
        # Torsion indicator recorded for the doubled story.
        self.assertIn(3, out.torsion_indicator)

    def test_two_sensors_one_story_averaged(self):
        res = modal.ExperimentalModalResult(
            frequencies_hz=[2.3],
            mode_shapes_sensor=[[0.4, 0.6, 1.0]],
            n_modes_found=1,
            success=True,
        )
        # sensors 1 & 2 both on story 1 -> averaged to 0.5; sensor 3 on story 2.
        out = modal.map_to_stories(res, sensor_story_map=[1, 1, 2], n_story=2)
        self.assertTrue(out.mode_shapes_available)
        # Before normalization story1=0.5, story2=1.0 -> normalized [0.5, 1.0].
        self.assertAlmostEqual(out.mode_shapes_ux["1"][0], 0.5, places=6)
        self.assertAlmostEqual(out.mode_shapes_ux["1"][1], 1.0, places=6)


class TestToExperimentalDict(unittest.TestCase):
    def test_includes_shapes_when_available(self):
        out = modal.map_to_stories(
            modal.ExperimentalModalResult(
                frequencies_hz=[2.3],
                mode_shapes_sensor=[[0.3, 1.0]],
                n_modes_found=1,
                success=True,
            ),
            sensor_story_map=[1, 2],
            n_story=2,
        )
        d = modal.to_experimental_dict(out)
        self.assertIn("frequencies_hz", d)
        self.assertIn("mode_shapes_ux", d)

    def test_omits_shapes_when_partial(self):
        out = modal.map_to_stories(
            modal.ExperimentalModalResult(
                frequencies_hz=[2.3], mode_shapes_sensor=[[0.3, 1.0]],
                n_modes_found=1, success=True,
            ),
            sensor_story_map=[1, 1], n_story=3,
        )
        d = modal.to_experimental_dict(out)
        self.assertIn("frequencies_hz", d)
        self.assertNotIn("mode_shapes_ux", d)


if __name__ == "__main__":
    unittest.main()
