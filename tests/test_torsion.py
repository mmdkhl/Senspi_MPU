"""Torsion indicators from a differenced pair and from the gyroscope.

Built on a floor whose motion is *constructed* as a known mix of sway and twist,
so "the indicator rises with twist" can be checked rather than assumed.
"""
import unittest

import numpy as np

from sensepi.analysis import torsion as tor

FS = 100.0
N = 6000
T = np.arange(N) / FS
F_MODE = 3.0


def _floor(twist_fraction, cell_a="A1", cell_b="C3", channel="ax", f=F_MODE):
    """Two sensors on one floor: common sway plus a rotation of known size.

    ``u_x = sway - theta * (y - y_c)`` and ``u_y = sway + theta * (x - x_c)`` —
    the floor rotates about its own centre, so the offsets are centre-relative.
    ``twist_fraction`` is the rotation amplitude relative to the sway.
    """
    centre = 1.0
    ya, yb = tor.cell_offsets(cell_a)[1] - centre, tor.cell_offsets(cell_b)[1] - centre
    xa, xb = tor.cell_offsets(cell_a)[0] - centre, tor.cell_offsets(cell_b)[0] - centre
    sway = np.sin(2 * np.pi * f * T)
    theta = twist_fraction * np.sin(2 * np.pi * f * T)
    if channel == "ay":
        return sway + theta * xa, sway + theta * xb
    return sway - theta * ya, sway - theta * yb


class TestPairGeometry(unittest.TestCase):
    def test_lever_arm_is_the_perpendicular_separation(self):
        # Rotation moves a point along x in proportion to its y offset.
        self.assertEqual(abs(tor.lever_arm_cells("A1", "C3", "ax")), 2.0)
        self.assertEqual(abs(tor.lever_arm_cells("A1", "A3", "ax")), 2.0)

    def test_same_row_pair_is_blind_to_torsion_in_ax(self):
        # A1 and C1 are far apart along x but at the SAME y, so a rotation moves
        # them identically and their difference contains no torsion.
        self.assertEqual(tor.lever_arm_cells("A1", "C1", "ax"), 0.0)
        a, b = _floor(0.5, "A1", "C1", channel="ax")
        res = tor.pair_torsion(a, b, FS, [F_MODE], cell_a="A1", cell_b="C1",
                               channel="ax", floor=3)
        self.assertFalse(res.usable)
        self.assertIn("cannot see torsion", res.note)
        self.assertTrue(np.isnan(res.per_mode[0]))

    def test_the_same_blind_pair_sees_torsion_in_ay(self):
        self.assertEqual(abs(tor.lever_arm_cells("A1", "C1", "ay")), 2.0)
        a, b = _floor(0.5, "A1", "C1", channel="ay")
        res = tor.pair_torsion(a, b, FS, [F_MODE], cell_a="A1", cell_b="C1",
                               channel="ay", floor=3)
        self.assertTrue(res.usable)
        self.assertGreater(res.per_mode[0], 0.1)

    def test_two_sensors_in_one_cell_are_blind(self):
        res = tor.pair_torsion(*_floor(0.5), FS, [F_MODE], cell_a="B2",
                               cell_b="B2", channel="ax", floor=2)
        self.assertFalse(res.usable)


class TestPairIndicator(unittest.TestCase):
    def test_pure_sway_reads_zero(self):
        a, b = _floor(0.0)
        res = tor.pair_torsion(a, b, FS, [F_MODE], cell_a="A1", cell_b="C3",
                               channel="ax", floor=3)
        self.assertTrue(res.usable)
        self.assertLess(res.per_mode[0], 0.02)

    def test_indicator_rises_monotonically_with_twist(self):
        got = []
        for frac in (0.0, 0.05, 0.1, 0.2, 0.4):
            a, b = _floor(frac)
            res = tor.pair_torsion(a, b, FS, [F_MODE], cell_a="A1", cell_b="C3",
                                   channel="ax", floor=3)
            got.append(res.per_mode[0])
        for lo, hi in zip(got, got[1:]):
            self.assertLess(lo, hi, f"not monotonic: {got}")

    def test_indicator_recovers_the_twist_fraction(self):
        # differential = theta * arm, common = sway, so ratio/arm = theta/sway.
        for frac in (0.1, 0.25, 0.5):
            a, b = _floor(frac)
            res = tor.pair_torsion(a, b, FS, [F_MODE], cell_a="A1", cell_b="C3",
                                   channel="ax", floor=3)
            self.assertAlmostEqual(res.per_mode[0], frac, delta=0.02 + 0.05 * frac)

    def test_a_frequency_the_floor_does_not_move_at_is_nan_not_huge(self):
        a, b = _floor(0.2)
        res = tor.pair_torsion(a, b, FS, [F_MODE, 17.0], cell_a="A1",
                               cell_b="C3", channel="ax", floor=3)
        self.assertFalse(np.isnan(res.per_mode[0]))
        self.assertTrue(np.isnan(res.per_mode[1]),
                        "0/0 at an unexcited frequency should be NaN")

    def test_lever_arm_is_divided_out_so_floors_are_comparable(self):
        # The same physical twist read by a wide symmetric pair and a narrow
        # OFF-CENTRE one must give the same indicator, or floors with different
        # pair placements could not be compared with each other.
        wide = tor.pair_torsion(*_floor(0.2, "A1", "C3"), FS, [F_MODE],
                                cell_a="A1", cell_b="C3", channel="ax", floor=3)
        narrow = tor.pair_torsion(*_floor(0.2, "A1", "C2"), FS, [F_MODE],
                                  cell_a="A1", cell_b="C2", channel="ax", floor=2)
        self.assertAlmostEqual(wide.per_mode[0], narrow.per_mode[0], delta=0.02)

    def test_centre_weight_reduces_to_the_mean_for_a_symmetric_pair(self):
        self.assertAlmostEqual(tor.centre_weight("A1", "C3", "ax"), 0.5)
        # An off-centre pair leans on the sensor nearer the centre instead.
        self.assertAlmostEqual(tor.centre_weight("A1", "C2", "ax"), 1.0)


class TestGyro(unittest.TestCase):
    def _rows(self, amps, f=F_MODE):
        return np.vstack([a * np.sin(2 * np.pi * f * T) for a in amps])

    def test_profile_is_normalised_to_the_largest_floor(self):
        res = tor.gyro_torsion(self._rows([0.25, 0.5, 1.0]), FS, [F_MODE], [1, 2, 3])
        vals = [r.per_mode[0] for r in res]
        self.assertAlmostEqual(max(vals), 1.0, places=6)
        self.assertAlmostEqual(vals[0] / vals[2], 0.25, delta=0.02)

    def test_a_silent_mode_gives_nan_not_a_spurious_profile(self):
        res = tor.gyro_torsion(np.zeros((2, N)), FS, [F_MODE], [1, 2])
        self.assertTrue(all(np.isnan(r.per_mode[0]) for r in res))


class TestAssembly(unittest.TestCase):
    def test_reports_when_there_is_no_torsion_source(self):
        res = tor.identify_torsion([F_MODE], channel="ax", fs=FS)
        self.assertFalse(res.success)
        self.assertIn("No torsion source", res.message)

    def test_combines_pair_and_gyro_and_names_a_blind_pair(self):
        a, b = _floor(0.3)
        blind_a, blind_b = _floor(0.3, "A1", "C1")
        gz = np.vstack([0.5 * np.sin(2 * np.pi * F_MODE * T)])
        res = tor.identify_torsion(
            [F_MODE], channel="ax", fs=FS,
            pair_series={3: (a, b, "A1", "C3", (3, 4)),
                         2: (blind_a, blind_b, "A1", "C1", (1, 2))},
            gyro_series={3: (gz[0], 4)})
        self.assertTrue(res.success)
        self.assertIn("blind pair on floor(s) 2", res.message)
        self.assertIn("not calibrated rotation", res.message)
        self.assertTrue(res.by_floor(3, "pair").usable)
        self.assertFalse(res.by_floor(2, "pair").usable)
        self.assertIsNotNone(res.by_floor(3, "gyro"))

    def test_no_frequencies_is_reported_not_crashed(self):
        res = tor.identify_torsion([], channel="ax", fs=FS)
        self.assertFalse(res.success)
        self.assertIn("No identified frequencies", res.message)


if __name__ == "__main__":
    unittest.main()
