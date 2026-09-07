"""Instructions for the physical structure, derived from a calibration.

The direction is the whole point and is easy to get backwards, so it is pinned
first: a calibration that needed MORE mass means the real structure is carrying
more than the design says, so the instruction is to REMOVE it.
"""
import unittest

from sensepi.digital_twin.decisions import (ADD, NOTHING, REMOVE, SOFTEN,
                                            STIFFEN, decide, rebalance_plan)

DESIGN = {"floor_masses": [0.20, 0.20, 0.20], "E": 2.0e11,
          "m_scale_lb": 0.5, "m_scale_ub": 2.0,
          "E_scale_lb": 0.5, "E_scale_ub": 2.0, "use_mode_shapes": True}


def _calib(masses=None, e=2.0e11):
    return {"floor_masses": list(masses or DESIGN["floor_masses"]), "E": e}


class TestDirection(unittest.TestCase):
    def test_the_users_example(self):
        """Design 0.2, calibration says 0.36 -> remove mass from storey 1."""
        d = decide(DESIGN, _calib([0.36, 0.20, 0.20]))
        self.assertEqual(d.masses[0].action, REMOVE)
        self.assertAlmostEqual(d.masses[0].delta, 0.16, places=6)
        self.assertIn("REMOVE", d.masses[0].headline())

    def test_a_lighter_fit_means_add(self):
        d = decide(DESIGN, _calib([0.10, 0.20, 0.20]))
        self.assertEqual(d.masses[0].action, ADD)

    def test_agreement_means_do_nothing(self):
        d = decide(DESIGN, _calib())
        self.assertTrue(all(m.action == NOTHING for m in d.masses))
        self.assertEqual(d.actions, [])
        self.assertIn("no change needed", d.summary())

    def test_softer_model_means_stiffen_the_structure(self):
        d = decide(DESIGN, _calib(e=1.76e11))
        self.assertEqual(d.stiffness.action, STIFFEN)
        self.assertIn("12%", d.stiffness.headline())

    def test_stiffer_model_means_soften(self):
        d = decide(DESIGN, _calib(e=2.4e11))
        self.assertEqual(d.stiffness.action, SOFTEN)


class TestTolerance(unittest.TestCase):
    def test_small_differences_are_not_actions(self):
        d = decide(DESIGN, _calib([0.204, 0.198, 0.201], e=2.02e11))
        self.assertEqual(d.actions, [])

    def test_the_tolerance_is_configurable(self):
        loose = decide(DESIGN, _calib([0.30, 0.20, 0.20]), mass_tolerance=0.9)
        self.assertEqual(loose.masses[0].action, NOTHING)


class TestSaturation(unittest.TestCase):
    """A scale on its bound is a lower limit, not a measurement."""

    def test_mass_on_its_bound_is_reported_as_at_least(self):
        d = decide(DESIGN, _calib([0.40, 0.20, 0.20]))   # scale 2.0 == m_scale_ub
        self.assertTrue(d.masses[0].saturated)
        self.assertIn("at least", d.masses[0].headline())
        self.assertTrue(any("bound" in n for n in d.notes))

    def test_an_interior_solution_is_not_saturated(self):
        d = decide(DESIGN, _calib([0.30, 0.20, 0.20]))
        self.assertFalse(d.masses[0].saturated)
        self.assertNotIn("at least", d.masses[0].headline())


class TestRebalance(unittest.TestCase):
    def test_it_offers_the_opposite_move(self):
        d = decide(DESIGN, _calib([0.36, 0.20, 0.20]))
        self.assertAlmostEqual(d.rebalance[1], -0.16, places=6)

    def test_the_net_imbalance_is_reported_under_key_zero(self):
        plan = rebalance_plan(decide(DESIGN, _calib([0.36, 0.10, 0.20])).masses)
        self.assertIn(0, plan)
        self.assertAlmostEqual(plan[0], -0.16 + 0.10, places=6)

    def test_nothing_to_rebalance_when_everything_agrees(self):
        self.assertEqual(decide(DESIGN, _calib()).rebalance, {})


class TestHonesty(unittest.TestCase):
    def test_the_confounding_note_is_always_present(self):
        d = decide(DESIGN, _calib([0.36, 0.20, 0.20]))
        self.assertTrue(any("confounded" in n for n in d.notes))

    def test_frequency_only_calibration_is_flagged_as_weak(self):
        design = dict(DESIGN, use_mode_shapes=False)
        d = decide(design, _calib([0.36, 0.20, 0.20]))
        self.assertTrue(any("frequencies only" in n.lower() for n in d.notes))

    def test_detail_gives_the_numbers_not_only_the_verdict(self):
        d = decide(DESIGN, _calib([0.36, 0.20, 0.20]))
        text = d.masses[0].detail()
        self.assertIn("0.36", text)
        self.assertIn("0.2", text)


class TestRefusesNonsense(unittest.TestCase):
    def test_no_calibration_is_reported_not_crashed(self):
        for bad in (None, {}, "no"):
            d = decide(DESIGN, bad)
            self.assertFalse(d.ok)
            self.assertTrue(d.message)

    def test_mismatched_storey_counts_are_refused(self):
        d = decide(DESIGN, _calib([0.2, 0.2]))
        self.assertFalse(d.ok)
        self.assertIn("same number of storeys", d.message)

    def test_a_zero_designed_mass_does_not_divide_by_zero(self):
        design = dict(DESIGN, floor_masses=[0.0, 0.2, 0.2])
        d = decide(design, _calib([0.1, 0.2, 0.2]))
        self.assertTrue(d.ok)


if __name__ == "__main__":
    unittest.main()
