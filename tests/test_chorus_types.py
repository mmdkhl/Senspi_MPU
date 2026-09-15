"""User-chosen animal types: one per mode, one per structural case.

The rule under test: the user decides WHAT KIND of animal sings each mode and
each case; the structure's frequency still decides WHICH species inside that
type. No Qt, no audio device.
"""
from __future__ import annotations

import unittest

import numpy as np

from sensepi.sonification.chorus import (ChorusConfig, ChorusEngine, TYPES, TYPE_ORDER,
                                         available_types, carrier_for_mode, cast_meadow,
                                         catalog_available, load_catalog, type_span)
from sensepi.sonification.chorus.types import ModalState, type_of_group

HAVE_DATA = catalog_available()
FREQS = np.array([1.9, 8.37, 12.77])


def _leads(cast):
    return {c.mode: c for c in cast if c.role == "lead"}


class TestRegistry(unittest.TestCase):
    def test_every_type_maps_at_least_one_group_and_back(self):
        for key, (label, groups, blurb) in TYPES.items():
            self.assertTrue(label and groups and blurb, key)
            for g in groups:
                self.assertEqual(type_of_group(g), key)
        self.assertEqual(type_of_group("nonsense"), "")

    def test_defaults_name_real_types(self):
        cfg = ChorusConfig()
        for t in cfg.type_of_mode:
            self.assertIn(t, TYPES)
        for attr in ("resonance_type", "torsion_type", "alarm_type", "drift_type"):
            self.assertIn(getattr(cfg, attr), TYPES, attr)
        self.assertEqual(cfg.ambient_type, "auto")
        self.assertEqual(tuple(TYPE_ORDER), tuple(TYPES))


@unittest.skipUnless(HAVE_DATA, "species catalog / grain bank not installed")
class TestTypeSpans(unittest.TestCase):
    def test_catalog_species_carry_a_type(self):
        for s in load_catalog():
            self.assertIn(s.type, TYPES, s.species)

    def test_each_available_type_has_a_usable_span(self):
        for key in available_types():
            lo, hi = type_span(key)
            self.assertGreater(lo, 100.0, key)
            self.assertGreaterEqual(np.log2(hi / lo), 1.0 - 1e-9, key)

    def test_map_lands_inside_the_types_own_span(self):
        cfg = ChorusConfig()
        for key in available_types():
            lo, hi = type_span(key)
            prev = -1.0
            for f in np.linspace(cfg.f_lo, cfg.f_hi, 25):
                c = carrier_for_mode(float(f), cfg, key)
                self.assertGreaterEqual(c, lo - 1e-6)
                self.assertLessEqual(c, hi + 1e-6)
                self.assertGreater(c, prev)
                prev = c
            self.assertAlmostEqual(carrier_for_mode(cfg.f_lo, cfg, key), lo, places=3)
            self.assertAlmostEqual(carrier_for_mode(cfg.f_hi, cfg, key), hi, places=3)


@unittest.skipUnless(HAVE_DATA, "species catalog / grain bank not installed")
class TestUserChosenCast(unittest.TestCase):
    def test_each_mode_sings_as_the_chosen_type(self):
        avail = list(available_types())
        if len(avail) < 3:
            self.skipTest("fewer than three types shipped")
        cfg = ChorusConfig()
        cfg.type_of_mode = (avail[2], avail[0], avail[1])
        leads = _leads(cast_meadow(FREQS, cfg))
        self.assertEqual(set(leads), {0, 1, 2})
        for m, entry in leads.items():
            self.assertEqual(entry.info.type, cfg.type_of_mode[m], f"mode {m + 1}")

    def test_two_users_two_meadows(self):
        """Different choices -> different animals for the same structure."""
        avail = list(available_types())
        if len(avail) < 2:
            self.skipTest("fewer than two types shipped")
        a, b = ChorusConfig(), ChorusConfig()
        a.type_of_mode = (avail[0],) * 3
        b.type_of_mode = (avail[1],) * 3
        la = {m: e.info.species for m, e in _leads(cast_meadow(FREQS, a)).items()}
        lb = {m: e.info.species for m, e in _leads(cast_meadow(FREQS, b)).items()}
        self.assertTrue(la and lb)
        self.assertFalse(set(la.values()) & set(lb.values()))

    def test_frequency_picks_the_species_inside_the_type(self):
        """Softening -> lower carrier -> a different, lower species of the SAME type."""
        for key in available_types():
            n = sum(1 for s in load_catalog() if s.type == key)
            if n < 4:
                continue
            cfg = ChorusConfig()
            cfg.type_of_mode = (key,)
            healthy = _leads(cast_meadow(np.array([8.0]), cfg))[0]
            damaged = _leads(cast_meadow(np.array([8.0 * 0.5]), cfg))[0]
            self.assertEqual(healthy.info.type, key)
            self.assertEqual(damaged.info.type, key)
            self.assertLess(damaged.target_carrier, healthy.target_carrier, key)
            self.assertLessEqual(damaged.info.carrier, healthy.info.carrier, key)

    def test_case_voices_come_from_their_chosen_types(self):
        avail = list(available_types())
        cfg = ChorusConfig()
        cfg.resonance_type = avail[-1]
        cfg.torsion_type = avail[0]
        cfg.drift_type = avail[min(1, len(avail) - 1)]
        cfg.alarm_type = avail[min(2, len(avail) - 1)]
        cfg.ambient_type = avail[min(3, len(avail) - 1)]
        roles = {}
        for c in cast_meadow(FREQS, cfg):
            roles.setdefault(c.role, []).append(c)
        for role, attr in (("resonance", "resonance_type"), ("torsion", "torsion_type"),
                           ("drift", "drift_type"), ("alarm", "alarm_type"),
                           ("ambient", "ambient_type")):
            self.assertIn(role, roles, role)
            for c in roles[role]:
                self.assertEqual(c.info.type, getattr(cfg, attr), role)

    def test_torsion_and_drift_follow_the_structure(self):
        """Their species are matched to the identified frequencies, not fixed."""
        cfg = ChorusConfig()
        cast = cast_meadow(FREQS, cfg)
        tors = [c for c in cast if c.role == "torsion"]
        drift = [c for c in cast if c.role == "drift"]
        self.assertTrue(tors and drift)
        self.assertAlmostEqual(tors[0].mode_freq, float(FREQS.max()), places=6)
        self.assertAlmostEqual(drift[0].mode_freq,
                               float(np.exp(np.mean(np.log(FREQS)))), places=6)
        for e in tors + drift:
            octaves = abs(np.log2(e.info.carrier / e.target_carrier))
            self.assertLess(octaves, 2.0, e.role)

    def test_never_the_same_species_twice(self):
        cfg = ChorusConfig()
        cfg.type_of_mode = (available_types()[0],) * 4
        names = [c.info.species for c in cast_meadow(np.array([1.0, 3.0, 7.0, 15.0]), cfg)]
        self.assertEqual(len(names), len(set(names)))

    def test_unknown_type_falls_back_instead_of_silence(self):
        cfg = ChorusConfig()
        cfg.type_of_mode = ("dragons",)
        cfg.torsion_type = "dragons"
        cast = cast_meadow(FREQS, cfg)
        self.assertTrue(_leads(cast))
        self.assertTrue([c for c in cast if c.role == "torsion"])

    def test_hysteresis_uses_the_config_value(self):
        """The catalog must not carry its own constant any more."""
        from sensepi.sonification.chorus import catalog
        self.assertFalse(hasattr(catalog, "CAST_HYSTERESIS_OCT"))
        cfg = ChorusConfig()
        self.assertLess(cfg.cast_hysteresis_oct, 0.2)


@unittest.skipUnless(HAVE_DATA, "species catalog / grain bank not installed")
class TestLiveTypeChange(unittest.TestCase):
    def test_set_option_type_of_mode_recasts_on_next_tick(self):
        avail = list(available_types())
        if len(avail) < 2:
            self.skipTest("fewer than two types shipped")
        engine = ChorusEngine(ChorusConfig())
        engine.tracker._state = ModalState(frequencies_hz=FREQS, ok=True,
                                           damping=np.array([0.016, 0.004, 0.002]))
        engine._recast()
        before = _leads(engine.cast)[0].info.type
        new = avail[0] if before != avail[0] else avail[1]
        engine.set_option("type_of_mode", [new, new, new])   # a list, as the tab sends
        self.assertEqual(engine.cfg.type_of_mode, (new, new, new))
        self.assertTrue(engine._recast_pending)
        engine.tick(0.1, None, None)
        self.assertEqual(_leads(engine.cast)[0].info.type, new)

    def test_case_type_options_are_live(self):
        engine = ChorusEngine(ChorusConfig())
        key = available_types()[0]
        for attr in ("resonance_type", "torsion_type", "alarm_type",
                     "drift_type", "ambient_type"):
            engine.set_option(attr, key)
            self.assertEqual(getattr(engine.cfg, attr), key)


if __name__ == "__main__":
    unittest.main()
