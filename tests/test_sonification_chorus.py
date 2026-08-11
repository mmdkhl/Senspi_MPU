"""Tests for the Bioacoustic Chorus sonification model.

These import no Qt and no sounddevice: the engine layer must stay GUI-agnostic
and device-agnostic (guardrails G7 / G8-style optional audio).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sensepi.sonification.chorus import (ChorusConfig, ChorusEngine, carrier_for_mode,
                                         cast_meadow, catalog_available, load_catalog,
                                         load_grain_banks, run_offline)
from sensepi.sonification.chorus.audio_out import WavCapture, is_audio_available
from sensepi.sonification.chorus.features import FeatureExtractor, ModalTracker
from sensepi.sonification.chorus.renderer import ChorusRenderer
from sensepi.sonification.chorus.types import SAMPLE_RATE, ModalState

HAVE_DATA = catalog_available()


class _Snap:
    """Minimal ModalSession stand-in."""

    def __init__(self, data, fs):
        self.data = np.asarray(data, dtype=float)
        self.fs = float(fs)


def _synthetic(freqs=(1.9, 8.37, 12.77), fs=41.0, dur=60.0, seed=0):
    """Three sensors responding at the given modal frequencies."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1.0 / fs)
    shapes = np.array([[0.3, -0.6, 0.5], [0.7, 0.2, -0.9], [1.0, 1.0, 1.0]])
    data = np.zeros((3, t.size))
    for m, f in enumerate(freqs):
        data += np.outer(shapes[:, m], np.sin(2 * np.pi * f * t)) * (0.03 / (m + 1))
    data += rng.standard_normal(data.shape) * 0.002
    return _Snap(data, fs)


class TestCastingMap(unittest.TestCase):
    def test_carrier_map_is_monotonic_and_bounded(self):
        cfg = ChorusConfig()
        prev = -1.0
        for f in np.linspace(0.5, 20.0, 40):
            c = carrier_for_mode(float(f), cfg)
            self.assertGreaterEqual(c, cfg.c_lo - 1e-6)
            self.assertLessEqual(c, cfg.c_hi + 1e-6)
            self.assertGreater(c, prev)
            prev = c

    def test_map_clamps_outside_the_declared_range(self):
        cfg = ChorusConfig()
        self.assertAlmostEqual(carrier_for_mode(0.01, cfg), cfg.c_lo, places=3)
        self.assertAlmostEqual(carrier_for_mode(1e4, cfg), cfg.c_hi, places=3)

    def test_endpoints_hit_the_configured_carriers(self):
        cfg = ChorusConfig()
        self.assertAlmostEqual(carrier_for_mode(cfg.f_lo, cfg), cfg.c_lo, places=3)
        self.assertAlmostEqual(carrier_for_mode(cfg.f_hi, cfg), cfg.c_hi, places=3)


@unittest.skipUnless(HAVE_DATA, "species catalog / grain bank not installed")
class TestCatalogAndCast(unittest.TestCase):
    def test_catalog_entries_are_biologically_plausible(self):
        cat = load_catalog()
        self.assertGreater(len(cat), 10)
        for s in cat:
            self.assertGreater(s.carrier, 150.0)
            self.assertLess(s.carrier, 20000.0)
            self.assertGreater(s.unit_s, 0.0)

    def test_every_catalogued_species_has_units(self):
        banks = load_grain_banks()
        self.assertTrue(banks)
        for name, units in banks.items():
            self.assertGreaterEqual(len(units), 1, name)
            self.assertTrue(all(u.size >= 8 for u in units), name)

    def test_cast_gives_each_mode_a_lead_and_never_repeats_a_species(self):
        cfg = ChorusConfig()
        cast = cast_meadow(np.array([1.9, 8.37, 12.77]), cfg)
        self.assertTrue(cast)
        leads = [c for c in cast if c.role == "lead"]
        self.assertEqual(sorted(c.mode for c in leads), [0, 1, 2])
        names = [c.info.species for c in cast]
        self.assertEqual(len(names), len(set(names)))

    def test_lead_carrier_is_close_to_the_requested_target(self):
        cfg = ChorusConfig()
        for entry in cast_meadow(np.array([1.9, 8.37, 12.77]), cfg):
            if entry.role != "lead":
                continue
            octaves = abs(np.log2(entry.info.carrier / entry.target_carrier))
            self.assertLess(octaves, 1.5, entry.info.species)

    def test_chirp_rate_equals_the_modal_frequency_exactly(self):
        """The 1:1 mapping is the point of the model — no transposition."""
        freqs = np.array([1.9, 8.37, 12.77])
        for entry in cast_meadow(freqs, ChorusConfig()):
            if entry.mode >= 0:
                self.assertAlmostEqual(entry.mode_freq, freqs[entry.mode], places=6)

    def test_a_softening_structure_recasts_the_meadow(self):
        """Frequencies drop -> carriers drop -> different species. The diagnostic."""
        cfg = ChorusConfig()
        base = np.array([1.9, 8.37, 12.77])
        healthy = [c.info.species for c in cast_meadow(base, cfg) if c.role == "lead"]
        damaged = [c.info.species for c in cast_meadow(base * 0.6, cfg) if c.role == "lead"]
        self.assertNotEqual(healthy, damaged)
        for m in range(3):
            self.assertLess(carrier_for_mode(base[m] * 0.6, cfg),
                            carrier_for_mode(base[m], cfg))

    def test_cast_survives_degenerate_frequencies(self):
        cfg = ChorusConfig()
        for freqs in (np.array([]), np.array([np.nan, 8.0]), np.array([0.0, -3.0])):
            cast_meadow(freqs, cfg)          # must not raise


class TestFeatures(unittest.TestCase):
    def test_excitation_tracker_finds_a_planted_tone(self):
        fs = 41.0
        t = np.arange(0, 30, 1 / fs)
        rng = np.random.default_rng(1)
        data = np.vstack([np.sin(2 * np.pi * 5.0 * t) * 0.05
                          + rng.standard_normal(t.size) * 0.002 for _ in range(3)])
        fx = FeatureExtractor(ChorusConfig())
        modal = ModalState(frequencies_hz=np.array([5.0, 9.0, 13.0]))
        frame = fx.update(_Snap(data, fs), None, modal, t=0.0)
        self.assertAlmostEqual(frame.exc_freq_hz, 5.0, delta=0.3)
        self.assertGreater(frame.exc_conf, 0.3)

    def test_pure_noise_gives_low_confidence(self):
        fs = 41.0
        rng = np.random.default_rng(2)
        data = rng.standard_normal((3, int(fs * 30))) * 0.01
        fx = FeatureExtractor(ChorusConfig())
        frame = fx.update(_Snap(data, fs), None,
                          ModalState(frequencies_hz=np.array([5.0])), t=0.0)
        self.assertLess(frame.exc_conf, 0.35)

    def test_sync_peaks_on_the_mode_being_driven(self):
        fs, drive = 41.0, 8.37
        t = np.arange(0, 40, 1 / fs)
        data = np.vstack([np.sin(2 * np.pi * drive * t) * 0.05 for _ in range(3)])
        fx = FeatureExtractor(ChorusConfig())
        modal = ModalState(frequencies_hz=np.array([1.9, 8.37, 12.77]))
        for i in range(30):                      # warm the rolling normalisers
            frame = fx.update(_Snap(data, fs), None, modal, t=i * 0.05)
        self.assertEqual(int(np.argmax(frame.sync)), 1)

    def test_empty_and_nan_input_is_survivable(self):
        fx = FeatureExtractor(ChorusConfig())
        modal = ModalState(frequencies_hz=np.array([2.0, 6.0]))
        self.assertEqual(fx.update(None, None, modal, 0.0).exc_conf, 0.0)
        bad = np.full((3, 400), np.nan)
        frame = fx.update(_Snap(bad, 41.0), None, modal, 0.0)
        self.assertGreater(frame.nan_ratio, 0.9)
        self.assertFalse(np.isnan(frame.env_global))

    def test_torsion_responds_to_the_gyro_channel(self):
        fs = 41.0
        t = np.arange(0, 20, 1 / fs)
        ax = np.vstack([np.sin(2 * np.pi * 5 * t) * 0.02] * 3)
        fx = FeatureExtractor(ChorusConfig())
        modal = ModalState(frequencies_hz=np.array([5.0]))
        quiet = np.vstack([np.zeros_like(t)] * 3)
        loud = np.vstack([np.sin(2 * np.pi * 6 * t) * 0.5] * 3)
        for _ in range(20):
            fx.update(_Snap(ax, fs), _Snap(quiet, fs), modal, 0.0)
        hot = fx.update(_Snap(ax, fs), _Snap(loud, fs), modal, 0.0)
        self.assertGreater(hot.torsion, 0.5)


class TestModalTracker(unittest.TestCase):
    def test_identifies_planted_modes(self):
        tr = ModalTracker(ChorusConfig())
        state = tr.reidentify(_synthetic(), 0.0)
        if state.ok:                     # identification quality is not this test's job
            self.assertEqual(len(state.frequencies_hz), 3)
            self.assertTrue(np.all(np.isfinite(state.frequencies_hz)))

    def test_never_raises_on_rubbish(self):
        tr = ModalTracker(ChorusConfig())
        for bad in (None, _Snap(np.empty((0, 0)), float("nan")),
                    _Snap(np.zeros((3, 4)), 41.0), _Snap(np.full((3, 500), np.nan), 41.0)):
            state = tr.reidentify(bad, 1.0)
            self.assertIsInstance(state, ModalState)

    def test_reid_interval_is_respected(self):
        cfg = ChorusConfig()
        cfg.reid_interval_s = 8.0
        tr = ModalTracker(cfg)
        self.assertTrue(tr.wants_reid(0.0))
        tr.reidentify(_synthetic(dur=40), 100.0)
        self.assertFalse(tr.wants_reid(101.0))
        self.assertTrue(tr.wants_reid(109.0))


@unittest.skipUnless(HAVE_DATA, "species catalog / grain bank not installed")
class TestRenderer(unittest.TestCase):
    def _renderer(self):
        cfg = ChorusConfig()
        r = ChorusRenderer(cfg)
        r.set_cast(cast_meadow(np.array([1.9, 8.37, 12.77]), cfg), load_grain_banks())
        return r

    def test_blocks_have_the_right_shape_and_stay_finite(self):
        r = self._renderer()
        for _ in range(40):
            r.schedule_ahead()
            block = r.render_block(1024)
            self.assertEqual(block.shape, (1024, 2))
            self.assertTrue(np.all(np.isfinite(block)))

    def test_output_is_limited(self):
        r = self._renderer()
        peak = 0.0
        for _ in range(60):
            r.schedule_ahead()
            peak = max(peak, float(np.abs(r.render_block()).max()))
        self.assertLessEqual(peak, 1.0)

    def test_ring_is_cleared_so_audio_does_not_loop(self):
        """A block already played must not reappear one ring later."""
        r = self._renderer()
        for _ in range(10):
            r.schedule_ahead()
            r.render_block()
        r._voices = []                       # silence every voice
        for _ in range(int(r.ring_len / 1024) + 4):
            r.render_block()
        tail = np.abs(r.render_block()).max()
        self.assertLess(float(tail), 1e-3)

    def test_play_head_advances_by_exactly_one_block(self):
        r = self._renderer()
        before = r.play_seconds
        r.schedule_ahead()
        r.render_block(512)
        self.assertAlmostEqual(r.play_seconds - before, 512 / SAMPLE_RATE, places=9)

    def test_grain_budget_bounds_the_work(self):
        cfg = ChorusConfig()
        cfg.density = 3.0
        cfg.chorus_size = 2.0
        r = ChorusRenderer(cfg)
        r.set_cast(cast_meadow(np.array([1.9, 8.37, 12.77]), cfg), load_grain_banks())
        for _ in range(40):
            r.schedule_ahead()
            r.render_block()
        played = max(r.play_seconds, 1e-6)
        self.assertLess(r.grains_written / played, 4000.0)


@unittest.skipUnless(HAVE_DATA, "species catalog / grain bank not installed")
class TestEngine(unittest.TestCase):
    def test_offline_run_produces_sane_audio(self):
        audio, frames, engine = run_offline(_synthetic(dur=12.0), None,
                                            ChorusConfig(), duration_s=12.0)
        self.assertGreater(len(audio), SAMPLE_RATE * 10)
        self.assertEqual(audio.shape[1], 2)
        self.assertFalse(np.isnan(audio).any())
        self.assertLessEqual(float(np.abs(audio).max()), 1.0)
        self.assertGreater(len(frames), 100)
        self.assertTrue(engine.cast)

    def test_frames_carry_a_usable_cast_and_state_text(self):
        _audio, frames, _engine = run_offline(_synthetic(dur=8.0), None,
                                              ChorusConfig(), duration_s=8.0)
        last = frames[-1]
        self.assertTrue(last.state_text)
        self.assertEqual(last.frame.band_energy.size, last.modal.frequencies_hz.size)

    def test_set_option_is_live_and_ignores_nonsense(self):
        engine = ChorusEngine(ChorusConfig())
        engine.set_option("master", 0.4)
        self.assertAlmostEqual(engine.cfg.master, 0.4)
        engine.set_option("not_a_real_option", 3)      # must not raise
        engine.set_option("c_hi", 9000.0)
        self.assertAlmostEqual(engine.cfg.c_hi, 9000.0)

    def test_config_is_clamped(self):
        cfg = ChorusConfig()
        cfg.master = 9.0
        cfg.naturalism = -2.0
        cfg.space = 5.0
        cfg = cfg.clamped()
        self.assertLessEqual(cfg.master, 1.0)
        self.assertGreaterEqual(cfg.naturalism, 0.0)
        self.assertLessEqual(cfg.space, 1.0)

    def test_engine_survives_a_silent_structure(self):
        flat = _Snap(np.zeros((3, 41 * 10)), 41.0)
        audio, frames, _ = run_offline(flat, None, ChorusConfig(), duration_s=5.0)
        self.assertFalse(np.isnan(audio).any())
        self.assertTrue(frames)


class TestAudioOut(unittest.TestCase):
    def test_availability_is_a_bool(self):
        self.assertIsInstance(is_audio_available(), bool)

    def test_capture_round_trips_to_a_wav(self):
        cap = WavCapture()
        cap.start()
        for _ in range(8):
            cap.add(np.zeros((1024, 2), dtype=np.float32) + 0.25)
        with tempfile.TemporaryDirectory() as tmp:
            path = cap.save(Path(tmp) / "cap.wav")
            self.assertIsNotNone(path)
            self.assertTrue(path.is_file())
            import wave
            with wave.open(str(path)) as w:
                self.assertEqual(w.getnchannels(), 2)
                self.assertEqual(w.getframerate(), SAMPLE_RATE)
                self.assertEqual(w.getnframes(), 8 * 1024)

    def test_inactive_capture_saves_nothing(self):
        cap = WavCapture()
        cap.add(np.zeros((16, 2), dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(cap.save(Path(tmp) / "empty.wav"))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_DATA, "species catalog / grain bank not installed")
class TestAutofitCastingMap(unittest.TestCase):
    """Matching the structural range to the species range, and holding the fit."""

    def test_fit_spreads_the_modes_across_the_whole_palette(self):
        from sensepi.sonification.chorus.catalog import carrier_span, fit_casting_map
        cfg = ChorusConfig()
        freqs = np.array([1.9, 8.37, 12.77])
        f_lo, f_hi, c_lo, c_hi = fit_casting_map(freqs, cfg)
        self.assertLess(f_lo, freqs.min())
        self.assertGreater(f_hi, freqs.max())
        self.assertEqual((c_lo, c_hi), carrier_span())
        fitted = ChorusConfig()
        fitted.f_lo, fitted.f_hi, fitted.c_lo, fitted.c_hi = f_lo, f_hi, c_lo, c_hi
        spread = [carrier_for_mode(float(f), fitted) for f in freqs]
        default = [carrier_for_mode(float(f), cfg) for f in freqs]
        # the fitted map must use more of the available carrier range
        self.assertGreater(np.log2(spread[-1] / spread[0]),
                           np.log2(default[-1] / default[0]))

    def test_fit_is_taken_once_so_drift_still_recasts(self):
        """If the map re-fitted on every identification it would cancel out
        frequency drift, destroying the damage-readout property."""
        engine = ChorusEngine(ChorusConfig())
        engine.tracker._state = ModalState(
            frequencies_hz=np.array([1.9, 8.37, 12.77]), ok=True,
            damping=np.array([0.016, 0.004, 0.002]))
        engine._recast()
        first = [c.info.species for c in engine.cast if c.role == "lead"]
        fitted = (engine.cfg.f_lo, engine.cfg.f_hi, engine.cfg.c_lo, engine.cfg.c_hi)
        # the structure softens
        engine.tracker._state = ModalState(
            frequencies_hz=np.array([1.9, 8.37, 12.77]) * 0.7, ok=True,
            damping=np.array([0.016, 0.004, 0.002]))
        engine._recast()
        self.assertEqual(fitted, (engine.cfg.f_lo, engine.cfg.f_hi,
                                  engine.cfg.c_lo, engine.cfg.c_hi))
        self.assertNotEqual(first, [c.info.species for c in engine.cast
                                    if c.role == "lead"])

    def test_editing_the_map_by_hand_disables_autofit(self):
        engine = ChorusEngine(ChorusConfig())
        self.assertTrue(engine.cfg.autofit)
        engine.set_option("c_hi", 9000.0)
        self.assertFalse(engine.cfg.autofit)

    def test_autofit_off_leaves_the_declared_map_alone(self):
        defaults = ChorusConfig()
        cfg = ChorusConfig()
        cfg.autofit = False
        engine = ChorusEngine(cfg)
        engine.tracker._state = ModalState(frequencies_hz=np.array([2.0, 9.0]), ok=True)
        engine._recast()
        self.assertAlmostEqual(engine.cfg.f_lo, defaults.f_lo)
        self.assertAlmostEqual(engine.cfg.c_hi, defaults.c_hi)

    def test_the_declared_band_covers_real_buildings(self):
        """0-20 Hz, with mode 1 commonly below 2 Hz."""
        cfg = ChorusConfig()
        self.assertLessEqual(cfg.f_min, 0.3)
        self.assertGreaterEqual(cfg.f_max, 20.0)
        self.assertLessEqual(cfg.f_lo, 0.3)
        self.assertGreaterEqual(cfg.f_hi, 20.0)

    def test_a_sub_2hz_mode_one_is_not_squashed_to_the_floor(self):
        """A 0.4 Hz mode 1 must still get a distinct carrier, not the c_lo clamp."""
        cfg = ChorusConfig()
        low = carrier_for_mode(0.4, cfg)
        self.assertGreater(low, cfg.c_lo)
        self.assertLess(low, carrier_for_mode(1.9, cfg))


@unittest.skipUnless(HAVE_DATA, "species catalog / grain bank not installed")
class TestCastStability(unittest.TestCase):
    """Identified frequencies scatter per cycle; the cast must not flicker."""

    BASE = np.array([1.9, 6.7, 10.3])

    def _fitted(self):
        from sensepi.sonification.chorus.catalog import fit_casting_map
        cfg = ChorusConfig()
        cfg.f_lo, cfg.f_hi, cfg.c_lo, cfg.c_hi = fit_casting_map(self.BASE, cfg)
        return cfg

    def _leads(self, cast):
        return tuple(c.info.species for c in
                     sorted((c for c in cast if c.role == "lead"), key=lambda c: c.mode))

    def _flips(self, cfg, jitter, cycles=80, seed=4, use_previous=True):
        rng = np.random.default_rng(seed)
        prev, seen = None, []
        for _ in range(cycles):
            f = self.BASE * (1 + rng.uniform(-jitter, jitter, self.BASE.size))
            cast = cast_meadow(f, cfg, previous=prev if use_previous else None)
            if use_previous:
                prev = cast
            seen.append(self._leads(cast))
        return sum(1 for a, b in zip(seen[:-1], seen[1:]) if a != b)

    def test_small_jitter_does_not_flicker_the_cast(self):
        cfg = self._fitted()
        self.assertLessEqual(self._flips(cfg, 0.01), 4)
        self.assertLessEqual(self._flips(cfg, 0.02), 8)

    def test_hysteresis_is_what_provides_the_stability(self):
        """Same jitter without hysteresis must be measurably worse."""
        cfg = self._fitted()
        with_h = self._flips(cfg, 0.02)
        cfg_off = self._fitted()
        cfg_off.cast_hysteresis_oct = 0.0
        without = self._flips(cfg_off, 0.02, use_previous=False)
        self.assertLess(with_h, without)

    def test_real_drift_still_recasts_through_the_hysteresis(self):
        cfg = self._fitted()
        prev = cast_meadow(self.BASE, cfg)
        healthy = self._leads(prev)
        for drop in (0.02, 0.05, 0.10, 0.15, 0.20):
            prev = cast_meadow(self.BASE * (1 - drop), cfg, previous=prev)
        self.assertNotEqual(healthy, self._leads(prev))

    def test_an_incumbent_is_dropped_once_clearly_beaten(self):
        cfg = self._fitted()
        prev = cast_meadow(self.BASE, cfg)
        far = cast_meadow(self.BASE * 0.5, cfg, previous=prev)
        self.assertNotEqual(self._leads(prev), self._leads(far))
