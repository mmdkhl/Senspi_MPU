"""Structure Pulse engine: analysis, response spectra, and curve-to-audio.

No Qt and no audio device — the engine must stay importable and testable
without either (guardrails G7 / G8-style optional audio).
"""
from __future__ import annotations

import unittest

import numpy as np

from sensepi.sonification.structure_pulse import (BELL_TIMBRES, SCALE_MODES,
                                           VOICE_MODES,
                                           PulseConfig, PulseView, Curve,
                                           Marker, peak_of_spectrum, render_view,
                                           response_spectrum, sdof_response,
                                           strike)

FS = 100.0


def _sine(f, dur=40.0, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(0, dur, 1 / fs)
    return t, amp * np.sin(2 * np.pi * f * t + phase)


def _view(kind="spectrum", n=200, markers=(), amp_follows=True):
    x = np.linspace(0.5, 20.0, n)
    y = np.exp(-((x - 6.0) ** 2) / 2.0) + 0.1
    return PulseView(name="t", kind=kind, curves=[Curve(x, y, "c")],
                        markers=list(markers), amplitude_follows_y=amp_follows)


class TestResponseSpectrum(unittest.TestCase):
    """The one piece of structural analysis that was missing."""

    def test_resonant_drive_is_amplified_by_one_over_two_zeta(self):
        """Textbook steady-state result: Sa/a_drive -> 1/(2z) at resonance."""
        for z in (0.02, 0.05, 0.10):
            T0 = 0.25
            _t, a = _sine(1 / T0, dur=60.0, fs=200.0)
            u = sdof_response(a, 200.0, T0, z)
            sa = (2 * np.pi / T0) ** 2 * np.abs(u).max()
            self.assertAlmostEqual(sa, 1 / (2 * z), delta=0.12 / (2 * z),
                                   msg=f"z={z}")

    def test_spectrum_peaks_at_the_driving_period(self):
        T0 = 0.5
        _t, a = _sine(1 / T0, dur=60.0, fs=200.0)
        spec = response_spectrum(a, 200.0, damping=0.05, t_min=0.1, t_max=2.0,
                                 n_periods=180)
        Tp, _ = peak_of_spectrum(spec)
        self.assertAlmostEqual(Tp, T0, delta=0.03)

    def test_outputs_are_consistent_and_finite(self):
        _t, a = _sine(4.0, dur=20.0)
        spec = response_spectrum(a, FS, n_periods=40)
        for key in ("periods", "Sd", "Sv", "Sa"):
            self.assertTrue(np.isfinite(spec[key]).all(), key)
        wn = 2 * np.pi / spec["periods"]
        np.testing.assert_allclose(spec["Sv"], wn * spec["Sd"], rtol=1e-9)
        np.testing.assert_allclose(spec["Sa"], wn ** 2 * spec["Sd"], rtol=1e-9)

    def test_unresolvable_period_returns_zero_not_nonsense(self):
        """Below the sampling limit the answer is 'cannot say', not a number."""
        _t, a = _sine(4.0, dur=10.0)
        u = sdof_response(a, FS, period_s=1e-4)        # far above Nyquist
        self.assertTrue(np.all(u == 0.0))

    def test_degenerate_input_never_raises(self):
        for a in (np.zeros(0), np.zeros(5), np.full(50, np.nan)):
            response_spectrum(a, FS, n_periods=5)      # must not raise


class TestRenderBasics(unittest.TestCase):
    def test_every_voice_produces_finite_bounded_stereo(self):
        for voice in VOICE_MODES:
            kind = "time" if voice == "audify" else "spectrum"
            v = _view(kind=kind)
            r = render_view(v, PulseConfig(duration_s=2.0, voice=voice),
                            modal_frequencies=[2.0], data_fs=FS)
            self.assertGreater(r.audio.size, 0, voice)
            self.assertEqual(r.audio.shape[1], 2, voice)
            self.assertTrue(np.isfinite(r.audio).all(), voice)
            self.assertLessEqual(float(np.abs(r.audio).max()), 1.0, voice)

    def test_audify_refuses_a_non_time_view_instead_of_faking_it(self):
        r = render_view(_view(kind="spectrum"),
                        PulseConfig(voice="audify"), data_fs=FS)
        self.assertEqual(r.audio.size, 0)
        self.assertIn("time-domain", r.message)

    def test_empty_view_is_handled(self):
        v = PulseView(name="e", kind="time", curves=[])
        r = render_view(v, PulseConfig())
        self.assertEqual(r.audio.size, 0)
        self.assertTrue(r.message)

    def test_duration_is_honoured_by_the_sweep_voices(self):
        for voice in ("arc", "pulsar"):
            r = render_view(_view(), PulseConfig(duration_s=3.0, voice=voice),
                            modal_frequencies=[2.0])
            self.assertAlmostEqual(r.duration_s, 3.0, places=6, msg=voice)
            self.assertAlmostEqual(r.audio.shape[0] / r.sample_rate, 3.0,
                                   delta=0.01, msg=voice)


class TestUpicMapping(unittest.TestCase):
    """x is time, y is frequency — the thing the whole tab rests on."""

    def test_playhead_track_spans_the_plot_exactly(self):
        v = _view()
        x0, x1 = v.x_range()
        r = render_view(v, PulseConfig(duration_s=2.0))
        self.assertAlmostEqual(r.x_at(0.0), x0, places=4)
        self.assertAlmostEqual(r.x_at(r.duration_s), x1, places=3)
        self.assertAlmostEqual(r.x_at(r.duration_s / 2), (x0 + x1) / 2, places=2)

    def test_a_higher_curve_sounds_higher(self):
        """The mapping must be monotonic, or the plot lies about the sound."""
        cfg = PulseConfig(duration_s=1.0, bells=False, harmonics=1)
        x = np.linspace(0, 10, 50)
        centroids = []
        for level in (0.0, 0.5, 1.0):
            v = PulseView(name="f", kind="custom",
                             curves=[Curve(x, np.full_like(x, level))])
            # a flat curve at three heights, against a common 0..1 reference
            v.curves[0].y[0], v.curves[0].y[-1] = 0.0, 1.0
            v.curves[0].y[1:-1] = level
            r = render_view(v, cfg)
            mono = r.audio[:, 0]
            spec = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
            f = np.fft.rfftfreq(mono.size, 1 / r.sample_rate)
            centroids.append(float((spec * f).sum() / max(spec.sum(), 1e-12)))
        self.assertLess(centroids[0], centroids[1])
        self.assertLess(centroids[1], centroids[2])

    def test_flat_curve_lands_inside_the_configured_band(self):
        cfg = PulseConfig(duration_s=1.0, f_lo=200.0, f_hi=1600.0,
                             bells=False, harmonics=1, glide=0.0)
        x = np.linspace(0, 1, 100)
        v = PulseView(name="f", kind="custom", curves=[Curve(x, np.sin(2 * np.pi * x))])
        r = render_view(v, cfg)
        mono = r.audio[:, 0]
        spec = np.abs(np.fft.rfft(mono))
        f = np.fft.rfftfreq(mono.size, 1 / r.sample_rate)
        peak = float(f[int(np.argmax(spec))])
        self.assertGreaterEqual(peak, cfg.f_lo * 0.8)
        self.assertLessEqual(peak, cfg.f_hi * 1.2)


class TestPulsarVoice(unittest.TestCase):
    """The building's own rate must be audible AS a rate."""

    def test_pulse_rate_equals_the_eigenfrequency(self):
        for fp in (2.0, 5.0):
            cfg = PulseConfig(duration_s=4.0, voice="pulsar", bells=False,
                                 pulsar_fundamental_hz=fp)
            r = render_view(_view(amp_follows=False), cfg)
            env = np.abs(r.audio[:, 0])
            # envelope modulation spectrum: the peak is the repetition rate
            env = env - env.mean()
            spec = np.abs(np.fft.rfft(env))
            f = np.fft.rfftfreq(env.size, 1 / r.sample_rate)
            band = (f > 0.5) & (f < 30.0)
            got = float(f[band][int(np.argmax(spec[band]))])
            self.assertAlmostEqual(got, fp, delta=0.4, msg=f"fp={fp}")

    def test_falls_back_to_mode_one_when_unset(self):
        cfg = PulseConfig(duration_s=3.0, voice="pulsar", bells=False)
        r = render_view(_view(amp_follows=False), cfg, modal_frequencies=[3.0])
        env = np.abs(r.audio[:, 0])
        env = env - env.mean()
        spec = np.abs(np.fft.rfft(env))
        f = np.fft.rfftfreq(env.size, 1 / r.sample_rate)
        band = (f > 0.5) & (f < 30.0)
        self.assertAlmostEqual(float(f[band][int(np.argmax(spec[band]))]), 3.0,
                               delta=0.5)


class TestBells(unittest.TestCase):
    def test_a_bell_fires_at_each_marker_in_plot_order(self):
        marks = [Marker(x=2.0, label="f1", mode=0, timbre="bell"),
                 Marker(x=6.0, label="f2", mode=1, timbre="organ"),
                 Marker(x=9.0, label="f3", mode=2, timbre="triangle")]
        v = _view(markers=marks)
        x0, x1 = v.x_range()
        r = render_view(v, PulseConfig(duration_s=10.0, bells=True))
        self.assertEqual([s[1] for s in r.strikes], ["f1", "f2", "f3"])
        for mk, (t_hit, _lab) in zip(marks, r.strikes):
            expect = (mk.x - x0) / (x1 - x0) * 10.0
            self.assertAlmostEqual(t_hit, expect, places=4)

    def test_bells_can_be_switched_off(self):
        v = _view(markers=[Marker(x=6.0, label="f1")])
        r = render_view(v, PulseConfig(duration_s=4.0, bells=False))
        self.assertEqual(r.strikes, [])

    def test_timbre_off_skips_that_mode_only(self):
        v = _view(markers=[Marker(x=2.0, label="f1", timbre="off"),
                           Marker(x=6.0, label="f2", timbre="bell")])
        r = render_view(v, PulseConfig(duration_s=4.0))
        self.assertEqual([s[1] for s in r.strikes], ["f2"])

    def test_every_timbre_renders_and_decays(self):
        for name in BELL_TIMBRES:
            if name == "off":
                continue
            note = strike(440.0, name, decay_s=1.0)
            self.assertGreater(note.size, 0, name)
            self.assertTrue(np.isfinite(note).all(), name)
            head = float(np.abs(note[: note.size // 8]).max())
            tail = float(np.abs(note[-note.size // 8:]).max())
            self.assertGreater(head, tail, f"{name} must decay")

    def test_a_marker_outside_the_plot_is_ignored(self):
        v = _view(markers=[Marker(x=999.0, label="off-plot")])
        r = render_view(v, PulseConfig(duration_s=2.0))
        self.assertEqual(r.strikes, [])


class TestConfig(unittest.TestCase):
    def test_clamped_returns_a_copy_and_bounds_values(self):
        cfg = PulseConfig(master=9.0, f_lo=1.0, f_hi=1.0, duration_s=1e6,
                             voice="nonsense")
        c = cfg.clamped()
        self.assertIsNot(c, cfg)
        self.assertLessEqual(c.master, 1.0)
        self.assertGreater(c.f_hi, c.f_lo)
        self.assertLessEqual(c.duration_s, 600.0)
        self.assertIn(c.voice, VOICE_MODES)
        self.assertEqual(cfg.master, 9.0)          # original untouched

    def test_engine_imports_without_qt(self):
        import sys
        mods = [m for m in sys.modules
                if m.startswith("sensepi.sonification.structure_pulse")]
        self.assertTrue(mods)
        for m in mods:
            src = getattr(sys.modules[m], "__file__", "") or ""
            if not src.endswith(".py"):
                continue
            with open(src, encoding="utf-8") as fh:
                text = fh.read()
            self.assertNotIn("PySide6", text, m)


if __name__ == "__main__":
    unittest.main()


class TestPersistence(unittest.TestCase):
    """A saved session must replay exactly, and re-analyse from its raw half."""

    def _dataset(self):
        from sensepi.sonification.structure_pulse import build_dataset_from_arrays

        rng = np.random.default_rng(1)
        t = np.arange(0, 30, 1 / FS)
        rows = [sum(np.sin(2 * np.pi * f * t) * a
                    for f, a in ((2.0, 0.05 * (k + 1)), (6.0, 0.02)))
                + rng.standard_normal(t.size) * 1e-3 for k in range(3)]
        chans = {"ax": np.vstack(rows), "gz": np.vstack(rows) * 0.1}
        return build_dataset_from_arrays(chans, fs=FS, sensor_ids=[1, 2, 3],
                                         source="synthetic")

    def test_round_trip_preserves_views_curves_and_modes(self):
        import shutil
        import tempfile

        from sensepi.sonification.structure_pulse import (load_dataset,
                                                          save_dataset)
        ds = self._dataset()
        tmp = tempfile.mkdtemp()
        try:
            folder = save_dataset(ds, tmp, name="unit test")
            back = load_dataset(folder)
            self.assertEqual(ds.view_names(), back.view_names())
            np.testing.assert_allclose(ds.modal.frequencies_hz,
                                       back.modal.frequencies_hz, rtol=1e-6)
            self.assertEqual(ds.sensor_ids, back.sensor_ids)
            for name in ds.view_names():
                a, b = ds.views[name], back.views[name]
                self.assertEqual(a.x_label, b.x_label, name)
                self.assertEqual(a.x_log, b.x_log, name)
                self.assertEqual(len(a.curves), len(b.curves), name)
                for ca, cb in zip(a.curves, b.curves):
                    np.testing.assert_allclose(ca.x, cb.x, rtol=1e-5)
                    np.testing.assert_allclose(ca.y, cb.y, rtol=1e-4, atol=1e-9)
                self.assertEqual([(m.label, m.mode, m.timbre) for m in a.markers],
                                 [(m.label, m.mode, m.timbre) for m in b.markers],
                                 name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_reloaded_session_still_plays(self):
        import shutil
        import tempfile

        from sensepi.sonification.structure_pulse import (load_dataset,
                                                          save_dataset)
        ds = self._dataset()
        tmp = tempfile.mkdtemp()
        try:
            back = load_dataset(save_dataset(ds, tmp))
            name = "Spectrum" if "Spectrum" in back.view_names() else back.view_names()[0]
            r = render_view(back.views[name], PulseConfig(duration_s=2.0),
                            modal_frequencies=back.modal.frequencies_hz,
                            data_fs=back.fs)
            self.assertGreater(r.audio.size, 0)
            self.assertTrue(np.isfinite(r.audio).all())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_raw_half_is_kept_and_enables_reanalysis(self):
        import shutil
        import tempfile

        from sensepi.sonification.structure_pulse import reanalyse, save_dataset
        ds = self._dataset()
        self.assertTrue(ds.has_raw)
        tmp = tempfile.mkdtemp()
        try:
            folder = save_dataset(ds, tmp)
            again = reanalyse(folder, n_modes=2)
            self.assertTrue(again.modal.ok, again.modal.message)
            self.assertLessEqual(again.modal.frequencies_hz.size, 2)
            self.assertTrue(again.view_names())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_without_raw_it_replays_but_refuses_to_reanalyse(self):
        import shutil
        import tempfile

        from sensepi.sonification.structure_pulse import (load_dataset, reanalyse,
                                                          save_dataset)
        ds = self._dataset()
        tmp = tempfile.mkdtemp()
        try:
            folder = save_dataset(ds, tmp, include_raw=False)
            back = load_dataset(folder)
            self.assertFalse(back.has_raw)
            self.assertTrue(back.view_names())          # still replayable
            with self.assertRaises(ValueError):
                reanalyse(folder)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_listing_and_describing_saved_sessions(self):
        import shutil
        import tempfile

        from sensepi.sonification.structure_pulse import (describe, list_sessions,
                                                          save_dataset)
        tmp = tempfile.mkdtemp()
        try:
            self.assertEqual(list_sessions(tmp), [])
            folder = save_dataset(self._dataset(), tmp, name="my run")
            found = list_sessions(tmp)
            self.assertEqual([p.name for p in found], [folder.name])
            text = describe(folder)
            self.assertIn("my run", text)
            self.assertIn("raw", text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rejects_a_folder_that_is_not_a_session(self):
        import shutil
        import tempfile

        from sensepi.sonification.structure_pulse import load_dataset
        tmp = tempfile.mkdtemp()
        try:
            with self.assertRaises(Exception):
                load_dataset(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSweepFollowsTheAxis(unittest.TestCase):
    """The playhead must cross the SCREEN evenly, whatever the axis."""

    def _log_view(self):
        x = np.geomspace(0.02, 2.0, 200)
        return PulseView(name="rs", kind="response", x_log=True,
                         curves=[Curve(x, np.ones_like(x))])

    def test_a_log_axis_sweeps_logarithmically(self):
        v = self._log_view()
        r = render_view(v, PulseConfig(duration_s=10.0, bells=False))
        x0, x1 = v.x_range()
        span = np.log(x1 / x0)
        for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
            x = r.x_at(frac * r.duration_s)
            screen = float(np.log(x / x0) / span)
            self.assertAlmostEqual(screen, frac, places=3,
                                   msg=f"at t={frac} the line sits at {screen}")

    def test_a_linear_axis_still_sweeps_linearly(self):
        x = np.linspace(0.0, 60.0, 200)
        v = PulseView(name="t", kind="time", curves=[Curve(x, np.ones_like(x))])
        r = render_view(v, PulseConfig(duration_s=6.0, bells=False))
        x0, x1 = v.x_range()
        for frac in (0.25, 0.5, 0.75):
            got = (r.x_at(frac * r.duration_s) - x0) / (x1 - x0)
            self.assertAlmostEqual(got, frac, places=3)

    def test_markers_ring_where_the_line_actually_is(self):
        v = self._log_view()
        v.markers = [Marker(x=0.2, label="f1", mode=0, timbre="bell")]
        r = render_view(v, PulseConfig(duration_s=10.0, bells=True))
        self.assertEqual(len(r.strikes), 1)
        x0, x1 = v.x_range()
        expect = float(np.log(0.2 / x0) / np.log(x1 / x0)) * 10.0
        self.assertAlmostEqual(r.strikes[0][0], expect, places=3)


class TestCurveShaping(unittest.TestCase):
    """A record swings through zero; the pitch must not swing with it."""

    def _record(self, n=4000):
        t = np.linspace(0, 20, n)
        return t, np.sin(2 * np.pi * 2.0 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.1 * t))

    def test_envelope_is_far_smoother_than_the_raw_value(self):
        from sensepi.sonification.structure_pulse.render import _norm01, _shape

        _t, y = self._record()
        yy = np.interp(np.linspace(0, 1, 44100), np.linspace(0, 1, y.size), y)
        jit = {}
        for mode in ("signed", "envelope"):
            n = _norm01(_shape(yy, mode, 44100, 1.0, 0.02, "time"))
            jit[mode] = float(np.mean(np.abs(np.diff(n))))
        self.assertLess(jit["envelope"], jit["signed"] / 5.0,
                        f"envelope must settle the pitch: {jit}")

    def test_auto_uses_the_contour_for_a_time_view(self):
        from sensepi.sonification.structure_pulse.render import _shape

        _t, y = self._record()
        self.assertFalse(np.allclose(_shape(y, "auto", 44100, 1.0, 0.02, "time"), y))

    def test_auto_leaves_a_mode_shape_signed(self):
        """A mode shape's sign is the node; rectifying it would erase it."""
        from sensepi.sonification.structure_pulse.render import _shape

        shape = np.array([0.0, 1.0, 0.23, -0.81, -0.78])
        out = _shape(shape, "auto", 44100, 1.0, 0.02, "shape")
        np.testing.assert_allclose(out, shape)
        np.testing.assert_array_equal(np.sign(out), np.sign(shape))

    def test_auto_leaves_a_spectrum_signed(self):
        from sensepi.sonification.structure_pulse.render import _shape

        x = np.linspace(0, 1, 200)
        y = np.exp(-((x - 0.5) ** 2) / 0.01)
        np.testing.assert_allclose(_shape(y, "auto", 44100, 1.0, 0.02, "spectrum"), y)

    def test_a_dc_biased_gyro_counts_as_oscillating(self):
        """gz rests near -1.65 deg/s and never changes sign, but it oscillates."""
        from sensepi.sonification.structure_pulse.render import _is_oscillating

        t = np.linspace(0, 10, 2000)
        gz = -1.65 + 0.07 * np.sin(2 * np.pi * 2.0 * t)
        self.assertTrue(np.all(gz < 0))
        self.assertTrue(_is_oscillating(gz))

    def test_every_scale_mode_renders(self):
        from sensepi.sonification.structure_pulse import SCALE_MODES

        t, y = self._record(400)
        v = PulseView(name="t", kind="time", curves=[Curve(t, y)])
        for mode in ("auto",) + tuple(SCALE_MODES):
            cfg = PulseConfig(duration_s=1.0, bells=False)
            cfg.scale = mode
            r = render_view(v, cfg, modal_frequencies=[2.0], data_fs=100.0)
            self.assertGreater(r.audio.size, 0, mode)
            self.assertTrue(np.isfinite(r.audio).all(), mode)

    def test_smoothing_is_linear_in_the_signal_length(self):
        """The envelope must not depend on the WINDOW size for its cost.

        A windowed convolution is O(n*m), and the adaptive window is a fraction
        of the sweep: at 20 s and 44.1 kHz it reached 441 000 samples, which
        never finished and hung the live worker inside numpy.
        """
        import time

        from sensepi.sonification.structure_pulse.render import _shape

        y = np.sin(2 * np.pi * 40 * np.linspace(0, 1, 44100 * 10))
        t0 = time.time()
        out = _shape(y, "envelope", 44100, 10.0, 0.4, "time")   # a huge window
        elapsed = time.time() - t0
        self.assertEqual(out.size, y.size)
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(elapsed, 2.0, f"smoothing took {elapsed:.1f}s")

    def test_box_smoothing_preserves_length_and_range(self):
        from sensepi.sonification.structure_pulse.render import _box2

        for n in (3, 101, 5001):
            y = np.abs(np.sin(np.linspace(0, 20, 20000)))
            out = _box2(y, n)
            self.assertEqual(out.size, y.size, n)
            self.assertTrue(np.isfinite(out).all(), n)
            self.assertGreaterEqual(float(out.min()), -1e-9, n)
            self.assertLessEqual(float(out.max()), float(y.max()) + 1e-9, n)
