"""Base-referenced (input-output) identification.

The point of this method is not that it is another way to find peaks — it is
that it can tell a mode from a peak in the shaker's own drive, which an
output-only method structurally cannot. These tests are built around a 3-DOF
shear frame with known modes so that claim can be checked rather than asserted.
"""
import unittest

import numpy as np
from scipy import signal

from sensepi.analysis import modal as modal_id
from sensepi.analysis import transmissibility as tr

FS = 100.0
N = 6000                       # 60 s
K = 400.0 * np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 1]], float)
_W2, _PHI = np.linalg.eigh(K)
TRUE_HZ = np.sqrt(_W2) / (2 * np.pi)          # ~1.417, 3.969, 5.736


def _bandnoise(rng, lo, hi, n=N):
    sos = signal.butter(2, [lo, hi], btype="band", fs=FS, output="sos")
    return signal.sosfilt(sos, rng.normal(0, 1, n))


def _simulate(seed=7, zeta=0.02, drive_tone=0.0, ambient=0.0):
    """Base-driven 3-DOF frame. Returns (responses, base)."""
    rng = np.random.default_rng(seed)
    base = _bandnoise(rng, 0.4, 20.0)
    if drive_tone:                     # a peak in the DRIVE, not the structure
        a = _bandnoise(rng, 4.8, 5.2)
        base = base + drive_tone * a / (np.std(a) + 1e-12) * np.std(base)
    resp = np.zeros((3, N))
    for wn, phi in zip(np.sqrt(_W2), _PHI.T):
        b, a = signal.bilinear([1.0], [1.0, 2 * zeta * wn, wn ** 2], FS)
        resp += np.outer(phi, -wn ** 2 * signal.lfilter(b, a, base))
    resp = resp + base
    if ambient:                        # energy in the RESPONSES only
        amb = _bandnoise(rng, 8.8, 9.2)
        resp = resp + ambient * amb / (np.std(amb) + 1e-12) * np.std(resp)
    resp = resp + rng.normal(0, 0.02 * np.std(resp), resp.shape)
    return resp, base


def _matches(got, want, tol=0.15):
    return any(abs(g - want) <= tol for g in got)


def _mac(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(abs(a @ b) ** 2 / ((a @ a) * (b @ b)))


class TestFindsTheRealModes(unittest.TestCase):
    def test_identifies_all_three_known_frequencies(self):
        resp, base = _simulate()
        br = tr.identify_modes_base_referenced(
            resp, base, FS, n_modes=3, f_min=0.5, f_max=20.0)
        self.assertTrue(br.success, br.message)
        got = br.modal.frequencies_hz
        self.assertEqual(len(got), 3)
        for want in TRUE_HZ:
            self.assertTrue(_matches(got, want), f"missed {want:.3f} Hz in {got}")

    def test_mode_shapes_match_the_eigenvectors(self):
        resp, base = _simulate()
        br = tr.identify_modes_base_referenced(
            resp, base, FS, n_modes=3, f_min=0.5, f_max=20.0)
        for k, shape in enumerate(br.modal.mode_shapes_sensor):
            self.assertGreater(_mac(shape, _PHI[:, k]), 0.9,
                               f"mode {k + 1} shape {shape} vs {_PHI[:, k]}")


class TestRejectsWhatIsNotAMode(unittest.TestCase):
    """The reason this method exists."""

    def test_a_peak_in_the_drive_is_not_reported_as_a_mode(self):
        resp, base = _simulate(drive_tone=6.0)      # strong 5.0 Hz drive artefact
        br = tr.identify_modes_base_referenced(
            resp, base, FS, n_modes=4, f_min=0.5, f_max=20.0)
        self.assertTrue(br.success, br.message)
        # Nothing within 0.3 Hz of the artefact, which sits between modes 2 and 3.
        for f in br.modal.frequencies_hz:
            self.assertGreater(abs(f - 5.0), 0.3,
                               f"reported the drive artefact at {f:.3f} Hz")

    def test_output_only_is_fooled_by_the_same_drive(self):
        # Not a test of our code — it pins WHY the base-referenced path is needed.
        # If this ever starts passing, the extra method has lost its purpose.
        resp, _ = _simulate(drive_tone=6.0)
        res = modal_id.identify_modes(
            resp, FS, method="fdd", n_modes=4, f_min=0.5, f_max=20.0)
        self.assertTrue(any(abs(f - 5.0) < 0.3 for f in res.frequencies_hz),
                        f"FDD no longer picks the artefact: {res.frequencies_hz}")

    def test_ambient_energy_loses_the_prominence_ranking(self):
        """Not the coherence gate's doing — the H1 estimator's.

        Energy uncorrelated with the reference averages toward zero in the
        cross-spectrum, so an ambient peak comes out weak and is not among the
        strongest. The coherence gate does NOT reliably catch this case (see
        DEFAULT_MIN_COHERENCE), which is why the assertion is about ranking.
        """
        resp, base = _simulate(ambient=5.0)         # 9 Hz in the responses only
        br = tr.identify_modes_base_referenced(
            resp, base, FS, n_modes=3, f_min=0.5, f_max=20.0)
        self.assertTrue(br.success, br.message)
        for f in br.modal.frequencies_hz:
            self.assertGreater(abs(f - 9.0), 0.5,
                               f"ambient peak was reported as a mode: {f:.3f} Hz")

    def test_rejected_peaks_carry_their_coherence(self):
        resp, base = _simulate(ambient=5.0)
        br = tr.identify_modes_base_referenced(resp, base, FS, n_modes=3)
        for _f, coh in br.rejected:
            self.assertLess(coh, tr.DEFAULT_MIN_COHERENCE)


class TestCoherenceIsMeaningful(unittest.TestCase):
    def test_spectra_are_averaged_so_coherence_is_not_identically_one(self):
        resp, base = _simulate()
        _f, _h, coh, n_seg, _np = tr.estimate_transmissibility(resp, base, FS)
        self.assertGreaterEqual(n_seg, tr.MIN_SEGMENTS)
        self.assertLess(float(np.mean(coh)), 0.999,
                        "coherence is degenerate — the spectra were not averaged")

    def test_driven_modes_have_high_coherence(self):
        resp, base = _simulate()
        br = tr.identify_modes_base_referenced(resp, base, FS, n_modes=3)
        self.assertTrue(all(c >= tr.DEFAULT_MIN_COHERENCE for c in br.mode_coherence))

    def test_an_unrelated_reference_identifies_nothing(self):
        """The failure the gate exists for: a mismapped or dead base sensor."""
        resp, _base = _simulate()
        noise = np.random.default_rng(1).normal(0, 1, N)
        br = tr.identify_modes_base_referenced(resp, noise, FS, n_modes=3)
        self.assertFalse(br.success)
        self.assertIn("coherence", br.message)

    def test_coherence_is_read_beside_the_peak_not_at_it(self):
        """A lightly damped mode dips at its own peak bin (estimator bias)."""
        resp, base = _simulate()
        br = tr.identify_modes_base_referenced(resp, base, FS, n_modes=3)
        f, _h, coh, _n, _p = tr.estimate_transmissibility(resp, base, FS)
        first = br.modal.frequencies_hz[0]
        at_peak = float(coh[int(np.argmin(abs(f - first)))])
        self.assertLess(at_peak, br.mode_coherence[0],
                        "neighbourhood reading gave no improvement over the peak bin")
        self.assertGreaterEqual(br.mode_coherence[0], tr.DEFAULT_MIN_COHERENCE)

    def test_a_short_record_still_finds_the_modes_and_says_where_it_can_judge(self):
        """Coarse resolution narrows where the full threshold applies.

        A 20 s slice resolves modes 2 and 3 but not mode 1, so mode 1 is judged
        at the relaxed floor rather than being either trusted or silently
        dropped. All three must still be found.
        """
        resp, base = _simulate()
        short = tr.identify_modes_base_referenced(
            resp[:, :2000], base[:2000], FS, n_modes=3, min_duration=10.0)
        self.assertTrue(short.success, short.message)
        self.assertEqual(len(short.modal.frequencies_hz), 3)
        self.assertIn("gate applies above", short.message)

    def test_gate_is_enforced_on_a_long_record(self):
        resp, base = _simulate()
        br = tr.identify_modes_base_referenced(resp, base, FS, n_modes=3)
        self.assertTrue(br.gate_enforced)
        self.assertNotIn("gate applies above", br.message)


class TestSegmentation(unittest.TestCase):
    def test_nperseg_buys_the_required_number_of_averages(self):
        for n in (1200, 2000, 6000, 30000):
            nperseg = tr._choose_frf_nperseg(n)
            self.assertGreaterEqual(tr._segment_count(n, nperseg), tr.MIN_SEGMENTS)

    def test_true_resolution_is_reported_not_the_padded_grid(self):
        resp, base = _simulate()
        br = tr.identify_modes_base_referenced(resp, base, FS, n_modes=3)
        # Zero-padding makes the grid finer than the record can justify; both
        # numbers are reported so neither is mistaken for the other.
        self.assertLess(br.freq_resolution_hz, br.true_resolution_hz)


class TestDamping(unittest.TestCase):
    def test_damping_is_never_claimed(self):
        # Half-power bandwidth was measured against known damping and was wrong
        # across the whole range (DEBT-8), so this path reports NaN by design.
        resp, base = _simulate()
        br = tr.identify_modes_base_referenced(resp, base, FS, n_modes=3)
        self.assertTrue(all(np.isnan(z) for z in br.modal.damping_ratios))


class TestRefusesWhatItCannotDo(unittest.TestCase):
    def test_short_record_is_refused(self):
        resp, base = _simulate()
        br = tr.identify_modes_base_referenced(resp[:, :300], base[:300], FS)
        self.assertFalse(br.success)
        self.assertIn("too short", br.message)

    def test_a_dead_base_sensor_is_named_as_the_problem(self):
        resp, _ = _simulate()
        br = tr.identify_modes_base_referenced(resp, np.zeros(N), FS)
        self.assertFalse(br.success)
        self.assertIn("not moving", br.message)

    def test_mismatched_lengths_are_refused(self):
        resp, base = _simulate()
        br = tr.identify_modes_base_referenced(resp, base[:-10], FS)
        self.assertFalse(br.success)
        self.assertIn("length", br.message)

    def test_bad_sampling_rate_is_refused(self):
        resp, base = _simulate()
        self.assertFalse(tr.identify_modes_base_referenced(resp, base, 0.0).success)

    def test_result_is_the_shared_type_so_map_to_stories_accepts_it(self):
        resp, base = _simulate()
        br = tr.identify_modes_base_referenced(resp, base, FS, n_modes=3)
        story = modal_id.map_to_stories(br.modal, [1, 2, 3], 3)
        self.assertEqual(story.coverage_stories, [1, 2, 3])
        self.assertTrue(story.mode_shapes_available)


if __name__ == "__main__":
    unittest.main()
