"""Tests for the Gaussian Bayesian calibration engine (PHASE 3 / A1, A2, MU-4).

The pure math (prior build, augmented residual, MAP + Laplace) imports only
numpy/scipy and runs everywhere. The MAP-vs-deterministic validation (T3.6) needs
the calibrator forward model and is skipped where openseespy is unavailable
(canonical env: senspi_mpu).
"""

import unittest

import numpy as np

from opensees_model_updating.calibration import bayesian

try:  # calibrator import requires openseespy (transitively)
    import openseespy.opensees  # noqa: F401
    _HAS_OPENSEES = True
except Exception:  # pragma: no cover - env-dependent
    _HAS_OPENSEES = False


class TestBayesianPureMath(unittest.TestCase):
    """No OpenSees: prior, augmented residual, MAP, Laplace covariance."""

    def test_build_prior(self):
        prior = bayesian.build_prior({"nStory": 3})
        self.assertEqual(prior.mean.shape, (4,))      # E + 3 masses
        self.assertEqual(prior.cov.shape, (4, 4))
        self.assertTrue(np.allclose(prior.mean, 1.0))
        self.assertTrue(np.allclose(prior.sigma, [0.30, 0.15, 0.15, 0.15]))
        # cov is diagonal sigma^2
        self.assertTrue(np.allclose(np.diag(prior.cov), prior.sigma ** 2))

    def test_augmented_residual_length(self):
        prior = bayesian.build_prior({"nStory": 3})       # length 4

        def res(theta):
            return np.zeros(5)                            # synthetic 5-term data block

        r = bayesian.augmented_residual(np.ones(4), res, prior)
        self.assertEqual(r.size, 5 + 4)                  # data + (1 + nStory)

    def test_map_matches_analytic_linear(self):
        # Linear gaussian: 3 measurements, 4 params (rank-deficient data) -> the
        # classic underdetermined case. MAP must match the closed form, and the
        # data-unconstrained 4th param must fall back to its prior variance.
        M = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0]])                    # 3x4, rank 3
        theta_true = np.array([1.1, 0.9, 1.05, 1.2])
        y = M @ theta_true
        prior = bayesian.GaussianPrior(
            mean=np.array([1.0, 1.0, 1.0, 1.0]),
            cov=np.diag(np.array([0.30, 0.15, 0.15, 0.15]) ** 2),
        )

        def res(theta):
            return M @ theta - y

        out = bayesian.gaussian_map_update(res, x0=np.ones(4), prior=prior, bounds=None)

        D = np.diag(1.0 / prior.sigma ** 2)
        H = M.T @ M + D
        closed_mean = np.linalg.solve(H, M.T @ y + D @ prior.mean)
        self.assertTrue(np.allclose(out.mean, closed_mean, atol=1e-6))
        self.assertTrue(np.allclose(out.cov, np.linalg.inv(H), atol=1e-6))
        # 4th param is unconstrained by the data -> posterior var == prior var.
        self.assertAlmostEqual(out.cov[3, 3], prior.sigma[3] ** 2, places=6)

    def test_freq_only_covariance_finite_and_psd(self):
        # Data alone is rank-deficient; the prior block keeps Sigma_post finite + PSD.
        M = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0]])
        prior = bayesian.build_prior({"nStory": 3})

        def res(theta):
            return M @ theta - np.array([2.0, 6.0, 9.0])

        out = bayesian.gaussian_map_update(res, x0=np.ones(4), prior=prior, bounds=None)
        self.assertTrue(np.all(np.isfinite(out.cov)))
        eig = np.linalg.eigvalsh(out.cov)
        self.assertTrue(np.all(eig > 0.0))               # positive-definite
        # bands are mean +/- z*sigma
        lo, hi = out.bands_95
        self.assertTrue(np.allclose(hi - lo, 2 * 1.959963984540054 * out.sigma))


class TestMassRegularization(unittest.TestCase):
    """R1 (mass-similarity) reduces non-physical spread but allows genuine changes."""

    def test_residual_zero_when_uniform(self):
        r = bayesian.mass_similarity_residual([1.0, 1.1, 1.1, 1.1], n_story=3, weight=0.5)
        self.assertTrue(np.allclose(r, 0.0))                  # uniform masses -> no penalty
        r2 = bayesian.mass_similarity_residual([1.0, 1.3, 1.0, 0.7], n_story=3, weight=0.5)
        self.assertGreater(np.linalg.norm(r2), 0.0)          # spread -> penalty
        self.assertEqual(r2.size, 3)

    def _run(self, target, alpha, w_sim):
        # Weak prior so the data + R1 dominate; isolates the R1 effect.
        prior = bayesian.build_prior({"nStory": 3}, sigma_E=1e3, sigma_m=1e3)
        tgt = np.asarray([1.0] + list(target), dtype=float)   # [E_target, m_targets...]

        def res(theta):
            data = alpha * (np.asarray(theta, float) - tgt)
            reg = bayesian.mass_similarity_residual(theta, 3, w_sim)
            return np.concatenate([data, reg])

        out = bayesian.gaussian_map_update(res, x0=np.ones(4), prior=prior, bounds=None)
        return out.mean[1:]                                   # the three mass scales

    def test_R1_reduces_spurious_spread(self):
        # Weak data pull toward a spread distribution; R1 should shrink the spread.
        masses_no_r1 = self._run(target=[1.2, 1.0, 0.8], alpha=0.3, w_sim=0.0)
        masses_r1 = self._run(target=[1.2, 1.0, 0.8], alpha=0.3, w_sim=0.5)
        self.assertLess(np.std(masses_r1), np.std(masses_no_r1))

    def test_R1_does_not_block_genuine_change(self):
        # Strong, single-floor data evidence; soft R1 must NOT suppress it (Baban).
        masses = self._run(target=[1.0, 1.5, 1.0], alpha=5.0, w_sim=0.5)
        self.assertGreater(masses[1], 1.25)                  # floor-2 change survives
        self.assertGreater(masses[1] - masses[0], 0.2)      # still clearly distinct


class TestPerModeSigma(unittest.TestCase):
    """CU-4: per-mode sigma down-weights a noisy mode (pure scaling + band effect)."""

    def test_scale_residual_per_mode_blocks(self):
        # Layout: 2 freq terms + 2 modes * 2 measured-DOF shape terms = 6 entries.
        exp = {"n_modes_used": 2, "use_mode_shapes": True,
               "measured_dof_indices": [0, 1], "nStory": 2}
        r = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
        sig = np.array([0.10, 0.50])           # mode-2 is 5x noisier
        out = bayesian._scale_residual_per_mode(r, exp, sig, sigma_floor=0.02)
        # freq block: divided by per-mode sigma.
        self.assertAlmostEqual(out[0], 1.0 / 0.10)
        self.assertAlmostEqual(out[1], 1.0 / 0.50)
        # shape block: mode-1 dofs /0.10, mode-2 dofs /0.50.
        self.assertAlmostEqual(out[2], 1.0 / 0.10)
        self.assertAlmostEqual(out[3], 1.0 / 0.10)
        self.assertAlmostEqual(out[4], 1.0 / 0.50)
        self.assertAlmostEqual(out[5], 1.0 / 0.50)

    def test_sigma_floor_applied(self):
        exp = {"n_modes_used": 2, "use_mode_shapes": False}
        r = np.array([1.0, 1.0])
        # First mode sigma below the floor -> clamped to the floor (no over-confidence).
        out = bayesian._scale_residual_per_mode(r, exp, np.array([0.001, 0.20]), sigma_floor=0.02)
        self.assertAlmostEqual(out[0], 1.0 / 0.02)   # floored
        self.assertAlmostEqual(out[1], 1.0 / 0.20)

    def test_nan_or_zero_sigma_falls_back_to_floor(self):
        exp = {"n_modes_used": 2, "use_mode_shapes": False}
        r = np.array([1.0, 1.0])
        out = bayesian._scale_residual_per_mode(r, exp, np.array([np.nan, 0.0]), sigma_floor=0.05)
        self.assertAlmostEqual(out[0], 1.0 / 0.05)
        self.assertAlmostEqual(out[1], 1.0 / 0.05)


class TestRecursivePrior(unittest.TestCase):
    """C1: recursive prior propagation — track narrows, soft anchor limits drift."""

    def test_recursive_track_narrows(self):
        # Well-determined synthetic data folded over cycles -> posterior covariance
        # shrinks vs the single-cycle one-shot (evidence accumulates).
        M = np.eye(3)
        y = M @ np.array([1.1, 0.9, 1.05])

        def res(theta):
            return M @ theta - y

        init_prior = bayesian.build_prior({"nStory": 2})       # 3 params
        anchor = bayesian.build_anchor_prior({"nStory": 2})
        one_shot = bayesian.gaussian_map_update(res, np.ones(3), init_prior, bounds=None)

        post_mean, post_cov = one_shot.mean, one_shot.cov
        for _ in range(3):
            pri = bayesian.propagate_recursive_prior(post_mean, post_cov, anchor)
            out = bayesian.gaussian_map_update(res, post_mean, pri, bounds=None)
            post_mean, post_cov = out.mean, out.cov

        self.assertLess(np.trace(post_cov), np.trace(one_shot.cov))

    def test_soft_anchor_limits_drift(self):
        # A drifted posterior is pulled back toward the initial params (1.0) by the
        # soft anchor when forming the next prior.
        anchor = bayesian.build_anchor_prior({"nStory": 2})
        post_mean = np.array([1.5, 0.6, 1.4])
        post_cov = np.diag([0.05, 0.05, 0.05])
        pri = bayesian.propagate_recursive_prior(post_mean, post_cov, anchor)
        self.assertTrue(np.all(np.abs(pri.mean - 1.0) <= np.abs(post_mean - 1.0) + 1e-9))
        self.assertTrue(np.any(np.abs(pri.mean - 1.0) < np.abs(post_mean - 1.0)))


def _synthetic_freqs(x):
    # Diagonal, invertible forward model -> a UNIQUE data minimizer so MAP (weak
    # prior) and least-squares must converge to the same point. nStory=2 ->
    # x = [E, m1, m2]; 3 "frequencies".
    b = np.array([2.0, 6.0, 9.0])
    return list(b * np.asarray(x, dtype=float))


@unittest.skipUnless(_HAS_OPENSEES, "calibrator import requires openseespy (use senspi_mpu)")
class TestMapMatchesLeastSquares(unittest.TestCase):
    """T3.6 / MU-4: with a weak prior the MAP mean ~= deterministic result.x."""

    def setUp(self):
        from opensees_model_updating.calibration import calibrator
        self.calibrator = calibrator
        self._orig_apply = calibrator.apply_calibration_vector
        self._orig_extract = calibrator.extract_modal_results
        calibrator.apply_calibration_vector = lambda bp, x: {"x": np.asarray(x, dtype=float)}
        calibrator.extract_modal_results = lambda p, **k: {
            "freqs": _synthetic_freqs(p["x"]),
            "mode_shapes_ux_master": [],
        }

    def tearDown(self):
        self.calibrator.apply_calibration_vector = self._orig_apply
        self.calibrator.extract_modal_results = self._orig_extract

    def _base(self):
        return {
            "nStory": 2,
            "E_scale_lb": 0.5, "E_scale_ub": 2.0,
            "m_scale_lb": 0.5, "m_scale_ub": 2.0,
            "w_freq": 1.0, "w_mode": 0.35, "max_nfev": 400,
        }

    def _exp(self, theta_true):
        return {
            "freqs": np.asarray(_synthetic_freqs(theta_true), dtype=float),
            "modes": [],
            "use_mode_shapes": False,
            "mode_shapes_available": False,
            "n_modes_used": 3,
            "measured_dof_indices": None,
        }

    def test_map_close_to_lsq(self):
        theta_true = np.array([1.1, 0.9, 1.05])  # [E, m1, m2]
        base = self._base()
        exp = self._exp(theta_true)

        lsq_result, _ = self.calibrator.run_calibration(base, exp)

        # Weak (near-flat) prior AND R1 disabled so the data alone drives the fit
        # -> MAP ~= deterministic LSQ. (With the default R1 on, MAP deliberately
        # differs — that's the point of regularization; see TestMassRegularization.)
        weak = bayesian.build_prior(base, sigma_E=1e3, sigma_m=1e3)
        bayes_result, _ = bayesian.run_bayesian_calibration(
            base, exp, prior=weak, sigma_data_scale=1.0, mass_similarity_weight=0.0)

        self.assertTrue(np.allclose(bayes_result.mean, lsq_result.x, atol=1e-3))
        self.assertTrue(np.allclose(bayes_result.mean, theta_true, atol=1e-3))


@unittest.skipUnless(_HAS_OPENSEES, "calibrator import requires openseespy (use senspi_mpu)")
class TestPerModeSigmaEndToEnd(unittest.TestCase):
    """CU-4: a noisy mode (large per-mode sigma) widens the band of its parameter."""

    def setUp(self):
        from opensees_model_updating.calibration import calibrator
        self.calibrator = calibrator
        self._orig_apply = calibrator.apply_calibration_vector
        self._orig_extract = calibrator.extract_modal_results
        calibrator.apply_calibration_vector = lambda bp, x: {"x": np.asarray(x, dtype=float)}
        calibrator.extract_modal_results = lambda p, **k: {
            "freqs": _synthetic_freqs(p["x"]),     # diagonal: freq2 <-> m2
            "mode_shapes_ux_master": [],
        }

    def tearDown(self):
        self.calibrator.apply_calibration_vector = self._orig_apply
        self.calibrator.extract_modal_results = self._orig_extract

    def test_noisy_mode_widens_its_param_band(self):
        base = {"nStory": 2, "E_scale_lb": 0.5, "E_scale_ub": 2.0,
                "m_scale_lb": 0.5, "m_scale_ub": 2.0,
                "w_freq": 1.0, "w_mode": 0.35, "max_nfev": 400}
        exp = {"freqs": np.asarray(_synthetic_freqs([1.0, 1.0, 1.0]), dtype=float),
               "modes": [], "use_mode_shapes": False, "mode_shapes_available": False,
               "n_modes_used": 3, "measured_dof_indices": None}
        prior = bayesian.build_prior(base)

        uniform, _ = bayesian.run_bayesian_calibration(
            base, exp, prior=prior, sigma_data_per_mode=[0.02, 0.02, 0.02],
            mass_similarity_weight=0.0)
        noisy3, _ = bayesian.run_bayesian_calibration(
            base, exp, prior=prior, sigma_data_per_mode=[0.02, 0.02, 0.50],
            mass_similarity_weight=0.0)

        # m2 (index 2, tied to freq-3) gets a wider posterior sigma when mode 3 is noisy.
        self.assertGreater(noisy3.sigma[2], uniform.sigma[2])


@unittest.skipUnless(_HAS_OPENSEES, "calibrator import requires openseespy (use senspi_mpu)")
class TestNStoryGeneralization(unittest.TestCase):
    """NS / GATE-10: the calibration vector sizes to nStory (not a fixed 3)."""

    def setUp(self):
        from opensees_model_updating.calibration import calibrator
        self.calibrator = calibrator
        self._orig_apply = calibrator.apply_calibration_vector
        self._orig_extract = calibrator.extract_modal_results

    def tearDown(self):
        self.calibrator.apply_calibration_vector = self._orig_apply
        self.calibrator.extract_modal_results = self._orig_extract

    def _run(self, n_story):
        cal = self.calibrator
        cal.apply_calibration_vector = lambda bp, x: {
            "E": float(x[0]),
            "floor_masses": [float(m) for m in x[1:1 + n_story]],
        }
        cal.extract_modal_results = lambda p, **k: {
            "freqs": [2.0 + i for i in range(n_story)],
            "mode_shapes_ux_master": [],
        }
        base = {
            "nStory": n_story, "E": 1.0,
            "E_scale_lb": 0.5, "E_scale_ub": 2.0,
            "m_scale_lb": 0.5, "m_scale_ub": 2.0,
            "w_freq": 1.0, "w_mode": 0.35, "max_nfev": 100,
        }
        exp = {
            "freqs": np.array([2.0 + i for i in range(n_story)]),
            "modes": [], "use_mode_shapes": False, "mode_shapes_available": False,
            "n_modes_used": n_story, "measured_dof_indices": None,
        }
        return bayesian.run_bayesian_calibration(
            base, exp, sigma_data_scale=1.0, mass_similarity_weight=0.0)

    def test_two_story(self):
        res, calib = self._run(2)
        self.assertEqual(res.mean.size, 1 + 2)
        self.assertEqual(len(calib["floor_masses"]), 2)

    def test_four_story(self):
        res, calib = self._run(4)
        self.assertEqual(res.mean.size, 1 + 4)
        self.assertEqual(len(calib["floor_masses"]), 4)


if __name__ == "__main__":
    unittest.main()
