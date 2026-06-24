# -*- coding: utf-8 -*-
"""Simple Gaussian Bayesian model updating (MAP + Laplace covariance).

v26.1.1 headline upgrade (Q-B). This is a thin, **additive** wrapper around the
existing deterministic calibration: the data-misfit term is the very same
``modal_residuals`` / OpenSees forward model used by ``run_calibration`` — we only
append a Gaussian prior block and read a covariance out of the Jacobian. The
deterministic ``run_calibration`` is left untouched (Bayesian is a sibling entry
point, Q-F).

Math
----
- Prior:     theta ~ N(theta0, Sigma_prior), theta0 = [1, 1, ..., 1] (E + masses,
             all ACTIVE per Baban), Sigma_prior diagonal from sigma_E / sigma_m.
- Likelihood: d ~ N(g(theta), Sigma_data); g = OpenSees eigen-solve (modal_residuals).
- MAP:       minimize  1/2||(d-g)/sigma_d||^2 + 1/2||(theta-theta0)/sigma_theta||^2
             = least_squares on the AUGMENTED residual r_aug = [data ; prior].
- Laplace:   Sigma_post = (J_aug^T J_aug)^-1, where J_aug = result.jac of the
             augmented residual (it ALREADY contains the prior block).

Guardrails / audit fixes baked in
---------------------------------
- G7: pure numpy/scipy, no Qt. G8: no new dependency (scipy already shipped).
- BLOCKER-5: the MAP ``least_squares`` MUST use ``loss='linear'`` so ``result.jac``
  is the raw Jacobian (the deterministic path keeps ``soft_l1`` — do not change it).
- RISK-MU-3: with the augmented residual, ``Sigma_post = inv(J^T J)`` directly —
  the prior is already inside J_aug; do NOT add ``Sigma_prior^-1`` a second time.
- GAP-MU-3: the data residual is scaled by a physical ``sigma_data_scale`` (the
  likelihood noise), which is DISTINCT from the term-balancing ``w_freq``/``w_mode``.
  Per the PRE-3 caveat, bands are honest *relative* uncertainty until grounded.

The pure pieces (``build_prior``, ``augmented_residual``, ``gaussian_map_update``)
import only numpy/scipy, so they unit-test without OpenSees. Only
``run_bayesian_calibration`` touches the forward model, and it imports the
calibrator lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares


# --- Default knobs (RESOLVED OPEN ENDS table; provisional until PRE-3 grounded) ---
DEFAULT_SIGMA_E = 0.30        # prior sigma on E_scale (E best-constrained: f ~ sqrt(E))
DEFAULT_SIGMA_M = 0.15        # prior sigma on each m_i_scale (weaker, not pinned)
DEFAULT_SIGMA_DATA = 0.02     # likelihood noise scale (~2% frequency noise)
DEFAULT_MASS_SIMILARITY_WEIGHT = 0.5   # R1: soft penalty on floor-to-floor mass spread
DEFAULT_FORGETTING = 0.98      # C3: covariance inflation per cycle (stay adaptive, no lock-up)
DEFAULT_ANCHOR_SIGMA_E = 0.5   # soft anchor to the INITIAL E (weak: prevents long-run drift)
DEFAULT_ANCHOR_SIGMA_M = 0.3   # soft anchor to the INITIAL masses (weak)
DEFAULT_TEMPORAL_SMOOTHNESS_WEIGHT = 0.3   # R2: inter-cycle smoothness (non-recursive paths)
_Z95 = 1.959963984540054      # 97.5th percentile of N(0,1) for a 95% band

# --- Physical bound convention (R3) -------------------------------------------
# The hard bounds come from ``base_params`` (NOT hard-coded here): ``E_scale_lb/ub``
# and ``m_scale_lb/ub`` are GUI knobs (panel spinboxes in tab_model_updating.py).
# They are FRACTIONAL scales on the as-defined model: E_scale = 1.0 means nominal
# Young's modulus, m_i_scale = 1.0 means the defined floor mass (self + added). A
# typical physical range is ~[0.7, 1.3] (+/-30%). Masses stay ACTIVE within these
# bounds (Baban) — regularization (R1 similarity here + R2 temporal smoothness in
# Mode B + the Gaussian prior) keeps them physical without pinning them.


@dataclass
class GaussianPrior:
    """Diagonal Gaussian prior on theta = [E_scale, m1_scale, ..., mn_scale]."""

    mean: np.ndarray                         # theta0, length 1 + nStory
    cov: np.ndarray                          # diagonal Sigma_prior, (k, k)
    sigma: np.ndarray = field(default_factory=lambda: np.empty(0))  # sqrt(diag(cov))

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float)
        self.cov = np.asarray(self.cov, dtype=float)
        if self.sigma is None or np.asarray(self.sigma).size == 0:
            self.sigma = np.sqrt(np.clip(np.diag(self.cov), 0.0, None))
        else:
            self.sigma = np.asarray(self.sigma, dtype=float)


@dataclass
class BayesianResult:
    """Posterior summary of a Gaussian MAP + Laplace update."""

    mean: np.ndarray                         # posterior mean theta_hat
    cov: np.ndarray                          # posterior covariance Sigma_post
    sigma: np.ndarray                        # per-parameter std = sqrt(diag(cov))
    bands_1sigma: tuple                      # (mean - sigma, mean + sigma)
    bands_95: tuple                          # (mean - 1.96 sigma, mean + 1.96 sigma)
    success: bool = False
    status: int = 0
    cost: float = float("nan")
    n_data: int = 0
    nfev: int = 0
    message: str = ""
    raw_result: object = None


def build_prior(base_params: dict,
                sigma_E: float = DEFAULT_SIGMA_E,
                sigma_m: float = DEFAULT_SIGMA_M) -> GaussianPrior:
    """Build the diagonal Gaussian prior for ``theta = [E_scale, m1..mn_scale]``.

    Centred on 1.0 (the model as-defined). Masses stay ACTIVE (Baban) — the prior
    keeps them physically sane without pinning them. Length is ``1 + nStory``.
    """
    n_story = int(base_params["nStory"])
    mean = np.ones(1 + n_story, dtype=float)
    sigma = np.array([float(sigma_E)] + [float(sigma_m)] * n_story, dtype=float)
    cov = np.diag(sigma ** 2)
    return GaussianPrior(mean=mean, cov=cov, sigma=sigma)


def augmented_residual(theta: Sequence[float],
                       residual_fn: Callable[[np.ndarray], np.ndarray],
                       prior: GaussianPrior) -> np.ndarray:
    """``r_aug(theta) = [ data_block ; (theta - theta0) / sigma_theta ]``.

    ``residual_fn(theta)`` must already return the data residual scaled by the
    likelihood noise (i.e. ``(d - g(theta)) / sigma_d``). The appended prior block
    is what makes the MAP well-posed and the Laplace covariance non-singular even
    when the data alone is rank-deficient (e.g. frequency-only, 3 freqs / 4 params).
    """
    theta = np.asarray(theta, dtype=float)
    data = np.asarray(residual_fn(theta), dtype=float).ravel()
    prior_block = (theta - prior.mean) / prior.sigma
    return np.concatenate([data, prior_block])


def mass_similarity_residual(theta: Sequence[float],
                             n_story: int,
                             weight: float = DEFAULT_MASS_SIMILARITY_WEIGHT) -> np.ndarray:
    """R1 mass-similarity penalty (Baban): discourage NON-PHYSICAL floor-mass spread.

    Returns a length-``n_story`` residual block ``weight * (m_i - mean(m))`` where
    ``m = theta[1 : 1 + n_story]``. It is zero when the floor mass scales are uniform
    and grows with their spread, so the optimizer is discouraged from the "one floor
    grossly over-corrected, another under-corrected" distributions seen before. It is
    **soft**: a genuine mass change supported by modal evidence (the data block) still
    moves the masses — the penalty only resists spread that the data does not require.
    It does NOT prevent mass changes, only non-physical ones (Baban authority).

    Penalizing deviation-from-mean (not drift-from-1.0) keeps this orthogonal to the
    Gaussian prior, which already anchors masses toward their defined values.
    """
    theta = np.asarray(theta, dtype=float)
    masses = theta[1:1 + int(n_story)]
    if masses.size == 0:
        return np.empty(0, dtype=float)
    return float(weight) * (masses - float(np.mean(masses)))


def build_anchor_prior(base_params: dict,
                       sigma_E: float = DEFAULT_ANCHOR_SIGMA_E,
                       sigma_m: float = DEFAULT_ANCHOR_SIGMA_M) -> GaussianPrior:
    """A weak Gaussian prior centred on the INITIAL parameters (theta0 = 1.0).

    Used as the soft anchor in :func:`propagate_recursive_prior` so a long recursive
    run cannot drift arbitrarily far from the as-defined model. Wider sigmas than the
    data prior (``build_prior``) => a gentle pull, not a pin.
    """
    return build_prior(base_params, sigma_E=sigma_E, sigma_m=sigma_m)


def propagate_recursive_prior(posterior_mean, posterior_cov, anchor_prior,
                              forgetting: float = DEFAULT_FORGETTING) -> GaussianPrior:
    """Build the next cycle's prior from this cycle's posterior (recursive Bayesian, C1).

    Inflate the posterior covariance by ``1/forgetting`` (so the filter stays adaptive
    and never locks up), then combine it with the soft ``anchor_prior`` via a Gaussian
    product. The result narrows as consistent evidence accumulates yet is pinned weakly
    to the initial parameters, which is the honest digital-twin behaviour (Q-C). This is
    also where R2 (temporal smoothness) comes from for free on the recursive path.
    """
    mean = np.asarray(posterior_mean, dtype=float)
    cov = np.asarray(posterior_cov, dtype=float)
    lam = float(forgetting)
    if lam <= 0.0:
        lam = 1.0
    cov_inflated = cov / lam
    P_prev = np.linalg.inv(cov_inflated)
    P_anchor = np.linalg.inv(anchor_prior.cov)
    P = P_prev + P_anchor
    new_cov = np.linalg.inv(P)
    new_cov = 0.5 * (new_cov + new_cov.T)
    new_mean = new_cov @ (P_prev @ mean + P_anchor @ anchor_prior.mean)
    return GaussianPrior(mean=new_mean, cov=new_cov)


def _laplace_cov(jac: np.ndarray) -> np.ndarray:
    """Laplace covariance from the augmented Jacobian: ``inv(J^T J)``.

    The augmented Jacobian already contains the prior block, so ``J^T J`` is the
    Gauss-Newton Hessian of the full MAP objective (data + prior) — do NOT add the
    prior precision again (RISK-MU-3). The prior block keeps it positive-definite,
    so the inverse exists even with rank-deficient data; pinv is a safety fallback.
    """
    J = np.asarray(jac, dtype=float)
    H = J.T @ J
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    return 0.5 * (cov + cov.T)  # symmetrize


def gaussian_map_update(residual_fn: Callable[[np.ndarray], np.ndarray],
                        x0: Sequence[float],
                        prior: GaussianPrior,
                        bounds: Optional[tuple] = None,
                        *,
                        max_nfev: int = 200) -> BayesianResult:
    """Core Gaussian MAP + Laplace update (pure: numpy/scipy only).

    ``residual_fn`` is the noise-scaled data residual ``(d - g(theta)) / sigma_d``;
    it is the only thing that knows about the forward model, so this function is
    fully testable with a synthetic ``g``.
    """
    x0 = np.asarray(x0, dtype=float)

    def r_aug(theta):
        return augmented_residual(theta, residual_fn, prior)

    # loss='linear' is MANDATORY (BLOCKER-5): a robust loss would scale result.jac
    # and bias the Laplace covariance.
    kwargs = dict(method="trf", loss="linear", max_nfev=int(max_nfev))
    if bounds is not None:
        result = least_squares(r_aug, x0, bounds=bounds, **kwargs)
    else:
        result = least_squares(r_aug, x0, **kwargs)

    cov = _laplace_cov(result.jac)
    mean = np.asarray(result.x, dtype=float)
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    n_data = int(np.asarray(result.jac).shape[0] - prior.mean.size)
    return BayesianResult(
        mean=mean,
        cov=cov,
        sigma=sigma,
        bands_1sigma=(mean - sigma, mean + sigma),
        bands_95=(mean - _Z95 * sigma, mean + _Z95 * sigma),
        success=bool(result.success),
        status=int(getattr(result, "status", 0)),
        cost=float(result.cost),
        n_data=n_data,
        nfev=int(getattr(result, "nfev", 0)),
        message=str(result.message),
        raw_result=result,
    )


def _scale_residual_per_mode(r: np.ndarray, exp_data: dict, sigma_per_mode: np.ndarray,
                             sigma_floor: float) -> np.ndarray:
    """Divide each mode's freq + shape residual block by that mode's own sigma (CU-4).

    ``modal_residuals`` returns ``[freq(n_use) ; shapes(n_use * n_meas)? ; anchor?]``.
    The per-mode (relative) sigma comes from Stage 1's observed scatter, so a noisy
    mode gets *less* weight in the Stage-2 fit and a wider posterior band. The
    sigma is floored at ``sigma_floor`` (the global physical noise) so a tight Stage-1
    cluster cannot make the likelihood over-confident — honest while PRE-3 is open.
    Any trailing block beyond the freq+shape terms (e.g. R2 anchor) is left untouched.
    """
    r = np.asarray(r, dtype=float).copy()
    n_use = int(exp_data["n_modes_used"])
    sig = np.asarray(sigma_per_mode, dtype=float)
    # Per-mode scale: finite & >0, floored; fall back to the floor where unusable.
    s = np.array([
        max(float(sig[i]), float(sigma_floor)) if (i < sig.size and np.isfinite(sig[i]) and sig[i] > 0)
        else float(sigma_floor)
        for i in range(n_use)
    ], dtype=float)

    n = r.size
    # Frequency block: one term per used mode.
    for i in range(min(n_use, n)):
        r[i] /= s[i]
    # Shape block (if present): n_use * n_meas terms, mode i contiguous.
    if exp_data.get("use_mode_shapes"):
        dofs = exp_data.get("measured_dof_indices")
        n_meas = len(dofs) if dofs else int(exp_data.get("nStory", 0) or 0)
        if n_meas > 0 and n >= n_use + n_use * n_meas:
            for i in range(n_use):
                start = n_use + i * n_meas
                r[start:start + n_meas] /= s[i]
    return r


def run_bayesian_calibration(base_params: dict,
                             exp_data: dict,
                             *,
                             prior: Optional[GaussianPrior] = None,
                             sigma_data_scale: float = DEFAULT_SIGMA_DATA,
                             sigma_data_per_mode: Optional[Sequence[float]] = None,
                             sigma_E: float = DEFAULT_SIGMA_E,
                             sigma_m: float = DEFAULT_SIGMA_M,
                             mass_similarity_weight: float = DEFAULT_MASS_SIMILARITY_WEIGHT,
                             show_info: bool = False):
    """Bayesian (Gaussian MAP) sibling of ``run_calibration``.

    Reuses the deterministic ``modal_residuals`` + OpenSees forward model unchanged;
    only the prior block, the R1 mass-similarity regularization block, and the
    covariance read-out are new. Returns ``(BayesianResult, calibrated_params)`` —
    mirroring ``run_calibration``'s ``(result, calib_params)`` shape so callers can
    swap engines.

    Likelihood noise (GAP-MU-3 / CU-4):
    - ``sigma_data_scale`` (scalar, default 0.02) scales the whole data residual —
      the back-compatible behaviour used by Mode A / the one-shot path.
    - ``sigma_data_per_mode`` (optional vector, length ``n_modes_used``) overrides
      the scalar with a **per-mode** relative sigma (e.g. Stage-1 ``sigma_f / f_hat``
      from the continuous loop). Noisy modes are down-weighted in the fit and get a
      wider posterior band, which is how the two-stage digital twin propagates the
      observed scatter into the parameter uncertainty. Each entry is floored at
      ``sigma_data_scale`` so a tight cluster cannot over-state confidence.

    Bounds (R3): ``E_scale_lb/ub`` and ``m_scale_lb/ub`` come from ``base_params``
    (GUI knobs), NOT hard-coded — see the "Physical bound convention" note above.
    Set ``mass_similarity_weight=0`` to disable R1 (e.g. to reproduce the pure prior).
    """
    # Lazy import: keeps the pure math above importable without openseespy (G8).
    from . import calibrator

    n_story = int(base_params["nStory"])
    if prior is None:
        prior = build_prior(base_params, sigma_E=sigma_E, sigma_m=sigma_m)

    x0 = np.array([1.0] + [1.0] * n_story, dtype=float)
    lb = np.array([base_params["E_scale_lb"]] + [base_params["m_scale_lb"]] * n_story, dtype=float)
    ub = np.array([base_params["E_scale_ub"]] + [base_params["m_scale_ub"]] * n_story, dtype=float)
    w_freq = float(base_params.get("w_freq", 1.0))
    w_mode = float(base_params.get("w_mode", 0.35))
    max_nfev = int(base_params.get("max_nfev", 200))
    sd = float(sigma_data_scale)
    w_sim = float(mass_similarity_weight)
    sig_pm = None if sigma_data_per_mode is None else np.asarray(sigma_data_per_mode, dtype=float)

    def data_residual(theta):
        # GAP-MU-3: term-balanced residual scaled by the physical likelihood noise,
        # then the R1 mass-similarity block (loss='linear' applies to all blocks,
        # consistent with BLOCKER-5).
        r = calibrator.modal_residuals(theta, base_params, exp_data, w_freq, w_mode, show_info)
        r = np.asarray(r, dtype=float)
        if sig_pm is not None and sig_pm.size:
            r = _scale_residual_per_mode(r, exp_data, sig_pm, sd)   # CU-4 per-mode noise
        else:
            r = r / sd                                              # scalar (back-compat)
        if w_sim > 0.0 and n_story > 0:
            r = np.concatenate([r, mass_similarity_residual(theta, n_story, w_sim)])
        return r

    result = gaussian_map_update(data_residual, x0, prior, bounds=(lb, ub), max_nfev=max_nfev)
    calib_params = calibrator.apply_calibration_vector(base_params, result.mean)
    return result, calib_params
