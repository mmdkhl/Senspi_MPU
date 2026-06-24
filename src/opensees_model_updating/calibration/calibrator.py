# -*- coding: utf-8 -*-
"""
Calibration engine - wraps the scipy optimization from DigitalTwin_V8.py.

Provides functions for:
- Preparing experimental data for calibration
- Applying calibration parameter vectors
- Computing modal residuals (objective function)
- Running the least-squares optimization
"""

import copy
import traceback
import numpy as np
from scipy.optimize import least_squares

from ..analysis.modal import extract_modal_results
from ..io.loaders import EXPERIMENTAL_MODAL_JSON, load_experimental_modal_data
from ..utils.math_utils import normalize_mode_maxabs, align_mode_sign, pair_modes_by_mac


def prepare_experimental_modal_data(params):
    """
    Load and prepare experimental modal data for calibration.

    Parameters
    ----------
    params : dict
        Must contain: nStory, nCalibModes, use_mode_shapes.

    Returns
    -------
    dict with keys: freqs, modes, use_mode_shapes, mode_shapes_available,
                    source_file, raw_data, n_modes_used
    """
    loaded = load_experimental_modal_data(
        EXPERIMENTAL_MODAL_JSON,
        params["nStory"],
        require_mode_shapes=params["use_mode_shapes"]
    )

    if params["nCalibModes"] > len(loaded["frequencies_hz"]):
        raise ValueError(
            f"Experimental file contains only {len(loaded['frequencies_hz'])} frequencies, "
            f"but nCalibModes = {params['nCalibModes']}."
        )

    exp_freqs = loaded["frequencies_hz"][:params["nCalibModes"]]
    exp_modes = []
    mode_shapes_available = bool(loaded.get("mode_shapes_available", False))

    if params["use_mode_shapes"]:
        exp_modes = [normalize_mode_maxabs(v) for v in loaded["mode_shapes_ux"][:params["nCalibModes"]]]

    return {
        "freqs": exp_freqs,
        "modes": exp_modes,
        "use_mode_shapes": params["use_mode_shapes"],
        "mode_shapes_available": mode_shapes_available,
        "source_file": EXPERIMENTAL_MODAL_JSON,
        "raw_data": loaded["raw_data"],
        "n_modes_used": params["nCalibModes"]
    }


def apply_calibration_vector(base_params, x):
    """
    Apply an optimization parameter vector to the base model parameters.

    x[0] = E_scale
    x[1:] = mass_scale per story

    Parameters
    ----------
    base_params : dict
        Original model parameters.
    x : array-like
        Optimization variable vector.

    Returns
    -------
    dict : Updated parameters copy.
    """
    p = copy.deepcopy(base_params)
    E_scale = float(x[0])
    mass_scales = np.array(x[1:], dtype=float)
    scope = base_params.get("mass_calibration_scope", "self_weight_only")

    p["E"] = base_params["E"] * E_scale

    p["floor_masses"] = [
        base_params["floor_masses"][i] * float(mass_scales[i])
        for i in range(base_params["nStory"])
    ]

    if scope == "total_mass":
        additional = copy.deepcopy(base_params.get("additional_masses", {}))
        for story in range(1, base_params["nStory"] + 1):
            scale = float(mass_scales[story - 1])
            vals = additional.get(story, [0.0, 0.0, 0.0, 0.0, 0.0])
            additional[story] = [float(v) * scale for v in vals]
        p["additional_masses"] = additional

    return p


def modal_residuals(x, base_params, exp_data, w_freq=1.0, w_mode=0.35, show_info=False):
    """
    Compute the residual vector for the calibration objective function.

    Parameters
    ----------
    x : array-like
        Optimization variable vector [E_scale, m1_scale, m2_scale, ...].
    base_params : dict
        Original model parameters.
    exp_data : dict
        Experimental data from prepare_experimental_modal_data().
    w_freq : float
        Term-balancing weight on the frequency residual block (default 1.0).
    w_mode : float
        Term-balancing weight on the mode-shape residual block (default 0.35).
    show_info : bool
        Print debug info on failure.

    Returns
    -------
    np.ndarray : residual vector

    Weighting convention (B3 / GAP-MU-3)
    ------------------------------------
    ``w_freq`` and ``w_mode`` are **relative term-balancing weights, NOT
    measurement-noise standard deviations.** The frequency block holds ``n_use``
    *relative* errors ``(fn - ft) / ft`` (each ~O(1e-2)); the mode-shape block
    holds ``n_use x n_measured`` shape differences (each ~O(1e-1..1)). The default
    ``w_mode = 0.35 < w_freq = 1.0`` down-weights the more numerous, larger-
    magnitude shape terms so neither block dominates the least-squares cost.
    These are distinct from the physical likelihood scale ``sigma_data_scale``
    used by the Bayesian path (``bayesian.py``), which forms ``(d - g)/sigma_d``;
    do not conflate the two. Defaults mirror the GUI panel spinboxes
    (``tab_model_updating.py`` Frequency/Mode-shape weight). Adjust only if a
    freq+shape run shows one block dominating.
    """
    try:
        p = apply_calibration_vector(base_params, x)
        modal = extract_modal_results(p, normalize_modes=True, show_info=False)

        n_use = exp_data["n_modes_used"]
        use_shapes = exp_data["use_mode_shapes"]
        measured_dofs = np.asarray(exp_data.get("measured_dof_indices") or [], dtype=int)

        modal_freqs = np.asarray(modal["freqs"], dtype=float)
        exp_freqs = np.asarray(exp_data["freqs"][:n_use], dtype=float)

        def _slice_model_shape(j):
            # Partial coverage (B1): slice a full-length model shape to the measured
            # stories and RE-NORMALIZE on that support so it matches how phi_exp was
            # normalized (RISK-MU-2). Full coverage leaves it untouched (slice
            # skipped), keeping results byte-identical.
            s = np.asarray(modal["mode_shapes_ux_master"][j], dtype=float)
            if 0 < measured_dofs.size < s.size:
                s = s[measured_dofs]
                peak = float(np.max(np.abs(s))) if s.size else 0.0
                if peak > 0.0:
                    s = s / peak
            return s

        # Mode pairing (B2): match each experimental mode to the model mode whose
        # (measured) shape is most similar by MAC, so a swapped or missed
        # identification cannot corrupt the fit via frequency order. Falls back to
        # frequency order without shapes or with < 3 measured DOFs (MAC degenerate).
        # A non-swapped case yields the identity perm == the prior behavior.
        if use_shapes:
            exp_shapes = [np.asarray(exp_data["modes"][i], dtype=float) for i in range(n_use)]
            model_shapes_cmp = [_slice_model_shape(j) for j in range(len(modal_freqs))]
            perm = pair_modes_by_mac(model_shapes_cmp, exp_shapes)
        else:
            perm = list(range(n_use))

        res = []
        for i in range(n_use):
            fn = float(modal_freqs[perm[i]])
            ft = float(exp_freqs[i])
            res.append(w_freq * (fn - ft) / ft)

        if use_shapes:
            for i in range(n_use):
                phi_num = _slice_model_shape(perm[i])
                a = align_mode_sign(phi_num, exp_shapes[i])
                res.extend(list(w_mode * (a - exp_shapes[i])))

        # R2 temporal-smoothness (MU-13): optional block toward the previous accepted
        # estimate, OFF by default. Mode B's non-recursive paths set
        # base_params["temporal_anchor"] = (theta_prev, weight); recursive Bayesian
        # gets R2 from the prior instead, so it leaves this unset (no double-count).
        anchor = base_params.get("temporal_anchor")
        if anchor is not None:
            theta_prev, w_smooth = anchor
            theta_prev = np.asarray(theta_prev, dtype=float)
            xv = np.asarray(x, dtype=float)
            if w_smooth and theta_prev.size == xv.size:
                res.extend(list(float(w_smooth) * (xv - theta_prev)))

        return np.array(res, dtype=float)

    except Exception:
        if show_info:
            print("Calibration trial failed. Penalizing this point.")
            traceback.print_exc()
        # BLOCKER-7: the penalty vector MUST match the success-branch length on
        # every call (least_squares requires a fixed residual length). Under
        # partial coverage the shape block has one term per MEASURED DOF, not per
        # story — derive the per-mode length from measured_dof_indices (falls back
        # to nStory for full coverage / no indices).
        if exp_data["use_mode_shapes"]:
            measured_dofs = exp_data.get("measured_dof_indices")
            n_per_mode = len(measured_dofs) if measured_dofs else base_params["nStory"]
            n_mode_terms = exp_data["n_modes_used"] * n_per_mode
        else:
            n_mode_terms = 0
        # Match the success branch if the R2 temporal-smoothness block is active.
        n_anchor = 0
        anchor = base_params.get("temporal_anchor")
        if anchor is not None:
            theta_prev, w_smooth = anchor
            if w_smooth and np.asarray(theta_prev).size == (1 + base_params["nStory"]):
                n_anchor = 1 + base_params["nStory"]
        return np.ones(exp_data["n_modes_used"] + n_mode_terms + n_anchor, dtype=float) * 1.0e3


def run_calibration(base_params, exp_data, show_info=False):
    """
    Run the least-squares calibration optimization.

    Parameters
    ----------
    base_params : dict
        Original model parameters with calibration settings.
    exp_data : dict
        Experimental data from prepare_experimental_modal_data().
    show_info : bool
        Print verbose optimizer output.

    Returns
    -------
    tuple : (scipy OptimizeResult, calibrated_params dict)
    """
    nStory = base_params["nStory"]

    x0 = np.array([1.0] + [1.0] * nStory, dtype=float)
    lb = np.array([base_params["E_scale_lb"]] + [base_params["m_scale_lb"]] * nStory, dtype=float)
    ub = np.array([base_params["E_scale_ub"]] + [base_params["m_scale_ub"]] * nStory, dtype=float)

    loss_name = 'soft_l1'

    result = least_squares(
        modal_residuals,
        x0,
        bounds=(lb, ub),
        args=(base_params, exp_data, base_params["w_freq"], base_params["w_mode"], show_info),
        method='trf',
        loss=loss_name,
        max_nfev=base_params["max_nfev"],
        verbose=2 if show_info else 0
    )

    calib_params = apply_calibration_vector(base_params, result.x)
    return result, calib_params
