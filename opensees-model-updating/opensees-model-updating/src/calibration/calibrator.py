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
from ..utils.math_utils import normalize_mode_maxabs, align_mode_sign


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
        Weight for frequency residuals.
    w_mode : float
        Weight for mode-shape residuals.
    show_info : bool
        Print debug info on failure.

    Returns
    -------
    np.ndarray : residual vector
    """
    try:
        p = apply_calibration_vector(base_params, x)
        modal = extract_modal_results(p, normalize_modes=True, show_info=False)

        n_use = exp_data["n_modes_used"]

        num_freqs = np.asarray(modal["freqs"][:n_use], dtype=float)
        exp_freqs = np.asarray(exp_data["freqs"][:n_use], dtype=float)

        res = []

        for fn, ft in zip(num_freqs, exp_freqs):
            res.append(w_freq * (fn - ft) / ft)

        if exp_data["use_mode_shapes"]:
            for i in range(n_use):
                phi_num = np.asarray(modal["mode_shapes_ux_master"][i], dtype=float)
                phi_exp = np.asarray(exp_data["modes"][i], dtype=float)
                a = align_mode_sign(phi_num, phi_exp)
                mode_diff = w_mode * (a - phi_exp)
                res.extend(list(mode_diff))

        return np.array(res, dtype=float)

    except Exception:
        if show_info:
            print("Calibration trial failed. Penalizing this point.")
            traceback.print_exc()
        n_mode_terms = exp_data["n_modes_used"] * base_params["nStory"] if exp_data["use_mode_shapes"] else 0
        return np.ones(exp_data["n_modes_used"] + n_mode_terms, dtype=float) * 1.0e3


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
