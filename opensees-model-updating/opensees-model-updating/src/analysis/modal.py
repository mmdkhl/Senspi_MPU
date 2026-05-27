# -*- coding: utf-8 -*-
"""Modal (eigenvalue) analysis for the OpenSees frame model."""

import math
import numpy as np
import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt

from ..model.builder import build_model
from ..analysis.gravity import run_gravity_analysis
from ..utils.math_utils import normalize_mode_maxabs
from ..utils.formatters import r3


def run_eigen_with_fallback(num_modes, show_info=False):
    """
    Run OpenSees eigen analysis with ARPACK, falling back to fullGenLapack.

    The default OpenSees eigensolver is ARPACK. It is fast, but it can fail for
    small models, repeated/clustered eigenvalues, or when too many modes are
    requested relative to the number of inertial DOFs. If ARPACK fails, this
    function retries with fullGenLapack, which is slower but more robust.

    Parameters
    ----------
    num_modes : int
        Number of eigenvalues to extract.
    show_info : bool
        Print solver info to console.

    Returns
    -------
    np.ndarray : eigenvalues (rad²/s²)
    """
    try:
        return np.array(ops.eigen(num_modes), dtype=float)
    except Exception as arpack_error:
        if show_info:
            print("Default ARPACK eigen solver failed. Retrying with fullGenLapack...")
            print("ARPACK error:", arpack_error)

        lapack_errors = []
        for solver_name in ("-fullGenLapack", "fullGenLapack"):
            try:
                return np.array(ops.eigen(solver_name, num_modes), dtype=float)
            except Exception as e:
                lapack_errors.append(f"{solver_name}: {e}")

        raise RuntimeError(
            "OpenSees eigen analysis failed with both the default ARPACK solver and fullGenLapack. "
            "For a 2-story model, try setting 'Number of modes in model' = 2 and "
            "'Number of modes used in calibration' = 2. Details: "
            + " | ".join(lapack_errors)
        ) from arpack_error


def extract_modal_results(params, normalize_modes=True, show_info=False):
    """
    Build model, run gravity, and extract modal analysis results.

    Parameters
    ----------
    params : dict
        All model parameters.
    normalize_modes : bool
        Normalize mode shapes by max absolute value.
    show_info : bool
        Print progress to console.

    Returns
    -------
    dict with keys: lam, omegas, freqs, periods, mode_shapes_ux_master, ctx
    """
    ctx = build_model(params, show_info=show_info)
    run_gravity_analysis(ctx, show_info=show_info)

    lam = run_eigen_with_fallback(params["numModes"], show_info=show_info)
    if len(lam) < params["numModes"]:
        raise RuntimeError(
            f"OpenSees returned only {len(lam)} eigenvalues, but {params['numModes']} modes were requested. "
            "Reduce 'Number of modes in model'."
        )
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        raise RuntimeError(
            "OpenSees returned non-positive or non-finite eigenvalues. "
            "Check that all active stories have enough columns, positive mass, and stable boundary conditions."
        )

    omegas = np.sqrt(lam)
    freqs = omegas / (2.0 * math.pi)
    periods = 2.0 * math.pi / omegas

    master_nodes = ctx["master_nodes"]

    mode_shapes_ux_master = []
    for mode in range(1, params["numModes"] + 1):
        phi = np.array([ops.nodeEigenvector(node, mode, 1) for node in master_nodes], dtype=float)
        if normalize_modes:
            phi = normalize_mode_maxabs(phi)
        mode_shapes_ux_master.append(phi)

    return {
        "lam": lam,
        "omegas": omegas,
        "freqs": freqs,
        "periods": periods,
        "mode_shapes_ux_master": mode_shapes_ux_master,
        "ctx": ctx,
    }


def plot_modal_results_calibrated_only(modal_data, title_prefix="Calibrated", block=True):
    """
    Plot 3D mode shapes using opsvis.

    Parameters
    ----------
    modal_data : dict
        Output from extract_modal_results().
    title_prefix : str
        Prefix for subplot titles.
    block : bool
        Whether plt.show() should block.
    """
    freqs = modal_data["freqs"]
    periods = modal_data["periods"]
    numModes = len(freqs)

    fig_modes = plt.figure(figsize=(4 * numModes, 5))
    try:
        fig_modes.canvas.manager.window.move(100, 800)
    except Exception:
        pass

    axes = [fig_modes.add_subplot(1, numModes, i + 1, projection='3d')
            for i in range(numModes)]

    for i, mode in enumerate(range(1, numModes + 1)):
        opsv.plot_mode_shape(mode, ax=axes[i], az_el=(-70, 25))

        axes[i].grid(False)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_zticks([])

        for coll in list(axes[i].collections):
            coll.remove()

        axes[i].set_title(
            f"{title_prefix} Mode {mode}\nT={periods[i]:.4f} s, f={freqs[i]:.3f} Hz"
        )

    plt.tight_layout()
    plt.show(block=block)


def export_modal_files(modal_data, output_prefix):
    """
    Export modal results to text files in the output/ directory.

    Parameters
    ----------
    modal_data : dict
        Output from extract_modal_results().
    output_prefix : str
        Prefix for output filenames (e.g., 'original', 'calibrated').
    """
    lam = modal_data["lam"]
    omegas = modal_data["omegas"]
    freqs = modal_data["freqs"]
    periods = modal_data["periods"]
    ctx = modal_data["ctx"]
    master_nodes = ctx["master_nodes"]

    with open(f"output/{output_prefix}_periods.out", "w", encoding="utf-8") as f:
        f.write("Mode  Lambda(rad^2/s^2)  Omega(rad/s)  Frequency(Hz)  Period(s)\n")
        for i in range(len(freqs)):
            f.write(
                f"{i+1}  {r3(lam[i])}  {r3(omegas[i])}  {r3(freqs[i])}  {r3(periods[i])}\n"
            )

    with open(f"output/{output_prefix}_mode_shapes_normalized.out", "w", encoding="utf-8") as f:
        f.write("Mode  StoryMasterNode  UX_normalized\n")
        for mode in range(1, len(freqs) + 1):
            phi = modal_data["mode_shapes_ux_master"][mode - 1]
            for node, ux in zip(master_nodes, phi):
                f.write(f"{mode}  {node}  {r3(ux)}\n")
