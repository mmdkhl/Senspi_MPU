# -*- coding: utf-8 -*-
"""
Report generation for calibration results.

Contains make_modal_comparison_report(), report_to_text(),
and save_calibration_summary_figure() from DigitalTwin_V8.py.
"""

from ..paths import output_path as _out_path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from ..io.loaders import EXPERIMENTAL_MODAL_JSON
from ..utils.formatters import (
    r3,
    sci3,
    column_orientation_layout_to_text,
    additional_masses_to_text,
)
from ..utils.math_utils import align_mode_sign, safe_percent_error


# ── Partial-coverage mode-shape helpers (T8.3 / B1) ───────────────────────────
# Experimental shapes may cover only the MEASURED stories (partial coverage). The
# model shape is full-length, so model-vs-measured comparison and plotting must
# both happen on the measured support, with one shared normalization (RISK-MU-2).

def _measured_dofs(exp_data, n_story):
    """0-based DOF indices the experimental shapes cover (all stories if full)."""
    dofs = exp_data.get("measured_dof_indices")
    if dofs:
        return np.asarray(dofs, dtype=int)
    return np.arange(int(n_story), dtype=int)


def _align_norm_on_support(model_full, phi_exp, idx):
    """Sign-align a FULL-length model shape to ``phi_exp`` on the measured support
    ``idx`` and normalize the full vector by its measured-support max-abs, so the
    model line and the measured markers share one normalization. Returns the full
    aligned/normalized vector (length = len(model_full))."""
    model_full = np.asarray(model_full, dtype=float)
    phi_exp = np.asarray(phi_exp, dtype=float)
    support = model_full[idx]
    sign = -1.0 if float(np.dot(support, phi_exp)) < 0.0 else 1.0
    v = sign * model_full
    peak = float(np.max(np.abs(v[idx]))) if idx.size else 0.0
    return v / peak if peak > 0.0 else v


def _plot_measured_vs_model(ax, exp_data, modal_before, modal_after, stories, n_use):
    """Overlay sensor-measured mode shapes (red markers at the MEASURED stories) on
    the original (black) and calibrated (green) MODEL shapes (full lines). Partial-
    coverage safe (T8.3): measured points appear only where a sensor exists, and the
    sensor positions are mapped onto the model's story axis."""
    idx = _measured_dofs(exp_data, len(stories))
    meas_stories = [int(d) + 1 for d in idx]
    line_styles = ['-', '--', ':', '-.']
    for i in range(n_use):
        phi_exp = np.asarray(exp_data["modes"][i], dtype=float)
        before_full = _align_norm_on_support(modal_before["mode_shapes_ux_master"][i], phi_exp, idx)
        after_full = _align_norm_on_support(modal_after["mode_shapes_ux_master"][i], phi_exp, idx)
        ls = line_styles[i % len(line_styles)]
        ax.plot(before_full, stories, color="black", linestyle=ls, marker="s",
                linewidth=1.6, label=f"Orig M{i+1}")
        ax.plot(after_full, stories, color="green", linestyle=ls, marker="^",
                linewidth=1.8, label=f"Calib M{i+1}")
        ax.plot(phi_exp, meas_stories, color="red", linestyle="none", marker="o",
                markersize=8, label=f"Measured M{i+1}")


def make_modal_comparison_report(exp_data, modal_before, modal_after,
                                  base_params, final_params, calib_result=None):
    """
    Build a structured comparison report dict.

    Parameters
    ----------
    exp_data : dict
        Experimental data from prepare_experimental_modal_data().
    modal_before : dict
        Modal results before calibration.
    modal_after : dict
        Modal results after calibration.
    base_params : dict
        Original model parameters.
    final_params : dict
        Calibrated model parameters.
    calib_result : scipy OptimizeResult or None

    Returns
    -------
    dict : comparison report
    """
    n_use = exp_data["n_modes_used"]

    target_freqs = np.asarray(exp_data["freqs"][:n_use], dtype=float)
    before_freqs = np.asarray(modal_before["freqs"][:n_use], dtype=float)
    after_freqs = np.asarray(modal_after["freqs"][:n_use], dtype=float)

    report = {
        "experimental_data_source_file": exp_data.get("source_file", ""),
        "experimental_modal_data_raw": exp_data.get("raw_data", {}),
        "n_modes_used_in_calibration": n_use,
        "experimental_frequencies_hz": target_freqs.tolist(),
        "before_calibration_frequencies_hz": before_freqs.tolist(),
        "after_calibration_frequencies_hz": after_freqs.tolist(),
        "before_calibration_frequency_error_percent": [
            safe_percent_error(before_freqs[i], target_freqs[i]) for i in range(n_use)
        ],
        "after_calibration_frequency_error_percent": [
            safe_percent_error(after_freqs[i], target_freqs[i]) for i in range(n_use)
        ],
        "frequency_tolerance_percent": base_params["freq_tol_percent"],
        "before_calibration_within_tolerance": all(
            abs(safe_percent_error(before_freqs[i], target_freqs[i])) <= base_params["freq_tol_percent"]
            for i in range(n_use)
        ),
        "after_calibration_within_tolerance": all(
            abs(safe_percent_error(after_freqs[i], target_freqs[i])) <= base_params["freq_tol_percent"]
            for i in range(n_use)
        ),
        "original_E": base_params["E"],
        "calibrated_E": final_params["E"],
        "mass_calibration_scope": base_params.get("mass_calibration_scope", "self_weight_only"),
        "original_self_weight_floor_masses": base_params["floor_masses"],
        "calibrated_self_weight_floor_masses": final_params["floor_masses"],
        "column_orientation_layout": base_params.get("column_orientation_layout", {}),
        "original_additional_masses": base_params.get("additional_masses", {}),
        "calibrated_additional_masses": final_params.get("additional_masses", {}),
        "additional_masses": base_params.get("additional_masses", {}),
        "used_mode_shapes_in_calibration": bool(exp_data["use_mode_shapes"]),
    }

    mode_shape_comparison = []
    mode_shapes_can_compare = (
        bool(exp_data.get("use_mode_shapes", False))
        and len(exp_data.get("modes", [])) >= n_use
    )

    if mode_shapes_can_compare:
        # Compare on the measured support so partial coverage (B1) doesn't mismatch
        # lengths: slice the full model shapes to the measured DOFs after aligning.
        idx = _measured_dofs(exp_data, base_params["nStory"])
        for i in range(n_use):
            phi_exp = np.asarray(exp_data["modes"][i], dtype=float)
            before_meas = _align_norm_on_support(
                modal_before["mode_shapes_ux_master"][i], phi_exp, idx)[idx]
            after_meas = _align_norm_on_support(
                modal_after["mode_shapes_ux_master"][i], phi_exp, idx)[idx]

            mode_shape_comparison.append({
                "mode": i + 1,
                "experimental_normalized": phi_exp.tolist(),
                "before_calibration_normalized": before_meas.tolist(),
                "after_calibration_normalized": after_meas.tolist(),
                "before_l2_mismatch": float(np.linalg.norm(before_meas - phi_exp)),
                "after_l2_mismatch": float(np.linalg.norm(after_meas - phi_exp)),
            })

    report["mode_shape_comparison_available"] = mode_shapes_can_compare
    report["mode_shape_comparison"] = mode_shape_comparison

    if calib_result is not None:
        report["optimizer"] = {
            "success": bool(calib_result.success),
            "status": int(calib_result.status),
            "message": str(calib_result.message),
            "nfev": int(calib_result.nfev),
            "cost": float(calib_result.cost),
            "x": np.asarray(calib_result.x, dtype=float).tolist(),
        }

    return report


def report_to_text(report):
    """
    Format a comparison report dict as a human-readable text string.

    Parameters
    ----------
    report : dict
        Output from make_modal_comparison_report().

    Returns
    -------
    str : formatted report text
    """
    n_use = report["n_modes_used_in_calibration"]

    lines = []
    lines.append("MODAL CALIBRATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Experimental data source file: {report.get('experimental_data_source_file', '')}")
    lines.append(f"Number of modes used in calibration: {n_use}")
    lines.append("")

    lines.append("Experimental frequencies (Hz):")
    for i, f in enumerate(report["experimental_frequencies_hz"], start=1):
        lines.append(f"  Mode {i}: {r3(f)}")
    lines.append("")

    lines.append("Before calibration:")
    for i, (f, e) in enumerate(zip(
        report["before_calibration_frequencies_hz"],
        report["before_calibration_frequency_error_percent"]
    ), start=1):
        lines.append(f"  Mode {i}: f = {r3(f)} Hz, error = {r3(e)} %")
    lines.append("")

    lines.append("After calibration:")
    for i, (f, e) in enumerate(zip(
        report["after_calibration_frequencies_hz"],
        report["after_calibration_frequency_error_percent"]
    ), start=1):
        lines.append(f"  Mode {i}: f = {r3(f)} Hz, error = {r3(e)} %")
    lines.append("")

    lines.append(f"Tolerance used: {r3(report['frequency_tolerance_percent'])} %")
    lines.append(f"Before within tolerance: {report['before_calibration_within_tolerance']}")
    lines.append(f"After within tolerance:  {report['after_calibration_within_tolerance']}")
    lines.append(f"Used mode shapes in calibration: {report['used_mode_shapes_in_calibration']}")
    lines.append("")

    scope_text = (
        "Total mass including additional masses"
        if report.get("mass_calibration_scope") == "total_mass"
        else "Self-weight mass only"
    )
    lines.append("Parameter update:")
    lines.append(f"  Mass calibration target = {scope_text}")
    lines.append(f"  Original E   = {sci3(report['original_E'])}")
    lines.append(f"  Calibrated E = {sci3(report['calibrated_E'])}")
    lines.append(f"  Original self-weight masses   = {[r3(v) for v in report['original_self_weight_floor_masses']]}")
    lines.append(f"  Calibrated self-weight masses = {[r3(v) for v in report['calibrated_self_weight_floor_masses']]}")
    lines.append("")

    if report.get("column_orientation_layout"):
        nStory = len(report["original_self_weight_floor_masses"])
        lines.append("Column orientation layout:")
        lines.append(f"  {column_orientation_layout_to_text(report['column_orientation_layout'], nStory)}")
        lines.append("")

    if report.get("original_additional_masses"):
        nStory = len(report["original_self_weight_floor_masses"])
        lines.append("Original additional masses [center,c1,c2,c3,c4] kg:")
        lines.append(f"  {additional_masses_to_text(report['original_additional_masses'], nStory)}")
        lines.append("Calibrated additional masses [center,c1,c2,c3,c4] kg:")
        lines.append(f"  {additional_masses_to_text(report['calibrated_additional_masses'], nStory)}")
        lines.append("")

    if report.get("mode_shape_comparison_available", False):
        lines.append("Mode-shape comparison (normalized, max abs = 1):")
        for item in report["mode_shape_comparison"]:
            lines.append(f"  Mode {item['mode']}:")
            lines.append(f"    Experimental          = {[r3(v) for v in item['experimental_normalized']]}")
            lines.append(f"    Before calibration    = {[r3(v) for v in item['before_calibration_normalized']]}")
            lines.append(f"    After calibration     = {[r3(v) for v in item['after_calibration_normalized']]}")
            lines.append(f"    Before L2 mismatch    = {r3(item['before_l2_mismatch'])}")
            lines.append(f"    After L2 mismatch     = {r3(item['after_l2_mismatch'])}")
        lines.append("")
    else:
        lines.append(
            "Mode-shape comparison was not produced because mode shapes were not used "
            "or were not available for this story count."
        )
        lines.append("")

    if "optimizer" in report:
        lines.append("Optimizer summary:")
        lines.append(f"  Success = {report['optimizer']['success']}")
        lines.append(f"  Status  = {report['optimizer']['status']}")
        lines.append(f"  Message = {report['optimizer']['message']}")
        lines.append(f"  nfev    = {report['optimizer']['nfev']}")
        lines.append(f"  Cost    = {r3(report['optimizer']['cost'])}")
        lines.append(f"  x       = {[r3(v) for v in report['optimizer']['x']]}")
        lines.append("")

    return "\n".join(lines)


def generate_calibration_summary_png(exp_data, modal_before, modal_after,
                                      original_params, calibrated_params):
    """
    Generate the calibration summary figure (2 subplots) and return PNG bytes.

    Shows only: Frequencies comparison (left) + Normalized Mode Shapes (right).

    Parameters
    ----------
    exp_data : dict
        Experimental data from prepare_experimental_modal_data().
    modal_before : dict
        Modal results before calibration.
    modal_after : dict
        Modal results after calibration.
    original_params : dict
        Original model parameters.
    calibrated_params : dict
        Calibrated model parameters.

    Returns
    -------
    bytes : PNG image data
    """
    import io as _io
    n_use = exp_data["n_modes_used"]
    stories = np.arange(1, original_params["nStory"] + 1)
    modes = np.arange(1, n_use + 1)

    fs = 15   # base font size — readable after ~50 % downscale

    fig = Figure(figsize=(10, 4.5))
    FigureCanvas(fig)

    # --- Left: Frequency comparison ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(modes, exp_data["freqs"][:n_use], marker="o", linewidth=2.2,
             color="red", label="Target")
    ax1.plot(modes, modal_before["freqs"][:n_use], marker="s", linewidth=2.0,
             color="black", linestyle="--", label="Original")
    ax1.plot(modes, modal_after["freqs"][:n_use], marker="^", linewidth=2.0,
             color="green", linestyle="-.", label="Calibrated")
    ax1.set_title("Frequencies", fontsize=fs + 1)
    ax1.set_xlabel("Mode", fontsize=fs)
    ax1.set_ylabel("Frequency (Hz)", fontsize=fs)
    ax1.set_xticks(modes)
    ax1.tick_params(labelsize=fs - 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=fs - 2)

    # --- Right: Normalized mode shapes ---
    ax2 = fig.add_subplot(1, 2, 2)
    mode_shapes_can_plot = (
        bool(exp_data.get("use_mode_shapes", False))
        and len(exp_data.get("modes", [])) >= n_use
    )

    if mode_shapes_can_plot:
        # T8.3: measured (sensor) shapes as markers at their stories vs the model
        # lines; partial-coverage safe via the shared helper.
        _plot_measured_vs_model(ax2, exp_data, modal_before, modal_after, stories, n_use)
        ax2.set_title("Normalized Mode Shapes (UX)", fontsize=fs + 1)
        ax2.set_xlabel("Normalized amplitude", fontsize=fs)
        ax2.set_ylabel("Story", fontsize=fs)
        ax2.set_yticks(stories)
        ax2.tick_params(labelsize=fs - 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(ncol=3, fontsize=fs - 4)
    else:
        ax2.axis("off")
        ax2.set_title("Mode Shapes", fontsize=fs + 1)
        ax2.text(0.05, 0.95,
                 "Mode shapes were not used\nor not available for this story count.\n\n"
                 "Frequencies are still used for calibration.",
                 va="top", ha="left", fontsize=fs - 1)

    fig.tight_layout(pad=1.8)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def save_calibration_summary_figure(exp_data, modal_before, modal_after,
                                     original_params, calibrated_params,
                                     save_path=None, output_base=None):
    """
    Generate and save the calibration summary figure (4 subplots).

    Uses a non-interactive Agg backend so no window is opened.

    Parameters
    ----------
    exp_data : dict
        Experimental data from prepare_experimental_modal_data().
    modal_before : dict
        Modal results before calibration.
    modal_after : dict
        Modal results after calibration.
    original_params : dict
        Original model parameters.
    calibrated_params : dict
        Calibrated model parameters.
    save_path : str or Path, optional
        Explicit destination. When omitted the file goes to the resolved output
        directory (see ``opensees_model_updating.paths``).
    output_base : str or Path, optional
        Output directory to resolve ``save_path`` against when it is omitted.
    _unused : str
        Output file path for the figure.
    """
    n_use = exp_data["n_modes_used"]
    stories = np.arange(1, original_params["nStory"] + 1)
    modes = np.arange(1, n_use + 1)

    fig = Figure(figsize=(16, 11))
    canvas = FigureCanvas(fig)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(modes, exp_data["freqs"][:n_use], marker='o', linewidth=2.2, color='red', label="Target")
    ax1.plot(modes, modal_before["freqs"][:n_use], marker='s', linewidth=2.0, color='black', linestyle='--', label="Original")
    ax1.plot(modes, modal_after["freqs"][:n_use], marker='^', linewidth=2.0, color='green', linestyle='-.', label="Calibrated")
    ax1.set_title("Frequencies", fontsize=14)
    ax1.set_xlabel("Mode", fontsize=12)
    ax1.set_ylabel("Frequency (Hz)", fontsize=12)
    ax1.set_xticks(modes)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(original_params["nStory"])
    w = 0.36

    def total_floor_masses(params):
        additional = params.get("additional_masses", {})
        return [
            params["floor_masses"][i] + sum(additional.get(i + 1, [0.0, 0.0, 0.0, 0.0, 0.0]))
            for i in range(params["nStory"])
        ]

    if original_params.get("mass_calibration_scope") == "total_mass":
        original_bar_masses = total_floor_masses(original_params)
        calibrated_bar_masses = total_floor_masses(calibrated_params)
        mass_plot_title = "Total Floor Masses"
    else:
        original_bar_masses = original_params["floor_masses"]
        calibrated_bar_masses = calibrated_params["floor_masses"]
        mass_plot_title = "Self-Weight Floor Masses"

    ax2.bar(x - w/2, original_bar_masses, width=w, color='black', label="Original")
    ax2.bar(x + w/2, calibrated_bar_masses, width=w, color='green', label="Calibrated")
    ax2.set_title(mass_plot_title, fontsize=14)
    ax2.set_xlabel("Story", fontsize=12)
    ax2.set_ylabel("Mass (kg)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"M{i}" for i in range(1, original_params["nStory"] + 1)])
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(fontsize=11)

    ax3 = fig.add_subplot(2, 2, 3)

    mode_shapes_can_plot = (
        bool(exp_data.get("use_mode_shapes", False))
        and len(exp_data.get("modes", [])) >= n_use
    )

    if mode_shapes_can_plot:
        # Partial-coverage safe (B1/T8.3): measured markers at measured stories vs
        # the full model lines, one shared normalization.
        _plot_measured_vs_model(ax3, exp_data, modal_before, modal_after, stories, n_use)

        ax3.set_title("Normalized Mode Shapes (UX, max abs = 1)", fontsize=14)
        ax3.set_xlabel("Normalized amplitude", fontsize=12)
        ax3.set_ylabel("Story", fontsize=12)
        ax3.set_yticks(stories)
        ax3.grid(True, alpha=0.3)
        ax3.legend(ncol=3, fontsize=9)
    else:
        ax3.axis("off")
        ax3.set_title("Mode Shapes", fontsize=14)
        ax3.text(
            0.02, 0.95,
            "Mode shapes were not used or are not available\nfor the selected number of stories.\n\n"
            "Frequencies are still used for calibration.",
            va="top", ha="left", fontsize=12
        )

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    txt = [
        "Calibration summary",
        "",
        f"Experimental file: {EXPERIMENTAL_MODAL_JSON}",
        f"Modes used in calibration: {n_use}",
        "",
        f"Original E:   {sci3(original_params['E'])}",
        f"Calibrated E: {sci3(calibrated_params['E'])}",
        "",
        "Mass calibration target:",
        "Total mass incl. additional" if original_params.get("mass_calibration_scope") == "total_mass" else "Self-weight mass only",
        "",
        f"Original self-weight masses:   {[r3(v) for v in original_params['floor_masses']]}",
        f"Calibrated self-weight masses: {[r3(v) for v in calibrated_params['floor_masses']]}",
        "",
        "Column orientation:",
        column_orientation_layout_to_text(
            original_params.get('column_orientation_layout', {}), original_params['nStory']
        ),
        "",
        "Original additional masses [center,c1,c2,c3,c4] kg:",
        additional_masses_to_text(original_params.get('additional_masses', {}), original_params['nStory']),
        "",
        "Calibrated additional masses [center,c1,c2,c3,c4] kg:",
        additional_masses_to_text(calibrated_params.get('additional_masses', {}), calibrated_params['nStory']),
        "",
        "Frequency errors before (%):",
        f"{[r3(safe_percent_error(modal_before['freqs'][i], exp_data['freqs'][i])) for i in range(n_use)]}",
        "",
        "Frequency errors after (%):",
        f"{[r3(safe_percent_error(modal_after['freqs'][i], exp_data['freqs'][i])) for i in range(n_use)]}",
    ]
    ax4.text(0.02, 0.98, "\n".join(txt), va="top", ha="left", fontsize=10)

    fig.tight_layout()
    canvas.draw()
    # Resolved rather than hardcoded: "output/calibration_summary.png" as a
    # default argument put the file wherever the process happened to be running.
    if save_path is None:
        save_path = _out_path("calibration_summary.png", output_base)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
