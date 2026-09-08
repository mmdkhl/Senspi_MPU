# -*- coding: utf-8 -*-
"""
CLI entry point for the OpenSees Model Updating package.

This mirrors the main() function from DigitalTwin_V8.py.
"""

from ..paths import output_dir as _out_dir, output_path as _out_path
import os
import copy

import openseespy.opensees as ops

from ..gui.main_window import launch_input_window
from ..io.loaders import EXPERIMENTAL_MODAL_JSON, write_json
from ..analysis.modal import (
    extract_modal_results,
    export_modal_files,
    plot_modal_results_calibrated_only,
)
from ..analysis.transient import run_transient_analysis_with_visualization


def main():
    """
    Main CLI entry point: launch GUI, then run analysis.
    """
    params = launch_input_window()

    os.makedirs("input", exist_ok=True)
    _out_dir()

    show_info = params["show_info"]

    if show_info:
        print("Current working directory:", os.getcwd())
        print("Excitation file exists:", os.path.exists(params["gmFile"]))
        print("Experimental modal JSON exists:", os.path.exists(EXPERIMENTAL_MODAL_JSON))

    use_precalibrated = bool(params.get("use_precalibrated", False))
    overlay_response = params.get("overlay_uncalibrated_response", None)

    run_params = copy.deepcopy(params)
    run_params.pop("gui_action", None)
    run_params.pop("use_precalibrated", None)
    run_params.pop("precalibrated_params", None)
    run_params.pop("overlay_uncalibrated_response", None)

    write_json(_out_path("current_run_inputs.json"), run_params)

    if use_precalibrated:
        print("\nUsing calibrated parameters stored by the Calibrate button.")
        final_params = copy.deepcopy(params["precalibrated_params"])
        write_json(_out_path("calibrated_inputs_used_for_run.json"), final_params)
    else:
        print("\nNo stored calibration was selected. Running the current input model.")
        final_params = run_params

    print("\nPreparing model for analysis...")
    final_modal = extract_modal_results(final_params, normalize_modes=True, show_info=show_info)
    export_modal_files(final_modal, "run_model")

    if params["run_transient"]:
        mode_title = "Calibrated" if use_precalibrated else "Run Model"
        print("\nShowing mode shapes...")
        plot_modal_results_calibrated_only(
            final_modal, title_prefix=mode_title, block=False
        )

        print("\nRunning transient analysis...")
        run_transient_analysis_with_visualization(
            final_params, final_modal, show_info=show_info,
            overlay_response=overlay_response
        )
    else:
        plot_modal_results_calibrated_only(final_modal, title_prefix="Run Model")

    ops.wipe()
    print("\nDone. Exported files are in the 'output' folder.")
    print("Main summary figure: output/calibration_summary.png")


if __name__ == "__main__":
    main()
