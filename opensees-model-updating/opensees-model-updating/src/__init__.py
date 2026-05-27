"""
opensees_model_updating
=======================

Automated OpenSees modal calibration and model updating for 3-D aluminum frames.

Typical usage::

    # From the project root with the opensees conda environment active:
    python run.py
"""

__version__ = "1.0.0"

from .cli.main import main
from .gui.main_window import launch_input_window
from .analysis.modal import extract_modal_results
from .analysis.transient import run_transient_analysis_with_visualization
from .calibration.calibrator import run_calibration
from .io.loaders import load_experimental_modal_data

__all__ = [
    "main",
    "launch_input_window",
    "extract_modal_results",
    "run_transient_analysis_with_visualization",
    "run_calibration",
    "load_experimental_modal_data",
]
