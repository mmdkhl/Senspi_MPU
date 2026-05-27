# -*- coding: utf-8 -*-
"""I/O utilities for loading and saving data files."""

import json
import os
import numpy as np

from ..utils.formatters import deep_round


# Fixed path for experimental modal data
EXPERIMENTAL_MODAL_JSON = "input/experimental_modal_data.json"


def write_json(path, data, round_values=True):
    """Write data to a JSON file with optional rounding."""
    data_out = deep_round(data) if round_values else data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_out, f, indent=2)


def write_text(path, text):
    """Write text content to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_experimental_modal_data(json_path, nStory, require_mode_shapes=True):
    """
    Load and validate experimental modal data from a JSON file.

    Parameters
    ----------
    json_path : str
        Path to the experimental modal data JSON file.
    nStory : int
        Number of stories in the model.
    require_mode_shapes : bool
        If True, raises an error when mode shapes are missing.

    Returns
    -------
    dict with keys:
        frequencies_hz, mode_shapes_ux, raw_data, mode_shapes_available
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Experimental modal data file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "frequencies_hz" not in data:
        raise ValueError("Experimental JSON must contain 'frequencies_hz'.")

    freqs = data["frequencies_hz"]
    if not isinstance(freqs, list) or len(freqs) < 1:
        raise ValueError("'frequencies_hz' must be a list with at least 1 value.")

    if "mode_shapes_ux" not in data:
        if require_mode_shapes:
            raise ValueError(
                "Experimental JSON must contain 'mode_shapes_ux' when 'Use mode shapes in calibration' is checked. "
                "Uncheck that option to calibrate using frequencies only."
            )
        return {
            "frequencies_hz": np.array([float(x) for x in freqs], dtype=float),
            "mode_shapes_ux": [],
            "raw_data": data,
            "mode_shapes_available": False,
        }

    mode_shapes = data["mode_shapes_ux"]
    if not isinstance(mode_shapes, dict):
        if require_mode_shapes:
            raise ValueError("'mode_shapes_ux' must be a dictionary.")
        return {
            "frequencies_hz": np.array([float(x) for x in freqs], dtype=float),
            "mode_shapes_ux": [],
            "raw_data": data,
            "mode_shapes_available": False,
        }

    modes = []
    for mode_id in range(1, len(freqs) + 1):
        key = str(mode_id)
        if key not in mode_shapes:
            if require_mode_shapes:
                raise ValueError(f"'mode_shapes_ux' must contain mode '{key}'.")
            return {
                "frequencies_hz": np.array([float(x) for x in freqs], dtype=float),
                "mode_shapes_ux": [],
                "raw_data": data,
                "mode_shapes_available": False,
            }

        vals = mode_shapes[key]
        if len(vals) != nStory:
            if require_mode_shapes:
                raise ValueError(
                    f"Experimental mode shape {key} in '{json_path}' has {len(vals)} story values, "
                    f"but the current model has {nStory} stories. "
                    "Update input/experimental_modal_data.json so each mode shape has one value per story, "
                    "or uncheck 'Use mode shapes in calibration' to calibrate using frequencies only."
                )
            return {
                "frequencies_hz": np.array([float(x) for x in freqs], dtype=float),
                "mode_shapes_ux": [],
                "raw_data": data,
                "mode_shapes_available": False,
            }

        modes.append(np.array([float(x) for x in vals], dtype=float))

    return {
        "frequencies_hz": np.array([float(x) for x in freqs], dtype=float),
        "mode_shapes_ux": modes,
        "raw_data": data,
        "mode_shapes_available": True,
    }
