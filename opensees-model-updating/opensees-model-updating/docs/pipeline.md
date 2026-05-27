# Data Pipeline Reference

This document describes every file that flows **into** and **out of** the application,
where each is consumed in code, and how to connect an external data source in the future.

---

## Overview

```
[Experiment / Sensor system]
         │
         ▼
  input/experimental_modal_data.json   ◄── PRIMARY INTEGRATION POINT
         │
         ▼
  opensees_model_updating/io/loaders.py  (src/io/loaders.py)
  load_experimental_modal_data()
         │
         ▼
  opensees_model_updating/calibration/calibrator.py  (src/calibration/calibrator.py)
  prepare_experimental_modal_data()
  modal_residuals()  ◄── optimizer objective function
  run_calibration()
         │
         ▼
  output/calibrated_inputs.json
  output/modal_comparison_report.json
  output/calibration_summary.png
         │
         ▼
  [Transient analysis with calibrated model]
         │
         ▼
  output/*.out  (node recorders)
  output/original_transient_response_overlay.npz
```

---

## Input files

### 1. `input/experimental_modal_data.json` — **main integration point**

**What it contains:**

```json
{
  "frequencies_hz": [2.1, 5.8, 11.3],
  "mode_shapes_ux": [
    [0.33, 0.67, 1.00],
    [0.87, 0.50, -1.00],
    [1.00, -0.78, 0.34]
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `frequencies_hz` | `list[float]` | **Yes** | Identified natural frequencies in Hz, ordered from lowest to highest |
| `mode_shapes_ux` | `list[list[float]]` | No | UX mode shape values at each floor level, one list per mode; roof value should be largest magnitude |

**Rules:**
- `frequencies_hz` must have at least as many entries as `nCalibModes` (set in the GUI).
- `mode_shapes_ux[i]` must have exactly `nStory` values (one per floor). If omitted, the calibrator uses frequencies only.
- Values are not required to be normalized — the code normalizes internally using max-abs normalization.

**Where it is read:**

```
src/io/loaders.py
  → EXPERIMENTAL_MODAL_JSON = "input/experimental_modal_data.json"   (line 12)
  → load_experimental_modal_data(json_path, nStory, require_mode_shapes)
```

```
src/calibration/calibrator.py
  → prepare_experimental_modal_data(params)   (calls load_experimental_modal_data)
```

**How to connect an automated pipeline:**

Replace or overwrite `input/experimental_modal_data.json` before calling the calibrator.
If running headlessly (no GUI), call the Python API directly:

```python
from opensees_model_updating.calibration.calibrator import (
    prepare_experimental_modal_data,
    run_calibration,
)
from opensees_model_updating.analysis.modal import extract_modal_results

# Write your sensor data here
import json, pathlib
pathlib.Path("input/experimental_modal_data.json").write_text(
    json.dumps({"frequencies_hz": [2.1, 5.8], "mode_shapes_ux": [[0.5, 1.0], [1.0, -0.6]]})
)

# Then run calibration
params = { ... }   # model parameters dict (same keys as GUI)
exp_data = prepare_experimental_modal_data(params)
calib_result, calibrated_params = run_calibration(params, exp_data)
```

---

### 2. `input/sine_*Hz_accel.txt` — ground-motion records

Plain-text, one acceleration value per line (m/s²), sampled at `dtGM` seconds.

Selected in the GUI **Ground-motion file** field.  
Read in:

```
src/analysis/transient.py
  → setup_dynamic_excitation(params)
  → ops.timeSeries('Path', ..., '-filePath', params["gmFile"])
```

To connect a real accelerometer feed, write the record as a plain-text file and point the GUI field (or `params["gmFile"]`) to it.

---

## Output files

All outputs land in `output/` relative to the working directory.

| File | Produced by | Description |
|---|---|---|
| `original_inputs.json` | `gui/main_window.py` → `calibrate_clicked()` | Raw GUI parameters before calibration |
| `calibrated_inputs.json` | same | Parameters after calibration |
| `calibrated_inputs_used_for_run.json` | `cli/main.py` | Exact params sent to transient analysis |
| `current_run_inputs.json` | `cli/main.py` | Params for the current run |
| `experimental_modal_data_loaded.json` | `gui/main_window.py` → `calibrate_clicked()` | Experimental data as parsed |
| `modal_comparison_report.json` | `reporting/calibration_report.py` | Structured before/after comparison |
| `modal_comparison_report.txt` | same | Human-readable version |
| `calibration_summary.png` | same | 4-panel figure (frequencies, masses, mode shapes, text) |
| `original_transient_response_overlay.npz` | `analysis/transient.py` | Roof time-history before calibration (for overlay plot) |
| `original_periods.out` | `analysis/modal.py` | OpenSees period file, pre-calibration |
| `calibrated_periods.out` | same | Post-calibration |
| `*_mode_shapes_normalized.out` | same | Normalized UX mode shapes |
| `*time_floor_disp_X.out` | `analysis/transient.py` | Floor displacement recorder |
| `*time_floor_accel_X.out` | same | Floor acceleration recorder |
| `*time_roof_disp_X.out` | same | Roof displacement recorder |
| `*time_roof_accel_X.out` | same | Roof acceleration recorder |

---

## Future integration checklist

To fully automate this pipeline without the GUI:

- [ ] Write `input/experimental_modal_data.json` from your sensor/identification software
- [ ] Write the ground-motion record to `input/<name>.txt`
- [ ] Build a `params` dict (all keys listed in `src/gui/main_window.py → collect_gui_inputs()`)
- [ ] Call `prepare_experimental_modal_data` → `run_calibration` → `extract_modal_results`
- [ ] Read outputs from `output/` or consume the return values directly in Python
