# opensees-model-updating

> Automated finite-element model updating for 3-D aluminum frames using OpenSeesPy and scipy.

A desktop application that builds a parametric OpenSees model of a multi-story 3-D moment frame, runs an automatic modal calibration against experimentally identified frequencies (and optionally mode shapes), then performs a Newmark transient analysis with real-time visualization — all driven by a three-tab Tkinter GUI.

---

## Table of contents

- [Background](#background)
- [How model updating works](#how-model-updating-works)
  - [The finite-element model](#the-finite-element-model)
  - [The calibration algorithm](#the-calibration-algorithm)
  - [The transient analysis](#the-transient-analysis)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Running the application](#running-the-application)
- [GUI walkthrough](#gui-walkthrough)
- [Input files](#input-files)
- [Output files](#output-files)
- [Repository layout](#repository-layout)
- [Python API](#python-api)
- [Legacy reference](#legacy-reference)
- [Contributing](#contributing)

---

## Background

Structural model updating is the process of adjusting the parameters of a numerical finite-element model so that its dynamic behaviour matches measurements taken from a real structure. This repository implements a practical workflow for a scaled laboratory aluminum frame:

1. **System identification** — modal frequencies (and optionally mode shapes) are extracted from measured vibration data and stored in a JSON file.
2. **FE model construction** — OpenSeesPy builds a 3-D Euler–Bernoulli beam-column frame with rigid diaphragms, self-weight masses, and optional point masses.
3. **Automatic calibration** — a bounded nonlinear least-squares optimizer adjusts the Young's modulus scale and per-story mass scales to minimise the mismatch between numerical and experimental modal properties.
4. **Transient analysis** — the calibrated model is excited by a recorded ground-motion and the roof displacement response is animated in real time.

---

## How model updating works

### The finite-element model

The frame is a 1-bay-by-1-bay, N-story structure built entirely in OpenSeesPy:

- **Nodes** — four corner nodes and one master node per floor (plus four fixed base nodes). The master node sits at the geometric centroid of each floor.
- **Elements** — `elasticBeamColumn` elements for all columns and beams, using tabulated section properties derived from user-supplied dimensions (thickness × width for columns, height × width for beams).
- **Column orientation** — each column can be individually set to *weak-axis* or *strong-axis* bending relative to the X-direction of excitation. Internally this changes the geometric transformation vector from `(1,0,0)` to `(0,1,0)`.
- **Rigid diaphragm** — `ops.rigidDiaphragm(3, master, *corner_nodes)` constrains each floor to in-plane rigid-body motion in the horizontal plane.
- **Mass** — translational mass is lumped at the master node (self-weight) and optionally at each corner (additional point masses). The rotational mass moment of inertia for each floor is computed from its plan dimensions and total mass.
- **Gravity** — a static load case applies gravity to the corner masses, and `ops.loadConst('-time', 0.0)` resets the time origin before dynamic analysis.

Section properties are computed from standard thin-walled mechanics:

```
I = (b * h^3)/12  -  (b - 2t) * (h - 2t)^3 / 12
A = b*h  -  (b - 2t)*(h - 2t)
J ≈ 2*t*(b-t)^2*(h-t)^2 / (b + h - 2t)
```

### The calibration algorithm

The calibration is a **bounded nonlinear least-squares** problem solved by `scipy.optimize.least_squares` with the Trust-Region Reflective (`trf`) method and `soft_l1` loss (robust to outliers).

**Calibration parameters** — the optimizer searches over a vector:

```
x = [alpha_E,  alpha_m1,  alpha_m2,  ...,  alpha_mN]
```

where `alpha_E` scales Young's modulus and `alpha_mi` scales the mass of story `i`.
All scales start at `1.0` (no change) and are bounded to `[lb, ub]` (default 0.70–1.30).

**Objective function** — at each optimizer iteration the full OpenSees model is rebuilt and an eigenvalue analysis is run. The residual vector is:

```
r = [w_freq * (f_num_k - f_exp_k) / f_exp_k]   for k = 1 .. n_calib_modes
  + [w_mode * (phi_num_kj - phi_exp_kj)]        for each mode k, floor j  (optional)
```

where frequencies are in Hz and mode shapes are max-abs-normalised and sign-aligned before differencing. If the model fails during an optimizer iteration, a large penalty (1×10³) is returned to steer the solver away from that region.

**Eigenvalue solver fallback** — OpenSees uses ARPACK by default. For small models or when many modes are requested relative to the number of inertial DOFs, ARPACK can be numerically unstable. The code automatically retries with `fullGenLapack` if ARPACK fails.

**Result** — the optimizer returns the best-fit `x*` and the calibrated parameters become:

```
E_cal   = alpha_E*   * E_nominal
m_i_cal = alpha_mi*  * m_i_nominal    for each story i
```

### The transient analysis

After calibration, Rayleigh damping coefficients are computed from the first two numerical modes to target a uniform damping ratio `zeta`:

```
alpha = 2 * zeta * omega1 * omega2 / (omega1 + omega2)
beta  = 2 * zeta / (omega1 + omega2)
```

Ground motion is applied as a uniform base excitation in the X-direction via `ops.UniformExcitation`. The time integration uses the **Newmark average acceleration method** (`gamma = 0.5`, `beta = 0.25`, unconditionally stable). Floor displacement and acceleration recorders are written to `output/*.out`. During the run, a live matplotlib animation updates every 10 time steps, showing the deformed frame shape at 20× amplification alongside the roof displacement time-history. If a pre-calibration response was stored, it is overlaid in grey for direct visual comparison.

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/opensees-model-updating.git
cd opensees-model-updating

# 2. Create the conda environment
conda env create -f environment.yml
conda activate opensees

# 3. Run
python run.py
```

---

## Installation

### Requirements

| Dependency | Minimum version | Purpose |
|---|---|---|
| Python | 3.9 | Language runtime |
| openseespy | 3.4.0 | FE model + analysis |
| numpy | 1.20 | Numerical arrays |
| scipy | 1.7 | Least-squares optimizer |
| matplotlib | 3.3 | Plots + real-time animation |
| opsvis | 1.0 | Mode-shape visualisation |

### Step by step

```bash
# Using the provided environment file (recommended)
conda env create -f environment.yml
conda activate opensees

# Or manually
conda create -n opensees python=3.11
conda activate opensees
pip install openseespy numpy scipy matplotlib opsvis
```

---

## Running the application

```bash
# From the repository root, with the opensees environment active:
python run.py
```

---

## GUI walkthrough

The GUI has three tabs and two action buttons.

### Tab 1 — Basic Model

| Field | Description |
|---|---|
| Building length X / width Y | Plan dimensions in metres |
| Number of stories | Updating this field instantly rebuilds all story tables |
| Column thickness / width | Cross-section of each column (m) |
| Beam thickness / height | Cross-section of each beam (m) |
| Story height table | Per-story height (m) and self-weight floor mass (kg) |
| Columns present | Checkboxes C1–C4 per story; unchecked columns are omitted from the model |
| Column orientation | Per-story, per-column: Weak axis or Strong axis relative to X-shaking |

The 3-D sketch on the right updates live as you change the story count or column orientation.

### Tab 2 — Mass + Analysis

| Field | Description |
|---|---|
| Elastic modulus E | Young's modulus in Pa (e.g. `200e9`) |
| Poisson ratio ν | Used to derive shear modulus G |
| Additional mass table | Per-story extra masses (kg) at the floor centroid and corners C1–C4 |
| Number of modes | Total eigenvalues to extract from the FE model |
| Damping ratio ζ | Target Rayleigh damping ratio |
| Ground-motion file | Path to a plain-text acceleration record (m/s²) |
| Ground-motion dt | Sampling time step of the record (s) |
| Ground-motion scale factor | Multiplier applied to the record (e.g. `9.81` to convert g to m/s²) |
| Run transient after calibration | If unchecked, only modal analysis is performed |

### Tab 3 — Calibration

| Field | Description |
|---|---|
| Enable automatic calibration | Toggle the entire calibration step |
| Use mode shapes | Include mode-shape residuals in the objective (requires shapes in JSON) |
| Mass calibration target | Scale self-weight masses only, or self-weight + additional masses |
| Number of calibration modes | How many experimental modes to match (must be ≤ numModes and ≤ nStory) |
| Frequency / mode-shape weights | Relative weighting of the two residual components |
| E scale bounds | Lower and upper multiplier limits for Young's modulus |
| Mass scale bounds | Lower and upper multiplier limits for each story mass |
| Max optimizer evaluations | Hard cap on function evaluations; increase for tighter convergence |
| Show uncalibrated response | Overlay the pre-calibration roof response on the final animation |

### Action buttons

| Button | What it does |
|---|---|
| **Calibrate** | Runs calibration, saves all reports to `output/`, stores calibrated state in memory |
| **Run Analysis** | Uses the stored calibrated parameters (or current inputs if not yet calibrated) to run modal + transient analysis |
| Cancel | Exits the application |

> **Tip:** always click **Calibrate** before **Run Analysis**. If you change any input after calibrating, the application warns you and lets you choose whether to proceed uncalibrated or go back and recalibrate.

---

## Input files

### `input/experimental_modal_data.json`

The primary integration point. See [docs/pipeline.md](docs/pipeline.md) for the full data-flow reference and how to connect an automated sensor pipeline.

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

| Field | Required | Description |
|---|---|---|
| `frequencies_hz` | **Yes** | Identified natural frequencies in Hz, ordered lowest to highest |
| `mode_shapes_ux` | No | UX mode shape at each floor (bottom to top), one list per mode. Values need not be normalised. |

Rules:
- The list must contain at least `nCalibModes` entries.
- If `mode_shapes_ux` is provided, each inner list must have exactly `nStory` values.
- If omitted, calibration runs on frequencies only.

### `input/sine_*Hz_accel.txt`

Plain-text ground-motion records, one acceleration value per line (m/s²), sampled uniformly at `dtGM` seconds. Point the **Ground-motion file** field to whichever record you want to use.

---

## Output files

All files are written to `output/` relative to the working directory. The folder is gitignored (only `.gitkeep` is committed).

| File | Description |
|---|---|
| `original_inputs.json` | Full parameter dict before calibration |
| `calibrated_inputs.json` | Full parameter dict after calibration |
| `calibrated_inputs_used_for_run.json` | Exact parameters sent to transient analysis |
| `current_run_inputs.json` | Parameters for the most recent run |
| `experimental_modal_data_loaded.json` | Experimental data as parsed and validated |
| `modal_comparison_report.json` | Structured before/after comparison (frequencies, percent errors) |
| `modal_comparison_report.txt` | Human-readable version of the same report |
| `calibration_summary.png` | 4-panel figure: frequency bar chart, mass bar chart, mode shapes, text summary |
| `original_transient_response_overlay.npz` | Pre-calibration roof time-history for overlay plot |
| `original_periods.out` | OpenSees period file, original model |
| `calibrated_periods.out` | OpenSees period file, calibrated model |
| `*_mode_shapes_normalized.out` | Max-abs-normalised UX mode shapes |
| `*time_floor_disp_X.out` | Floor node displacement recorder |
| `*time_floor_accel_X.out` | Floor node acceleration recorder |
| `*time_roof_disp_X.out` | Roof master-node displacement recorder |
| `*time_roof_accel_X.out` | Roof master-node acceleration recorder |

---

## Repository layout

```
opensees-model-updating/
├── docs/
│   ├── pipeline.md                  # Data-flow reference + automation guide
│   ├── developer_guide.md           # Architecture, call graph, extension guide
│   └── legacy_monolith.py           # Original single-file script (reference only)
├── input/                           # Runtime input data (committed)
│   ├── experimental_modal_data.json
│   └── sine_*Hz_accel.txt
├── output/                          # Generated at run time (gitignored)
│   └── .gitkeep
├── src/                             # The Python package (opensees_model_updating)
│   ├── __init__.py
│   ├── __main__.py
│   ├── analysis/
│   │   ├── gravity.py           — static gravity + loadConst reset
│   │   ├── modal.py             — eigenvalue extraction, ARPACK fallback, plots
│   │   └── transient.py         — Rayleigh damping, Newmark integration, animation
│   ├── calibration/
│   │   └── calibrator.py        — TRF optimizer, residual function, bounds
│   ├── cli/
│   │   └── main.py              — post-GUI analysis orchestration
│   ├── domain/
│   │   └── enums.py             — ColumnOrientation, AnalysisType, MassCalibrationScope
│   ├── gui/
│   │   ├── main_window.py       — 3-tab Tkinter GUI, Calibrate + Run buttons
│   │   └── widgets/
│   │       └── canvas_widget.py — plan-view legend, 3-D column sketch
│   ├── io/
│   │   └── loaders.py           — JSON/text I/O, experimental data validation
│   ├── model/
│   │   └── builder.py           — nodes, elements, rigid diaphragm, mass
│   ├── reporting/
│   │   └── calibration_report.py — report dicts, text formatting, summary figure
│   ├── utils/
│   │   ├── formatters.py        — r3(), sci3(), deep_round(), layout text helpers
│   │   └── math_utils.py        — mode normalisation, sign alignment, percent error
│   └── visualization/
│       └── __init__.py          — re-exports from analysis modules
├── .gitattributes
├── .gitignore
├── environment.yml
├── pyproject.toml
├── README.md
└── run.py                           — one-line launcher
```

---

## Python API

The package can be used without the GUI for automated pipelines:

```python
import json
import pathlib
from opensees_model_updating.calibration.calibrator import (
    prepare_experimental_modal_data,
    run_calibration,
)
from opensees_model_updating.analysis.modal import extract_modal_results, export_modal_files
from opensees_model_updating.reporting.calibration_report import (
    make_modal_comparison_report,
    save_calibration_summary_figure,
)

# Write experimental data (replace with your sensor pipeline output)
pathlib.Path("input/experimental_modal_data.json").write_text(
    json.dumps({
        "frequencies_hz": [2.1, 5.8],
        "mode_shapes_ux": [[0.5, 1.0], [1.0, -0.6]],
    })
)

# Define model parameters (same keys that the GUI collects)
params = {
    "nStory": 2,
    "Lx": 0.245, "Ly": 0.23,
    "story_heights": [0.24, 0.24],
    "floor_masses": [0.27, 0.27],
    "t_column": 0.001, "b_column": 0.006,
    "b_beam": 0.001,   "h_beam": 0.008,
    "E": 200e9, "nu": 0.33,
    "numModes": 2, "nCalibModes": 2,
    "zeta": 0.005, "gmFactor": 9.81,
    "gmFile": "input/sine_1Hz_accel.txt", "dtGM": 0.01,
    "enable_calibration": True, "use_mode_shapes": False,
    "mass_calibration_scope": "self_weight_only",
    "freq_tol_percent": 5.0, "w_freq": 1.0, "w_mode": 0.35,
    "E_scale_lb": 0.70, "E_scale_ub": 1.30,
    "m_scale_lb": 0.70, "m_scale_ub": 1.30,
    "max_nfev": 200, "show_info": False,
    "story_column_layout": {1: [1,2,3,4], 2: [1,2,3,4]},
    "column_orientation_layout": {
        1: {1: "weak", 2: "weak", 3: "weak", 4: "weak"},
        2: {1: "weak", 2: "weak", 3: "weak", 4: "weak"},
    },
    "additional_masses": {1: [0,0,0,0,0], 2: [0,0,0,0,0]},
}

exp_data     = prepare_experimental_modal_data(params)
modal_before = extract_modal_results(params)
calib_result, calibrated_params = run_calibration(params, exp_data)
modal_after  = extract_modal_results(calibrated_params)

export_modal_files(modal_before, "original")
export_modal_files(modal_after,  "calibrated")

report = make_modal_comparison_report(
    exp_data, modal_before, modal_after, params, calibrated_params, calib_result
)
save_calibration_summary_figure(
    exp_data, modal_before, modal_after, params, calibrated_params,
    save_path="output/calibration_summary.png",
)
print("Calibrated frequencies (Hz):", modal_after["freqs"])
```

See [docs/pipeline.md](docs/pipeline.md) for the complete integration reference including how to feed live sensor data.

---

## Legacy reference

[docs/legacy_monolith.py](docs/legacy_monolith.py) is the original single-file script (`DigitalTwin_V8.py`) from which this package was refactored. It is kept for historical reference only — do not run or import it. The developer guide contains a [function-by-function mapping](docs/developer_guide.md#legacy-reference) from the old script to the current package modules.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Follow the existing code style — `black` for formatting, `ruff` for linting.
3. Add or update tests under `src/tests/`.
4. Open a pull request with a clear description of the change.

```bash
# Format and lint before committing
black src/
ruff check src/
pytest src/tests/
```
