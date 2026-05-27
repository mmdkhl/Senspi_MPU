# Developer Guide

This guide is for developers picking up the codebase. It covers environment setup,
project layout, how the pieces connect, and how to extend the package.

---

## Repository layout

```
opensees-model-updating/
├── .claude/                    # AI-agent planning docs — do not delete
├── docs/
│   ├── pipeline.md             # Data flow + integration points (start here)
│   └── developer_guide.md     # This file
├── input/                      # Runtime input data (committed)
│   ├── experimental_modal_data.json
│   └── sine_*Hz_accel.txt
├── output/                     # Runtime outputs (gitignored, created at run time)
│   └── .gitkeep
├── src/                        # The Python package (maps to opensees_model_updating)
│   ├── __init__.py
│   ├── __main__.py
│   ├── analysis/
│   │   ├── gravity.py          — static gravity + load-const reset
│   │   ├── modal.py            — eigenvalue extraction, mode-shape plots
│   │   └── transient.py        — Newmark integration, real-time animation
│   ├── calibration/
│   │   └── calibrator.py       — scipy TRF optimizer, objective function
│   ├── cli/
│   │   └── main.py             — main() entry point (post-GUI analysis flow)
│   ├── domain/
│   │   └── enums.py            — ColumnOrientation, AnalysisType, MassCalibrationScope
│   ├── gui/
│   │   ├── main_window.py      — launch_input_window() full Tkinter GUI
│   │   └── widgets/
│   │       └── canvas_widget.py — plan-view + 3D sketch helpers
│   ├── io/
│   │   └── loaders.py          — JSON/text I/O, experimental data loading/validation
│   ├── model/
│   │   └── builder.py          — OpenSees model construction (nodes, elements, mass)
│   ├── reporting/
│   │   └── calibration_report.py — report dicts, text, 4-panel figure
│   ├── utils/
│   │   ├── formatters.py       — rounding, layout-to-text helpers
│   │   └── math_utils.py       — mode normalization, sign alignment, percent error
│   └── visualization/
│       └── __init__.py         — re-exports from analysis modules
├── environment.yml             # Conda environment spec
├── pyproject.toml              # Build config + entry points
├── README.md
└── run.py                      # One-file launcher
```

---

## Setup

```bash
# 1. Create the environment from the spec
conda env create -f environment.yml
conda activate opensees

# 2. Launch
python run.py
```

---

## Entry points

| Command | Resolves to |
|---|---|
| `python run.py` | `src/cli/main.py → main()` |

---

## Call graph (high level)

```
main()                          cli/main.py
  └── launch_input_window()     gui/main_window.py
        ├── [Calibrate button]
        │     ├── prepare_experimental_modal_data()   calibration/calibrator.py
        │     ├── extract_modal_results()             analysis/modal.py
        │     ├── run_calibration()                   calibration/calibrator.py
        │     ├── make_modal_comparison_report()      reporting/calibration_report.py
        │     └── run_transient_analysis_collect_data()  analysis/transient.py
        └── [Run Analysis button]
              ├── extract_modal_results()
              ├── plot_modal_results_calibrated_only()
              └── run_transient_analysis_with_visualization()

extract_modal_results()         analysis/modal.py
  ├── build_model()             model/builder.py
  ├── run_gravity_analysis()    analysis/gravity.py
  └── run_eigen_with_fallback()
```

---

## Key data structures

### `params` dict

The central configuration object built by `collect_gui_inputs()` in `gui/main_window.py`.
All downstream functions accept it. Keys:

| Key | Type | Description |
|---|---|---|
| `nStory` | `int` | Number of stories |
| `Lx`, `Ly` | `float` | Plan dimensions (m) |
| `story_heights` | `list[float]` | Height of each story (m), length = nStory |
| `floor_masses` | `list[float]` | Self-weight mass per floor (kg), length = nStory |
| `t_column`, `b_column` | `float` | Column cross-section dimensions (m) |
| `b_beam`, `h_beam` | `float` | Beam cross-section dimensions (m) |
| `E` | `float` | Elastic modulus (Pa) |
| `nu` | `float` | Poisson ratio |
| `numModes` | `int` | Number of modes to extract |
| `zeta` | `float` | Rayleigh damping ratio |
| `gmFile` | `str` | Path to ground-motion record |
| `dtGM` | `float` | Time step of ground-motion record (s) |
| `gmFactor` | `float` | Scale factor applied to record |
| `story_column_layout` | `dict[int, list[int]]` | `{story: [col_ids present]}` |
| `column_orientation_layout` | `dict[int, dict[int, str]]` | `{story: {col_id: "weak"/"strong"}}` |
| `additional_masses` | `dict[int, list[float]]` | `{story: [center, C1, C2, C3, C4]}` in kg |
| `enable_calibration` | `bool` | Whether to run calibration |
| `nCalibModes` | `int` | Modes used in calibration objective |
| `mass_calibration_scope` | `str` | `"self_weight_only"` or `"total_mass"` |
| `w_freq`, `w_mode` | `float` | Objective function weights |
| `E_scale_lb/ub`, `m_scale_lb/ub` | `float` | Optimizer bounds |

### `modal_data` dict

Returned by `extract_modal_results()`. Keys:

| Key | Description |
|---|---|
| `freqs` | `np.array` — natural frequencies (Hz) |
| `periods` | `np.array` — periods (s) |
| `mode_shapes_ux_master` | `list[np.array]` — UX mode shapes at master nodes |
| `ctx` | Model context dict from `build_model()` |

### `exp_data` dict

Returned by `prepare_experimental_modal_data()`. Keys:

| Key | Description |
|---|---|
| `frequencies_hz` | `np.array` — experimental frequencies |
| `mode_shapes_ux` | `list[np.array]` — normalized experimental mode shapes (or `None`) |
| `mode_shapes_available` | `bool` |
| `raw_data` | Original parsed JSON |

---

## Adding a new analysis type

1. Add a value to `domain/enums.py → AnalysisType`.
2. Create `src/analysis/<new_type>.py`.
3. Export from `src/analysis/__init__.py`.
4. Wire a button or parameter in `src/gui/main_window.py`.

## Adding a new calibration parameter

1. Add the bound entries to the GUI in `gui/main_window.py → Tab 3`.
2. Extend `collect_gui_inputs()` to read and validate the new field.
3. Extend `apply_calibration_vector()` in `calibration/calibrator.py` to apply it.
4. Update `modal_residuals()` if it changes the objective.

---

## Running tests

```bash
conda activate opensees
pytest src/tests/ -v
```

Tests live in `src/tests/`.

---

## Linting and formatting

```bash
black src/
ruff check src/
```

---

## Legacy reference

`docs/legacy_monolith.py` is the original single-file implementation (`DigitalTwin_V8.py`)
from which this package was refactored. It is kept **for reference only** and must not
be run or imported.

Useful for tracing how a specific behaviour was originally coded:

| Original function | Now lives in |
|---|---|
| `build_model()` | `src/model/builder.py` |
| `run_gravity_analysis()` | `src/analysis/gravity.py` |
| `run_eigen_with_fallback()` | `src/analysis/modal.py` |
| `run_transient_analysis_*()` | `src/analysis/transient.py` |
| `run_calibration()` / `modal_residuals()` | `src/calibration/calibrator.py` |
| `launch_input_window()` | `src/gui/main_window.py` |
| `make_modal_comparison_report()` | `src/reporting/calibration_report.py` |
| `load_experimental_modal_data()` | `src/io/loaders.py` |
| `r3()`, `sci3()`, `deep_round()` | `src/utils/formatters.py` |
| `normalize_mode_maxabs()` | `src/utils/math_utils.py` |
