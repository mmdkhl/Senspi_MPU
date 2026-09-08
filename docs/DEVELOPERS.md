# SensePi Developer Notes

This document is for developers who want to modify SensePi or understand how it works internally.

---

## 1) Quick dev setup (PC)

From the repo root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

Run the GUI:

```bash
python -m sensepi.gui.application
# or
sensepi-gui
```

Run unit tests:

```bash
python -m unittest discover -s tests
```

---

## 2) Repository layout (what lives where)

- `src/sensepi/gui/`  
  Qt application + the six tabs, `MainWindow`, and `RecorderController`

- `src/sensepi/remote/`  
  SSH + remote process control + log sync

- `raspberrypi_scripts/`  
  The scripts copied to the Raspberry Pi (e.g. `mpu6050_multi_logger.py`)

- `src/sensepi/config/`  
  YAML + dataclasses for hosts/sensors/sampling and shared path conventions

- `src/sensepi/analysis/`  
  Pure NumPy/SciPy, no Qt: modal identification (`modal.py`), the sensor-map
  adapter (`sensor_layout.py`), base-referenced identification
  (`transmissibility.py`) and torsion indicators (`torsion.py`)

- `src/sensepi/digital_twin/`, `src/sensepi/sonification/`  
  The engines behind those two tabs, also Qt-free so they stay testable

- `src/opensees_model_updating/`  
  The OpenSees calibration package. Also runs standalone — see `paths.py` for how
  its output location is resolved without assuming a working directory

- `docs/`  
  `DEVELOPERS.md` (this file), `gui_pipeline_diagram.mmd`, the user manual, and
  `hardware/` — enclosure and bracket design source (Fusion `.f3d`, PCB `.dxf`,
  printable `.stl`, plus the `trimesh` script that generates the OLED bracket).
  Both the manual and the hardware files predate the current software and are
  kept for reference, not as a description of it.

- `output/` and `logs/`  
  Local runtime folders (created automatically; git-ignored). Everything the
  application *produces* lives under one root:

  ```
  output/
    sensor_recordings/<host>/mpu/<stamp>[_<name>]/   one folder per Smart Record
                                                     (main CSVs + audit_<stamp>/ six-channel copy)
    model/                          the default model workspace: input/ and output/
                                    (OpenSees recorders and calibration files)
    digital_twin/<stamp>/           experiment runs
    sonification/                   audio captures
  logs/                             diagnostics — deliberately separate
  ```

  Smart Record writes the samples **as received** (timestamps, decimation) plus an
  audit of rate, gaps and clock drift; alignment onto a common grid happens on the
  read side, in `dataio.modal_session_loader.align_per_sensor_series`, every time a
  session is loaded.

  Resolve these through `AppPaths` (`src/sensepi/config/app_config.py`), never by
  building a relative path. `SENSEPI_OUTPUT_ROOT` moves the whole tree,
  `SENSEPI_DATA_ROOT` moves the recordings folder alone, `SENSEPI_LOG_DIR` the logs.

---

## 3) Config files

### Desktop-side
- `src/sensepi/config/hosts.yaml`  
  List of Pis (host, user, password, paths)

- `src/sensepi/config/sensors.yaml`  
  Sensor defaults + sampling defaults  
  Tracked template. Per-machine state (the sensor placement map, the sensor count)
  is written to the gitignored `sensors.local.yaml`, which takes precedence when
  present — the same pattern as `hosts.local.yaml`.

### Pi-side
- `pi_config.yaml` (uploaded to each Pi)  
  Built from the desktop defaults and uploaded to `HostConfig.pi_config_path`

The GUI treats the desktop config as the source of truth.

---

## 4) Runtime architecture (end-to-end)

### Tabs in the current GUI
The main window builds these tabs:
- **Live Signals** (`src/sensepi/gui/tabs/tab_signals.py`)
- **Spectrum / FFT** (`src/sensepi/gui/tabs/tab_fft.py`)
- **Model Updating** (`src/sensepi/gui/tabs/tab_model_updating.py`)
- **Sonification** (`src/sensepi/gui/tabs/tab_sonification.py`) — a container with
  one sub-tab per model
- **Digital Twin Experiment** (`src/sensepi/gui/tabs/tab_digital_twin.py`) — runs
  the calibrated model in wall-clock time beside the real structure and compares them
- **Settings** (`src/sensepi/gui/tabs/tab_settings.py`)

### Sensor placement: defined once, in Settings
Where each sensor sits — its floor and its plan cell on a 3x3 grid — is set **only**
in Settings (`gui/widgets/sensor_map.py`) and fanned out by `MainWindow` to every
tab that needs it, through `apply_sensor_map()`.

**Do not add a sensor, floor or storey picker to any other tab.** Four of them used
to exist, they could disagree about the same rig, and they were removed. If a tab
needs to know where a sensor is, it takes the map and converts it with
`analysis/sensor_layout.py`, which answers the questions that actually matter:
which sensors are structural responses, which one is the shaker (floor 0, an
*input*, excluded from output-only identification), which channel the excitation
axis implies, how many modes the sensor count can support, and where a
differenceable pair exists for torsion.

### The “controller” layer
The GUI does not run SSH logic directly from the plotting tabs. Instead:
- `src/sensepi/gui/recorder_controller.py` owns start/stop, ingest, and shared buffers.
- Tabs talk to the controller via Qt signals/slots.

### Data flow (live streaming)
1. **User clicks Start** in the Live Signals tab.
2. `MainWindow` forwards this to `RecorderController.start_live_stream(...)`.
3. `RecorderController` creates a `PiRecorder` (SSH wrapper) and starts the Pi process.
4. The remote script (`mpu6050_multi_logger.py`) streams **one JSON object per line** on stdout.
5. `SensorIngestWorker` runs in a background thread, reads stdout, parses lines into `MpuSample`,
   and pushes samples into a shared `StreamingDataBuffer`.
6. `SignalsTab` and `FftTab` pull from that buffer on timers and update plots.

Key points:
- SSH / parsing / IO must stay off the Qt GUI thread.
- Plot refresh rates are independent of device sampling rate.

---

## 5) Raspberry Pi logger (what the GUI starts)

The default sensor is the MPU6050 logger:

- `raspberrypi_scripts/mpu6050_multi_logger.py`

The desktop starts it over SSH roughly like:

```text
python3 <base_path>/mpu6050_multi_logger.py --config <pi_config.yaml> ... --stream-stdout
```

(See `src/sensepi/remote/pi_recorder.py`.)

### Pi deployment helper
`deploy_pi.bat` copies:
- `raspberrypi_scripts/*` → `<REMOTE_DIR>/`
- `src/sensepi/config/*` + `src/sensepi/__init__.py` → `<REMOTE_DIR>/sensepi/`

This “mini sensepi package” on the Pi exists because the Pi scripts import shared helpers
like `sensepi.config.log_paths`.

---

## 6) Log conventions + syncing

Shared log/file naming rules live in:

- `src/sensepi/config/log_paths.py`

Syncing logs from a Pi to the PC is handled by:

- `src/sensepi/remote/log_sync.py`
- `src/sensepi/remote/log_sync_worker.py`

The GUI uses the host’s `data_dir` and the sensor prefix (e.g. `mpu`) to decide what to download.

---

## 7) Where to make common changes

### A) Add / change UI controls
- Add widgets in the relevant `tab_*.py`
- Wire actions into `RecorderController` (preferred) instead of doing SSH inside tabs
- Cross-tab wiring belongs in `MainWindow._wire_signals()` — that is the only place
  tabs are connected to each other. Do not reach into another tab's private members;
  add a public method instead (`ModelUpdatingTab.model_definition_snapshot()` is the
  worked example of replacing three such reach-ins with one documented seam).

### B) Add a new plot based on the live stream
- Subscribe to the shared buffer (or expose a signal from the controller)
- Keep plotting updates timer-driven (don’t redraw per-sample)

### C) Add a new sensor type
You typically need:
1. A Pi-side logger script under `raspberrypi_scripts/`
2. A desktop-side parser under `src/sensepi/sensors/`
3. A config entry in `sensors.yaml` + a `PiLoggerConfig` update
4. A way for `PiRecorder` / `RecorderController` to select the correct script + prefix

---

## 8) Tips for performance and stability

- Keep all SSH + file sync in worker threads.
- **Never call `os.chdir()`.** It is process-global: it moves the working directory
  for every thread, not the caller. This application runs SSH ingest, an audio
  worker and a model worker at once, so a relative path in any of them can resolve
  somewhere else for the duration. Pass an explicit directory instead. A test fails
  if `os.chdir` reappears under `src/sensepi`.
- Avoid building new Matplotlib objects every frame; update existing lines/curves.
- Use bounded buffers (ring buffers) for live views.
- Be conservative with default sampling/plot rates so slower PCs stay responsive.
