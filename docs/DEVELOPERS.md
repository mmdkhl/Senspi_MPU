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

To enable the `Digital Twin` tab:

```bash
pip install -e ".[digital-twin]"
```

Run unit tests:

```bash
python -m unittest discover -s tests
```

---

## 2) Repository layout (what lives where)

- `src/sensepi/gui/`  
  Qt application + tabs (Live Signals / Spectrum / Settings)

- `src/sensepi/remote/`  
  SSH + remote process control + log sync

- `raspberrypi_scripts/`  
  The scripts copied to the Raspberry Pi (e.g. `mpu6050_multi_logger.py`)

- `src/sensepi/config/`  
  YAML + dataclasses for hosts/sensors/sampling and shared path conventions

- `data/` and `logs/`  
  Local runtime folders (created automatically; usually git-ignored)

---

## 3) Config files

### Desktop-side
- `src/sensepi/config/hosts.yaml`  
  List of Pis (host, user, password, paths)

- `src/sensepi/config/sensors.yaml`  
  Sensor defaults + sampling defaults

### Pi-side
- `pi_config.yaml` (uploaded to each Pi)  
  Built from the desktop defaults and uploaded to `HostConfig.pi_config_path`

The GUI treats the desktop config as the source of truth.

---

## 4) Runtime architecture (end-to-end)

### Tabs in the current GUI
The main window builds three tabs:
- **Live Signals** (`src/sensepi/gui/tabs/tab_signals.py`)
- **Spectrum / FFT** (`src/sensepi/gui/tabs/tab_fft.py`)
- **Digital Twin** (`src/sensepi/gui/tabs/tab_digital_twin.py`)
- **Settings** (`src/sensepi/gui/tabs/tab_settings.py`)

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
- Add widgets in `tab_signals.py`, `tab_fft.py`, or `tab_settings.py`
- Wire actions into `RecorderController` (preferred) instead of doing SSH inside tabs

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
- Avoid building new Matplotlib objects every frame; update existing lines/curves.
- Use bounded buffers (ring buffers) for live views.
- Be conservative with default sampling/plot rates so slower PCs stay responsive.
