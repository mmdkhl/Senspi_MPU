# SensePi (GUI) — Raspberry Pi MPU6050 recording + live view

SensePi is a desktop GUI (PySide6) that connects to a Raspberry Pi over SSH, starts the MPU6050 logger, shows live plots, and (optionally) records logs on the Pi and downloads them to your PC.

This README is for **operators/users** who just want to run the GUI.

---

## What you need

### On your PC
- Windows 10/11 (or Linux/macOS) with **Python 3.9+**
- Network access to the Raspberry Pi (same LAN)
- The SensePi project folder (clone or unzip)

### On the Raspberry Pi
- Raspberry Pi OS
- MPU6050 wired and **I2C enabled**
- SSH access (username + password)

> If the Pi already has the SensePi scripts in place, you can skip the “Deploy to Pi” section.

---

## Install on the PC (GUI)

From the project root:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the GUI:

```bat
python -m sensepi.gui.application
```

Optional: install as an editable package (lets you run `sensepi-gui`):

```bat
pip install -e .
sensepi-gui
```

If you want to use the **Digital Twin** tab, install the optional OpenSees dependency:

```bat
pip install ".[digital-twin]"
```

---

## Configure your Raspberry Pi in the GUI

1. Start the GUI
2. Go to **Settings**
3. Add / edit your Pi in the **Raspberry Pi hosts** list
4. Save

The host list is stored in:

- `src/sensepi/config/hosts.local.yaml` for your real local settings
- `src/sensepi/config/hosts.yaml` as the sanitized repo template
- Public template: `src/sensepi/config/hosts.example.yaml`

Typical fields:
- `name`: friendly name (shown in the GUI)
- `host`: IP/hostname
- `user` / `password`
- `base_path`: where the Pi scripts live (example: `/home/pi/sensor`)
- `data_dir`: where logs should be written (example: `/home/pi/logs`)
- `pi_config_path`: where the GUI uploads `pi_config.yaml` (usually `<base_path>/pi_config.yaml`)

---

## Deploy to the Pi (only if needed)

If your Pi does **not** already have the scripts, use the provided Windows deploy script:

### 1) Prerequisites
- Install **PuTTY** (you need `plink.exe` + `pscp.exe`)

### 2) Edit `deploy_pi.bat`
Open `deploy_pi.bat` and update:
- `PUTTY_DIR` (where plink/pscp live)
- `LOCAL_ROOT` (your repo path)
- `PI_USER`, `PI_HOST`, `PI_PASS`
- `REMOTE_DIR` (where files will be copied)

For GitHub/public sharing, treat `deploy_pi.bat` as a template. Keep your real credentials in an untracked copy such as `deploy_pi.local.bat`.

⚠️ **Important:** the script **wipes** the remote directory before copying. Read it before running.

### 3) Run it
Double-click `deploy_pi.bat` (or run it from a terminal).

### 4) Install Pi Python dependencies
On the Pi (SSH):

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-smbus i2c-tools
# If requirements-pi.txt is not on the Pi yet, copy it from the repo root (or install the packages manually).
# Option A: use the requirements file
pip3 install --user -r requirements-pi.txt
# Option B (manual): pip3 install --user numpy smbus2 PyYAML RPi.GPIO
mkdir -p ~/logs/mpu
```

### 5) Enable I2C (if you haven’t already)
On the Pi:

```bash
sudo raspi-config
# Interface Options -> I2C -> Enable
```

---

## Using the GUI (basic workflow)

1. Go to **Live Signals**
2. Pick your Pi host
3. Click **Sync config to Pi** (in Settings) after you change sensors/rates
4. Click **Start** to begin live streaming
5. Enable **Recording** (if you want the Pi to write `.csv`/`.jsonl` files)
6. Click **Stop** when done
7. Click **Sync logs** to download new logs to your PC

The GUI also includes a **Digital Twin** tab for running an OpenSeesPy structural model from inside the app. It can:
- load a ground-motion text file
- run modal + transient analysis
- write OpenSees-style output files
- plot roof displacement / acceleration directly in the GUI

If `openseespy` is not installed, the tab stays visible but shows a dependency warning when you try to run it.

---

## Where your data goes

- **On the Pi:** `<data_dir>/mpu/...`
- **On the PC (after Sync):** `data/raw/...`

(You can override the PC folders using environment variables: `SENSEPI_DATA_ROOT` and `SENSEPI_LOG_DIR`.)

---

## Troubleshooting

**SSH connection fails**
- Check the IP/hostname in Settings
- Make sure SSH is enabled on the Pi
- Confirm username/password

**No sensor data**
- Confirm I2C is enabled
- Check wiring and address (`i2cdetect -y 1`)
- Try running the logger directly on the Pi:
  ```bash
  cd <base_path>
  python3 mpu6050_multi_logger.py --list
  ```

**Plots are laggy**
- Reduce the sample rate in Settings
- Reduce the number of sensors/channels being streamed

---

## GitHub / public repo note

This repository is prepared for public sharing:
- `src/sensepi/config/hosts.yaml` contains sanitized example values
- `src/sensepi/config/hosts.example.yaml` is a copyable template
- `deploy_pi.bat` / `deploy_pi.example.bat` contain placeholder values only
- The GUI saves real host edits to `src/sensepi/config/hosts.local.yaml`

Do not commit real hostnames, passwords, or private deployment paths. Keep real local values in untracked files such as:
- `src/sensepi/config/hosts.local.yaml`
- `deploy_pi.local.bat`

---

## Sonification (merged from your structural-response workflow)

The repo now includes a sonification module for structural CSV data (header row + units row + data rows).

Run via module:

```bash
python -m sensepi.sonification.cli --input path/to/data.csv --out out.wav --mode melody --joint 28 --measurement U1
```

Or after editable install:

```bash
sensepi-sonify --input path/to/data.csv --out out.wav --mode harmonic --joint 28 --r1-measurement R1 --u1-measurement U1
```

Modes:
- `melody`: maps one measurement (default `U1`) to MIDI-note-like tones
- `harmonic`: additive synthesis where `R1` modulates harmonic spacing and `U1` modulates base frequency
