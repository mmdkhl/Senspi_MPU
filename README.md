# SensePi — ProTELC Science Camp

SensePi is a desktop app for measuring how a model structure vibrates. It connects over the network to a
Raspberry Pi that has **four MPU6050 motion sensors** attached. You can watch the signals live, record
them, look at their frequency content, and compare them with a computer model of the structure.

This page shows you how to get the app running on **your own computer** for the first time. It takes
about 15 minutes, and most of that is waiting for downloads.

> **At the camp, the Raspberry Pis are already set up for you.** All you need is the app on your laptop
> and the connection details (IP address, username, password) your instructor gives you. To talk to
> a Pi, your laptop must be connected to the **`MissionControl`** Wi-Fi network.

---

## Quick start

### Step 1: Install the tools (one time only)

You need two free programs:

| Program | Where to get it | Notes |
|---|---|---|
| **Python 3.12** | https://www.python.org/downloads/release/python-31210/ | On Windows, tick **"Add python.exe to PATH"** on the first installer screen. |
| **Git** | https://git-scm.com/downloads | The default options are fine. |

Use **Python 3.12**. Newer versions (3.13, 3.14) install without errors, but on Windows the *Model
Updating* tab then fails, because its OpenSees engine is only built for 3.12. If you already have a
different Python, install 3.12 next to it.

To check that both are installed, open a **new** terminal (Windows: press the Start key and type
`PowerShell`; macOS: open *Terminal*) and run:

```bash
python --version
git --version
```

The first command should print `Python 3.12.x`. On macOS/Linux, use `python3.12` in place of
`python`.

### Step 2: Download the app

The commands below download only the parts of the project the app needs, so the download stays small:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/mmdkhl/Senspi_MPU.git
cd Senspi_MPU
git sparse-checkout set src
```

This creates a folder called `Senspi_MPU` inside whatever folder your terminal was in (usually your
user folder). The rest of this guide runs commands **from inside that folder**.

<details>
<summary>No Git? Download a ZIP instead</summary>

Open https://github.com/mmdkhl/Senspi_MPU and click **Code → Download ZIP**. Unzip it, then open a
terminal in the unzipped folder. You get the whole project this way, which is fine but a bigger download.
</details>

### Step 3: Install the app

Create a private Python environment for the app so it doesn't interfere with anything else on your
computer, then install the app into it.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs **everything** — every tab works after this, including *Spectrum*,
*Model Updating* and the sound in *Sonification*. There is nothing else to add later.

When the environment is active, your prompt starts with `(.venv)`. The install downloads a few hundred MB
and can take several minutes.

> **Windows: "running scripts is disabled on this system"?** Run this once, then try
> `Activate.ps1` again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### Step 4: Start the app

```bash
python main.py
```

The SensePi window opens. (On macOS/Linux use `python3` if `python` is not found.)

### Step 5: Connect to your Raspberry Pi

> **Connect your laptop to the `MissionControl` Wi-Fi network first.** The Raspberry Pis are only
> reachable on `MissionControl`, so the app can't find them from any other network. Do Steps 1–3 on a
> network with internet access, because `MissionControl` may not have any.

1. Open the **Settings** tab.
2. Under **Raspberry Pi hosts**, fill in the example entry like this:

   | Field | Value |
   |---|---|
   | **Name** | any name you like, for example `Pi-1` |
   | **Host / IP** | the Pi's IP address, from your instructor (the Pi's small screen also shows it) |
   | **User** | `verwalter` |
   | **SSH port** | `22` |
   | **Password** | from your instructor |
   | **Scripts base path** | `/home/verwalter/sensor4` |
   | **Data directory** | `/home/verwalter/logs` |
   | **Pi config path** | `/home/verwalter/sensor4/pi_config.yaml` |

   Type the paths exactly as shown. They point to the **4-sensor** scripts on the camp Pis.
3. Click **Save host config**.
4. Check that **Number of sensors** is set to **4 sensors**. This is the default.
5. In the **sensor map**, set the floor and position of each sensor to match your structure. Every other
   tab uses this map, so get it right here. Then click **Save sensors.yaml**.
6. Click **Sync Pi defaults (pi_config.yaml)** to send these settings to the Pi.

### Step 6: Measure!

1. Go to **Live Signals** and click **Start**. Tap the structure and watch the signals move.
2. Click **Stop** when you're done.
3. To keep a measurement, set the recording length and click **Smart Record**. The app checks the real
   sampling rate for a few seconds, then records for the length you chose.

---

## Starting the app again later

You only do Steps 1–3 once. After that, open a terminal and run:

**Windows:**
```powershell
cd Senspi_MPU
.venv\Scripts\Activate.ps1
python main.py
```

**macOS / Linux:**
```bash
cd Senspi_MPU
source .venv/bin/activate
python main.py
```

To get the latest version of the app (for example when your instructor announces an update), run this in
the `Senspi_MPU` folder:

```bash
git pull
```

---

## What's in the app

| Tab | What it does |
|---|---|
| **Live Signals** | Streams the four sensors live and records measurements |
| **Spectrum** | Shows which frequencies the structure vibrates at (its natural frequencies and mode shapes) |
| **Model Updating** | Tunes a computer (OpenSees) model of the structure so it matches your measurements. Its sub-tabs are Model, Additional Mass, Analysis, Calibration, **Digital Twin** (the tuned model's response beside the measurement) and **Digital Shadow** (the model run live alongside the real structure) |
| **Sonification** | Turns the vibrations into sound |
| **Settings** | Raspberry Pi connection, number of sensors, sensor positions and sampling rate |

### Where your data is saved

Everything the app produces goes into the `output/` folder inside `Senspi_MPU`:

```text
output/
  sensor_recordings/   your recorded measurements (one folder per recording)
  model/               computer-model files
  digital_twin/        Digital Shadow runs
  sonification/        audio you captured
```

---

## Troubleshooting

**`python` or `git` is "not recognized" / "command not found"**
Close the terminal and open a new one. If it still fails, reinstall Python and make sure **"Add
python.exe to PATH"** is ticked. On macOS/Linux, try `python3`.

**`No module named 'PySide6'` (or another package) when starting the app**
The environment isn't active. Run the `Activate` line from Step 3 — your prompt should show `(.venv)` —
then `python main.py` again. Run `python main.py` from inside the `Senspi_MPU` folder.

**The *Model Updating* tab says OpenSees is missing, or *Sonification* has no sound**
The dependencies didn't all install. With the environment active, run
`pip install -r requirements.txt` again and read the output for errors.

**Can't connect to the Pi**
- Is your laptop connected to the **`MissionControl`** Wi-Fi? (Laptops sometimes switch back to another
  saved network on their own.)
- Check the IP address, username and password in **Settings**.
- Ask your instructor whether the Pi is switched on. Its small screen shows its IP address.

**No signal / a sensor is missing**
- Check that **Number of sensors** is set to 4 and click **Sync Pi defaults (pi_config.yaml)** again.
- Check that the sensor cables are firmly connected, then tell your instructor.

**Plots are slow or jerky**
Lower the sample rate in **Settings**, or close other heavy programs.

---

## For instructors: preparing a Raspberry Pi

The Pi-side scripts for the 4-sensor setup (four MPU6050s on I2C buses 0 and 1, plus an OLED status
display) are in [`raspberrypi_scripts_4_sensor/`](raspberrypi_scripts_4_sensor/). Its
[`README_rpi.md`](raspberrypi_scripts_4_sensor/README_rpi.md) covers wiring, copying the scripts to the
Pi, installing dependencies (offline wheels included) and checking the sensors.

A student clone made with Step 2 doesn't include that folder. To get it, run
`git sparse-checkout add raspberrypi_scripts_4_sensor`, or clone the full repository.

Notes:
- The camp Pis use the user `verwalter`, with the 4-sensor scripts copied to `/home/verwalter/sensor4`.
  The student setup (Step 5) depends on these paths, so keep them the same on every Pi.
- `output_dir` in `pi_config.yaml` must be `/home/verwalter/logs/mpu` (inside the `verwalter` home
  directory).
- `deploy_pi.bat` / `deploy_pi.example.bat` copy the 4-sensor folder to the Pi. Set `REMOTE_DIR`
  (and the matching "Refusing to wipe" guard line) to the Pi's real folder first, for example
  `/home/verwalter/sensor4`: the script empties that folder before copying.

### Keeping credentials out of Git

The GUI saves real host details to `src/sensepi/config/hosts.local.yaml`, which Git ignores. The tracked
`hosts.yaml` / `hosts.example.yaml` contain placeholder values only. Never commit real IP addresses or
passwords.

---

## For developers

See [`docs/DEVELOPERS.md`](docs/DEVELOPERS.md) for the architecture, config files and tests.

`requirements.txt` is the complete dependency set and keeps step with `[project].dependencies` plus the
`model-updating` extra in `pyproject.toml`. OpenSees stays an extra in the package metadata (guardrail
G8) so `sensepi` can be imported as a library without it; `requirements.txt` installs it anyway, so the
app is always complete.

Installing the package itself adds the console commands:

```bash
pip install -e .          # adds sensepi-gui, sensepi-sonify, opensees-calibrate
sensepi-sonify --input path/to/data.csv --out out.wav --mode harmonic --joint 28 --r1-measurement R1 --u1-measurement U1
```
