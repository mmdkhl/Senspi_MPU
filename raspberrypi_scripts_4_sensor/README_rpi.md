# SensePi Raspberry Pi scripts — 4-sensor + OLED setup

This folder is the development copy of the Pi-side scripts, extended from the
working 3-sensor deployment (`raspberrypi_scripts/`) to support a 4th
MPU6050 sensor and an I2C OLED status display. The original 3-sensor folder
is untouched; this one is safe to test independently.

## Hardware this matches

- Sensor 1: bus 1, address 0x68 (AD0 floating)
- Sensor 2: bus 1, address 0x69 (AD0 → 3.3V)
- Sensor 3: bus 0, address 0x68 (AD0 floating)
- Sensor 4: bus 0, address 0x69 (AD0 → 3.3V)
- OLED: bus 1, address 0x3C (SSD1306/SSD1315, 0.96", 128x64)

## Files

- `mpu6050_multi_logger.py` — multi-sensor MPU6050 logger, now mapping
  sensors 1–4 by default. Also writes a small status heartbeat file
  (`/tmp/sensepi_status.json`) while running, for the OLED display to read.
- `oled_status.py` — standalone script that renders IP address, publishing
  state, and sampling rate on the OLED. Runs as its own process, independent
  of the logger, by design — a crash in either one can't take down the other.
- `pi_logger_common.py` — shared helpers and configuration loading.
- `pi_config.yaml` — configuration file for rates, channels, paths, and the
  enabled sensor list (now `1, 2, 3, 4`).
- `install_pi_deps.sh` — installs Python dependencies, including the OLED's
  (`requirements-oled.txt`).
- `requirements-oled.txt` — `luma.oled` + `pillow`, kept separate from the
  base `requirements-pi.txt` so the 3-sensor deployment is unaffected.
- `run_all_sensors.sh` — launches the OLED display in the background, then
  the logger in the foreground; stopping the logger (Ctrl-C or a stop
  signal) also stops the OLED script.

## Setup

Replace `<user>@<host>` below with your actual Pi login (e.g. `verwalter@192.168.0.111`)
and `<target_dir>` with wherever you want this folder to live on the Pi (e.g. `~/sensor4`,
as a sibling of your existing 3-sensor folder — do not reuse the same folder name).

On your workstation:

```bash
scp -r raspberrypi_scripts_4_sensor <user>@<host>:<target_dir>
ssh <user>@<host> "chmod +x <target_dir>/run_all_sensors.sh <target_dir>/install_pi_deps.sh"
```

This folder does not ship its own `sensepi/` package — `mpu6050_multi_logger.py` imports
`sensepi.config.log_paths`, and Python auto-adds a script's own directory to its import
path, so `sensepi/` just needs to sit next to the script. If your existing 3-sensor folder
already has a working `sensepi/` copy (check with `find <existing_folder> -maxdepth 1 -name
sensepi`), reuse it instead of transferring a fresh one:

```bash
ssh <user>@<host> "cp -r <existing_3sensor_dir>/sensepi <target_dir>/sensepi"
```

Before installing, confirm `output_dir` in `pi_config.yaml` matches your actual home
directory on the Pi (it defaults to `/home/pi/logs/mpu`, which will fail with a permission
error if your Pi user isn't literally named `pi` — check with `ssh <user>@<host> whoami`).

```bash
ssh <user>@<host> "bash <target_dir>/install_pi_deps.sh"
```

If this fails with `error: externally-managed-environment`, the script's `pip3 install`
calls need `--break-system-packages` added (safe here since it's a `--user` install, not a
system-wide one) — this is already fixed in this folder's `install_pi_deps.sh` as of the
last update.

Verify wiring before running anything:

```bash
ssh <user>@<host> "cd <target_dir> && python3 mpu6050_multi_logger.py --list"
```

You should see `WHO_AM_I=0x68` on both `0x68` and `0x69`, on both bus 0 and bus 1 — not
just an address match, the actual register read matters (a stuck/shorted bus can make every
address falsely appear present). Confirm the OLED separately:

```bash
ssh <user>@<host> "sudo /usr/sbin/i2cdetect -y 1"
```

(look for `3c`; use the full `/usr/sbin/i2cdetect` path if you get `command not found` —
`/usr/sbin` often isn't on `PATH` for non-interactive SSH sessions even when the package is
installed). `--list` above only checks 0x68/0x69, so it won't catch OLED wiring issues.

Then run everything (logger + OLED display together):

```bash
ssh <user>@<host> "cd <target_dir> && ./run_all_sensors.sh"
```

Or run the OLED display on its own, e.g. while testing the logger separately:

```bash
python3 oled_status.py --interval 1.0
```

## Notes

- INT, XDA, and XCL are unused on every sensor — nothing in this codebase
  reads them, so they can stay unconnected.
- This folder only changes the Pi-side scripts. The desktop GUI
  (`src/sensepi/gui/...`) still assumes 3 sensors and hasn't been touched;
  that's a separate follow-up once this 4-sensor acquisition path is
  validated on the Pi.
