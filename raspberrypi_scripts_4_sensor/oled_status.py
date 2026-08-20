#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oled_status.py
===============
Standalone status display for the SensePi rig's OLED (SSD1306/SSD1315,
0.96", I2C address 0x3C, sharing bus 1 with sensor 1 and sensor 2).

Deliberately kept as its own process, separate from mpu6050_multi_logger.py:
a display glitch can never take down data acquisition, and a slow/hung
acquisition run can never freeze the display. The two talk to each other
only through a small heartbeat file the logger writes
(/tmp/sensepi_status.json by default) -- no sockets, no shared memory.

Shows exactly three things, per current requirements:
  1) IP address
  2) Whether data is currently being published (recording) or not
  3) The sensors' sampling rate

Install (on the Pi):
    pip3 install luma.oled pillow

Run manually:
    python3 oled_status.py

Run alongside the logger:
    python3 mpu6050_multi_logger.py --config pi_config.yaml &
    python3 oled_status.py &

Options:
    --status-file PATH   heartbeat file written by the logger
                          (default: /tmp/sensepi_status.json)
    --interval SECONDS   display refresh period (default: 1.0)
    --i2c-port N         I2C bus the OLED is on (default: 1, matches bus 1)
    --i2c-address HEX    OLED I2C address (default: 0x3C)
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

try:
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import ssd1306
except Exception:
    print(
        "ERROR: luma.oled (and pillow) are required. "
        "Install with: pip3 install luma.oled pillow",
        file=sys.stderr,
    )
    raise

# A heartbeat older than this counts as "not publishing" -- covers both a
# stopped logger (clean exit already writes publishing=False) and a hung
# or killed -9 logger process (heartbeat simply stops updating).
STALE_AFTER_S = 3.0


def get_ip_address() -> str:
    """Best-effort local IP lookup. Doesn't actually send any packets --
    connect() on a UDP socket just resolves the outbound interface/route."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "no network"
    finally:
        s.close()


def read_status(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # No file yet, or logger never started -> nothing is publishing.
        return {"publishing": False, "sample_rate_hz": None, "sensor_ids": []}

    age_s = time.time() - data.get("updated_unix_s", 0)
    if age_s > STALE_AFTER_S:
        data["publishing"] = False
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="SensePi OLED status display")
    ap.add_argument("--status-file", default="/tmp/sensepi_status.json")
    ap.add_argument("--interval", type=float, default=1.0, help="Refresh period in seconds")
    ap.add_argument("--i2c-port", type=int, default=1, help="I2C bus for the OLED (bus 1)")
    ap.add_argument("--i2c-address", default="0x3C")
    args = ap.parse_args()

    status_path = Path(args.status_file)

    # I2C/display init failures (OLED not wired, wrong bus/address, bad
    # solder joint, etc.) previously raised an uncaught traceback here. Since
    # run_all_sensors.sh launches this script in the background with `&` and
    # doesn't redirect its output anywhere persistent, that traceback was
    # easy to miss entirely -- the display just silently never comes on.
    # Fail loudly and specifically instead.
    try:
        serial = i2c(port=args.i2c_port, address=int(args.i2c_address, 16))
        device = ssd1306(serial)
    except Exception as exc:
        print(
            f"ERROR: Could not initialize OLED on bus {args.i2c_port} at "
            f"address {args.i2c_address}: {exc}\n"
            "Check wiring (SDA/SCL/VCC/GND) and confirm the display is "
            f"detected with: i2cdetect -y {args.i2c_port}",
            file=sys.stderr,
        )
        sys.exit(1)

    ip_address = get_ip_address()
    last_ip_refresh = time.monotonic()
    IP_REFRESH_PERIOD_S = 30.0  # IP rarely changes; no need to check every frame

    # Consecutive-failure tracking for the render loop below: a single I2C
    # hiccup shouldn't kill the display process (the logger script treats
    # I2C errors the same way -- log and keep going), but a display that's
    # been unplugged mid-run should eventually exit instead of spinning
    # forever on a dead bus.
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 20

    while True:
        if time.monotonic() - last_ip_refresh > IP_REFRESH_PERIOD_S:
            ip_address = get_ip_address()
            last_ip_refresh = time.monotonic()

        status = read_status(status_path)
        publishing = bool(status.get("publishing"))
        rate = status.get("sample_rate_hz")
        rate_text = f"{rate:.0f} Hz" if (publishing and rate) else "--"

        try:
            with canvas(device) as draw:
                draw.text((0, 0), "SensePi Status", fill="white")
                draw.line((0, 12, 128, 12), fill="white")
                draw.text((0, 20), f"IP:   {ip_address}", fill="white")
                draw.text((0, 34), f"Data: {'PUBLISHING' if publishing else 'stopped'}", fill="white")
                draw.text((0, 48), f"Rate: {rate_text}", fill="white")
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            print(
                f"[WARN] OLED render/write failed ({exc}); "
                f"consecutive_errors={consecutive_errors}",
                file=sys.stderr,
            )
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(
                    f"ERROR: OLED failed {consecutive_errors} times in a row; "
                    "assuming the display was disconnected. Exiting.",
                    file=sys.stderr,
                )
                sys.exit(1)

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
