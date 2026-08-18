#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/pi_config.yaml"

# mpu6050_multi_logger.py imports sensepi.config.log_paths, which lives in
# the main project's src/ tree, not in this folder. This assumes the repo
# is deployed to the Pi with the same layout as on the workstation --
# i.e. this script's parent directory has a sibling "src/" folder. If your
# deployment layout differs, override PYTHONPATH before calling this script.
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -d "$REPO_ROOT/src" ]; then
  export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
fi

# Launch the OLED status display as its own background process. It's
# deliberately independent of the logger (reads a heartbeat file, see
# write_status_file() in mpu6050_multi_logger.py) so a problem in either
# process can never take the other down.
OLED_PID=""
if [ -f "$SCRIPT_DIR/oled_status.py" ]; then
  python3 "$SCRIPT_DIR/oled_status.py" &
  OLED_PID=$!
fi

cleanup() {
  if [ -n "$OLED_PID" ] && kill -0 "$OLED_PID" 2>/dev/null; then
    kill "$OLED_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Run the logger in the foreground so Ctrl-C / systemd stop signals reach it
# directly; the trap above takes the OLED script down with it.
python3 "$SCRIPT_DIR/mpu6050_multi_logger.py" --config "$CONFIG"
