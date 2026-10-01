#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Logger dependencies. Read from this folder: it is the folder that gets
# deployed to the Pi. The file used to be looked up one level above, where
# neither the README's scp nor deploy_pi.bat puts it, so this step was
# silently skipped.
pip3 install --user --break-system-packages -r "$SCRIPT_DIR/requirements-pi.txt"

# OLED status display deps (luma.oled, pillow) -- kept in a separate
# requirements file.
if [ -f "$SCRIPT_DIR/requirements-oled.txt" ]; then
  pip3 install --user --break-system-packages -r "$SCRIPT_DIR/requirements-oled.txt"
fi
