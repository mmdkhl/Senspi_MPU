#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_ROOT="$(dirname "$SCRIPT_DIR")"

sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

if [ -f "$REQUIREMENTS_ROOT/requirements-pi.txt" ]; then
  pip3 install --user --break-system-packages -r "$REQUIREMENTS_ROOT/requirements-pi.txt"
fi

# OLED status display deps (luma.oled, pillow) -- kept in a separate
# requirements file so the base 3-sensor deployment's dependencies are
# untouched.
if [ -f "$SCRIPT_DIR/requirements-oled.txt" ]; then
  pip3 install --user --break-system-packages -r "$SCRIPT_DIR/requirements-oled.txt"
fi
