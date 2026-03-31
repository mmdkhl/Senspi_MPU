#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_ROOT="$(dirname "$SCRIPT_DIR")"

sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

if [ -f "$REQUIREMENTS_ROOT/requirements-pi.txt" ]; then
  pip3 install --user -r "$REQUIREMENTS_ROOT/requirements-pi.txt"
fi
