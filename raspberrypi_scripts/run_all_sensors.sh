#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/pi_config.yaml"

python3 "$SCRIPT_DIR/mpu6050_multi_logger.py" --config "$CONFIG"
