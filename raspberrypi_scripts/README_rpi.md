# SensePi Raspberry Pi scripts

This folder contains the lightweight logger scripts that should be copied to a Raspberry Pi. They record MPU6050 sensor data locally on the device and can optionally stream lines over stdout for live viewing.

## Files

- `mpu6050_multi_logger.py` – multi-sensor MPU6050 logger.
- `pi_logger_common.py` – shared helpers and configuration loading.
- `pi_config.yaml` – example configuration file for rates, channels, and paths.
- `install_pi_deps.sh` – installs Python dependencies on the Pi.
- `run_all_sensors.sh` – simple launcher for the MPU6050 logger.

## Setup

On your workstation:

```bash
scp -r raspberrypi_scripts pi@<host>:/home/pi/
ssh pi@<host> "bash /home/pi/raspberrypi_scripts/install_pi_deps.sh"
```

Then edit pi_config.yaml on the Pi to match your logging directory and sensor layout, and run:

```bash
ssh pi@<host> "cd /home/pi/raspberrypi_scripts && ./run_all_sensors.sh"
```
