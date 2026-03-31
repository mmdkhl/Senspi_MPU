"""Configuration objects and helpers for SensePi.

This package knows how to load/save YAML descriptors that capture the current
lab hardware setup, including:
- ``hosts.yaml`` with Raspberry Pi targets
- ``sensors.yaml`` and ``sampling.py`` defaults for each device
The resulting typed dataclasses (see :mod:`runtime`) are imported everywhere
else to configure recorders, streamers, and GUI defaults consistently.
"""

from .runtime import SensePiConfig, config_from_mapping, load_config

__all__ = ["SensePiConfig", "config_from_mapping", "load_config"]
