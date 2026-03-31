"""Helpers for querying local process performance metrics."""

from __future__ import annotations

import os
from typing import Final

import psutil

_PROCESS: Final[psutil.Process] = psutil.Process(os.getpid())


def get_process_cpu_percent() -> float:
    """
    Return the current CPU usage of the GUI process.

    psutil's cpu_percent needs to be called periodically; the first call
    may return 0.0 which is acceptable for the lightweight HUD display.
    """
    try:
        return float(_PROCESS.cpu_percent(interval=None))
    except Exception:
        return 0.0
