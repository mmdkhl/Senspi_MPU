"""Helpers for consuming live data from remote loggers."""

from typing import Any, Callable, Iterable
import logging

from ..sensors.mpu6050 import parse_line as parse_mpu

logger = logging.getLogger(__name__)


def stream_lines(
    lines: Iterable[str],
    parser: Callable[[str], Any],
    callback: Callable[[Any], None],
) -> None:
    """
    Parse incoming lines and forward decoded samples to a callback.

    This function is intentionally defensive:
    - It strips whitespace from each line.
    - It skips empty lines.
    - It logs and skips malformed lines instead of raising.
    - It logs callback errors instead of crashing the stream loop.
    """
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        try:
            sample = parser(line)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Error parsing line %r: %s", line, exc)
            continue

        if sample is None:
            continue

        try:
            callback(sample)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Error in stream callback for sample %r: %s", sample, exc)


def select_parser(sensor_type: str) -> Callable[[str], Any]:
    """
    Select the appropriate line parser for a given sensor type name.

    Currently only the MPU6050 logger is supported.
    """
    st = sensor_type.strip().lower()
    if st in {"mpu6050", "mpu-6050"}:
        return parse_mpu
    raise ValueError(f"Unknown sensor_type {sensor_type!r}")
