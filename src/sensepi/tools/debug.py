"""Minimal helpers for opt-in debug/instrumentation hooks."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Callable, Iterator

DEBUG_SENSEPI = os.getenv("SENSEPI_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def debug_enabled() -> bool:
    """Return True when lightweight instrumentation should run."""
    return DEBUG_SENSEPI


@contextmanager
def time_block(label: str, *, emitter: Callable[[str], None] | None = None) -> Iterator[None]:
    """
    Context manager that emits elapsed time when debugging is enabled.

    The overhead is essentially a couple of perf_counter() calls when disabled.
    """
    if not DEBUG_SENSEPI:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        target = emitter or (lambda msg: print(msg, file=sys.stderr, flush=True))
        try:
            target(f"[DEBUG] {label} took {elapsed_ms:.3f} ms")
        except Exception:
            pass
