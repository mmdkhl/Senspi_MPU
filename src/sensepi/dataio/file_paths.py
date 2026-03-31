"""Helpers for constructing standard file paths."""

import re
from datetime import datetime
from pathlib import Path

from ..config.app_config import AppPaths

# Allow only alphanumerics, underscore, dot, and dash.
_SESSION_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_session_name(name: str) -> str:
    """
    Sanitize a session name for use as a directory name.

    - Replace disallowed characters with '_'.
    - Strip leading/trailing underscores.
    - Fall back to 'session' if nothing remains.
    """
    cleaned = _SESSION_NAME_RE.sub("_", name).strip("_")
    return cleaned or "session"


def session_directory(name: str, base: Path | None = None) -> Path:
    """
    Create a timestamped directory name for a recording session.

    Example: "shake_test_20251204_153045"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base or AppPaths().raw_data
    safe_name = _sanitize_session_name(name)
    return root / f"{safe_name}_{timestamp}"
