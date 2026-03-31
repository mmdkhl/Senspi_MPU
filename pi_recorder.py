from __future__ import annotations

"""
Legacy shim module.

This file exists for backward compatibility only.
New code should import from :mod:`sensepi.remote` instead, e.g.:

    from sensepi.remote import PiRecorder
"""

import warnings

from sensepi.remote import PiRecorder

__all__ = ["PiRecorder"]

# Emit a deprecation warning when this shim is imported
warnings.warn(
    "Importing 'PiRecorder' from 'pi_recorder' is deprecated. "
    "Use 'from sensepi.remote import PiRecorder' instead.",
    DeprecationWarning,
    stacklevel=2,
)
