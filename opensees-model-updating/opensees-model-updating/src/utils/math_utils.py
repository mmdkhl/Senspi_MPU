# -*- coding: utf-8 -*-
"""Math utilities for modal analysis."""

import numpy as np


def normalize_mode_maxabs(vec):
    """Normalize a mode shape vector by its maximum absolute value."""
    v = np.asarray(vec, dtype=float)
    m = np.max(np.abs(v))
    if m <= 0.0:
        return v.copy()
    return v / m


def align_mode_sign(phi_num, phi_exp):
    """Flip sign of numerical mode shape to align with experimental mode shape."""
    a = np.asarray(phi_num, dtype=float)
    b = np.asarray(phi_exp, dtype=float)
    if np.dot(a, b) < 0.0:
        return -a
    return a


def safe_percent_error(pred, target):
    """Compute percent error, returning NaN if target is near zero."""
    if abs(target) < 1e-14:
        return np.nan
    return 100.0 * (pred - target) / target
