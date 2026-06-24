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


# Minimum spatial points for a trustworthy MAC. On fewer DOFs the criterion is
# degenerate, so pairing/gating must fall back to frequency order (Baban caveat).
MAC_MIN_DOFS = 3


def mac(phi_a, phi_b):
    """Modal Assurance Criterion between two real mode-shape vectors.

    ``MAC = |a.b|^2 / ((a.a)(b.b))`` in ``[0, 1]``. Scale- and sign-invariant:
    1.0 means identical shape (up to scale/sign), 0.0 means orthogonal. Returns
    0.0 for an empty, zero, or length-mismatched vector (cannot compare).
    """
    a = np.asarray(phi_a, dtype=float).ravel()
    b = np.asarray(phi_b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return 0.0
    denom = float(np.dot(a, a) * np.dot(b, b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) ** 2 / denom)


def pair_modes_by_mac(model_shapes, exp_shapes, min_dofs=MAC_MIN_DOFS):
    """Pair each experimental mode to the most MAC-similar model mode (B2).

    Returns ``perm`` of length ``len(exp_shapes)``: experimental mode ``i`` is
    matched to model mode ``perm[i]``. Greedy max-MAC assignment without
    replacement, so a swapped or missed identification cannot corrupt the fit by
    relying on frequency order alone.

    **Few-DOF fallback (Baban / RESOLVED OPEN ENDS):** MAC on fewer than
    ``min_dofs`` spatial points is degenerate, so when any experimental shape is
    shorter than ``min_dofs`` the pairing falls back to **frequency order**
    (identity ``[0, 1, ...]``). Identity is also returned when there are no shapes.
    """
    n_exp = len(exp_shapes)
    if n_exp == 0:
        return []
    if any(np.asarray(s, dtype=float).size < int(min_dofs) for s in exp_shapes):
        return list(range(n_exp))  # frequency-order fallback (<3 measured DOFs)

    n_model = len(model_shapes)
    used: set[int] = set()
    perm: list[int] = []
    for i in range(n_exp):
        best_j, best_mac = None, -1.0
        for j in range(n_model):
            if j in used:
                continue
            m = mac(model_shapes[j], exp_shapes[i])
            if m > best_mac:
                best_mac, best_j = m, j
        if best_j is None:  # ran out of distinct model modes -> order fallback
            best_j = min(i, n_model - 1) if n_model > 0 else i
        used.add(best_j)
        perm.append(int(best_j))
    return perm
