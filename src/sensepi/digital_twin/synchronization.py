"""Time-alignment helpers for digital-twin experiments.

The first implementation intentionally estimates *start/onset lag* instead of
maximizing full-record cross-correlation.  For a periodic shaker command (for
example a 1 Hz sine), full cross-correlation has equally plausible peaks one
period apart and can also absorb a genuine model phase error.  Onset alignment
is therefore a safer default when no base accelerometer or hardware trigger is
available.
"""
from __future__ import annotations

import numpy as np


def _clean(t, y):
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = min(t.size, y.size)
    t, y = t[:n], y[:n]
    keep = np.isfinite(t) & np.isfinite(y)
    return t[keep], y[keep]


def _find_onset(
    t: np.ndarray,
    y: np.ndarray,
    *,
    threshold_fraction: float = 0.08,
    noise_multiplier: float = 5.0,
    minimum_run_s: float = 0.05,
) -> float | None:
    """Return the first sustained response onset time.

    A robust baseline is taken from the first 10 % (at least a few samples), and
    the threshold is the larger of a noise-based threshold and a fraction of
    the current record amplitude.  Requiring a short sustained run rejects
    isolated sensor spikes.
    """
    if t.size < 8 or y.size < 8:
        return None

    y0 = y - float(np.nanmedian(y[: max(3, min(y.size, y.size // 10 or 3))]))
    n_base = max(3, min(y0.size // 5, int(round(0.5 / max(np.median(np.diff(t)), 1e-6)))))
    base = y0[:n_base]
    med = float(np.nanmedian(base))
    mad = float(np.nanmedian(np.abs(base - med)))
    sigma = 1.4826 * mad
    peak = float(np.nanmax(np.abs(y0))) if y0.size else 0.0
    threshold = max(noise_multiplier * sigma, threshold_fraction * peak, 1.0e-9)

    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    run = max(1, int(round(minimum_run_s / max(dt, 1.0e-6))))
    active = np.abs(y0) >= threshold
    if run <= 1:
        idx = np.flatnonzero(active)
        return float(t[idx[0]]) if idx.size else None

    kernel = np.ones(run, dtype=int)
    sustained = np.convolve(active.astype(int), kernel, mode="valid") >= run
    idx = np.flatnonzero(sustained)
    return float(t[idx[0]]) if idx.size else None


def estimate_onset_lag(
    measured_t,
    measured_y,
    numerical_t,
    numerical_y,
    *,
    max_abs_lag_s: float = 3.0,
) -> float | None:
    """Estimate ``measured_time - numerical_time`` at response onset.

    Positive lag means the measured structure began responding *later* than the
    numerical model.  To align the measured curve to numerical time, plot it at
    ``measured_t - lag``.

    The estimate is deliberately bounded.  If the inferred delay is outside
    ``max_abs_lag_s`` it is rejected rather than silently shifting the record by
    an implausible amount.
    """
    mt, my = _clean(measured_t, measured_y)
    nt, ny = _clean(numerical_t, numerical_y)
    if mt.size < 8 or nt.size < 8:
        return None

    m_on = _find_onset(mt, my)
    n_on = _find_onset(nt, ny)
    if m_on is None or n_on is None:
        return None
    lag = float(m_on - n_on)
    if not np.isfinite(lag) or abs(lag) > float(max_abs_lag_s):
        return None
    return lag
