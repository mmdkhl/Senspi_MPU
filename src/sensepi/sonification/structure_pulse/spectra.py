"""Response spectra for a measured acceleration record.

The one piece of structural analysis the repo did not already have. Pure numpy
+ scipy, no Qt (guardrail G7).

Method: the single-degree-of-freedom oscillator

    u" + 2 z wn u' + wn^2 u = -a_g(t)

is discretised with a **first-order hold**, which is exact when the input is
taken as piecewise linear between samples — the same assumption as the classical
Nigam-Jennings recurrence, but obtained from ``scipy.signal.cont2discrete``
rather than hand-coded coefficients, so there is far less to get wrong.

Returned quantities are the usual ones:

====  ==========================================================
Sd    peak relative displacement,  max|u|
Sv    pseudo-velocity,             wn * Sd
Sa    pseudo-acceleration,         wn^2 * Sd
====  ==========================================================

Pseudo-acceleration is what "spectral acceleration" means in design practice and
is what gets plotted against period.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sg

#: Period range worth showing for a shake-table model structure. Its modes sit
#: around 1-20 Hz, i.e. 0.05-1 s, so this brackets them with room either side.
DEFAULT_T_MIN = 0.02
DEFAULT_T_MAX = 2.0
DEFAULT_N_PERIODS = 200
DEFAULT_DAMPING = 0.05


def sdof_response(accel: np.ndarray, fs: float, period_s: float,
                  damping: float = DEFAULT_DAMPING) -> np.ndarray:
    """Relative displacement history of one SDOF oscillator, in metres.

    ``accel`` is the base acceleration in m/s^2.
    """
    a = np.asarray(accel, dtype=float).ravel()
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.size < 4 or not np.isfinite(fs) or fs <= 0 or period_s <= 0:
        return np.zeros_like(a)
    wn = 2.0 * np.pi / float(period_s)
    z = float(np.clip(damping, 1e-4, 0.999))
    # A period far below the Nyquist sampling limit cannot be resolved; return
    # zeros rather than a confident-looking but meaningless number.
    if wn > np.pi * fs:
        return np.zeros_like(a)
    num = [-1.0]
    den = [1.0, 2.0 * z * wn, wn * wn]
    b, d, _ = sg.cont2discrete((num, den), 1.0 / float(fs), method="foh")
    u = sg.lfilter(np.asarray(b).ravel(), np.asarray(d).ravel(), a)
    return np.asarray(u, dtype=float)


def response_spectrum(accel: np.ndarray, fs: float, *,
                      periods: np.ndarray | None = None,
                      damping: float = DEFAULT_DAMPING,
                      t_min: float = DEFAULT_T_MIN,
                      t_max: float = DEFAULT_T_MAX,
                      n_periods: int = DEFAULT_N_PERIODS) -> dict:
    """Sd / Sv / Sa against period for one acceleration record.

    Returns a dict with ``periods``, ``Sd``, ``Sv``, ``Sa`` and ``damping``.
    ``Sa`` is pseudo-acceleration in m/s^2.
    """
    a = np.asarray(accel, dtype=float).ravel()
    if periods is None:
        periods = np.geomspace(max(t_min, 1e-4), max(t_max, t_min * 2),
                               int(max(n_periods, 2)))
    periods = np.asarray(periods, dtype=float).ravel()
    sd = np.zeros(periods.size)
    for i, T in enumerate(periods):
        u = sdof_response(a, fs, float(T), damping)
        sd[i] = float(np.max(np.abs(u))) if u.size else 0.0
    wn = 2.0 * np.pi / np.maximum(periods, 1e-12)
    return {
        "periods": periods,
        "Sd": sd,
        "Sv": wn * sd,
        "Sa": wn * wn * sd,
        "damping": float(damping),
    }


def peak_of_spectrum(spec: dict, key: str = "Sa") -> tuple:
    """``(period, value)`` of the largest ordinate — the dominant period."""
    y = np.asarray(spec.get(key, []), dtype=float)
    T = np.asarray(spec.get("periods", []), dtype=float)
    if y.size == 0 or T.size != y.size:
        return float("nan"), float("nan")
    k = int(np.argmax(y))
    return float(T[k]), float(y[k])
