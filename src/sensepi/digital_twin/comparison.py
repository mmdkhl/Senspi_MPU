"""Signal comparison and spectral metrics for digital-twin experiments."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ComparisonMetrics:
    rms_error: float = float("nan")
    nrmse_percent: float = float("nan")
    correlation: float = float("nan")
    peak_measured: float = float("nan")
    peak_numerical: float = float("nan")
    peak_ratio: float = float("nan")
    samples: int = 0


def _overlap_resample(measured_t, measured_y, numerical_t, numerical_y):
    mt = np.asarray(measured_t, dtype=float).reshape(-1)
    my = np.asarray(measured_y, dtype=float).reshape(-1)
    nt = np.asarray(numerical_t, dtype=float).reshape(-1)
    ny = np.asarray(numerical_y, dtype=float).reshape(-1)
    nm = min(mt.size, my.size)
    nn = min(nt.size, ny.size)
    mt, my, nt, ny = mt[:nm], my[:nm], nt[:nn], ny[:nn]
    km = np.isfinite(mt) & np.isfinite(my)
    kn = np.isfinite(nt) & np.isfinite(ny)
    mt, my, nt, ny = mt[km], my[km], nt[kn], ny[kn]
    if mt.size < 3 or nt.size < 3:
        return np.array([]), np.array([]), np.array([])
    lo = max(float(mt[0]), float(nt[0]))
    hi = min(float(mt[-1]), float(nt[-1]))
    if hi <= lo:
        return np.array([]), np.array([]), np.array([])
    # Use the numerical time grid because it is exactly the integration grid.
    mask = (nt >= lo) & (nt <= hi)
    grid = nt[mask]
    if grid.size < 3:
        return np.array([]), np.array([]), np.array([])
    m = np.interp(grid, mt, my)
    n = ny[mask]
    return grid, m, n


def compute_comparison_metrics(measured_t, measured_y, numerical_t, numerical_y) -> ComparisonMetrics:
    _, m, n = _overlap_resample(measured_t, measured_y, numerical_t, numerical_y)
    if m.size < 3:
        return ComparisonMetrics(samples=int(m.size))
    err = m - n
    rms = float(np.sqrt(np.mean(err * err)))
    denom = float(np.sqrt(np.mean(m * m)))
    nrmse = 100.0 * rms / denom if denom > 1.0e-12 else float("nan")
    if np.std(m) > 1.0e-12 and np.std(n) > 1.0e-12:
        corr = float(np.corrcoef(m, n)[0, 1])
    else:
        corr = float("nan")
    pm = float(np.max(np.abs(m)))
    pn = float(np.max(np.abs(n)))
    ratio = pm / pn if pn > 1.0e-12 else float("nan")
    return ComparisonMetrics(
        rms_error=rms,
        nrmse_percent=nrmse,
        correlation=corr,
        peak_measured=pm,
        peak_numerical=pn,
        peak_ratio=ratio,
        samples=int(m.size),
    )


def fft_amplitude(t, y):
    """Return one-sided Hann-window amplitude spectrum ``(frequency, amplitude)``."""
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = min(t.size, y.size)
    t, y = t[:n], y[:n]
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    if t.size < 8:
        return np.array([]), np.array([])
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.array([]), np.array([])
    fs = 1.0 / float(np.median(dt))
    # Resample to a uniform grid if timestamps are mildly irregular.
    grid = t[0] + np.arange(t.size) / fs
    yy = np.interp(grid, t, y)
    yy = yy - float(np.mean(yy))
    window = np.hanning(yy.size)
    coherent_gain = float(np.mean(window))
    spec = np.fft.rfft(yy * window)
    amp = 2.0 * np.abs(spec) / max(1.0, yy.size * coherent_gain)
    if amp.size:
        amp[0] *= 0.5
        if yy.size % 2 == 0:
            amp[-1] *= 0.5
    freq = np.fft.rfftfreq(yy.size, d=1.0 / fs)
    return freq, amp
