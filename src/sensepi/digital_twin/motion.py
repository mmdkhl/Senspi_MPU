"""Reconstruct a displacement-like signal from accelerometer data, for display.

An accelerometer cannot measure displacement. Integrating acceleration twice in
open loop drifts without bound: any constant offset becomes a ramp, and then a
parabola, and within seconds the reconstruction is dominated by an error that has
nothing to do with the structure.

Over a **short rolling window** it is tractable, and that is the only claim made
here. Three things keep it stable:

1. a band-pass first, so there is no DC term to integrate into a ramp, and
   nothing above the structural band to amplify;
2. detrending between the two integrations, which removes the linear drift the
   first integration introduces;
3. a window of a few seconds, so error has no time to accumulate.

What this is and is not, measured against sinusoids of known amplitude
----------------------------------------------------------------------
* **Relative motion between sensors is faithful: ~0.1 % error** across
  0.5-20 Hz, with noise, and with three modes superposed. That is the number
  that matters, because every channel goes through the *identical* filter chain,
  so whatever distortion the chain introduces is common to all of them and
  cancels in the ratio. A wireframe shows relative motion, so this is the claim
  it rests on.
* **Absolute amplitude is approximate** — a few percent at 1-2 Hz, tens of
  percent by 10 Hz, from the trapezoidal rule and residual filter phase. It is
  not corrected because the display is normalised anyway, and it must not be
  reported as a displacement measurement.

Two things were measured and fixed while building this, both worth stating so
they are not reintroduced: a linear ``detrend`` between integrations left the
result 39 % high (the least-squares line is dragged by the band-pass's edge
transients), and reading the newest sample of a short window gave errors in the
hundreds of percent (the 0.5 Hz filter needs seconds to settle at each end).
Hence :data:`WINDOW_S` and :data:`SETBACK_S` — the second is exactly the
"couple of seconds of delay" that makes this work at all.

``gz`` is an angular **rate**, so the yaw angle needs one integration, not two.

Pure and Qt-free (G7).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

#: Structural band. Below this is drift and tilt; above it is nothing a frame at
#: a few tens of Hz could show anyway.
DEFAULT_BAND = (0.5, 20.0)

#: Shortest window worth reconstructing from. Below roughly two cycles of the
#: low corner there is not enough signal for the band-pass to mean anything.
MIN_WINDOW_S = 2.0

#: Window handed to the reconstruction. Long enough that the 0.5 Hz filter has
#: settled well before the point actually read.
WINDOW_S = 12.0

#: How far back from the newest sample to read. The filter rings at both ends of
#: the window; at the very edge the error is hundreds of percent, and 2 s back it
#: is a few. This is the accepted display latency, spent deliberately.
SETBACK_S = 2.0


def _bandpass(x: np.ndarray, fs: float, band=DEFAULT_BAND) -> np.ndarray:
    lo, hi = float(band[0]), float(min(band[1], 0.45 * fs))
    if hi <= lo:
        return signal.detrend(x, type="linear")
    sos = signal.butter(2, [lo, hi], btype="band", fs=float(fs), output="sos")
    return signal.sosfiltfilt(sos, x)


def _highpass(x: np.ndarray, fs: float, lo: float) -> np.ndarray:
    """Remove what integration just added.

    A linear detrend is the obvious choice here and it is the wrong one: the
    band-pass leaves transients at both ends of the window, the least-squares
    line is dragged by them, and a residual offset survives in the middle where
    the signal is actually being read. Measured on a 2 Hz sinusoid of known
    amplitude, detrending left the velocity 39 % high. A high-pass removes the
    same low-frequency error without fitting anything to the edges.
    """
    lo = float(min(max(lo, 1e-3), 0.45 * fs))
    sos = signal.butter(2, lo, btype="high", fs=float(fs), output="sos")
    return signal.sosfiltfilt(sos, x)


def integrate_once(x: np.ndarray, fs: float, *, band=DEFAULT_BAND) -> np.ndarray:
    """One integration, band-limited and detrended. Rate -> angle."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 8 or not np.isfinite(fs) or fs <= 0:
        return np.zeros_like(x)
    band_limited = _bandpass(np.nan_to_num(x), fs, band)
    # Trapezoidal rather than rectangular: the half-sample bias of a plain
    # cumsum is small but it compounds through the second integration.
    y = np.concatenate(([0.0], np.cumsum(
        (band_limited[1:] + band_limited[:-1]) * 0.5))) / float(fs)
    return _highpass(y, fs, band[0])


def integrate_twice(x: np.ndarray, fs: float, *, band=DEFAULT_BAND) -> np.ndarray:
    """Two integrations. Acceleration -> a displacement-like signal.

    Detrended after each step, which is what keeps the second integration from
    turning the first one's residual slope into a parabola.
    """
    v = integrate_once(x, fs, band=band)
    if v.size < 8:
        return np.zeros_like(v)
    u = np.concatenate(([0.0], np.cumsum((v[1:] + v[:-1]) * 0.5))) / float(fs)
    # BAND-pass, not merely high-pass. Two integrations are a 1/f^2 gain, so a
    # residual sitting just above the corner is amplified by (f_signal/f_corner)^2
    # relative to the signal — measured at 280 % error on a 10 Hz input when only
    # a high-pass was applied here. Restricting the result to the analysis band
    # removes that amplified low-frequency junk instead of integrating it.
    return _bandpass(u, fs, band)


def latest_displacement(x: np.ndarray, fs: float, *, band=DEFAULT_BAND) -> float:
    """The most recent reconstructed displacement sample, or 0.0."""
    u = integrate_twice(x, fs, band=band)
    return float(u[-1]) if u.size else 0.0


def latest_angle(gz: np.ndarray, fs: float, *, band=DEFAULT_BAND) -> float:
    """The most recent yaw angle from a yaw-RATE channel, in radians."""
    a = integrate_once(gz, fs, band=band)
    return float(a[-1]) if a.size else 0.0


def read_back(x: np.ndarray, fs: float, setback_s: float = SETBACK_S) -> float:
    """The sample ``setback_s`` before the end — never the last one.

    The last sample of a filtered window is the least trustworthy one in it.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    i = x.size - 1 - int(max(0.0, float(setback_s)) * float(fs))
    return float(x[max(0, min(i, x.size - 1))])


def displacement_at(x: np.ndarray, fs: float, *, band=DEFAULT_BAND,
                    setback_s: float = SETBACK_S) -> float:
    """Reconstructed displacement, read back from the edge. For display only."""
    return read_back(integrate_twice(x, fs, band=band), fs, setback_s)


def angle_at(gz: np.ndarray, fs: float, *, band=DEFAULT_BAND,
             setback_s: float = SETBACK_S) -> float:
    """Reconstructed yaw angle in radians, read back from the edge."""
    return read_back(integrate_once(gz, fs, band=band), fs, setback_s)


def normalising_scale(values, target: float = 0.9) -> float:
    """A display scale so the largest motion fills roughly ``target`` of a cell.

    The reconstruction has no absolute meaning, so a fixed scale would show
    either nothing or chaos depending on how hard the rig is being driven. The
    scale is chosen from the data and held by the caller between frames, so the
    picture does not breathe.
    """
    arr = np.abs(np.asarray(list(values), dtype=float))
    arr = arr[np.isfinite(arr)]
    peak = float(arr.max()) if arr.size else 0.0
    if peak <= 1e-12:
        return 1.0
    return float(target / peak)
