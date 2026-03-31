"""Filtering helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal


def butter_lowpass(
    data: ArrayLike,
    cutoff_hz: float,
    sample_rate_hz: float,
    order: int = 4,
    *,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter using filtfilt.

    Parameters
    ----------
    data:
        Input data (array-like). Filtering is applied along `axis`.
    cutoff_hz:
        Cutoff frequency in Hz (0 < cutoff_hz < sample_rate_hz / 2).
    sample_rate_hz:
        Sampling rate in Hz. Must be > 0.
    order:
        Filter order (default: 4).
    axis:
        Axis along which to filter (default: last axis).

    Returns
    -------
    np.ndarray
        Filtered data with the same shape as the input.
    """
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be > 0, got {cutoff_hz}")

    nyquist = 0.5 * float(sample_rate_hz)
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz must be < Nyquist ({nyquist:.3f} Hz), got {cutoff_hz}"
        )

    data_arr = np.asarray(data, dtype=float)
    normal_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data_arr, axis=axis)


def detrend(data: ArrayLike, *, axis: int = -1, type: str = "linear") -> np.ndarray:
    """
    Remove a trend from data using scipy.signal.detrend.

    Parameters
    ----------
    data:
        Input data (array-like).
    axis:
        Axis along which to detrend (default: last axis).
    type:
        Type of detrending. One of {"linear", "constant"}; see scipy docs.

    Returns
    -------
    np.ndarray
        Detrended data.
    """
    data_arr = np.asarray(data, dtype=float)
    return signal.detrend(data_arr, axis=axis, type=type)
