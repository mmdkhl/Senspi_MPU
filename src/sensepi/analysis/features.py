"""Feature extraction helpers."""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike


Number = Union[float, np.floating]


def _to_1d_array(signal: ArrayLike) -> np.ndarray:
    """Convert input to a 1D float64 numpy array."""
    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        raise ValueError("signal must contain at least one sample")
    if arr.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {arr.shape}")
    return arr


def rms(signal: ArrayLike) -> Number:
    """
    Compute root-mean-square (RMS) value of a 1-D signal.

    Parameters
    ----------
    signal:
        1-D array-like of samples.

    Returns
    -------
    float
        RMS value of the signal.
    """
    arr = _to_1d_array(signal)
    return float(np.sqrt(np.mean(np.square(arr))))


def peak_to_peak(signal: ArrayLike) -> Number:
    """
    Compute peak-to-peak value (max - min) of a 1-D signal.

    Parameters
    ----------
    signal:
        1-D array-like of samples.

    Returns
    -------
    float
        Peak-to-peak amplitude.
    """
    arr = _to_1d_array(signal)
    return float(np.max(arr) - np.min(arr))
