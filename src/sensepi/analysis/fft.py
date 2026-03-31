"""FFT helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def compute_fft(
    signal: ArrayLike,
    sample_rate_hz: float,
    *,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency bins and magnitudes for a real-valued signal.

    Parameters
    ----------
    signal:
        Array-like input signal. Can be 1-D or ND.
    sample_rate_hz:
        Sampling rate in Hz. Must be > 0.
    axis:
        Axis along which to compute the FFT (default: last axis).

    Returns
    -------
    freqs : np.ndarray
        1-D array of frequency bins in Hz.
    magnitude : np.ndarray
        Magnitude of the one-sided FFT along the given axis.
    """
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")

    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        raise ValueError("signal must contain at least one sample")

    # Use rFFT along the specified axis
    fft_result = np.fft.rfft(arr, axis=axis)
    n_samples = arr.shape[axis]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / float(sample_rate_hz))
    magnitude = np.abs(fft_result)

    return freqs, magnitude
