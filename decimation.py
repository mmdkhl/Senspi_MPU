"""Decimation and smoothing helpers for SensePi sensor data.

This module exposes a streaming friendly :class:`Decimator` that transforms
high-rate sensor samples into plot-ready series by combining decimation,
optional exponential smoothing, and optional min/max envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

WindowOutputs = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]


@dataclass
class DecimationConfig:
    """Configuration for a decimator instance."""

    sensor_fs: float  # Sensor sampling frequency in Hz.
    plot_fs: float  # Desired plot refresh rate in Hz.
    use_envelope: bool = True
    window_mode: str = "block"  # "block" or "sliding".
    smoothing_alpha: Optional[float] = None  # Exponential smoothing factor.

    def __post_init__(self) -> None:
        if self.sensor_fs <= 0 or self.plot_fs <= 0:
            raise ValueError("sensor_fs and plot_fs must be positive.")
        if self.window_mode not in {"block", "sliding"}:
            raise ValueError(f"Unsupported window_mode '{self.window_mode}'.")
        if self.smoothing_alpha is not None:
            if not (0.0 < self.smoothing_alpha <= 1.0):
                raise ValueError("smoothing_alpha must be within (0, 1]; use None to disable.")

    def decimation_factor(self) -> int:
        """Return integer decimation factor (number of samples per output).

        Notes
        -----
        * Returns at least 1. If ``plot_fs >= sensor_fs``, the decimator
          effectively disables decimation and behaves like a pass-through.
        * Uses floor on the ratio to avoid undersampling the stream.
        """
        ratio = self.sensor_fs / self.plot_fs
        if ratio <= 0.0:
            raise ValueError(
                f"Invalid sampling/plot rates: sensor_fs={self.sensor_fs}, "
                f"plot_fs={self.plot_fs}"
            )

        D = int(ratio)
        if D <= 0:
            # Passthrough: sensor is already slower than or equal to the
            # requested plot rate.
            D = 1

        return D

    def window_step(self) -> int:
        """Return window stride measured in sensor samples."""
        D = self.decimation_factor()
        if self.window_mode == "block":
            return D
        # Sliding windows keep half of the previous window by default.
        return max(1, min(D, D // 2 or 1))


@dataclass
class Decimator:
    """Streaming decimator that converts raw samples to plot-friendly data."""

    config: DecimationConfig
    _buffer: np.ndarray = field(init=False, repr=False)
    _idx: int = field(init=False, default=0, repr=False)
    _y_lp: Optional[float] = field(init=False, default=None, repr=False)
    _buffer_t0: Optional[float] = field(init=False, default=None, repr=False)
    _window_step: int = field(init=False, repr=False)
    _dt: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        D = self.config.decimation_factor()
        self._buffer = np.empty(D, dtype=np.float32)
        self._idx = 0
        self._buffer_t0 = None
        self._window_step = self.config.window_step()
        self._dt = 1.0 / float(self.config.sensor_fs)
        if self.config.window_mode == "sliding" and self._window_step > D:
            self._window_step = D

    @property
    def dt(self) -> float:
        """Sensor sampling period in seconds."""
        return self._dt

    @property
    def window_size(self) -> int:
        """Number of samples per decimation window (D)."""
        return self._buffer.size

    @property
    def window_step(self) -> int:
        """Stride between successive windows in samples."""
        return self._window_step

    @property
    def effective_plot_fs(self) -> float:
        """Approximate decimated output rate in Hz."""
        return self.config.sensor_fs / float(self._window_step)

    def reset(self) -> None:
        """Reset internal buffers and smoothing state."""
        self._idx = 0
        self._buffer_t0 = None
        if self.config.smoothing_alpha is not None:
            self._y_lp = None

    def process_block(self, samples: np.ndarray, start_time: float) -> WindowOutputs:
        """Decimate a contiguous 1D array of samples into window statistics.

        Parameters
        ----------
        samples:
            Raw samples acquired at `sensor_fs`.
        start_time:
            Absolute time (seconds) of the first entry inside `samples`.

        Returns
        -------
        t_dec:
            1-D array with timestamps (seconds) located at the midpoint of each
            decimated interval.
        y_mean:
            Mean value for each interval.
        y_min, y_max:
            Optional envelope arrays when `use_envelope=True`.
        """
        flat = np.asarray(samples)

        # Accept scalars and 1-D arrays only; anything higher-dimensional
        # is almost certainly a caller bug (e.g. mixing channels together).
        if flat.ndim == 0:
            flat = flat.reshape(1)
        elif flat.ndim != 1:
            raise ValueError(
                f"Decimator expects a 1-D array of samples, got shape {flat.shape!r}. "
                "Run one Decimator per channel."
            )
        n_samples = flat.size
        if n_samples == 0:
            empty_t = np.empty(0, dtype=np.float64)
            empty_y = np.empty(0, dtype=np.float32)
            if self.config.use_envelope:
                empty_env = np.empty(0, dtype=np.float32)
                return empty_t, empty_y, empty_env, empty_env.copy()
            return empty_t, empty_y, None, None

        buffer = self._buffer
        idx = self._idx
        buf_t0 = self._buffer_t0
        dt = self._dt
        D = buffer.size
        stride = self._window_step
        alpha = self.config.smoothing_alpha
        y_lp = self._y_lp

        t_list = []
        mean_list = []
        min_list = [] if self.config.use_envelope else None
        max_list = [] if self.config.use_envelope else None

        sample_time = float(start_time)
        block_duration = D * dt

        for value in flat:
            sample_val = float(value)
            if alpha is not None:
                if y_lp is None:
                    y_lp = sample_val
                else:
                    y_lp = y_lp + alpha * (sample_val - y_lp)
                sample_val = y_lp

            if idx == 0 and buf_t0 is None:
                buf_t0 = sample_time

            buffer[idx] = sample_val
            idx += 1
            sample_time += dt

            if idx == D:
                block_view = buffer[:D]
                mean_val = float(block_view.mean(dtype=np.float64))
                mean_list.append(mean_val)
                if self.config.use_envelope:
                    min_val = float(block_view.min())
                    max_val = float(block_view.max())
                    min_list.append(min_val)
                    max_list.append(max_val)

                window_start = buf_t0 if buf_t0 is not None else (sample_time - block_duration)
                t_list.append(window_start + 0.5 * block_duration)

                if stride >= D:
                    idx = 0
                    buf_t0 = None
                else:
                    overlap = D - stride
                    if overlap > 0:
                        buffer[:overlap] = buffer[stride:D]
                        idx = overlap
                        buf_t0 = window_start + stride * dt
                    else:
                        idx = 0
                        buf_t0 = None

        self._idx = idx
        self._buffer_t0 = buf_t0
        if alpha is not None:
            self._y_lp = y_lp

        t_dec = np.asarray(t_list, dtype=np.float64)
        y_mean = np.asarray(mean_list, dtype=np.float32)
        if self.config.use_envelope:
            y_min = np.asarray(min_list, dtype=np.float32)
            y_max = np.asarray(max_list, dtype=np.float32)
        else:
            y_min = None
            y_max = None

        return t_dec, y_mean, y_min, y_max


def decimate_array(samples: np.ndarray, start_time: float, config: DecimationConfig) -> WindowOutputs:
    """Stateless convenience helper for one-off block processing."""
    decimator = Decimator(config)
    return decimator.process_block(samples, start_time)
