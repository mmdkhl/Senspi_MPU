"""
Helpers for managing mean-line, envelope, and spike markers on Matplotlib axes.

These utilities are designed for live plots that refresh ~20–60 Hz while the
incoming sensor samples are decimated to a smaller set of points.  The helpers
keep Matplotlib artists alive between frames to avoid unnecessary allocations.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.lines import Line2D


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return ``values`` as a NumPy array of ``float64``."""
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False)
    return np.asarray(values, dtype=float)


def _build_envelope_vertices(t: np.ndarray, y_min: np.ndarray, y_max: np.ndarray) -> np.ndarray:
    """Construct polygon vertices for a standard min/max envelope."""
    upper = np.column_stack((t, y_max))
    lower = np.column_stack((t[::-1], y_min[::-1]))
    return np.vstack((upper, lower))


def init_envelope_plot(
    ax: plt.Axes,
    color: str = "C0",
    alpha: float = 0.2,
    line_kwargs: Optional[dict[str, Any]] = None,
) -> Tuple[Line2D, PolyCollection]:
    """
    Initialize a line/envelope pair on ``ax``.

    Parameters
    ----------
    ax:
        Target axes for the artists.
    color:
        Base color applied to the mean line and envelope fill.
    alpha:
        Opacity for the envelope band.
    line_kwargs:
        Optional keyword arguments forwarded to ``Axes.plot`` for the mean line.
    """
    lw = 1.0
    if line_kwargs:
        lw = float(line_kwargs.get("lw", lw))
    line_opts = {"color": color, "lw": lw}
    if line_kwargs:
        line_opts.update(line_kwargs)

    (line,) = ax.plot([], [], **line_opts)

    face_color = mcolors.to_rgba(line.get_color(), alpha=alpha)
    envelope = PolyCollection(
        verts=[],
        facecolors=[face_color],
        edgecolors="none",
        antialiased=False,
    )
    ax.add_collection(envelope)
    return line, envelope


def update_envelope_plot(
    line: Line2D,
    envelope_coll: PolyCollection,
    t_dec: Sequence[float] | np.ndarray,
    y_mean: Sequence[float] | np.ndarray,
    y_min: Optional[Sequence[float] | np.ndarray],
    y_max: Optional[Sequence[float] | np.ndarray],
) -> PolyCollection:
    """
    Update the artists with newly decimated data.

    Returns the (possibly new) PolyCollection for the envelope so the caller can
    keep the latest reference.
    """
    t_arr = _as_array(t_dec)
    y_mean_arr = _as_array(y_mean)
    line.set_data(t_arr, y_mean_arr)

    if y_min is None or y_max is None or t_arr.size == 0:
        envelope_coll.set_verts([])
        return envelope_coll

    y_min_arr = _as_array(y_min)
    y_max_arr = _as_array(y_max)
    if t_arr.shape != y_min_arr.shape or t_arr.shape != y_max_arr.shape:
        raise ValueError("t_dec, y_min, and y_max must have matching lengths")

    verts = _build_envelope_vertices(t_arr, y_min_arr, y_max_arr)
    envelope_coll.set_verts([verts])
    return envelope_coll


def init_spike_markers(
    ax: plt.Axes,
    color: str = "red",
    marker: str = "x",
    size: float = 30.0,
    zorder: Optional[float] = None,
) -> PathCollection:
    """Create an empty scatter artist used to highlight spikes."""
    scatter = ax.scatter(
        [],
        [],
        s=size,
        color=color,
        marker=marker,
        zorder=zorder,
    )
    return scatter


def update_spike_markers(
    scatter: PathCollection,
    t_dec: Sequence[float] | np.ndarray,
    y_mean: Sequence[float] | np.ndarray,
    y_max: Optional[Sequence[float] | np.ndarray],
    spike_threshold: float,
) -> None:
    """
    Update spike markers based on an absolute threshold above the mean.

    ``spike_threshold`` is compared against ``(y_max - y_mean)`` per interval.
    """
    if y_max is None:
        scatter.set_offsets(np.empty((0, 2)))
        return

    t_arr = _as_array(t_dec)
    y_mean_arr = _as_array(y_mean)
    y_max_arr = _as_array(y_max)
    if t_arr.shape != y_mean_arr.shape or t_arr.shape != y_max_arr.shape:
        raise ValueError("t_dec, y_mean, and y_max must have matching lengths")

    mask = (y_max_arr - y_mean_arr) > float(spike_threshold)
    if not np.any(mask):
        scatter.set_offsets(np.empty((0, 2)))
        return

    t_spikes = t_arr[mask]
    y_spikes = y_max_arr[mask]
    points = np.column_stack((t_spikes, y_spikes))
    scatter.set_offsets(points)
