"""Torsion indicators — how much a mode twists rather than only sways.

Two independent readings, from different physics:

**Differenced pair.** Two sensors on the same floor in different plan cells. For
a rigid floor diaphragm rotating by ``theta`` about the vertical axis, the
in-plane displacements are ``u_x = -theta * y`` and ``u_y = +theta * x``. So the
*difference* between two points is proportional to the rotation and to their
separation **perpendicular to the measured direction**:

    theta  ~  (u_x,A - u_x,B) / (y_B - y_A)          when measuring ax
    theta  ~  (u_y,A - u_y,B) / (x_A - x_B)          when measuring ay

That perpendicular separation matters and is a real constraint, not a detail:
two sensors in the same *row* tell you nothing about torsion from ``ax``, however
far apart they are along x — they both sit at the same ``y``, so rotation moves
them identically. The map knows the plan cells, so this module can say that
outright instead of returning a confident number built on a blind pair.

**Gyroscope.** ``gz`` measures the yaw rate directly, one sensor being enough.
No differencing, no lever arm, no rigid-diaphragm assumption.

Why the output is relative
--------------------------
Nothing in the configuration records the building's plan dimensions, so the lever
arm is only known in **grid units** (the 3x3 cells are 0, 1, 2 apart), and the
accelerometer and gyroscope are in different units anyway. An absolute rotation
in rad/s^2 is therefore not computable and is not claimed. What is computable,
and is what these indicators are:

* ``pair`` — a dimensionless **twist-to-sway ratio**: differential motion over
  common motion, per grid unit of lever arm. 0 means the floor translates
  rigidly; larger means more of that floor's motion at that frequency is
  rotation. Comparable between floors and between modes because both numerator
  and denominator are the same quantity in the same units.
* ``gyro`` — a per-floor **profile**, normalised so the largest floor reads 1.
  Comparable *across floors within one mode*, which is the question worth asking
  ("where does this mode twist most?"), but not an absolute magnitude.

Pure and Qt-free (G7).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

#: Plan-grid axes. Columns A/B/C run along x, rows 1/2/3 along y — the same
#: convention the placement figure draws.
COLS = ("A", "B", "C")
ROWS = ("1", "2", "3")

#: Leakage tolerance when reading an amplitude at an identified frequency: the
#: largest bin within this many bins of the target is used. A mode sits between
#: bins as often as on one, and the peak of a lightly damped resonance is narrow.
PEAK_SEARCH_BINS = 2

#: Below this common-motion amplitude the twist-to-sway ratio is 0/0. It happens
#: at frequencies the floor barely moves at, where the ratio is meaningless
#: rather than large.
MIN_COMMON_FRACTION = 1e-3


@dataclass
class FloorTorsion:
    """One floor's torsion reading at each identified mode."""

    floor: int
    source: str                       # "pair" | "gyro"
    sensors: tuple = ()
    #: One value per identified mode, same order as ``TorsionResult.frequencies_hz``.
    #: NaN where it could not be measured at that frequency.
    per_mode: list = field(default_factory=list)
    #: Perpendicular lever arm in grid units. 0 means the pair is blind to
    #: torsion in the measured direction. Unused for the gyroscope.
    lever_arm_cells: float = 0.0
    usable: bool = True
    note: str = ""


@dataclass
class TorsionResult:
    frequencies_hz: list = field(default_factory=list)
    channel: str = "ax"
    floors: list = field(default_factory=list)
    success: bool = False
    message: str = ""

    def by_floor(self, floor: int, source: str | None = None):
        for f in self.floors:
            if f.floor == floor and (source is None or f.source == source):
                return f
        return None


def cell_offsets(cell: str) -> tuple:
    """``"A1"`` -> ``(0, 0)``; ``"B2"`` (centre) -> ``(1, 1)``. Unknown -> centre."""
    cell = str(cell or "B2")
    try:
        return (COLS.index(cell[0].upper()), ROWS.index(cell[1]))
    except (IndexError, ValueError):
        return (1, 1)


def _perpendicular_coord(cell: str, channel: str) -> float:
    """Grid coordinate that rotation acts through for the measured direction."""
    x, y = cell_offsets(cell)
    return float(x) if str(channel).lower() == "ay" else float(y)


def centre_weight(cell_a: str, cell_b: str, channel: str) -> float:
    """Weight ``w`` such that ``(1-w)*a + w*b`` is the motion at the plan centre.

    Averaging the two sensors is only the translation when the pair straddles the
    centre symmetrically. An off-centre pair — say one sensor at the centre row
    and one at the edge — has a mean that still contains part of the rotation,
    which biases the twist-to-sway ratio by a few percent and, worse, biases it
    *differently on each floor*, so floors stop being comparable. Interpolating
    linearly to the centre instead is exact for a rigid diaphragm, and reduces to
    the plain mean for a symmetric pair.
    """
    ca = _perpendicular_coord(cell_a, channel)
    cb = _perpendicular_coord(cell_b, channel)
    centre = float(len(ROWS) - 1) / 2.0          # 1.0 on the 3x3 grid
    if cb == ca:
        return 0.5
    return (centre - ca) / (cb - ca)


def lever_arm_cells(cell_a: str, cell_b: str, channel: str) -> float:
    """Separation **perpendicular** to the measured direction, in grid units.

    Rotation about the vertical axis moves a point along x in proportion to its
    *y* offset, and along y in proportion to its *x* offset. So the useful
    separation for ``ax`` is in y, and for ``ay`` it is in x. Returns 0.0 when
    the pair cannot see torsion in that direction at all.
    """
    ax_a, ay_a = cell_offsets(cell_a)
    ax_b, ay_b = cell_offsets(cell_b)
    if str(channel).lower() == "ay":
        return float(ax_a - ax_b)
    return float(ay_b - ay_a)          # u_x = -theta * y, hence the reversal


def _hann_spectrum(rows: np.ndarray, fs: float):
    """One-sided Hann FFT amplitudes, zero-padded for finer peak location."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    n = rows.shape[1]
    if n < 4:
        return np.empty(0), np.empty((0, 0))
    n_fft = int(1 << math.ceil(math.log2(max(4096, 4 * n))))
    n_fft = min(n_fft, 1 << 16)
    window = signal.windows.hann(n, sym=False)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    out = np.empty((rows.shape[0], freqs.size), dtype=float)
    for i, row in enumerate(rows):
        x = signal.detrend(np.asarray(row, dtype=float), type="linear")
        out[i] = np.abs(np.fft.rfft(x * window, n=n_fft)) / float(n)
    return freqs, out


def _amplitude_at(freqs: np.ndarray, amps: np.ndarray, f_target: float) -> float:
    """Amplitude at ``f_target``, taking the largest bin within the tolerance."""
    if freqs.size == 0 or not np.isfinite(f_target):
        return float("nan")
    i = int(np.argmin(np.abs(freqs - float(f_target))))
    lo = max(0, i - PEAK_SEARCH_BINS)
    hi = min(freqs.size, i + PEAK_SEARCH_BINS + 1)
    return float(np.max(amps[lo:hi]))


def pair_torsion(
    series_a: np.ndarray,
    series_b: np.ndarray,
    fs: float,
    frequencies_hz,
    *,
    cell_a: str,
    cell_b: str,
    channel: str = "ax",
    floor: int = 0,
    sensors: tuple = (),
) -> FloorTorsion:
    """Twist-to-sway ratio for one floor from a differenced sensor pair."""
    arm = lever_arm_cells(cell_a, cell_b, channel)
    out = FloorTorsion(floor=int(floor), source="pair", sensors=tuple(sensors),
                       lever_arm_cells=arm)

    if arm == 0.0:
        perpendicular = "y" if str(channel).lower() != "ay" else "x"
        out.usable = False
        out.note = (
            f"{cell_a} and {cell_b} sit at the same {perpendicular} position, so "
            f"rotation moves them identically — this pair cannot see torsion from "
            f"{channel}. Move one sensor, or analyse the other axis.")
        out.per_mode = [float("nan")] * len(list(frequencies_hz))
        return out

    a = np.asarray(series_a, dtype=float).ravel()
    b = np.asarray(series_b, dtype=float).ravel()
    n = min(a.size, b.size)
    if n < 4:
        out.usable = False
        out.note = "Too few samples on this floor's pair."
        out.per_mode = [float("nan")] * len(list(frequencies_hz))
        return out
    a, b = a[:n], b[:n]

    # Differential carries the rotation; the centre-interpolated combination
    # carries the sway. Taking both from the same two channels means sensor gain
    # and units cancel in the ratio.
    w = centre_weight(cell_a, cell_b, channel)
    common = (1.0 - w) * a + w * b
    freqs, amps = _hann_spectrum(np.vstack([a - b, common]), fs)
    if freqs.size == 0:
        out.usable = False
        out.note = "Could not compute the spectrum for this floor's pair."
        out.per_mode = [float("nan")] * len(list(frequencies_hz))
        return out

    diff_amp, common_amp = amps[0], amps[1]
    common_floor = float(np.max(common_amp)) * MIN_COMMON_FRACTION
    for f_k in frequencies_hz:
        d = _amplitude_at(freqs, diff_amp, f_k)
        c = _amplitude_at(freqs, common_amp, f_k)
        if not np.isfinite(c) or c <= common_floor:
            # The floor barely moves here; the ratio would be noise over noise.
            out.per_mode.append(float("nan"))
        else:
            out.per_mode.append(float(abs(d / c) / abs(arm)))
    return out


def gyro_torsion(
    gz_rows: np.ndarray,
    fs: float,
    frequencies_hz,
    floors,
    sensor_ids=None,
) -> list:
    """Per-floor yaw-rate profile at each mode, normalised across floors.

    ``gz`` is a direct rotation measurement, so a single sensor per floor is
    enough — no pair, no lever arm, no rigid-diaphragm assumption. The profile is
    normalised per mode so the largest floor reads 1: the meaningful question is
    which floors twist most in a given mode, and the raw rad/s amplitudes are not
    comparable with the accelerometer's units anyway.
    """
    gz_rows = np.atleast_2d(np.asarray(gz_rows, dtype=float))
    floors = [int(f) for f in floors]
    sensor_ids = list(sensor_ids or [0] * len(floors))
    freqs_list = [float(f) for f in frequencies_hz]

    results = [FloorTorsion(floor=f, source="gyro", sensors=(sensor_ids[i],)
                            if i < len(sensor_ids) else ())
               for i, f in enumerate(floors)]
    freqs, amps = _hann_spectrum(gz_rows, fs)
    if freqs.size == 0 or gz_rows.shape[0] != len(floors):
        for r in results:
            r.usable = False
            r.note = "Gyroscope data unavailable for this floor."
            r.per_mode = [float("nan")] * len(freqs_list)
        return results

    raw = np.array([[_amplitude_at(freqs, amps[i], f_k) for f_k in freqs_list]
                    for i in range(len(floors))], dtype=float)
    for k in range(raw.shape[1]):
        col = raw[:, k]
        peak = float(np.nanmax(np.abs(col))) if col.size else 0.0
        if not np.isfinite(peak) or peak <= 0:
            raw[:, k] = np.nan
        else:
            raw[:, k] = col / peak
    for i, r in enumerate(results):
        r.per_mode = [float(v) for v in raw[i]]
    return results


def identify_torsion(
    frequencies_hz,
    *,
    channel: str = "ax",
    pair_series=None,
    gyro_series=None,
    fs: float = 100.0,
) -> TorsionResult:
    """Assemble every torsion reading the placement makes available.

    ``pair_series`` maps ``floor -> (a, b, cell_a, cell_b, (sid_a, sid_b))``.
    ``gyro_series`` maps ``floor -> (gz_row, sensor_id)``.
    Both are optional: a rig may have one, the other, both, or neither.
    """
    freqs_list = [float(f) for f in frequencies_hz]
    out = TorsionResult(frequencies_hz=freqs_list, channel=str(channel))
    if not freqs_list:
        out.message = "No identified frequencies to evaluate torsion at."
        return out

    for floor in sorted(pair_series or {}):
        a, b, cell_a, cell_b, sids = (pair_series or {})[floor]
        out.floors.append(pair_torsion(
            a, b, fs, freqs_list, cell_a=cell_a, cell_b=cell_b,
            channel=channel, floor=floor, sensors=sids))

    gyro = gyro_series or {}
    if gyro:
        floors = sorted(gyro)
        rows = np.vstack([np.asarray(gyro[f][0], dtype=float).ravel() for f in floors])
        out.floors.extend(gyro_torsion(
            rows, fs, freqs_list, floors, [gyro[f][1] for f in floors]))

    usable = [f for f in out.floors if f.usable]
    out.success = bool(usable)
    if not out.floors:
        out.message = (
            "No torsion source. A floor needs either two sensors in different "
            "plan cells, or a gyroscope reading.")
    elif not usable:
        out.message = " · ".join(f.note for f in out.floors if f.note)
    else:
        n_pair = sum(1 for f in usable if f.source == "pair")
        n_gyro = sum(1 for f in usable if f.source == "gyro")
        bits = []
        if n_pair:
            bits.append(f"{n_pair} differenced pair(s)")
        if n_gyro:
            bits.append(f"{n_gyro} gyroscope floor(s)")
        blind = [f for f in out.floors if not f.usable and f.source == "pair"]
        msg = "Torsion from " + " and ".join(bits)
        if blind:
            msg += (" · blind pair on floor(s) "
                    + ", ".join(str(f.floor) for f in blind))
        out.message = msg + " · relative indicator, not calibrated rotation"
    return out
