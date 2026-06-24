"""Operational (output-only) modal identification from sensor acceleration.

This module is the bridge between raw sensor data and the OpenSees calibrator.
It takes synchronized acceleration time series (one ``ax`` channel per sensor)
and extracts natural frequencies and mode shapes using **Frequency Domain
Decomposition (FDD)** — the SVD of the cross-spectral-density (CSD) matrix.

FDD is used instead of plain Welch peak-picking because the first left singular
vector at each spectral peak gives a *signed* mode shape (floors moving out of
phase produce negative entries), which plain PSD magnitude cannot recover.

Design constraints (see project guardrails G7):
- Pure numpy/scipy. **No Qt, no SSH, no OpenSees imports.**
- Input: numpy arrays. Output: plain dataclasses / dicts.
- Fully testable without a Pi or OpenSees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal


# Minimum recording length we trust for identification. Below this there are
# too few cycles of the fundamental and the frequency resolution is too coarse.
MIN_DURATION_S = 10.0


@dataclass
class ExperimentalModalResult:
    """Sensor-level identification result (before sensor->story mapping).

    ``mode_shapes_sensor[k]`` is the signed, ``|max| = 1`` normalized shape of
    mode ``k``, with one entry per input sensor (same order as the input rows).
    """

    frequencies_hz: list[float] = field(default_factory=list)
    mode_shapes_sensor: list[list[float]] = field(default_factory=list)
    damping_ratios: list[float] = field(default_factory=list)
    # The identification spectrum that gets plotted. For ``method="fdd"`` this is
    # the first singular value of the CSD matrix; for ``method="fft"`` it is the
    # sensor-averaged Welch PSD. The field names keep the ``fdd_`` prefix for
    # backward compatibility; ``method`` says which spectrum it actually holds.
    fdd_freqs: np.ndarray = field(default_factory=lambda: np.empty(0))
    fdd_spectrum: np.ndarray = field(default_factory=lambda: np.empty(0))
    method: str = "fdd"
    n_modes_found: int = 0
    success: bool = False
    message: str = ""


@dataclass
class StoryModalData:
    """Sensor result mapped onto stories — ready for the OpenSees calibrator.

    ``mode_shapes_ux`` is populated (and ``mode_shapes_available`` True) whenever
    at least one story has a sensor — the vector then carries one entry per
    *measured* story, ordered by ``coverage_stories`` (B1, partial coverage).
    ``full_coverage`` says whether every story was measured. ``measured_points``
    still carries what was measured so the GUI can plot identified points over
    the FEM mode shape.
    """

    frequencies_hz: list[float] = field(default_factory=list)
    mode_shapes_ux: dict[str, list[float]] = field(default_factory=dict)
    mode_shapes_available: bool = False
    coverage_stories: list[int] = field(default_factory=list)
    full_coverage: bool = False
    n_story: int = 0
    # Per mode: {story: measured value} for the stories that have sensors.
    measured_points: list[dict[int, float]] = field(default_factory=list)
    # Per story that has >1 sensor: spread of the sensor values (torsion proxy).
    torsion_indicator: dict[int, float] = field(default_factory=dict)


def estimate_fs(timestamps_s: np.ndarray) -> float:
    """Estimate sampling rate (Hz) from a monotonic timestamp vector (seconds)."""
    t = np.asarray(timestamps_s, dtype=float)
    if t.size < 2:
        return float("nan")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def _choose_nperseg(n_samples: int) -> int:
    """Pick a Welch/CSD segment length: power of two, ~3 averaging segments."""
    target = max(256, n_samples // 3)
    nperseg = 1 << int(math.floor(math.log2(target)))
    return int(min(nperseg, n_samples))


def _normalize_signed(vec: np.ndarray) -> np.ndarray:
    """Normalize a real mode-shape vector so the largest-magnitude entry is +1."""
    if vec.size == 0:
        return vec
    idx = int(np.argmax(np.abs(vec)))
    peak = vec[idx]
    if peak == 0.0 or not np.isfinite(peak):
        return vec
    return vec / peak


def _align_phase_real(u: np.ndarray) -> np.ndarray:
    """Rotate a complex singular vector to maximize the real part, return real."""
    if u.size == 0:
        return np.real(u)
    idx = int(np.argmax(np.abs(u)))
    ref = u[idx]
    if ref != 0:
        u = u * np.exp(-1j * np.angle(ref))
    return np.real(u)


def _half_power_damping(freqs: np.ndarray, spectrum: np.ndarray, peak_idx: int) -> float:
    """Rough damping estimate from the half-power bandwidth around a peak."""
    peak_val = spectrum[peak_idx]
    if peak_val <= 0:
        return float("nan")
    half = peak_val / 2.0  # -3 dB on a power spectrum

    lo = peak_idx
    while lo > 0 and spectrum[lo] > half:
        lo -= 1
    hi = peak_idx
    while hi < spectrum.size - 1 and spectrum[hi] > half:
        hi += 1
    if hi <= lo:
        return float("nan")
    f_n = freqs[peak_idx]
    if f_n <= 0:
        return float("nan")
    bandwidth = freqs[hi] - freqs[lo]
    return float(bandwidth / (2.0 * f_n))


def _pick_peaks_and_build(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    shape_fn,
    *,
    f_min: float,
    f_max: float,
    n_modes: int,
    prominence_db: float,
    method: str,
) -> ExperimentalModalResult:
    """Shared tail for FDD and FFT: peak-pick a spectrum and build the result.

    ``shape_fn(idx)`` returns the (already normalized) mode-shape vector at the
    spectral bin ``idx`` — signed for FDD, magnitude-only for FFT.
    """
    nf = freqs.size
    band = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(band):
        return ExperimentalModalResult(
            fdd_freqs=freqs, fdd_spectrum=spectrum, method=method,
            success=False, message=f"No spectral lines in [{f_min}, {f_max}] Hz.",
        )

    # Peak-pick on the dB spectrum within the band.
    floor = np.max(spectrum[band]) * 1e-9 + 1e-30
    spec_db = 10.0 * np.log10(np.maximum(spectrum, floor))
    band_idx = np.where(band)[0]
    min_distance = max(1, int(round(0.25 / (freqs[1] - freqs[0])))) if nf > 1 else 1
    peaks_local, props = signal.find_peaks(
        spec_db[band_idx], prominence=prominence_db, distance=min_distance
    )
    peaks = band_idx[peaks_local]

    if peaks.size == 0:
        return ExperimentalModalResult(
            fdd_freqs=freqs, fdd_spectrum=spectrum, method=method,
            n_modes_found=0, success=False,
            message="No spectral peaks found above the prominence threshold.",
        )

    # Keep the strongest n_modes peaks, then order them by ascending frequency.
    order_by_strength = np.argsort(props["prominences"])[::-1]
    kept = peaks[order_by_strength][: max(1, n_modes)]
    kept = kept[np.argsort(freqs[kept])]

    frequencies: list[float] = []
    shapes: list[list[float]] = []
    dampings: list[float] = []
    for idx in kept:
        frequencies.append(float(freqs[idx]))
        shapes.append([float(v) for v in shape_fn(int(idx))])
        dampings.append(_half_power_damping(freqs, spectrum, int(idx)))

    n_found = len(frequencies)
    if n_found < n_modes:
        message = f"Found {n_found} of {n_modes} requested modes."
    else:
        message = f"Found {n_found} modes."

    return ExperimentalModalResult(
        frequencies_hz=frequencies,
        mode_shapes_sensor=shapes,
        damping_ratios=dampings,
        fdd_freqs=freqs,
        fdd_spectrum=spectrum,
        method=method,
        n_modes_found=n_found,
        success=n_found > 0,
        message=message,
    )


def _identify_fdd(work, fs, nperseg, noverlap, *, f_min, f_max, n_modes, prominence_db):
    """FDD: SVD of the cross-spectral-density matrix → signed mode shapes."""
    n_sensors = work.shape[0]

    # Build the cross-spectral-density matrix G(f), shape (nf, ns, ns).
    freqs, _ = signal.csd(work[0], work[0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    nf = freqs.size
    G = np.zeros((nf, n_sensors, n_sensors), dtype=complex)
    for i in range(n_sensors):
        for j in range(i, n_sensors):
            _, gij = signal.csd(work[i], work[j], fs=fs, nperseg=nperseg, noverlap=noverlap)
            G[:, i, j] = gij
            if i != j:
                G[:, j, i] = np.conj(gij)

    # First singular value (FDD spectrum) and first singular vector per frequency.
    s1 = np.zeros(nf)
    u1 = np.zeros((nf, n_sensors), dtype=complex)
    for k in range(nf):
        U, S, _ = np.linalg.svd(G[k])
        s1[k] = S[0]
        u1[k] = U[:, 0]

    def shape_fn(idx: int) -> np.ndarray:
        return _normalize_signed(_align_phase_real(u1[idx]))

    return _pick_peaks_and_build(
        freqs, s1, shape_fn, f_min=f_min, f_max=f_max,
        n_modes=n_modes, prominence_db=prominence_db, method="fdd",
    )


def _identify_fft(work, fs, nperseg, noverlap, *, f_min, f_max, n_modes, prominence_db):
    """FFT: sensor-averaged Welch PSD → frequencies; magnitude-only mode shapes.

    PSD magnitude is always positive, so the resulting shapes are unsigned (the
    sign of out-of-phase floors cannot be recovered). This is the natural fit for
    frequency-only calibration; use FDD when signed shapes are needed.
    """
    n_sensors = work.shape[0]
    psd = []
    freqs = np.empty(0)
    for i in range(n_sensors):
        freqs, pxx = signal.welch(work[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        psd.append(pxx)
    psd = np.asarray(psd)               # (n_sensors, nf)
    psd_avg = psd.mean(axis=0)          # combined FFT spectrum

    def shape_fn(idx: int) -> np.ndarray:
        mag = np.sqrt(np.maximum(psd[:, idx], 0.0))
        return _normalize_signed(mag)

    return _pick_peaks_and_build(
        freqs, psd_avg, shape_fn, f_min=f_min, f_max=f_max,
        n_modes=n_modes, prominence_db=prominence_db, method="fft",
    )


def identify_modes(
    data: np.ndarray,
    fs: float,
    *,
    f_min: float = 0.5,
    f_max: float = 20.0,
    n_modes: int = 3,
    nperseg: int | None = None,
    prominence_db: float = 3.0,
    detrend: bool = True,
    method: str = "fdd",
) -> ExperimentalModalResult:
    """Identify natural frequencies and mode shapes from sensor acceleration.

    Parameters
    ----------
    data : np.ndarray, shape ``(n_sensors, n_samples)``
        Acceleration time series, one ``ax`` channel per sensor, already
        resampled onto a common uniform time grid.
    fs : float
        Sampling rate in Hz.
    f_min, f_max : float
        Frequency band to search for modes.
    n_modes : int
        Maximum number of modes to return (strongest peaks, sorted by frequency).
    nperseg : int, optional
        CSD/Welch segment length. Auto-chosen if None.
    prominence_db : float
        Peak prominence threshold on the dB spectrum.
    method : {"fdd", "fft"}
        ``"fdd"`` (default) — SVD of the cross-spectral-density matrix; gives
        *signed* mode shapes. ``"fft"`` — sensor-averaged Welch PSD peak-picking;
        simpler and more transparent, but mode shapes are magnitude-only.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_sensors, n_samples = data.shape

    method = str(method).lower()
    if method not in ("fdd", "fft"):
        return ExperimentalModalResult(
            success=False, message=f"Unknown identification method: {method!r}.",
        )

    if not np.isfinite(fs) or fs <= 0:
        return ExperimentalModalResult(
            method=method, success=False, message="Invalid sampling rate.")
    duration = n_samples / fs
    if duration < MIN_DURATION_S:
        return ExperimentalModalResult(
            method=method,
            success=False,
            message=(
                f"Recording too short: {duration:.1f} s < {MIN_DURATION_S:.0f} s minimum. "
                "Record a longer segment (30 s recommended)."
            ),
        )
    if f_max >= fs / 2.0:
        f_max = 0.95 * (fs / 2.0)

    work = data
    if detrend:
        work = signal.detrend(work, axis=1, type="linear")

    if nperseg is None:
        nperseg = _choose_nperseg(n_samples)
    nperseg = int(min(nperseg, n_samples))
    noverlap = nperseg // 2

    if method == "fft":
        return _identify_fft(
            work, fs, nperseg, noverlap,
            f_min=f_min, f_max=f_max, n_modes=n_modes, prominence_db=prominence_db,
        )
    return _identify_fdd(
        work, fs, nperseg, noverlap,
        f_min=f_min, f_max=f_max, n_modes=n_modes, prominence_db=prominence_db,
    )


def map_to_stories(
    result: ExperimentalModalResult,
    sensor_story_map: list[int],
    n_story: int,
) -> StoryModalData:
    """Map a sensor-level result onto stories for the OpenSees calibrator.

    ``sensor_story_map[i]`` is the 1-based story that input sensor ``i`` sits on.
    Multiple sensors on one story are averaged for the story's ux value; their
    spread is recorded in ``torsion_indicator``.
    """
    sensor_story_map = [int(s) for s in sensor_story_map]
    coverage = sorted({s for s in sensor_story_map if 1 <= s <= n_story})
    full_coverage = len(coverage) == n_story and n_story > 0

    out = StoryModalData(
        frequencies_hz=list(result.frequencies_hz),
        coverage_stories=coverage,
        full_coverage=full_coverage,
        n_story=n_story,
    )

    mode_shapes_ux: dict[str, list[float]] = {}
    for m, shape in enumerate(result.mode_shapes_sensor):
        # Group this mode's sensor values by story.
        per_story_vals: dict[int, list[float]] = {}
        for sensor_i, story in enumerate(sensor_story_map):
            if sensor_i < len(shape) and 1 <= story <= n_story:
                per_story_vals.setdefault(story, []).append(float(shape[sensor_i]))

        measured = {story: float(np.mean(vals)) for story, vals in per_story_vals.items()}
        out.measured_points.append(measured)

        # Torsion proxy: spread among sensors sharing a story (computed once).
        if m == 0:
            for story, vals in per_story_vals.items():
                if len(vals) > 1:
                    out.torsion_indicator[story] = float(np.max(vals) - np.min(vals))

        # B1 (partial coverage): emit the mode-shape vector over the MEASURED
        # stories, ordered by sorted ``coverage`` and normalized on that measured
        # support. Unmeasured interior stories are simply omitted (never
        # interpolated). Full coverage is the special case where every story is
        # measured, and produces the same vector as before. ``measured`` always
        # holds exactly the coverage stories, so the guard is a safety net.
        if coverage and all(s in measured for s in coverage):
            vec = np.array([measured[s] for s in coverage], dtype=float)
            vec = _normalize_signed(vec)
            mode_shapes_ux[str(m + 1)] = [float(v) for v in vec]

    if mode_shapes_ux:
        out.mode_shapes_ux = mode_shapes_ux
        out.mode_shapes_available = True

    return out


def to_experimental_dict(story_data: StoryModalData, notes: str = "") -> dict:
    """Produce the dict the calibrator path expects.

    Matches the ``experimental_modal_data.json`` schema and the structure that
    ``ModelUpdatingTab._build_exp_data_from_gui_values`` consumes:
    ``{"frequencies_hz": [...], "mode_shapes_ux": {...},
    "measured_dof_indices": [...], "notes": "..."}``.

    Under partial coverage the shape vectors only span the measured stories, so
    ``measured_dof_indices`` (0-based, in the same sorted-story order) tells the
    calibrator which model DOFs to compare against (B1 / BLOCKER-8).
    """
    data: dict = {
        "frequencies_hz": list(story_data.frequencies_hz),
        "notes": notes or "Identified from sensor data (FDD)",
    }
    if story_data.mode_shapes_available:
        data["mode_shapes_ux"] = dict(story_data.mode_shapes_ux)
        # 0-based DOF indices the shape vectors correspond to, same order as
        # mode_shapes_ux. Always safe to include; full coverage → [0..n_story-1].
        data["measured_dof_indices"] = [s - 1 for s in story_data.coverage_stories]
    return data
