"""Base-referenced (input-output) modal identification.

Everything else in this app is **output-only** (OMA): ``identify_modes`` looks at
the responses alone and assumes the excitation is broadband and flat, so that a
peak in the response spectrum is a property of the structure. On a shake table
that assumption is not safe. The shaker has its own spectrum, and any peak,
roll-off or harmonic in the *drive* shows up in every response at once — looking
exactly like a mode, on every sensor, perfectly "coherent" between them.

When a sensor sits on floor 0 we are measuring that drive, so we can divide it
out and stop guessing. This module is that path.

The estimator
-------------
``H1(f) = S_ref,resp(f) / S_ref,ref(f)`` — the cross-spectrum between the base and
each response, over the base auto-spectrum. H1 is the standard choice when the
noise is on the *output* (sensor noise, ambient traffic, someone leaning on the
frame), which is the realistic case here; it is unbiased under that assumption in
a way that a plain amplitude ratio is not.

Because both channels are accelerations, ``H1`` is dimensionless: it is the
**transmissibility**, "how many g at this floor per g at the base". At a natural
frequency it peaks; well away from one it tends to 1 (the floor simply rides
along with the base) rather than to zero.

Coherence is the reason this is worth doing
-------------------------------------------
``γ²(f) = |S_ref,resp|² / (S_ref,ref · S_resp,resp)`` is the fraction of the
response at ``f`` that is linearly explained by the measured input. It is the
honest quality gate an output-only method cannot offer: a peak with γ² ≈ 0.95 is
the structure answering the shaker; a peak with γ² ≈ 0.2 is something else in the
room, and is rejected here rather than reported as a mode.

Coherence only means anything when the spectra are **averaged over several
segments** — with a single segment it is identically 1 by construction. That is
why this module picks its own segment length instead of reusing the one the FDD
path uses, which deliberately takes one long segment for resolution.

Nothing in :mod:`sensepi.analysis.modal` is modified. The result is the same
``ExperimentalModalResult`` the rest of the app already consumes, so
``map_to_stories`` and every downstream consumer work unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

# Shared internals, imported rather than duplicated so that the base-referenced
# path picks peaks by exactly the same rule as FDD and FFT. If that rule changes,
# it changes in one place.
from .modal import (ExperimentalModalResult, _align_phase_real,
                    _normalize_signed)

#: Reject a peak whose coherence is below this.
#:
#: What this gate does and does not do, measured rather than assumed. Against a
#: 3-DOF frame with known modes, driven modes score 0.91-0.98 while an
#: *unrelated* reference tops out at 0.73 — so the gate reliably catches the
#: failure that matters in practice: the floor-0 sensor is mismapped, fell off,
#: or the shaker was not running, and the whole result is meaningless.
#:
#: It does **not** reliably reject narrowband ambient contamination. Near such a
#: peak the structure is still responding to the shaker, so coherence stays high
#: there too; the separation was 0.00-0.05 and sometimes inverted. What suppresses
#: ambient energy is the H1 estimator itself — noise uncorrelated with the
#: reference averages toward zero in the cross-spectrum — so those peaks come out
#: weak and lose the prominence ranking. The gate is a backstop, not that defence.
DEFAULT_MIN_COHERENCE = 0.85

#: Segments to average for the coherence estimate. Below ~6 the estimate is so
#: biased upward that the gate stops discriminating.
MIN_SEGMENTS = 6

#: Coherence is read as the maximum within this many true resolution bins of the
#: peak, not at the peak bin itself. At a lightly damped resonance the Welch
#: coherence estimate *dips* exactly at the peak — the mode is narrower than the
#: segment resolution, so the estimate is biased down right where it is needed.
#: Measured on a known frame: at the peak bin a genuine mode read 0.56 while the
#: bins beside it read 0.98. Gating on the peak bin alone rejected real modes.
COHERENCE_NEIGHBOURHOOD_BINS = 3

#: A peak is judged at the full threshold only when the record resolves it:
#: resolution <= f_peak / this. Below that the coherence estimate is too biased
#: for the full threshold (at 20-30 s records spurious peaks outscored real ones).
GATE_RESOLUTION_RATIO = 10.0

#: Shortest record this method should be asked for. At 12-20 s the segment
#: length that buys enough averages leaves ~0.2-0.4 Hz resolution, which cannot
#: resolve a first mode near 1-2 Hz: the coherence gate then falls back to its
#: relaxed floor everywhere and the frequency itself carries a several-percent
#: error. 30 s is where the gate starts applying across the structural band.
BASE_REF_MIN_DURATION_S = 30.0

#: Fraction of the threshold applied to a peak too low in frequency to be
#: resolved. Not exempt, just judged leniently: the resolution bias pushes
#: coherence *down*, so a low reading there is ambiguous — but a reading near
#: zero is not, and letting those through unjudged reported pure noise as a mode.
RELAXED_GATE_FRACTION = 0.5

# --- why this module reports no damping ---------------------------------
# Half-power bandwidth was implemented here and then removed, because measured
# against a 3-DOF frame with known damping it was never trustworthy at the
# resolution a live capture can buy (100 Hz, 12-20 s -> ~0.1 Hz true resolution):
#
#   zeta_true   0.01   0.02   0.05   0.08   0.12
#   error        n/a    n/a    18 %   17 %   84-186 %
#
# Lightly damped peaks are narrower than the Hann window's own main lobe, so what
# gets measured is the window; heavily damped peaks overlap their neighbours, so
# the -3 dB crossing lands on the next mode. Neither is fixable by thresholding —
# there is no band of damping values where the answer is good. Reporting NaN is
# the honest result. Damping comes from the log-decrement fit
# (``modal.estimate_damping_first_mode_real_response``), which works in the time
# domain and is not resolution-limited in this way.


@dataclass
class BaseReferencedResult:
    """The modal result plus the diagnostics that justify believing it."""

    modal: ExperimentalModalResult = field(default_factory=ExperimentalModalResult)
    #: Frequency grid shared by every curve below.
    freqs: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Mean coherence across responses, per frequency.
    coherence: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Mean coherence at each *identified* frequency, same order as ``modal``.
    mode_coherence: list = field(default_factory=list)
    #: Peaks that were found but thrown out by the coherence gate, as
    #: ``(frequency_hz, coherence)`` — reported so a rejection is visible rather
    #: than looking like the structure simply has fewer modes.
    rejected: list = field(default_factory=list)
    #: Whether the record was long enough for the coherence gate to be enforced.
    #: When False the coherence values are still reported, but nothing was
    #: rejected on them — see ``GATE_RESOLUTION_RATIO``.
    gate_enforced: bool = False
    n_segments: int = 0
    #: Spacing of the returned (zero-padded) grid — how finely a peak is located.
    freq_resolution_hz: float = 0.0
    #: fs / nperseg — the resolution the record actually bought. Two peaks closer
    #: than this are not resolved, however fine the returned grid looks.
    true_resolution_hz: float = 0.0
    success: bool = False
    message: str = ""


def _choose_frf_nperseg(n_samples: int, min_segments: int = MIN_SEGMENTS) -> int:
    """Largest power-of-two segment that still yields ``min_segments`` averages.

    With 50 % overlap the segment count is ``floor(2n / nperseg) - 1``, so the
    constraint is ``nperseg <= 2n / (min_segments + 1)``. Resolution is traded for
    averaging deliberately: without averaging there is no coherence, and without
    coherence there is no reason to prefer this method over FDD.
    """
    if n_samples <= 0:
        return 1
    cap = max(64.0, 2.0 * n_samples / float(min_segments + 1))
    nperseg = 1 << int(math.floor(math.log2(cap)))
    return int(max(64, min(nperseg, n_samples)))


def _segment_count(n_samples: int, nperseg: int) -> int:
    if nperseg <= 0 or n_samples < nperseg:
        return 0
    return int((n_samples - nperseg) // (nperseg // 2)) + 1


def estimate_transmissibility(
    response: np.ndarray,
    reference: np.ndarray,
    fs: float,
    *,
    nperseg: int | None = None,
    zero_pad: int = 4,
):
    """H1 transmissibility and coherence of every response against the base.

    Parameters
    ----------
    response : ``(n_sensors, n_samples)``
        Structural responses, one row per sensor, on a common time grid.
    reference : ``(n_samples,)``
        The base/shaker channel, same grid and units.
    zero_pad : int
        FFT length as a multiple of the segment length. Zero-padding interpolates
        the spectrum onto a finer grid, which locates a peak more precisely. It
        does **not** create resolution — two modes closer together than one
        segment's bandwidth stay unresolved.

    Returns ``(freqs, H, coherence, n_segments, nperseg)`` where ``H`` is complex
    ``(n_sensors, n_freq)``.
    """
    response = np.atleast_2d(np.asarray(response, dtype=float))
    reference = np.asarray(reference, dtype=float).ravel()
    n_sensors, n_samples = response.shape
    if n_sensors == 0 or n_samples < 4 or reference.size != n_samples:
        return np.empty(0), np.empty((0, 0), dtype=complex), np.empty(0), 0, 0

    if nperseg is None:
        nperseg = _choose_frf_nperseg(n_samples)
    nperseg = int(max(8, min(nperseg, n_samples)))
    noverlap = nperseg // 2
    nfft = int(nperseg * max(1, int(zero_pad)))
    kw = dict(fs=float(fs), nperseg=nperseg, noverlap=noverlap, nfft=nfft,
              detrend="linear")

    freqs, p_rr = signal.welch(reference, **kw)
    # Guard the division: away from the drive band the base auto-spectrum can be
    # numerically zero, and 0/0 would put NaNs through the whole curve.
    floor = float(np.max(p_rr)) * 1e-12 + 1e-30
    p_rr_safe = np.maximum(p_rr, floor)

    h_rows = []
    coh_rows = []
    for row in response:
        _, p_ro = signal.csd(reference, row, **kw)
        _, p_oo = signal.welch(row, **kw)
        h_rows.append(p_ro / p_rr_safe)
        denom = p_rr_safe * np.maximum(p_oo, float(np.max(p_oo)) * 1e-12 + 1e-30)
        coh_rows.append(np.abs(p_ro) ** 2 / denom)

    h = np.asarray(h_rows, dtype=complex)
    coherence = np.clip(np.mean(np.asarray(coh_rows, dtype=float), axis=0), 0.0, 1.0)
    return freqs, h, coherence, _segment_count(n_samples, nperseg), nperseg


def identify_modes_base_referenced(
    response: np.ndarray,
    reference: np.ndarray,
    fs: float,
    *,
    f_min: float = 0.5,
    f_max: float = 20.0,
    n_modes: int = 3,
    prominence_db: float = 3.0,
    min_coherence: float = DEFAULT_MIN_COHERENCE,
    nperseg: int | None = None,
    min_duration: float | None = None,
) -> BaseReferencedResult:
    """Identify modes from the transmissibility between the base and the floors.

    Peaks are picked on ``Σ|H|²`` — power-like, so the half-power bandwidth has
    its usual meaning — then filtered by coherence. Mode shapes are the complex
    ``H`` column at the peak, phase-rotated to real and sign-normalised, which is
    the operating deflection shape *referenced to the measured input* rather than
    to whatever the excitation happened to look like.
    """
    response = np.atleast_2d(np.asarray(response, dtype=float))
    reference = np.asarray(reference, dtype=float).ravel()
    n_sensors, n_samples = response.shape

    if n_sensors == 0:
        return BaseReferencedResult(message="No structural sensor to identify from.")
    if reference.size != n_samples:
        return BaseReferencedResult(
            message="Base and response channels have different lengths.")
    if not np.isfinite(fs) or fs <= 0:
        return BaseReferencedResult(message="Invalid sampling rate.")

    floor_s = (BASE_REF_MIN_DURATION_S if min_duration is None
               else float(min_duration))
    duration = n_samples / float(fs)
    if duration < floor_s:
        return BaseReferencedResult(
            message=f"Recording too short: {duration:.1f} s < {floor_s:.1f} s minimum.")

    if float(np.std(reference)) <= 0.0:
        return BaseReferencedResult(
            message="The base sensor is not moving — nothing to reference against.")

    freqs, h, coherence, n_seg, used_nperseg = estimate_transmissibility(
        response, reference, fs, nperseg=nperseg)
    if freqs.size == 0:
        return BaseReferencedResult(message="Could not estimate the transmissibility.")

    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    df_true = float(fs) / used_nperseg if used_nperseg else 0.0
    spectrum = np.sum(np.abs(h) ** 2, axis=0)

    out = BaseReferencedResult(
        freqs=freqs, coherence=coherence, n_segments=n_seg,
        freq_resolution_hz=df, true_resolution_hz=df_true)

    band = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(band):
        out.message = f"No spectral lines in [{f_min}, {f_max}] Hz."
        return out

    # Same peak-picking rule as the FDD/FFT tail: prominence on the dB spectrum,
    # a 0.25 Hz minimum separation so one mode is not counted twice.
    floor_val = float(np.max(spectrum[band])) * 1e-9 + 1e-30
    spec_db = 10.0 * np.log10(np.maximum(spectrum, floor_val))
    band_idx = np.where(band)[0]
    min_distance = max(1, int(round(0.25 / df))) if df > 0 else 1
    peaks_local, props = signal.find_peaks(
        spec_db[band_idx], prominence=prominence_db, distance=min_distance)
    peaks = band_idx[peaks_local]
    if peaks.size == 0:
        out.message = "No transmissibility peaks above the prominence threshold."
        return out

    # Coherence per peak, read over a small neighbourhood rather than at the peak
    # bin (see COHERENCE_NEIGHBOURHOOD_BINS).
    half_bins = 1
    if df > 0 and df_true > 0:
        half_bins = max(1, int(round(COHERENCE_NEIGHBOURHOOD_BINS * df_true / df)))
    peak_coh = np.array([
        float(np.max(coherence[max(0, i - half_bins):i + half_bins + 1]))
        for i in peaks], dtype=float)

    # Enforce the gate PER PEAK, only where the record resolves that peak well
    # enough for its coherence estimate to be trustworthy. Deciding this once for
    # the whole spectrum let a single unresolvable peak near the band edge switch
    # the gate off for every real mode above it.
    resolvable = (np.asarray(freqs[peaks], dtype=float) / GATE_RESOLUTION_RATIO
                  >= df_true) if df_true > 0 else np.zeros(peaks.size, dtype=bool)
    out.gate_enforced = bool(np.any(resolvable))

    # Gate BEFORE the strongest-N cut, so a loud but undriven peak cannot crowd
    # out a quieter genuine mode.
    prominences = np.asarray(props["prominences"], dtype=float)
    thresholds = np.where(resolvable, float(min_coherence),
                          float(min_coherence) * RELAXED_GATE_FRACTION)
    keep_mask = peak_coh >= thresholds
    for i, idx in enumerate(peaks):
        if not keep_mask[i]:
            out.rejected.append((float(freqs[idx]), float(peak_coh[i])))
    coh_by_peak = {int(idx): float(peak_coh[i]) for i, idx in enumerate(peaks)}
    peaks = peaks[keep_mask]
    prominences = prominences[keep_mask]

    if peaks.size == 0:
        out.message = (
            f"Every peak failed the coherence gate (γ² < {min_coherence:.2f}). "
            "The floors are not moving with the sensor mapped to floor 0 — check "
            "that it really is the base sensor and that the shaker is running.")
        return out

    kept = peaks[np.argsort(prominences)[::-1]][: max(1, int(n_modes))]
    kept = kept[np.argsort(freqs[kept])]

    frequencies: list = []
    shapes: list = []
    for idx in kept:
        idx = int(idx)
        frequencies.append(float(freqs[idx]))
        shapes.append([float(v) for v in
                       _normalize_signed(_align_phase_real(h[:, idx]))])
        out.mode_coherence.append(coh_by_peak.get(idx, float(coherence[idx])))

    n_found = len(frequencies)
    out.modal = ExperimentalModalResult(
        frequencies_hz=frequencies,
        mode_shapes_sensor=shapes,
        # Deliberately not measured here — see the note at the top of the module.
        damping_ratios=[float("nan")] * n_found,
        fdd_freqs=freqs,
        fdd_spectrum=spectrum,
        method="base_ref",
        n_modes_found=n_found,
        success=True,
        message=f"Found {n_found} of {n_modes} requested modes.",
    )
    out.success = True
    bits = [f"{n_found} mode(s) from {n_seg} averaged segments "
            f"(resolution {df_true:.3f} Hz)"]
    if not out.gate_enforced:
        bits.append(
            f"coherence gate OFF — {df_true:.3f} Hz resolution is too coarse to "
            f"judge any peak in this band; record longer to enable it")
    elif not bool(np.all(resolvable)):
        floor_hz = df_true * GATE_RESOLUTION_RATIO
        bits.append(f"gate applies above {floor_hz:.2f} Hz only")
    if out.rejected:
        bits.append("rejected on coherence: "
                    + ", ".join(f"{f:.2f} Hz (γ²={c:.2f})" for f, c in out.rejected))
    out.message = " · ".join(bits)
    return out
