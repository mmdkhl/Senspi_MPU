# -*- coding: utf-8 -*-
"""Stage 1 of the continuous-update digital twin: a robust modal state estimator.

This is the **new** piece of the Mode-B redesign (Option B, two-stage). The old
continuous loop fed each single noisy identification window straight into the
model calibration and then tried to suppress the resulting jumpiness with a hard
gate that *stopped the loop* (diagnosis D1-D4 in
``.claude/model_updating_continuous_redesign.md``). The fix moves the accumulation
into **measurement space**, where the noise actually lives:

    per window:  raw peaks (f_j, phi_j)  ──▶  ModalStateTracker  ──▶  consolidated
                                                                       f_hat_i ± sigma_f_i
                                                                       phi_hat_i ± sigma_phi_i

The tracker keeps one running estimate per structural mode and folds each cycle's
raw peaks in with a **robust forgetting EWMA**, plus a small persistence-based
change detector so the two behaviours the digital twin must have can coexist:

- **A lone outlier barely moves the mean and raises sigma** -- a soft Cauchy /
  Student-t weight ``w = 1/(1+(r/c)^2)`` (floored at ``w_min``, never zero) folds a
  noisy in-gate reading with a tiny gain, and the running variance ``v`` (the
  observed scatter) inflates so the reported uncertainty goes up.
- **A persistent shift is tracked within a few cycles** -- a reading just outside
  the association gate is held as a *shift candidate*; when the same-direction
  candidate persists for ``shift_confirm`` cycles it is a real change (not noise),
  so the track re-locks onto it. A genuinely spurious peak (e.g. a 16 Hz noise bump
  when the mode is ~8 Hz, tens of sigma away) never becomes a candidate -- it is
  logged as unassociated and not folded.

- **Association** (fixes D2's mislabelling): each raw peak is matched to a tracked
  mode by gated nearest-neighbour in frequency (``|f - m| <= tau * sigma``), at
  most one peak per track. Tracks are *seeded from the lowest-frequency peaks* on
  the first valid cycle, because the structural modes are the low, strong ones
  (CU-3). Optionally a shape-MAC gate tightens association when >= 3 DOFs are
  measured (below that the MAC is degenerate -> frequency-NN only).

Guardrails: pure numpy, **no Qt, no OpenSees, no cross-package imports** (G7/G8).
A tiny local ``_mac`` / ``_align_sign`` keep this module a self-contained island so
it unit-tests without any optional dependency. Input: plain lists/arrays from the
identification result. Output: plain dataclasses -> the QThread worker turns them
into a calibrator ``exp_data`` dict and a Signal payload (G1/G4 stay in the worker).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# --- Default knobs (spec section 3.5; all tunable, all provisional per PRE-3) ---
DEFAULT_FORGETTING = 0.92      # lambda: responsiveness vs stability (gain base = 1-lambda)
DEFAULT_ROBUST_SCALE = 2.5     # c: innovation (in sigma) beyond which a reading is an outlier
DEFAULT_WEIGHT_FLOOR = 0.05    # w_min: outliers never fully discarded (still inform)
DEFAULT_ASSOC_GATE = 4.0       # tau: max sigma-distance to fold a peak into a track
DEFAULT_VAR_FLOOR_FRAC = 0.02  # variance floor = (frac * f)^2  (~2% freq); avoids over-confidence
# A reading between tau and shift_window (in sigma) is a "shift candidate"; beyond
# shift_window it is spurious. shift_confirm same-direction candidates -> re-lock.
# 25 sigma is wide because sigma floors at var_floor_frac (~2% f), so 25 sigma is
# ~50% of f at the floor -- enough to track a large but real structural change while
# a clearly-spurious peak (e.g. a 16 Hz bump on an 8 Hz mode, ~46 sigma) stays out.
DEFAULT_SHIFT_WINDOW = 25.0
DEFAULT_SHIFT_CONFIRM = 3
# Below this many measured DOFs the MAC is degenerate, so shape similarity is NOT
# used to gate association (frequency-NN only). Matches the project-wide caveat.
MAC_MIN_DOFS = 3
DEFAULT_MAC_GATE = 0.5         # reject a frequency-close peak whose shape MAC < this (>=3 DOFs only)


def _mac(a: np.ndarray, b: np.ndarray) -> float:
    """Modal Assurance Criterion, ``|a.b|^2 / ((a.a)(b.b))`` in ``[0, 1]``.

    Scale/sign-invariant. Returns 0.0 for empty / zero / length-mismatched inputs.
    Inlined (not imported) to keep this module dependency-free (G7/G8).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return 0.0
    denom = float(np.dot(a, a) * np.dot(b, b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) ** 2 / denom)


def _align_sign(incoming: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip ``incoming`` so it points the same way as ``reference`` (sign only)."""
    a = np.asarray(incoming, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.size == b.size and float(np.dot(a, b)) < 0.0:
        return -a
    return a


def _maxabs_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    m = float(np.max(np.abs(v))) if v.size else 0.0
    return v / m if m > 0.0 else v.copy()


@dataclass
class ReadingDiagnostic:
    """One raw peak's fate this cycle (for transparent logging, point 1 of the spec)."""

    freq: float
    track: int            # 0-based track index it folded into, or -1 if unassociated
    residual: float       # standardized innovation r (nan when seeded / candidate / unassoc)
    weight: float         # robust weight w applied (nan when not folded)
    note: str             # "seeded" | "folded" | "outlier" | "shift" | "shift-watch" | "unassociated"

    @property
    def folded(self) -> bool:
        """True when this reading actually moved a track's estimate."""
        return self.track >= 0 and self.note in ("folded", "outlier", "seeded", "shift")


@dataclass
class ConsolidatedModal:
    """The tracker's consolidated estimate after a cycle -> Stage-2 calibration input."""

    frequencies: list[float] = field(default_factory=list)     # f_hat per track, ascending
    freq_sigma: list[float] = field(default_factory=list)      # sigma_f per track (abs Hz)
    # Per-track consolidated shape (over coverage DOFs) + per-DOF sigma, when shapes are tracked.
    shapes: list[Optional[list[float]]] = field(default_factory=list)
    shape_sigma: list[Optional[list[float]]] = field(default_factory=list)
    coverage_stories: list[int] = field(default_factory=list)  # 1-based stories the shapes span
    n_obs: list[int] = field(default_factory=list)             # readings folded per track
    diagnostics: list[ReadingDiagnostic] = field(default_factory=list)
    shapes_available: bool = False

    @property
    def freq_sigma_rel(self) -> list[float]:
        """Relative per-mode sigma ``sigma_f / f_hat`` -> Stage-2 ``sigma_data_per_mode``."""
        return [
            (s / f) if f > 0 else float("nan")
            for f, s in zip(self.frequencies, self.freq_sigma)
        ]


class ModalStateTracker:
    """Running, robust, forgetting estimator of the structural modes (Stage 1).

    Maintain ``n_modes`` tracks. Each :meth:`update` folds one cycle's raw peaks in
    and returns the :class:`ConsolidatedModal` estimate. The loop **never** raises
    or rejects a whole cycle on disagreement: a bad reading down-weights itself, a
    peak that matches nothing is simply not folded, and a real shift re-locks.
    """

    def __init__(
        self,
        n_modes: int,
        *,
        forgetting: float = DEFAULT_FORGETTING,
        robust_scale: float = DEFAULT_ROBUST_SCALE,
        weight_floor: float = DEFAULT_WEIGHT_FLOOR,
        assoc_gate: float = DEFAULT_ASSOC_GATE,
        var_floor_frac: float = DEFAULT_VAR_FLOOR_FRAC,
        shift_window: float = DEFAULT_SHIFT_WINDOW,
        shift_confirm: int = DEFAULT_SHIFT_CONFIRM,
        track_shapes: bool = True,
        mac_gate: float = DEFAULT_MAC_GATE,
    ) -> None:
        self.n_modes = int(n_modes)
        self.lam = float(forgetting)
        self.c = float(robust_scale)
        self.w_min = float(weight_floor)
        self.tau = float(assoc_gate)
        self.var_floor_frac = float(var_floor_frac)
        self.shift_window = float(shift_window)
        self.shift_confirm = int(shift_confirm)
        self.track_shapes = bool(track_shapes)
        self.mac_gate = float(mac_gate)

        # Per-track state (parallel lists, ordered by ascending frequency).
        self._m: list[float] = []          # frequency means
        self._v: list[float] = []          # frequency variances (observed scatter)
        self._n: list[int] = []            # readings folded
        self._phi: list[Optional[np.ndarray]] = []      # shape means (maxabs-normalized)
        self._phi_v: list[Optional[np.ndarray]] = []    # per-DOF shape variances
        self._streak_sign: list[int] = []  # +/-1 direction of the current shift-candidate run
        self._streak_vals: list[list[float]] = []       # candidate freqs in the current run
        self._coverage: list[int] = []     # 1-based stories the shapes span
        self._seeded = False

    # -- helpers -----------------------------------------------------------------
    def _var_floor(self, freq: float) -> float:
        return float((self.var_floor_frac * max(freq, 1e-9)) ** 2)

    def _sigma_eff(self, i: int) -> float:
        """Reported / gating sigma: never below the variance floor (no over-confidence)."""
        return float(np.sqrt(max(self._v[i], self._var_floor(self._m[i]))))

    @property
    def initialized(self) -> bool:
        return self._seeded

    # -- the update --------------------------------------------------------------
    def update(
        self,
        frequencies: Sequence[float],
        shapes: Optional[Sequence[Optional[Sequence[float]]]] = None,
        coverage_stories: Optional[Sequence[int]] = None,
    ) -> ConsolidatedModal:
        """Fold one cycle's raw peaks into the running estimate.

        Parameters
        ----------
        frequencies : sequence of float
            Raw identified peak frequencies this cycle (any count; the strongest-N
            from the picker is fine -- the tracker keeps/associates the structural
            ones).
        shapes : sequence aligned with ``frequencies``, optional
            ``shapes[j]`` is peak ``j``'s mode-shape vector over ``coverage_stories``
            (or ``None``). Pass ``None`` to track frequency only.
        coverage_stories : sequence of int, optional
            1-based stories the shape vectors span (assumed stable across cycles for
            a fixed rig). Used to emit ``measured_dof_indices`` downstream.
        """
        finite = [j for j, f in enumerate(frequencies) if np.isfinite(f)]
        freqs = [float(frequencies[j]) for j in finite]
        if shapes is not None and self.track_shapes:
            shp = [self._as_shape(shapes[j] if j < len(shapes) else None) for j in finite]
        else:
            shp = None
        cov = [int(s) for s in coverage_stories] if coverage_stories else []

        diags: list[ReadingDiagnostic] = []

        if not self._seeded:
            self._seed(freqs, shp, cov, diags)
            self._seeded = True
            return self._consolidate(diags)

        if cov:
            self._coverage = cov

        self._associate_and_fold(freqs, shp, diags)
        return self._consolidate(diags)

    # -- internals ---------------------------------------------------------------
    @staticmethod
    def _as_shape(s) -> Optional[np.ndarray]:
        if s is None:
            return None
        v = np.asarray(s, dtype=float)
        return v if v.size else None

    def _seed(self, freqs, shp, cov, diags) -> None:
        """Seed tracks from the lowest-frequency peaks (CU-3: structural modes are low)."""
        order = list(np.argsort(freqs))     # ascending -> the lowest are the structural modes
        chosen = order[: self.n_modes]
        self._coverage = cov
        for idx in chosen:
            f = freqs[idx]
            self._m.append(f)
            self._v.append(self._var_floor(f))
            self._n.append(1)
            self._streak_sign.append(0)
            self._streak_vals.append([])
            if shp is not None and shp[idx] is not None:
                phi = _maxabs_normalize(shp[idx])
                self._phi.append(phi)
                self._phi_v.append(np.zeros_like(phi))
            else:
                self._phi.append(None)
                self._phi_v.append(None)
            diags.append(ReadingDiagnostic(freq=f, track=len(self._m) - 1,
                                           residual=float("nan"), weight=float("nan"),
                                           note="seeded"))
        for j, f in enumerate(freqs):
            if j not in chosen:
                diags.append(ReadingDiagnostic(freq=f, track=-1, residual=float("nan"),
                                               weight=float("nan"), note="unassociated"))
        self._reorder()

    def _associate_and_fold(self, freqs, shp, diags) -> None:
        n_tracks = len(self._m)
        n_peaks = len(freqs)
        use_mac = (
            shp is not None
            and self._coverage and len(self._coverage) >= MAC_MIN_DOFS
            and self.mac_gate > 0.0
        )

        # 1) Tight association within the gate (greedy NN, at most one peak / track).
        cands: list[tuple[float, int, int]] = []  # (distance, peak_j, track_i)
        for j in range(n_peaks):
            for i in range(n_tracks):
                d = abs(freqs[j] - self._m[i]) / self._sigma_eff(i)
                if d > self.tau:
                    continue
                if use_mac and shp[j] is not None and self._phi[i] is not None \
                        and self._phi[i].size == np.asarray(shp[j]).size:
                    if _mac(shp[j], self._phi[i]) < self.mac_gate:
                        continue  # frequency-close but shape disagrees -> not this track
                cands.append((d, j, i))
        cands.sort(key=lambda t: t[0])
        peak_taken = [False] * n_peaks
        track_taken = [False] * n_tracks
        assigned: dict[int, int] = {}
        for d, j, i in cands:
            if peak_taken[j] or track_taken[i]:
                continue
            peak_taken[j] = True
            track_taken[i] = True
            assigned[j] = i

        for j, i in assigned.items():
            r, w = self._fold(i, freqs[j], shp[j] if shp is not None else None)
            self._streak_sign[i] = 0           # a clean in-gate read clears any shift watch
            self._streak_vals[i] = []
            note = "outlier" if abs(r) > self.c else "folded"
            diags.append(ReadingDiagnostic(freq=freqs[j], track=i, residual=r, weight=w, note=note))

        # 2) Shift detection for tracks with no in-gate match: a same-direction
        #    candidate just outside the gate that PERSISTS is a real change.
        claimed = set(assigned)
        for i in range(n_tracks):
            if track_taken[i]:
                continue
            sigma = self._sigma_eff(i)
            best_j, best_d = None, None
            for j in range(n_peaks):
                if j in claimed or peak_taken[j]:
                    continue
                d = abs(freqs[j] - self._m[i]) / sigma
                if d <= self.shift_window and (best_d is None or d < best_d):
                    best_j, best_d = j, d
            if best_j is None:
                # Truly starved: uncertainty grows without fresh data; reset watch.
                self._v[i] = self._v[i] / max(self.lam, 1e-6)
                self._streak_sign[i] = 0
                self._streak_vals[i] = []
                continue
            claimed.add(best_j)
            peak_taken[best_j] = True
            self._shift_watch(i, freqs[best_j], shp[best_j] if shp is not None else None, diags)

        # 3) Anything still unclaimed is spurious / a new (untracked) mode.
        for j in range(n_peaks):
            if j not in claimed and not peak_taken[j]:
                diags.append(ReadingDiagnostic(freq=freqs[j], track=-1, residual=float("nan"),
                                               weight=float("nan"), note="unassociated"))

        self._reorder()

    def _shift_watch(self, i: int, x: float, shape, diags) -> None:
        """Hold an out-of-gate reading as a shift candidate; re-lock when it persists."""
        sign = 1 if x >= self._m[i] else -1
        if self._streak_sign[i] == sign:
            self._streak_vals[i].append(x)
        else:
            self._streak_sign[i] = sign
            self._streak_vals[i] = [x]
        # Something is persistently off -> uncertainty up while we decide.
        self._v[i] = self._v[i] / max(self.lam, 1e-6)

        if len(self._streak_vals[i]) >= self.shift_confirm:
            vals = np.asarray(self._streak_vals[i], dtype=float)
            new_m = float(np.mean(vals))
            scatter = float(np.var(vals)) if vals.size > 1 else 0.0
            self._m[i] = new_m                 # re-lock onto the confirmed new value
            self._v[i] = max(scatter, self._var_floor(new_m))
            self._n[i] += 1
            self._streak_sign[i] = 0
            self._streak_vals[i] = []
            # Shapes follow the frequency re-lock if the candidate carried one.
            if self.track_shapes and shape is not None and self._phi[i] is not None:
                phi_in = np.asarray(shape, dtype=float)
                if phi_in.size == self._phi[i].size:
                    self._phi[i] = _maxabs_normalize(_align_sign(phi_in, self._phi[i]))
            diags.append(ReadingDiagnostic(freq=x, track=i, residual=float("nan"),
                                           weight=float("nan"), note="shift"))
        else:
            diags.append(ReadingDiagnostic(freq=x, track=i, residual=float("nan"),
                                           weight=float("nan"), note="shift-watch"))

    def _fold(self, i: int, x: float, shape) -> tuple[float, float]:
        """Robust forgetting EWMA fold of reading ``x`` into track ``i``. Returns (r, w)."""
        sigma = self._sigma_eff(i)
        r = (x - self._m[i]) / sigma
        w = 1.0 / (1.0 + (r / self.c) ** 2)
        w = max(w, self.w_min)                 # NEVER zero -> an outlier still informs
        g = (1.0 - self.lam) * w               # gain = forgetting x robust weight
        m_old = self._m[i]
        self._m[i] = m_old + g * (x - m_old)   # robust EWMA mean
        # running variance == observed scatter (deviation from the OLD mean)
        self._v[i] = max((1.0 - g) * self._v[i] + g * (x - m_old) ** 2, 0.0)
        self._n[i] += 1

        if self.track_shapes and shape is not None and self._phi[i] is not None:
            phi_in = np.asarray(shape, dtype=float)
            if phi_in.size == self._phi[i].size:
                phi_in = _maxabs_normalize(_align_sign(phi_in, self._phi[i]))
                phi_old = self._phi[i].copy()
                self._phi[i] = phi_old + g * (phi_in - phi_old)
                self._phi_v[i] = (1.0 - g) * self._phi_v[i] + g * (phi_in - phi_old) ** 2
                self._phi[i] = _maxabs_normalize(self._phi[i])
        return r, w

    def _reorder(self) -> None:
        """Keep tracks ordered by ascending frequency (stable mode labels for display)."""
        if len(self._m) <= 1:
            return
        order = list(np.argsort(self._m))
        if order == list(range(len(self._m))):
            return
        self._m = [self._m[k] for k in order]
        self._v = [self._v[k] for k in order]
        self._n = [self._n[k] for k in order]
        self._phi = [self._phi[k] for k in order]
        self._phi_v = [self._phi_v[k] for k in order]
        self._streak_sign = [self._streak_sign[k] for k in order]
        self._streak_vals = [self._streak_vals[k] for k in order]

    def _consolidate(self, diags) -> ConsolidatedModal:
        shapes_avail = any(p is not None for p in self._phi)
        out = ConsolidatedModal(
            frequencies=[float(m) for m in self._m],
            freq_sigma=[self._sigma_eff(i) for i in range(len(self._m))],
            coverage_stories=list(self._coverage),
            n_obs=list(self._n),
            diagnostics=diags,
            shapes_available=bool(shapes_avail and self._coverage),
        )
        for i in range(len(self._m)):
            if self._phi[i] is not None:
                out.shapes.append([float(v) for v in self._phi[i]])
                out.shape_sigma.append([float(np.sqrt(max(v, 0.0))) for v in self._phi_v[i]])
            else:
                out.shapes.append(None)
                out.shape_sigma.append(None)
        return out
