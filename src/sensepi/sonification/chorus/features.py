"""Control-rate feature extraction and modal tracking.

Pure numpy. Consumes ``ModalSession``-shaped snapshots (``.data`` (n_sensors,
n_samples) and ``.fs``) so live and recorded data take an identical path.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

from .types import CONTROL_HZ, ChorusConfig, ControlFrame, ModalState

logger = logging.getLogger(__name__)

_FFT_N = 4096


def _band_peak(spectrum: np.ndarray, freqs: np.ndarray, f_mode: float) -> float:
    """Peak magnitude in a band around ``f_mode``, width scaled to the mode."""
    half = float(np.clip(f_mode * 0.35, 0.12, 0.6))
    sel = np.abs(freqs - f_mode) <= half
    if not np.any(sel):
        k = int(np.argmin(np.abs(freqs - f_mode)))
        return float(spectrum[k])
    return float(spectrum[sel].max())


def _as_array(snapshot) -> tuple[np.ndarray, float]:
    """Accept a ModalSession, a bare array, or None."""
    if snapshot is None:
        return np.empty((0, 0)), float("nan")
    data = getattr(snapshot, "data", snapshot)
    fs = float(getattr(snapshot, "fs", float("nan")))
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr, fs


class FeatureExtractor:
    """Turns short data snapshots into :class:`ControlFrame` observations."""

    def __init__(self, cfg: ChorusConfig) -> None:
        self.cfg = cfg
        self._be_hist: deque = deque(maxlen=400)
        self._tors_hist: deque = deque(maxlen=400)
        self._env_hist: deque = deque(maxlen=400)
        self._floor_hist: deque = deque(maxlen=200)
        self._drift_hist: deque = deque(maxlen=400)
        self._prox_prev: np.ndarray | None = None
        self._appr_hist: deque = deque(maxlen=400)
        self._impact_t = -1e9        # last impact, for the refractory window
        self._prev_ratio = 1.0       # a knock is a RISING edge, not a plateau
        self._t_prev = None
        self._rate_hist: deque = deque(maxlen=200)
        self._beat_hist: deque = deque(maxlen=600)   # 30 s of short-term envelope
        self._exc_hist: deque = deque(maxlen=80)     # 4 s of excitation frequency
        self._n_modes = 0

    def reset_modes(self, n_modes: int) -> None:
        if n_modes != self._n_modes:
            self._n_modes = n_modes
            self._be_hist.clear()

    @staticmethod
    def _normalise(hist: deque, value: np.ndarray | float):
        """Rolling percentile normalisation (10th -> 0, 95th -> 1)."""
        arr = np.asarray(list(hist), dtype=float)
        if arr.shape[0] < 6:
            return np.zeros_like(np.atleast_1d(value), dtype=float)
        lo = np.percentile(arr, 10, axis=0)
        hi = np.percentile(arr, 95, axis=0)
        span = np.maximum(hi - lo, 1e-12)
        return np.clip((np.atleast_1d(value) - lo) / span, 0.0, 1.0)

    def update(self, ax_snapshot, gz_snapshot, modal: ModalState,
               t: float, rate_hz: float = 0.0) -> ControlFrame:
        cfg = self.cfg
        frame = ControlFrame(t=float(t), rate_hz=float(rate_hz))
        data, fs = _as_array(ax_snapshot)
        freqs = np.asarray(modal.frequencies_hz, dtype=float).ravel()
        n_modes = max(1, freqs.size)
        self.reset_modes(n_modes)
        frame.band_energy = np.zeros(n_modes)
        frame.sync = np.zeros(n_modes)

        if data.size == 0 or not np.isfinite(fs) or fs <= 1.0:
            return frame

        finite = np.isfinite(data)
        frame.nan_ratio = float(1.0 - finite.mean()) if data.size else 0.0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = data - data.mean(axis=1, keepdims=True)

        frame.env_floor = np.sqrt(np.mean(data ** 2, axis=1))
        frame.env_global = float(np.sqrt(np.mean(data ** 2)))
        self._env_hist.append(frame.env_global)

        # excitation tracker: upper floors carry the sway
        ref = data[1:].mean(axis=0) if data.shape[0] > 1 else data[0]
        if ref.size >= 32:
            win = np.hanning(ref.size)
            spec = np.abs(np.fft.rfft(ref * win, n=_FFT_N))
            fx = np.fft.rfftfreq(_FFT_N, 1.0 / fs)
            # reach down to cfg.f_min: a mode-1 at 0.4 Hz is normal for a
            # building and an 0.8 Hz floor would simply never see it
            f_bot = max(0.15, min(cfg.f_min, 1.0))
            band = (fx >= f_bot) & (fx <= min(cfg.f_max, fs / 2 - 0.5))
            if band.sum() >= 4:
                sb, fb = spec[band], fx[band]
                k = int(np.argmax(sb))
                frame.exc_freq_hz = float(fb[k])
                self._exc_hist.append(frame.exc_freq_hz)
                med = float(np.median(sb)) + 1e-12
                frame.exc_conf = float(min(1.0, (sb[k] / med) / 20.0))
                frame.psd_freqs = fb
                frame.psd = sb
                # Band half-width must scale with the mode: a fixed +/-0.6 Hz
                # around a 0.8 Hz mode reaches into DC and swamps the estimate.
                raw_be = np.array([
                    _band_peak(sb, fb, float(fm)) for fm in freqs
                ]) if freqs.size else np.zeros(n_modes)
                self._be_hist.append(raw_be)
                frame.band_energy = np.asarray(
                    self._normalise(self._be_hist, raw_be), dtype=float).ravel()[:n_modes]

        # torsion from the gyro channel
        gz, gfs = _as_array(gz_snapshot)
        if gz.size:
            gz = np.nan_to_num(gz, nan=0.0, posinf=0.0, neginf=0.0)
            gz = gz - gz.mean(axis=1, keepdims=True)
            g = gz.mean(axis=0)
            raw_t = float(np.sqrt(np.mean(g ** 2)))
            self._tors_hist.append(raw_t)
            rel = float(np.asarray(self._normalise(self._tors_hist, raw_t)).ravel()[0])
            med = float(np.median(list(self._tors_hist))) if len(self._tors_hist) > 8 else 0.0
            # must be genuinely above its own typical level, not merely in the
            # top percentile of an otherwise quiet gyro
            # a gyro that has been perfectly still makes ANY rotation meaningful,
            # so an all-zero history must not gate the signal away entirely
            quiet_before = med <= 1e-9
            frame.torsion = rel if (raw_t > 1.8 * med or (quiet_before and raw_t > 1e-9)) else 0.0
            frame.torsion_pan = float(np.tanh(np.mean(g) * 3.0))

        # ---------------- IMPACT (E7) --------------------------------------
        # a knock shows up as a floor envelope jumping well above its own recent
        # median. Edge-triggered so one knock is one event, not a plateau.
        # Measured on a SHORT sub-window (last ~0.3 s), not the 6 s analysis
        # window. A knock smeared across 6 s of RMS barely moves the number and
        # is gone before the next tick; against the slow window it stands out.
        short_n = max(8, int(0.3 * fs))
        env_short = np.sqrt(np.mean(data[:, -short_n:] ** 2, axis=1))
        self._floor_hist.append(frame.env_floor.copy())
        if len(self._floor_hist) >= 12:
            hist = np.asarray(list(self._floor_hist))
            med = np.median(hist[:-1], axis=0) + 1e-12
            ratio = env_short / med
            k = int(np.argmax(ratio))
            # REFRACTORY, not "wait until quiet". Requiring the structure to
            # settle before re-arming meant one early trigger during a sweep
            # locked out every later knock for the rest of the run.
            rising = ratio[k] > self._prev_ratio * 1.05
            self._prev_ratio = float(ratio[k])
            if ratio[k] > 2.2 and rising and (t - self._impact_t) > 1.2:
                frame.impact = float(np.clip((ratio[k] - 2.2) / 3.0, 0.05, 1.0))
                frame.impact_floor = k
                self._impact_t = float(t)

        # ---------------- BEATING (E3) -------------------------------------
        # Two close frequencies amplitude-modulate the response at |f1-f2|.
        # Measured from a long history of SHORT-TERM envelope: the 6 s window's
        # own RMS averages the modulation away, and a slow 0.1 Hz beat needs far
        # more than one window to be visible at all.
        tail = ref[-max(8, int(0.25 * fs)):]
        self._beat_hist.append(float(np.sqrt(np.mean(tail ** 2))))
        if len(self._beat_hist) >= 128:
            env = np.asarray(list(self._beat_hist), dtype=float)
            env = env - env.mean()
            if float(np.std(env)) > 1e-12:
                spec = np.abs(np.fft.rfft(env * np.hanning(env.size)))
                fb = np.fft.rfftfreq(env.size, 1.0 / CONTROL_HZ)
                sel = (fb >= 0.05) & (fb <= 2.0)
                if sel.sum() > 3:
                    sb = spec[sel]
                    k = int(np.argmax(sb))
                    strength = float(sb[k] / (np.median(sb) + 1e-12))
                    # Only call it beating if the EXCITATION IS STEADY. A sweep
                    # continuously changes the response amplitude, which looks
                    # exactly like modulation and otherwise fires almost
                    # constantly (773 of 900 frames before this gate).
                    steady = (len(self._exc_hist) >= 40
                              and float(np.std(list(self._exc_hist)[-40:])) < 0.25)
                    if strength > 7.0 and steady:
                        frame.beating = float(np.clip((strength - 7.0) / 10.0, 0, 1))
                        frame.beat_hz = float(fb[sel][k])

        # ---------------- INTER-STOREY DRIFT (E6) --------------------------
        # adjacent floors moving against each other, normalised so it reports
        # CONCENTRATION of relative motion rather than absolute amplitude.
        if data.shape[0] >= 2:
            pairs = []
            for i in range(data.shape[0] - 1):
                rel = float(np.sqrt(np.mean((data[i] - data[i + 1]) ** 2)))
                tot = float(np.sqrt(np.mean(data[i] ** 2)) + np.sqrt(np.mean(data[i + 1] ** 2)))
                pairs.append(rel / (tot + 1e-12))
            raw = np.asarray(pairs)
            self._drift_hist.append(raw)
            rel = np.asarray(self._normalise(self._drift_hist, raw), dtype=float).ravel()
            # A percentile rank alone says "more than usual", which is true most
            # of the time during a sweep. Require real ABSOLUTE separation too,
            # so drift means the floors genuinely move apart.
            frame.drift = rel * (raw >= 0.60).astype(float)

        # ---------------- DATA DROPOUT (E9) --------------------------------
        sev = 0.0
        if frame.nan_ratio > 0.02:
            sev = max(sev, float(min(1.0, frame.nan_ratio / 0.2)))
        if rate_hz > 0:
            self._rate_hist.append(float(rate_hz))
            if len(self._rate_hist) >= 20:
                norm = float(np.median(list(self._rate_hist)))
                if norm > 0 and rate_hz < 0.8 * norm:
                    sev = max(sev, float(min(1.0, (0.8 - rate_hz / norm) / 0.5)))
        # a sensor that has gone quiet relative to its peers is effectively dead
        dead = []
        if frame.env_floor.size >= 2:
            peer = float(np.median(frame.env_floor))
            if peer > 1e-9:
                dead = [i for i, e in enumerate(frame.env_floor) if e < 0.05 * peer]
                if dead:
                    sev = max(sev, 0.6)
        frame.dropout = float(np.clip(sev, 0.0, 1.0))
        frame.dead_sensors = tuple(dead)

        # per-mode synchrony: is the excitation locked onto this mode?
        if freqs.size and frame.exc_freq_hz > 0:
            prox = np.clip(1.0 - np.abs(frame.exc_freq_hz / np.maximum(freqs, 1e-9) - 1.0)
                           / 0.18, 0.0, 1.0)
            be = frame.band_energy
            if be.size < freqs.size:
                be = np.pad(be, (0, freqs.size - be.size))
            frame.sync = np.clip(prox * frame.exc_conf * (0.35 + 0.65 * be), 0.0, 1.0) \
                * cfg.sync_strength

            # ---------- RESONANCE APPROACH (E1) ----------------------------
            # Driven by the RATE OF CHANGE of proximity, not proximity itself.
            # Parking on a mode is a lock, not an approach; only *closing in*
            # should build tension.
            if self._prox_prev is not None and self._prox_prev.size == prox.size:
                d = np.maximum(0.0, prox - self._prox_prev) * CONTROL_HZ
                raw = d * prox
                self._appr_hist.append(raw)
                frame.approach = np.asarray(
                    self._normalise(self._appr_hist, raw), dtype=float).ravel()
            self._prox_prev = prox.copy()

        # instantaneous per-sensor motion for the animated structure figure
        frame.motion = data[:, -1].astype(float) if data.shape[1] else np.zeros(data.shape[0])
        return frame


class ModalTracker:
    """Re-identifies modes on a slow cycle and median-smooths the result."""

    def __init__(self, cfg: ChorusConfig) -> None:
        self.cfg = cfg
        self._history: deque = deque(maxlen=3)
        self._state = ModalState(ok=False, message="listening — identifying modes…")
        self._last_t = -1e9

    @property
    def state(self) -> ModalState:
        return self._state

    def wants_reid(self, t: float) -> bool:
        return (t - self._last_t) >= self.cfg.reid_interval_s

    def reidentify(self, snapshot, t: float) -> ModalState:
        """Run identification on a long snapshot. Never raises."""
        self._last_t = float(t)
        cfg = self.cfg
        data, fs = _as_array(snapshot)
        if data.size == 0 or not np.isfinite(fs) or fs <= 2.0:
            self._state.message = "waiting for data…"
            return self._state
        try:
            from ...analysis.modal import identify_modes
            res = identify_modes(
                np.nan_to_num(data), fs,
                f_min=cfg.f_min, f_max=min(cfg.f_max, fs / 2 - 0.5),
                n_modes=cfg.n_modes,
            )
        except Exception as exc:                       # keep the sound alive
            logger.debug("chorus: identification failed: %s", exc)
            self._state.message = f"identification failed: {exc}"
            return self._state
        if not getattr(res, "success", False):
            self._state.message = getattr(res, "message", "no modes found")
            return self._state

        freqs = np.asarray(res.frequencies_hz, dtype=float).ravel()
        damp = np.asarray(res.damping_ratios, dtype=float).ravel()
        shapes = np.asarray(res.mode_shapes_sensor, dtype=float)
        if shapes.ndim == 2 and shapes.shape[0] != data.shape[0]:
            shapes = shapes.T
        self._history.append((freqs, damp))
        n = min(len(f) for f, _ in self._history)
        fs_stack = np.vstack([f[:n] for f, _ in self._history])
        dp_stack = np.vstack([d[:n] for _, d in self._history])
        self._state = ModalState(
            frequencies_hz=np.median(fs_stack, axis=0),
            damping=np.median(dp_stack, axis=0),
            shapes=shapes if shapes.ndim == 2 else None,
            fs=float(fs), t_identified=float(t), ok=True,
            message=getattr(res, "message", "ok"),
        )
        return self._state
