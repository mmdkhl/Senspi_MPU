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


def _layout_of(cfg):
    """The placement as analysis inputs, recomputed only when the map changes."""
    from ...analysis import sensor_layout as slayout
    return slayout.layout_from_mapping(getattr(cfg, "sensor_map", None),
                                       requested_modes=int(getattr(cfg, "n_modes", 3)))


def _row_geometry(layout, sensor_ids):
    """``(floor_of, pan_of)`` for the rows actually being rendered.

    ``pan_of`` turns the plan column into a stereo position — A left, B centre,
    C right — so four sensors produce an image of the rig rather than four
    voices in arbitrary places. Rows with no placement fall back to centre.
    """
    floors, pans = [], []
    for sid in sensor_ids:
        sid = int(sid)
        floors.append(int(layout.story_map.get(sid, 0)))
        cell = str(layout.cell_map.get(sid, "B2"))
        col = "ABC".find(cell[:1].upper())
        pans.append(0.0 if col < 0 else (col - 1) * 0.85)
    return tuple(floors), tuple(pans)


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
        self._cross_hist: deque = deque(maxlen=400)  # other horizontal axis, per mode
        self._vert_hist: deque = deque(maxlen=400)   # az
        self._rock_hist: deque = deque(maxlen=400)   # gx/gy per row
        self._n_modes = 0

    def reset_modes(self, n_modes: int) -> None:
        if n_modes != self._n_modes:
            self._n_modes = n_modes
            self._be_hist.clear()

    @staticmethod
    def _push(hist: deque, value):
        """Append to a rolling history, dropping it first if the width changed.

        Every history here is per-mode or per-sensor wide, and both counts move
        at runtime: identification returns two modes instead of three when a peak
        drops below the prominence threshold, and a sensor can stop streaming.
        A deque holding rows of two different widths cannot be stacked —
        ``np.asarray(list(hist), dtype=float)`` raises on the inhomogeneous
        shape, which killed the chorus worker mid-sound.

        Clearing is the right response, not padding: these histories exist to
        supply a rolling percentile, and a percentile taken across a column that
        meant "mode 3" before and "nothing" after is not a statistic. A cleared
        history simply returns 0 until it refills, which is what the ``< 6``
        guard in :meth:`_normalise` already does at start-up.
        """
        v = np.atleast_1d(np.asarray(value, dtype=float))
        if hist and np.shape(hist[-1]) != v.shape:
            hist.clear()
        hist.append(v)
        return v

    @staticmethod
    def _normalise(hist: deque, value: np.ndarray | float):
        """Rolling percentile normalisation (10th -> 0, 95th -> 1).

        Defensive about ragged content as well as :meth:`_push`: this runs inside
        the audio worker, where an exception is silence.
        """
        zero = np.zeros_like(np.atleast_1d(np.asarray(value, dtype=float)), dtype=float)
        rows = list(hist)
        if len(rows) < 6:
            return zero
        width = np.shape(rows[-1])
        if any(np.shape(r) != width for r in rows):
            return zero
        arr = np.asarray(rows, dtype=float)
        lo = np.percentile(arr, 10, axis=0)
        hi = np.percentile(arr, 95, axis=0)
        span = np.maximum(hi - lo, 1e-12)
        return np.clip((np.atleast_1d(value) - lo) / span, 0.0, 1.0)

    def update(self, ax_snapshot, gz_snapshot, modal: ModalState,
               t: float, rate_hz: float = 0.0, channels: dict | None = None) -> ControlFrame:
        """One control frame from the driven axis, the gyro yaw, and — when
        ``channels`` carries them — the other horizontal axis, ``az`` and
        ``gx``/``gy`` (see :meth:`_extra_channels`)."""
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

        # Which rows are responses and which is the shaker, from the placement.
        ids = getattr(ax_snapshot, "sensor_ids", None)
        if ids is None:
            ids = list(range(1, data.shape[0] + 1))
        ids = list(ids)[:data.shape[0]]
        layout = _layout_of(cfg)
        resp_rows = list(range(data.shape[0]))
        base_row = -1
        if layout.is_valid and ids:
            try:
                rr = list(layout.response_rows(ids))
            except Exception:
                rr = []
            if rr and len(rr) < len(ids):
                resp_rows = rr
                base_row = next(i for i in range(len(ids)) if i not in rr)

        # excitation tracker. With a base sensor the INPUT itself says what
        # frequency the table is driving at, which is exact; without one the
        # upper floors' sway is the best proxy.
        if base_row >= 0:
            ref = data[base_row]
        else:
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
                # A stated drive frequency wins over the tracker. Without a base
                # sensor the tracker reads the upper floors, which is a proxy for
                # the input rather than the input, and on a shake table the
                # operator knows the real number.
                stated = float(getattr(cfg, "exc_freq_hz_override", 0.0) or 0.0)
                frame.exc_freq_hz = stated if stated > 0.0 else float(fb[k])
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
                self._push(self._be_hist, raw_be)
                frame.band_energy = np.asarray(
                    self._normalise(self._be_hist, raw_be), dtype=float).ravel()[:n_modes]

        # ---------------- TORSION (per floor, from the gyro) ---------------
        # gz is a direct yaw rate, so each sensor already measures its own
        # floor's rotation — no differencing and no lever arm needed. This used
        # to average every gyro into ONE number, which is the one operation that
        # throws away where the twisting is: a top floor twisting hard and a
        # still ground floor average to "a bit of twist everywhere".
        gz, gfs = _as_array(gz_snapshot)
        if gz.size:
            gz = np.nan_to_num(gz, nan=0.0, posinf=0.0, neginf=0.0)
            gz = gz - gz.mean(axis=1, keepdims=True)
            raw_rows = np.sqrt(np.mean(gz ** 2, axis=1))
            self._push(self._tors_hist, raw_rows)
            rel_rows = np.asarray(
                self._normalise(self._tors_hist, raw_rows), dtype=float).ravel()
            rows = list(self._tors_hist)
            med_rows = (np.median(np.asarray(rows), axis=0)
                        if len(rows) > 8 and np.shape(rows[-1]) == raw_rows.shape
                        else np.zeros_like(raw_rows))
            # Must be genuinely above its own typical level, not merely in the
            # top percentile of an otherwise quiet gyro. A gyro that has been
            # perfectly still makes ANY rotation meaningful, so an all-zero
            # history must not gate the signal away entirely.
            quiet = med_rows <= 1e-9
            # Soft gate, not a cliff. The old hard ">1.8x median" test made
            # torsion a SURPRISE detector: a floor twisting steadily through a
            # resonance sweep sat just under the threshold and stayed silent,
            # which is the one moment it most needs to be heard. Ramping from
            # 1.0x to 1.8x keeps "above its own typical level" as the meaning
            # while letting a sustained twist sing.
            ratio = raw_rows / np.maximum(med_rows, 1e-12)
            gate = np.clip((ratio - 1.0) / 0.8, 0.0, 1.0)
            gate = np.where(quiet, (raw_rows > 1e-9).astype(float), gate)
            frame.torsion_floor = rel_rows * gate
            frame.torsion = float(np.max(frame.torsion_floor)) \
                if frame.torsion_floor.size else 0.0

            # Pan follows the LIVE direction of rotation, so the twist audibly
            # swings left and right in time with the structure. The old value
            # averaged a detrended window, which is ~0 by construction — the
            # torsion voice was pinned to the centre and never moved.
            lead = int(np.argmax(raw_rows)) if raw_rows.size else 0
            g = gz[lead]
            scale = float(raw_rows[lead]) * 2.5 + 1e-12
            frame.torsion_pan = float(np.tanh(float(g[-1]) / scale))

        # ---------------- IMPACT (E7) --------------------------------------
        # a knock shows up as a floor envelope jumping well above its own recent
        # median. Edge-triggered so one knock is one event, not a plateau.
        # Measured on a SHORT sub-window (last ~0.3 s), not the 6 s analysis
        # window. A knock smeared across 6 s of RMS barely moves the number and
        # is gone before the next tick; against the slow window it stands out.
        short_n = max(8, int(0.3 * fs))
        env_short = np.sqrt(np.mean(data[:, -short_n:] ** 2, axis=1))
        self._push(self._floor_hist, frame.env_floor)
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
            self._push(self._drift_hist, raw)
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
                self._push(self._appr_hist, raw)
                frame.approach = np.asarray(
                    self._normalise(self._appr_hist, raw), dtype=float).ravel()
            self._prox_prev = prox.copy()

        # instantaneous per-sensor motion for the animated structure figure
        frame.motion = data[:, -1].astype(float) if data.shape[1] else np.zeros(data.shape[0])

        # Where each row sits, so the renderer can place its voices like the rig.
        frame.floor_of, frame.pan_of = _row_geometry(layout, ids)

        # The other channels of the same sensors.
        self._extra_channels(frame, channels, data, fs, freqs, resp_rows)
        return frame

    def _extra_channels(self, frame: ControlFrame, channels: dict | None,
                        data: np.ndarray, fs: float, freqs: np.ndarray,
                        resp_rows: list) -> None:
        """Read the rest of the sensor into the frame.

        * the **other horizontal axis** → ``cross_energy`` per mode and
          ``cross_ratio``: how much the structure moves across the driven
          direction (a 2-D rig has cross modes; a 1-D one has almost none)
        * ``az`` → ``vertical``: vertical bounce / table pumping
        * ``gx``/``gy`` → ``rock_floor`` per row: a floor tilting, which the
          driven-axis difference between floors cannot separate from the
          mode shape itself

        Every value is a rolling-percentile rank (0..1), like the driven-axis
        features, so an unstreamed channel (all zeros) reads as 0 rather than
        as noise. Rows follow the driven snapshot's row order; the shaker row
        is excluded from the cross and vertical averages because it measures
        the input, not the response.
        """
        chan = channels or {}
        seen: list = []
        cfg = self.cfg
        n_rows = data.shape[0]
        rows = [r for r in resp_rows if r < n_rows] or list(range(n_rows))
        main_rms = float(np.sqrt(np.mean(data[rows] ** 2))) if rows else 0.0

        def prep(key: str):
            arr, cfs = _as_array(chan.get(key))
            if arr.size == 0 or not np.isfinite(cfs) or cfs <= 1.0:
                return None
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = arr - arr.mean(axis=1, keepdims=True)
            if not np.any(np.abs(arr) > 1e-12):
                return None                      # not streamed
            seen.append(key)
            return arr

        # --- cross axis -----------------------------------------------------
        other = "ay" if str(cfg.axis).lower() == "ax" else "ax"
        cr = prep(other)
        if cr is not None and freqs.size:
            crows = [r for r in rows if r < cr.shape[0]] or list(range(cr.shape[0]))
            ref = cr[crows].mean(axis=0)
            if ref.size >= 32:
                win = np.hanning(ref.size)
                spec = np.abs(np.fft.rfft(ref * win, n=_FFT_N))
                fx = np.fft.rfftfreq(_FFT_N, 1.0 / fs)
                raw = np.array([_band_peak(spec, fx, float(fm)) for fm in freqs])
                self._push(self._cross_hist, raw)
                frame.cross_energy = np.asarray(
                    self._normalise(self._cross_hist, raw), dtype=float).ravel()
            cross_rms = float(np.sqrt(np.mean(cr[crows] ** 2)))
            frame.cross_ratio = float(np.clip(cross_rms / (main_rms + cross_rms + 1e-12)
                                              * 2.0, 0.0, 1.0))

        # --- vertical -------------------------------------------------------
        vz = prep("az")
        if vz is not None:
            vrows = [r for r in rows if r < vz.shape[0]] or list(range(vz.shape[0]))
            raw_v = float(np.sqrt(np.mean(vz[vrows] ** 2)))
            self._vert_hist.append(raw_v)
            frame.vertical = float(np.asarray(
                self._normalise(self._vert_hist, raw_v), dtype=float).ravel()[0])

        # --- rocking (gx, gy) -----------------------------------------------
        gx, gy = prep("gx"), prep("gy")
        if gx is not None or gy is not None:
            parts = [g ** 2 for g in (gx, gy) if g is not None]
            m = min(p.shape[0] for p in parts)
            n = min(p.shape[1] for p in parts)
            mag = np.sqrt(sum(p[:m, :n] for p in parts))
            raw_r = np.sqrt(np.mean(mag ** 2, axis=1))
            self._push(self._rock_hist, raw_r)
            rel = np.asarray(self._normalise(self._rock_hist, raw_r), dtype=float).ravel()
            hist = list(self._rock_hist)
            med = (np.median(np.asarray(hist), axis=0)
                   if len(hist) > 8 and np.shape(hist[-1]) == raw_r.shape
                   else np.zeros_like(raw_r))
            ratio = raw_r / np.maximum(med, 1e-12)
            gate = np.clip((ratio - 1.0) / 0.8, 0.0, 1.0)
            gate = np.where(med <= 1e-9, (raw_r > 1e-9).astype(float), gate)
            frame.rock_floor = rel * gate

        frame.channels = tuple(seen)


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

        # The floor-0 sensor measures the shaker INPUT. Feeding it in as if it
        # were a response biases the picked modes toward the excitation — the
        # same correction that moved a real mode by ~2 % in the Spectrum tab.
        # Modes are also capped by how many measurement points there are: you
        # cannot identify more modes than sensors, and asking for more just
        # promotes noise peaks into animals that sing about nothing.
        layout = _layout_of(cfg)
        n_modes = int(cfg.n_modes)
        if layout.is_valid:
            ids = list(getattr(snapshot, "sensor_ids", []) or [])
            if ids:
                rows = layout.response_rows(ids)
                if rows and len(rows) < len(ids):
                    data = data[rows, :]
            n_modes = layout.max_modes(n_modes)

        try:
            from ...analysis.modal import identify_modes
            res = identify_modes(
                np.nan_to_num(data), fs,
                f_min=cfg.f_min, f_max=min(cfg.f_max, fs / 2 - 0.5),
                n_modes=n_modes,
            )
        except Exception as exc:                       # keep the sound alive
            logger.debug("chorus: identification failed: %s", exc)
            self._state.message = f"identification failed: {exc}"
            return self._state
        if not getattr(res, "success", False):
            self._state.message = getattr(res, "message", "no modes found")
            return self._state

        freqs = np.asarray(res.frequencies_hz, dtype=float).ravel()
        # Half-power ζ can be NaN (the base-referenced path reports it so, and
        # the DEBT-8 fix will make FDD/FFT do the same). The ring-down is an
        # aesthetic control, so a typical structural value stands in.
        damp = np.nan_to_num(np.asarray(res.damping_ratios, dtype=float).ravel(), nan=0.02)
        shapes = np.asarray(res.mode_shapes_sensor, dtype=float)
        if shapes.ndim == 2 and shapes.shape[0] != data.shape[0]:
            shapes = shapes.T
        self._history.append((freqs, damp))
        # Median per mode INDEX, over the history entries that actually have
        # that index, with the latest cycle setting how many modes to report.
        #
        # This used to be min(len(f)) across the history, which took the
        # FEWEST modes any recent cycle found. A single quiet cycle that
        # resolved only mode 1 then silenced modes 2 and 3 for the next three
        # cycles, while the Spectrum tab went on showing all three. On a rig
        # that is shaken in bursts, that is most of the time.
        target = int(freqs.size)
        med_f, med_d = [], []
        for i in range(target):
            fv = [f[i] for f, _ in self._history if len(f) > i]
            dv = [d[i] for _, d in self._history if len(d) > i]
            med_f.append(float(np.median(fv)) if fv else float(freqs[i]))
            med_d.append(float(np.median(dv)) if dv else float(damp[i]))
        self._state = ModalState(
            frequencies_hz=np.asarray(med_f, dtype=float),
            damping=np.asarray(med_d, dtype=float),
            shapes=shapes if shapes.ndim == 2 else None,
            fs=float(fs), t_identified=float(t), ok=True,
            message=getattr(res, "message", "ok"),
        )
        return self._state
