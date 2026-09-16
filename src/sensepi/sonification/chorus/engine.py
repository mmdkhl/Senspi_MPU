"""Qt-free facade wiring catalog + features + renderer.

Also provides :func:`run_offline`, which steps the identical tick/render path
over a recorded session faster than real time. Tests and dev listening use it;
the UI does not.
"""
from __future__ import annotations

import logging

import numpy as np

from .catalog import (cast_meadow, cast_signature, catalog_available,
                      fit_casting_map, load_grain_banks)
from .features import FeatureExtractor, ModalTracker
from .renderer import ChorusRenderer
from .types import (BLOCK_SIZE, CONTROL_HZ, SAMPLE_RATE, ChorusConfig,
                    ControlFrame, ModalState, VizFrame)

logger = logging.getLogger(__name__)


class ChorusEngine:
    """The Bioacoustic Chorus model, independent of Qt and of any audio device."""

    def __init__(self, cfg: ChorusConfig | None = None, seed: int = 23) -> None:
        self.cfg = (cfg or ChorusConfig()).clamped()
        self.features = FeatureExtractor(self.cfg)
        self.tracker = ModalTracker(self.cfg)
        self.renderer = ChorusRenderer(self.cfg, seed=seed)
        self._banks = load_grain_banks()
        self._cast: list = []
        self._cast_sig: tuple = ()
        self._frame = ControlFrame()
        self._t = 0.0
        self._recast_pending = False
        self._map_fitted = False
        self.available = catalog_available()

    # ------------------------------------------------------------------ config
    def set_config(self, cfg: ChorusConfig) -> None:
        self.cfg = cfg.clamped()
        self.features.cfg = self.cfg
        self.tracker.cfg = self.cfg
        self.renderer.set_config(self.cfg)
        self._recast(force=True)

    def set_option(self, key: str, value) -> None:
        """Live tweak by attribute name. Unknown keys are ignored."""
        if not hasattr(self.cfg, key):
            logger.debug("chorus: unknown option %r", key)
            return
        try:
            setattr(self.cfg, key, type(getattr(self.cfg, key))(value))
        except Exception:
            setattr(self.cfg, key, value)
        self.cfg = self.cfg.clamped()
        self.features.cfg = self.cfg
        self.tracker.cfg = self.cfg
        self.renderer.set_config(self.cfg)
        if key in ("f_lo", "f_hi"):
            self.cfg.autofit = False        # the user is driving the map now
        if key == "type_of_mode":
            # arrives as a list from the tab; keep the config hashable/stable
            self.cfg.type_of_mode = tuple(str(t) for t in (value or ()))
        if key in ("sensor_map", "axis", "n_modes"):
            # The placement decides which rows are responses and how many modes
            # may be claimed: re-identify on the next tick rather than singing
            # from the old identification for up to reid_interval_s.
            self.tracker._last_t = -1e9
        if key in ("f_lo", "f_hi", "n_chorus", "n_ambient",
                   "enabled_roles", "chorus_size", "pitch_rise", "depth",
                   "type_of_mode", "resonance_type", "torsion_type",
                   "alarm_type", "drift_type", "ambient_type"):
            # Do NOT recast here. A slider emits on every pixel of drag, and a
            # recast rebuilds every voice's pitch-variant banks (resampling every
            # grain). Mark it dirty and let the next tick do it once.
            self._recast_pending = True

    # -------------------------------------------------------------------- cast
    @property
    def cast(self) -> list:
        return list(self._cast)

    @property
    def modal(self) -> ModalState:
        return self.tracker.state

    def refit_casting_map(self) -> None:
        """Re-take the baseline fit against the current modes."""
        freqs = np.asarray(self.tracker.state.frequencies_hz, dtype=float).ravel()
        if freqs.size == 0:
            return
        self.cfg.f_lo, self.cfg.f_hi = fit_casting_map(freqs, self.cfg)
        self._map_fitted = True
        self._recast(force=True)

    def _recast(self, force: bool = False) -> None:
        state = self.tracker.state
        freqs = np.asarray(state.frequencies_hz, dtype=float).ravel()
        if freqs.size == 0:
            return
        # fit the two ranges to each other ONCE, on the first good identification
        if self.cfg.autofit and not self._map_fitted and state.ok:
            self.cfg.f_lo, self.cfg.f_hi = fit_casting_map(freqs, self.cfg)
            self._map_fitted = True
            force = True
        cast = cast_meadow(freqs, self.cfg, previous=self._cast)
        sig = cast_signature(cast)
        if force or sig != self._cast_sig:
            self._cast = cast
            self._cast_sig = sig
            self.renderer.set_cast(cast, self._banks, state.shapes, state.damping)

    # -------------------------------------------------------------------- tick
    def wants_reid(self, t: float) -> bool:
        return self.tracker.wants_reid(t)

    def reidentify(self, long_snapshot, t: float) -> ModalState:
        state = self.tracker.reidentify(long_snapshot, t)
        if state.ok:
            self._recast()
        return state

    def tick(self, t: float, ax_snapshot, gz_snapshot=None,
             rate_hz: float = 0.0, channels: dict | None = None) -> VizFrame:
        """Advance the control layer by one step and return a GUI frame.

        ``channels`` optionally carries the other sensor channels as
        ``{"ay": snapshot, "az": ..., "gx": ..., "gy": ...}`` (whichever the
        buffer has); each is read into the frame by the feature extractor.
        """
        self._t = float(t)
        state = self.tracker.state
        frame = self.features.update(ax_snapshot, gz_snapshot, state, t, rate_hz,
                                     channels=channels)
        self._frame = frame
        self.renderer.update_frame(frame)
        if self._recast_pending:
            self._recast_pending = False
            self._recast(force=True)
        elif not self._cast:
            self._recast()
        return self._viz(frame, state)

    def _viz(self, frame: ControlFrame, state: ModalState) -> VizFrame:
        locked = -1
        if frame.sync.size:
            k = int(np.argmax(frame.sync))
            if frame.sync[k] > 0.6:
                locked = k
        if not state.ok:
            text = state.message or "listening — identifying modes…"
        elif locked >= 0:
            text = (f"mode {locked + 1} LOCKED  {state.frequencies_hz[locked]:.2f} Hz"
                    f"  ·  sync {frame.sync[locked]:.2f}")
        else:
            text = (f"chorus  ·  excitation {frame.exc_freq_hz:.2f} Hz"
                    f"  ·  conf {frame.exc_conf:.2f}")
        # name the loudest active case, so the status line says what is happening
        cases = {
            "impact": float(frame.impact),
            "torsion": float(frame.torsion) if frame.torsion > 0.45 else 0.0,
            "drift": float(np.max(frame.drift)) if np.size(frame.drift) else 0.0,
            "beating": float(frame.beating),
            "dropout": float(frame.dropout),
            "approach": float(np.max(frame.approach)) if np.size(frame.approach) else 0.0,
        }
        active = [k for k, val in cases.items() if val > 0.5]
        if active:
            text += "  ·  " + " ".join(f"▲{k.upper()}" for k in active)
        return VizFrame(
            t=frame.t, frame=frame, modal=state, cast=list(self._cast),
            singing=self.renderer.singing_map(), state_text=text,
            spectro_col=frame.psd, spectro_freqs=frame.psd_freqs,
        )

    # ------------------------------------------------------------------- audio
    def schedule_ahead(self) -> None:
        self.renderer.schedule_ahead()

    def render_block(self, n: int = BLOCK_SIZE) -> np.ndarray:
        return self.renderer.render_block(n)

    @property
    def buffered_seconds(self) -> float:
        return self.renderer.scheduled_seconds_ahead


def run_offline(ax_session, gz_session=None, cfg: ChorusConfig | None = None,
                duration_s: float | None = None, seed: int = 23,
                channels: dict | None = None):
    """Render a recorded session through the identical live path.

    ``channels`` may carry further sessions keyed by axis (``"ay"``, ``"az"``,
    ``"gx"``, ``"gy"``), sampled like ``ax_session``; they are windowed the
    same way and reach the feature extractor exactly as in the live path.
    Returns ``(audio (n,2) float32, frames list[VizFrame], engine)``.
    """
    cfg = (cfg or ChorusConfig()).clamped()
    engine = ChorusEngine(cfg, seed=seed)
    data = np.asarray(getattr(ax_session, "data", ax_session), dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    fs = float(getattr(ax_session, "fs", 0.0) or 0.0)
    if fs <= 1.0 or data.size == 0:
        return np.zeros((0, 2), dtype=np.float32), [], engine
    total = duration_s if duration_s else data.shape[1] / fs
    gz = np.asarray(getattr(gz_session, "data", gz_session), dtype=float) \
        if gz_session is not None else None
    if gz is not None and gz.ndim == 1:
        gz = gz[None, :]
    extra: dict[str, np.ndarray] = {}
    for key, sess in (channels or {}).items():
        arr = np.asarray(getattr(sess, "data", sess), dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.size:
            extra[str(key)] = arr

    # Carries sensor_ids so the offline path honours the placement map exactly
    # as the live path does — otherwise a rendered recording would treat the
    # shaker as a floor while the live tab excluded it, and the two would not
    # sound the same.
    sensor_ids = list(getattr(ax_session, "sensor_ids", None)
                      or range(1, data.shape[0] + 1))

    class _Snap:
        __slots__ = ("data", "fs", "sensor_ids")

        def __init__(self, d, f):
            self.data = d
            self.fs = f
            self.sensor_ids = sensor_ids[:d.shape[0]] if d.ndim == 2 else sensor_ids

    n_ticks = max(1, int(total * CONTROL_HZ))
    fast = int(cfg.fast_window_s * fs)
    blocks: list[np.ndarray] = []
    frames: list[VizFrame] = []
    for i in range(n_ticks):
        t = i / CONTROL_HZ
        c = int(t * fs)
        a, b = max(0, c - fast), max(1, c)
        if engine.wants_reid(t):
            lo = max(0, c - int(cfg.id_window_s * fs))
            if c - lo > fs * 2:
                engine.reidentify(_Snap(data[:, lo:c], fs), t)
        ax_snap = _Snap(data[:, a:b], fs)
        gz_snap = _Snap(gz[:, a:b], fs) if gz is not None else None
        ch = {k: _Snap(v[:, a:b], fs) for k, v in extra.items()} or None
        frames.append(engine.tick(t, ax_snap, gz_snap, rate_hz=fs, channels=ch))
        # the play head is advanced by render_block alone; schedule_ahead keeps
        # the rings filled a pre-roll beyond it, exactly as in the live path
        engine.schedule_ahead()
        want = int((t + 1.0 / CONTROL_HZ) * SAMPLE_RATE)
        while engine.renderer.play_seconds * SAMPLE_RATE < want:
            blocks.append(engine.render_block())
    audio = np.concatenate(blocks) if blocks else np.zeros((0, 2), dtype=np.float32)
    return audio, frames, engine
