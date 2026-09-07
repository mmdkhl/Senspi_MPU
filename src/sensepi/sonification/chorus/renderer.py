"""Block-based chorus renderer.

Grains are scheduled AHEAD into per-depth-tier ring buffers; ``render_block``
only reads, filters and mixes. Nothing here allocates per grain in the audio
path, and the audio callback itself never calls into this module's scheduler
(guardrail G4 in spirit: no heavy work where it can stall).

Depth is what makes a chorus sound like a field rather than a speaker: each
individual sits in one of four tiers with its own air-absorption lowpass,
gain and reverb send.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
from scipy import signal as sg

from .types import (BLOCK_SIZE, DEPTH_TIERS, SAMPLE_RATE, CastEntry, ChorusConfig,
                    ControlFrame)

logger = logging.getLogger(__name__)

RING_SECONDS = 4.0
PREROLL_S = 0.25
MAX_GRAINS_PER_S = 1600          # real-time protection; excess is thinned
# Cut hard from the first build. 19 species x up to 18 individuals produced 252
# possible simultaneous voices, which masked every knob and squashed the mix flat
# (13 dB crest against 18-25 dB for real field recordings).
_MAX_IND = {"lead": 5, "chorus": 3, "resonance": 3, "torsion": 4,
            "drift": 3, "alarm": 2, "ambient": 3}
_TIER_P = {
    "drift": (0.55, 0.30, 0.12, 0.03),
    "alarm": (0.70, 0.22, 0.06, 0.02),
    "lead": (0.38, 0.32, 0.20, 0.10),
    "chorus": (0.26, 0.32, 0.26, 0.16),
    "resonance": (0.45, 0.35, 0.15, 0.05),
    "torsion": (0.50, 0.32, 0.13, 0.05),
    "ambient": (0.02, 0.16, 0.36, 0.46),
}
_BASE_GAIN = {"lead": 1.0, "chorus": 0.6, "resonance": 0.85, "torsion": 0.8,
              "drift": 0.75, "alarm": 1.0, "ambient": 0.30}


def _pitch_variants(bank: list, shifts) -> list:
    """Pre-render sharpened copies used for the resonance pitch rise."""
    out = []
    for s in shifts:
        if abs(s - 1.0) < 1e-6:
            out.append(list(bank))
            continue
        v = []
        for g in bank:
            n = max(8, int(len(g) / s))
            idx = np.linspace(0, len(g) - 1, n)
            v.append(np.interp(idx, np.arange(len(g)), g).astype(np.float32))
        out.append(v)
    return out


def _apply_damping(bank: list, zeta: float, amount: float) -> list:
    """Colour the call units with the mode's own damping.

    A heavily damped mode gives dull, quickly-cut calls; a lightly damped one
    lets them ring. This is the same ring-down the structure has, applied to the
    grain, which is what makes damping audible rather than decorative.
    """
    if amount <= 0.01 or not np.isfinite(zeta):
        return bank
    k = float(np.clip(zeta, 0.0, 0.08)) * 100.0 * float(amount)
    if k < 0.05:
        return bank
    out = []
    for g in bank:
        t = np.arange(len(g), dtype=np.float32) / SAMPLE_RATE
        tau = max(len(g) / SAMPLE_RATE, 1e-4) / (1.0 + k)
        out.append((g * np.exp(-t / tau)).astype(np.float32))
    return out


class _Voice:
    """One cast species and its population of individuals."""

    def __init__(self, entry: CastEntry, bank: list, cfg: ChorusConfig,
                 rng: np.random.Generator, shapes=None, zeta: float = 0.0) -> None:
        self.entry = entry
        self.cfg = cfg
        info = entry.info
        bank = _apply_damping(bank, zeta, cfg.damping_expression)
        self.variants = _pitch_variants(bank, (1.0, 1 + cfg.pitch_rise / 2, 1 + cfg.pitch_rise))
        self.train = info.is_train
        self.pulse_rate = float(np.clip(info.pulse_rate, 30.0, 240.0)) if self.train else 0.0
        self.unit_s = float(np.clip(info.unit_s, 0.0008, 0.25))
        self.fm = max(float(entry.mode_freq), 0.2)

        n = max(1, int(round(_MAX_IND.get(entry.role, 5) * cfg.chorus_size)))
        self.n = n
        nat = cfg.naturalism

        # WHERE each individual sits. Previously the whole species took one pan
        # from the sign of the shape's top entry and the individuals scattered
        # randomly around it — the mode shape barely reached the ears.
        #
        # Now each individual is assigned to a measurement row with probability
        # proportional to |mode shape| there, so the population *is* the mode
        # shape: a top-heavy first mode puts most of its animals at the top, and
        # a second mode with a sign change splits into two clusters with a quiet
        # band between them. That is the node, made audible.
        pan0 = 0.0
        if shapes is not None and getattr(shapes, "ndim", 0) == 2 \
                and entry.mode >= 0 and shapes.shape[1] > entry.mode:
            col = np.abs(np.asarray(shapes[:, entry.mode], dtype=float))
            total = float(col.sum())
            if np.isfinite(total) and total > 1e-12:
                w = col / total
                # A floor at a node still gets the occasional call; a shape that
                # is exactly zero there would otherwise silence it absolutely,
                # which sounds like a dead sensor rather than a node.
                w = 0.85 * w + 0.15 / w.size
                self.row_i = rng.choice(w.size, n, p=w / w.sum())
            else:
                self.row_i = rng.integers(0, max(col.size, 1), n)
            top = int(np.argmax(col)) if col.size else 0
            pan0 = float(np.clip((top / max(col.size - 1, 1)) * 1.2 - 0.6, -0.7, 0.7))
        else:
            self.row_i = rng.integers(0, 4, n)
            pan0 = float(rng.uniform(-0.35, 0.35))

        self.rate_i = self.fm * (1 + rng.uniform(-0.05, 0.05, n) * nat)
        # Jitter around whatever pan the placement gives at render time, so a
        # cluster on one floor spreads a little instead of stacking on a point.
        self.pan_jit = rng.uniform(-0.18, 0.18, n) * nat
        self.pan_i = np.clip(pan0 + rng.uniform(-0.6, 0.6, n), -0.95, 0.95)
        p = _TIER_P.get(entry.role, (0.25, 0.25, 0.25, 0.25))
        self.tier_i = rng.choice(len(DEPTH_TIERS), n, p=p)
        self.tier_i = np.where(rng.random(n) < (1.0 - cfg.depth), 0, self.tier_i)
        self.gain_i = 10 ** (rng.uniform(-7, 0, n) / 20)
        self.next_t = rng.uniform(0.0, 1.0, n) / self.fm
        self.singing = rng.random(n) < 0.6
        self.bout_end = rng.exponential(3.0, n)
        self.base = _BASE_GAIN.get(entry.role, 0.6)
        self.singing_count = 0

    def prime(self, t: float) -> None:
        self.next_t = t + (self.next_t % (1.0 / self.fm))
        self.bout_end = t + self.bout_end


class ChorusRenderer:
    """Schedules call units into ring buffers and mixes finished audio blocks."""

    def __init__(self, cfg: ChorusConfig, seed: int = 23) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.ring_len = int(RING_SECONDS * SAMPLE_RATE)
        self._tiers = [np.zeros((self.ring_len, 2), dtype=np.float32)
                       for _ in DEPTH_TIERS]
        self._voices: list[_Voice] = []
        self._frame = ControlFrame()
        self._play_pos = 0                 # samples handed to the audio device
        self._sched_pos = 0                # samples scheduled
        self._grain_budget = 0.0
        self.grains_written = 0
        self.thinned = 0
        self.dropped_late = 0
        self._duck = np.ones(8)          # per-mode gain: a locked mode ducks the rest
        self._prev_sync = np.zeros(8)    # for lock-onset detection
        self._burst_at: dict = {}        # mode -> time of a pending unison burst
        self._beat_phase = 0.0
        self._hush = 1.0                 # impact gate, smoothed in render_block
        self._hush_target = 1.0
        self._stutter = 0.0              # dropout severity
        self._alarm_at = -1.0            # pending startled call
        # render_block runs on the audio callback thread while schedule_ahead /
        # set_cast / set_config run on the worker thread. Without this lock a
        # grain can be written into a region the reader is clearing (a click),
        # or accepted behind a stale play head and then replayed a whole ring
        # later as a ghost. Hold time is short and bounded on both sides.
        self._lock = threading.RLock()
        self._init_filters()

    # ------------------------------------------------------------------ setup
    def _init_filters(self) -> None:
        self._tier_sos = []
        self._tier_zi = []
        for _, lp, _send in DEPTH_TIERS:
            cut = float(np.clip(lp * self.cfg.brightness, 500.0, SAMPLE_RATE / 2 * 0.95))
            sos = sg.butter(2, cut, "lowpass", fs=SAMPLE_RATE, output="sos")
            self._tier_sos.append(sos)
            self._tier_zi.append(np.zeros((sos.shape[0], 2, 2)))
        # Schroeder diffusion for the outdoor tail
        self._combs = []
        for d_ms, g in ((29.7, 0.75), (37.1, 0.72), (41.1, 0.69), (43.7, 0.66)):
            d = int(SAMPLE_RATE * d_ms / 1000)
            a = np.zeros(d + 1)
            a[0], a[d] = 1.0, -g * (0.55 + 0.45 * self.cfg.space)
            self._combs.append((np.array([1.0]), a, np.zeros((d, 2))))
        self._allpass = []
        for d_ms, g in ((5.0, 0.7), (1.7, 0.7)):
            d = int(SAMPLE_RATE * d_ms / 1000)
            b = np.zeros(d + 1); b[0], b[d] = -g, 1.0
            a = np.zeros(d + 1); a[0], a[d] = 1.0, -g
            self._allpass.append((b, a, np.zeros((d, 2))))

    def set_cast(self, cast: list[CastEntry], banks: dict, shapes=None,
                 damping=None) -> None:
        """Re-cast the meadow. Existing audio in the rings keeps playing out."""
        voices = []
        with self._lock:
            t_now = self._sched_pos / SAMPLE_RATE
        for entry in cast:
            bank = banks.get(entry.info.species)
            if not bank:
                continue
            try:
                zeta = 0.0
                if damping is not None and 0 <= entry.mode < len(damping):
                    zeta = float(damping[entry.mode])
                v = _Voice(entry, bank, self.cfg, self.rng, shapes, zeta)
            except Exception:
                logger.debug("chorus: could not build voice for %s", entry.info.species)
                continue
            v.prime(t_now)
            voices.append(v)
        with self._lock:
            self._voices = voices

    def set_config(self, cfg: ChorusConfig) -> None:
        with self._lock:
            self.cfg = cfg
            for v in self._voices:
                v.cfg = cfg
            self._init_filters()

    def update_frame(self, frame: ControlFrame) -> None:
        self._frame = frame

    # -------------------------------------------------------------- scheduling
    @property
    def scheduled_seconds_ahead(self) -> float:
        return (self._sched_pos - self._play_pos) / SAMPLE_RATE

    def schedule_ahead(self) -> None:
        """Fill the rings up to PREROLL_S beyond the play head."""
        with self._lock:
            target = self._play_pos + int((PREROLL_S + 0.05) * SAMPLE_RATE)
            if target <= self._sched_pos:
                return
            dt = (target - self._sched_pos) / SAMPLE_RATE
            self._grain_budget += MAX_GRAINS_PER_S * dt * max(1.0, self.cfg.density)
            self._update_cases(dt)
            t0 = self._sched_pos / SAMPLE_RATE
            t1 = target / SAMPLE_RATE
            for v in self._voices:
                try:
                    self._schedule_voice(v, t0, t1)
                except Exception:
                    logger.debug("chorus: scheduling failed for %s",
                                 v.entry.info.species, exc_info=True)
            # a burst is consumed once it has been handed to every voice
            self._burst_at = {k: b for k, b in self._burst_at.items()
                              if b >= target / SAMPLE_RATE}
            if 0 <= self._alarm_at < (target / SAMPLE_RATE) - 0.5:
                self._alarm_at = -1.0
            self._sched_pos = target

    def _update_cases(self, dt: float) -> None:
        """Cross-voice state: ducking, lock gestures, beating phase, gating."""
        frame = self._frame
        cfg = self.cfg
        sync = np.asarray(frame.sync, dtype=float).ravel()
        n = sync.size
        duck = np.ones(max(n, 1))
        if n:
            locked = sync > 0.6
            if locked.any():
                # DUCKING: the single biggest clarity win. When one mode locks,
                # everything else steps back so the locked mode is unmistakable.
                amt = 10 ** (-cfg.duck_db / 20.0)
                strength = float(sync[locked].max())
                for i in range(n):
                    if not locked[i]:
                        duck[i] = 1.0 + (amt - 1.0) * strength
            # LOCK GESTURE: a unison burst at the instant of locking, so the
            # transition is an event rather than a slow drift.
            if cfg.lock_gesture:
                for i in range(n):
                    if sync[i] > 0.6 and self._prev_sync[i] <= 0.6:
                        self._burst_at[i] = self._sched_pos / SAMPLE_RATE + 0.05
            self._prev_sync[:n] = sync
        self._duck = duck

        # BEATING: advance a phase at the measured beat rate so the chorus
        # pulses at the true |f1-f2| rather than at some arbitrary LFO speed.
        if frame.beating > 0.05 and frame.beat_hz > 0.02:
            self._beat_phase = (self._beat_phase + 2 * np.pi * frame.beat_hz * dt) % (2 * np.pi)

        # IMPACT: the meadow hushes, then recovers. Attack is immediate, the
        # release is slow because that is how a real meadow comes back.
        if frame.impact > 0.05:
            self._hush_target = float(np.clip(1.0 - 0.9 * frame.impact * cfg.impact_voice,
                                              0.05, 1.0))
        else:
            self._hush_target = 1.0
        self._stutter = float(frame.dropout) * cfg.dropout_voice
        # the startled call lands just after the hush, not on top of it
        if frame.impact > 0.15:
            self._alarm_at = self._sched_pos / SAMPLE_RATE + 0.35

    def _beat_gain(self) -> float:
        f = self._frame
        if f.beating <= 0.05 or f.beat_hz <= 0.02:
            return 1.0
        depth = 0.75 * float(f.beating) * self.cfg.beating_voice
        return float(1.0 - depth * (0.5 - 0.5 * np.cos(self._beat_phase)))

    def _schedule_voice(self, v: _Voice, t0: float, t1: float) -> None:
        cfg = self.cfg
        frame = self._frame
        entry = v.entry
        role = entry.role
        m = entry.mode
        be = frame.band_energy
        sy = frame.sync
        e = float(be[m]) if 0 <= m < be.size else float(np.clip(frame.env_global * 4, 0, 1))
        sync = float(sy[m]) if 0 <= m < sy.size else 0.0

        # SOLO: hear one mode at a time. The fastest way to learn what each
        # mode's animal sounds like.
        if cfg.solo_mode >= 0 and m != cfg.solo_mode:
            for k in range(v.n):
                period = 1.0 / max(v.rate_i[k], 0.2)
                while v.next_t[k] < t1:
                    v.next_t[k] += period
            v.singing_count = 0
            return

        if role == "resonance":
            drive = max(0.0, (sync - 0.35) / 0.65)
            n_sing = int(round(v.n * drive ** 0.8))
            loud = drive * cfg.resonance_voice
        elif role == "drift":
            d = float(np.max(frame.drift)) if np.size(frame.drift) else 0.0
            drive = max(0.0, (d - 0.45) / 0.55)
            n_sing = int(round(v.n * drive))
            loud = drive * cfg.drift_voice
        elif role == "alarm":
            # only after an impact, and only for a moment
            live = 0.0 <= self._alarm_at < t1 + 0.5
            n_sing = v.n if live else 0
            loud = (0.8 + 0.6 * float(frame.impact)) * cfg.impact_voice if live else 0.0
        elif role == "torsion":
            # Driven by the floor that is actually twisting, not by an average
            # over every gyro — averaging turned "the top floor is twisting
            # hard" into "a little twist everywhere" and the voice never rose.
            tf = frame.torsion_floor
            drive = float(np.max(tf)) if np.size(tf) else float(frame.torsion)
            n_sing = int(round(v.n * max(0.0, (drive - 0.4) / 0.6)))
            loud = max(0.0, (drive - 0.4) / 0.6) * cfg.torsion_voice
            # ...and it sings from that floor, so a twisting top floor is heard
            # at the top rather than across the whole structure.
            if np.size(tf):
                v.row_i = np.full(v.n, int(np.argmax(tf)))
        elif role == "ambient":
            n_sing = v.n
            loud = (0.5 + 0.5 * float(np.clip(frame.env_global * 4, 0, 1))) * cfg.ambient_bed
        else:
            # RESONANCE APPROACH: the chorus grows agitated as excitation closes
            # in, so the lock arrives with a build-up instead of out of nowhere.
            appr = 0.0
            if 0 <= m < np.size(frame.approach):
                appr = float(frame.approach[m]) * cfg.approach_voice
            n_sing = int(round(1 + (v.n - 1) * min(1.0, e ** 1.35 + 0.4 * appr)))
            loud = 0.28 + 0.72 * e + 0.25 * appr
        n_sing = int(np.clip(n_sing, 0, v.n))
        v.singing_count = n_sing
        loud *= v.base
        # DUCK: step back while another mode is locked
        if 0 <= m < self._duck.size and role not in ("alarm",):
            loud *= float(self._duck[m])
        # BEAT: pulse the whole scene at the true |f1-f2|
        if role in ("lead", "chorus"):
            loud *= self._beat_gain()

        if loud <= 1e-3 or n_sing <= 0:
            for k in range(v.n):                     # keep phases advancing
                period = 1.0 / max(v.rate_i[k], 0.2)
                while v.next_t[k] < t1:
                    v.next_t[k] += period
            return

        vi = int(np.clip(sync * (len(v.variants) - 1) + 0.5, 0, len(v.variants) - 1))
        bank = v.variants[vi]
        rate_boost = (1.0 + 3.5 * float(frame.torsion)) if role == "torsion" else 1.0
        nat = cfg.naturalism

        for k in range(n_sing):
            if t0 > v.bout_end[k]:
                v.singing[k] = self.rng.random() < (0.55 + 0.45 * sync)
                v.bout_end[k] = t0 + self.rng.exponential(2.5 * (0.5 + 0.5 * e) + 0.4)
            if role in ("resonance", "torsion"):
                v.singing[k] = True
            rate = (v.fm * sync + v.rate_i[k] * (1.0 - sync)) * rate_boost
            if role in ("lead", "chorus") and 0 <= m < np.size(frame.approach):
                rate *= 1.0 + 0.5 * float(frame.approach[m]) * cfg.approach_voice
            period = 1.0 / max(rate * max(cfg.density, 0.05), 0.15)
            # Place this individual where its sensor actually is: the plan
            # column becomes the stereo position, so four sensors give an image
            # of the rig rather than four voices in arbitrary places.
            row = int(v.row_i[k]) if k < np.size(v.row_i) else 0
            pans = frame.pan_of
            if pans and row < len(pans):
                pan = float(np.clip(pans[row] + v.pan_jit[k], -0.95, 0.95))
            else:
                pan = v.pan_i[k]
            if role == "torsion":
                pan = float(np.clip(0.75 * frame.torsion_pan + 0.25 * pan, -0.95, 0.95))
            elif role == "drift" and np.size(frame.drift):
                # sit between the pair that is separating: lower pairs left
                j = int(np.argmax(frame.drift))
                spread = (j / max(np.size(frame.drift) - 1, 1)) * 1.4 - 0.7
                pan = float(np.clip(spread + 0.25 * pan, -0.95, 0.95))
            gl = float(np.sqrt(1 - (pan + 1) / 2))
            gr = float(np.sqrt((pan + 1) / 2))
            # Height becomes distance: the higher the floor, the nearer and
            # brighter the animal sounds. Tier 0 is dry and close, so a high
            # floor maps to a low tier index. Without this the four sensors were
            # spread in stereo but flat in depth, and a first mode swaying the
            # top of the building sounded no closer than the ground floor.
            tier = int(v.tier_i[k])
            floors = frame.floor_of
            if floors and row < len(floors):
                top = max(max(floors), 1)
                near = 1.0 - (float(floors[row]) / top)
                tier = int(np.clip(round(near * (len(self._tiers) - 1)), 0,
                                   len(self._tiers) - 1))
                if cfg.depth < 1.0 and self.rng.random() < (1.0 - cfg.depth):
                    tier = 0
            buf = self._tiers[tier]

            burst = self._burst_at.get(m)
            if burst is not None and role in ("lead", "chorus") and t0 <= burst < t1:
                v.next_t[k] = burst          # everyone calls at the same instant
            while v.next_t[k] < t1:
                if v.next_t[k] >= t0 and v.singing[k] and self._grain_budget > 0:
                    jitter = 0.022 * nat * (1.0 - 0.8 * sync)
                    start = v.next_t[k] + self.rng.normal(0.0, jitter)
                    s0 = int(start * SAMPLE_RATE)
                    amp = v.gain_i[k] * loud * (0.55 + 0.45 * sync)
                    if v.train:
                        npul = int(np.clip(0.55 / max(rate, 0.3) * v.pulse_rate, 2, 26)
                                   * (0.55 + 0.45 * e))
                        npul = max(2, npul)
                        step = max(1, int(SAMPLE_RATE / v.pulse_rate))
                        arc = np.sin(np.linspace(0.3, np.pi - 0.3, npul)) ** 0.55
                        for p in range(npul):
                            if self._grain_budget <= 0:
                                self.thinned += 1
                                break
                            g = bank[int(self.rng.integers(len(bank)))]
                            pos = s0 + p * step + int(self.rng.integers(-4, 5) * nat)
                            a = amp * float(arc[p]) * float(self.rng.uniform(0.75, 1.0))
                            self._write(buf, pos, g, a * gl, a * gr)
                    else:
                        g = bank[int(self.rng.integers(len(bank)))]
                        a = amp * float(self.rng.uniform(0.8, 1.0))
                        self._write(buf, s0, g, a * gl, a * gr)
                v.next_t[k] += period

    def _write(self, buf: np.ndarray, pos: int, grain: np.ndarray,
               gl: float, gr: float) -> None:
        """Add one grain into a ring buffer, handling wrap-around.

        The audio thread reads and clears blocks concurrently, and timing
        jitter can push a scheduled grain slightly earlier than the window we
        asked for. Anything that lands inside the guard band would be raced by
        the reader (partially zeroed, i.e. an audible click), so it is dropped
        instead. This is rare by construction: we schedule a whole pre-roll
        ahead of the play head.
        """
        n = len(grain)
        if n <= 0 or pos < self._play_pos + 2 * BLOCK_SIZE:
            self.dropped_late += 1
            return
        self._grain_budget -= 1.0
        self.grains_written += 1
        start = pos % self.ring_len
        end = start + n
        if end <= self.ring_len:
            buf[start:end, 0] += grain * gl
            buf[start:end, 1] += grain * gr
        else:
            cut = self.ring_len - start
            buf[start:, 0] += grain[:cut] * gl
            buf[start:, 1] += grain[:cut] * gr
            rest = n - cut
            buf[:rest, 0] += grain[cut:] * gl
            buf[:rest, 1] += grain[cut:] * gr

    # ---------------------------------------------------------------- playback
    def render_block(self, n: int = BLOCK_SIZE) -> np.ndarray:
        """Pull one finished stereo block. Cheap: read, filter, mix, clear."""
        with self._lock:
            return self._render_block_locked(n)

    def _render_block_locked(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        wet = np.zeros((n, 2), dtype=np.float32)
        start = self._play_pos % self.ring_len
        end = start + n
        for i, (buf, (gain, _lp, send)) in enumerate(zip(self._tiers, DEPTH_TIERS)):
            if end <= self.ring_len:
                chunk = buf[start:end].copy()
                buf[start:end] = 0.0
            else:
                cut = self.ring_len - start
                rest = n - cut
                chunk = np.concatenate([buf[start:].copy(), buf[:rest].copy()])
                buf[start:] = 0.0
                buf[:rest] = 0.0
            y, self._tier_zi[i] = sg.sosfilt(self._tier_sos[i], chunk, axis=0,
                                             zi=self._tier_zi[i])
            y = y.astype(np.float32) * gain
            out += y
            wet += y * send
        self._play_pos += n

        if self.cfg.space > 0.001:
            rev = np.zeros((n, 2), dtype=np.float32)
            for j, (b, a, zi) in enumerate(self._combs):
                y, zf = sg.lfilter(b, a, wet, axis=0, zi=zi)
                self._combs[j] = (b, a, zf)
                rev += y.astype(np.float32)
            rev *= 0.25
            for j, (b, a, zi) in enumerate(self._allpass):
                rev, zf = sg.lfilter(b, a, rev, axis=0, zi=zi)
                rev = rev.astype(np.float32)
                self._allpass[j] = (b, a, zf)
            out += rev * (self.cfg.space * 1.25)

        # ---- IMPACT: the meadow hushes, then comes back -------------------
        # Applied here rather than at schedule time because the hush must affect
        # audio that was already scheduled 300 ms ago -- exactly like a real
        # meadow falling silent mid-call.
        if abs(self._hush - self._hush_target) > 1e-4 or self._hush < 0.999:
            # snap down fast, recover slowly
            coef = 0.55 if self._hush_target < self._hush else 0.995
            env = np.empty(n, dtype=np.float32)
            h = self._hush
            for i in range(n):
                h = coef * h + (1.0 - coef) * self._hush_target
                env[i] = h
            self._hush = float(h)
            out *= env[:, None]

        # ---- DROPOUT: lost samples make the scene stutter -----------------
        if self._stutter > 0.05:
            period = max(64, int(SAMPLE_RATE * 0.05))
            idx = (np.arange(n) + self._play_pos) % period
            duty = 1.0 - 0.85 * float(np.clip(self._stutter, 0, 1))
            gate = (idx < period * duty).astype(np.float32)
            out *= gate[:, None]

        out = np.tanh(out * 1.1) * 0.9
        np.clip(out, -0.98, 0.98, out=out)
        return (out * self.cfg.master).astype(np.float32)

    # ------------------------------------------------------------------ status
    @property
    def play_seconds(self) -> float:
        return self._play_pos / SAMPLE_RATE

    def singing_map(self) -> dict:
        return {v.entry.info.species: int(v.singing_count) for v in self._voices}
