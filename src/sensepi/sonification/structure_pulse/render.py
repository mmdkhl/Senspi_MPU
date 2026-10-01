"""Curve -> audio. The synthesis half of Structure Pulse.

Pure numpy/scipy, no Qt and no audio device (guardrail G7): this module returns
a finished stereo buffer, and whoever wants to hear it hands that to
``sonification.chorus.audio_out.AudioOutput``.

Rendering is **offline, whole-buffer**. The chorus streams because its source is
a live sensor feed; here the source is a finished recording, so a ring-buffer
scheduler would buy nothing and cost a lot of threading. The playhead simply
follows playback position.

Three voices, after the two systems this tab is modelled on:

``arc``
    Xenakis' UPIC (1977): x is time, y is frequency. The curve's height drives
    an oscillator as the playhead sweeps. The default, and the one that works
    for every view.

``pulsar``
    Curtis Roads' pulsar synthesis, as revived in Marcin Pietruszewski's nuPG.
    A pulsar has two independent frequencies: the *fundamental* is the pulse
    repetition rate, the *formant* is the pitch of each particle. Here the
    building's own eigenfrequency becomes the fundamental — so 2 Hz is heard
    as 2 pulses per second, its true rate, not transposed — while the curve
    drives the formant.

``audify``
    The classical seismology move: treat the samples as a waveform and speed
    them up until they are audible. Only meaningful for a time-domain view.

On top of any voice, ``bells`` strikes a resonator each time the playhead
crosses an eigenfrequency marker, with a different timbre per mode.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import signal as sg

from .types import SAMPLE_RATE, PulseConfig, PulseView, RenderResult

logger = logging.getLogger(__name__)

#: Partial ratios and decay multipliers per timbre. Inharmonic ratios are what
#: make a bell sound struck rather than merely pitched.
_TIMBRES = {
    "bell":      dict(partials=(1.0, 2.76, 5.40, 8.93), gains=(1.0, .6, .4, .25),
                      decay=(1.0, .7, .5, .4), attack=0.002),
    "organ":     dict(partials=(1.0, 2.0, 3.0, 4.0), gains=(1.0, .5, .35, .2),
                      decay=(3.0, 3.0, 2.6, 2.2), attack=0.06),
    "triangle":  dict(partials=(1.0, 4.1, 6.9, 10.7), gains=(.5, 1.0, .7, .5),
                      decay=(1.6, 1.4, 1.2, 1.0), attack=0.001),
    "woodblock": dict(partials=(1.0, 3.3, 5.9), gains=(1.0, .5, .3),
                      decay=(.12, .09, .07), attack=0.0008),
    "string":    dict(partials=(1.0, 2.0, 3.0, 5.0), gains=(1.0, .45, .3, .15),
                      decay=(1.8, 1.4, 1.1, .8), attack=0.012),
}


# --------------------------------------------------------------------- helpers
def _norm01(y: np.ndarray) -> np.ndarray:
    """Curve height as 0..1, robust to a single wild sample.

    Percentiles rather than min/max: one spike would otherwise squash the whole
    curve into a sliver of the pitch range.
    """
    y = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if y.size == 0:
        return y
    lo, hi = np.percentile(y, 1.0), np.percentile(y, 99.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-15:
        lo, hi = float(np.min(y)), float(np.max(y))
    if hi - lo < 1e-15:
        return np.full_like(y, 0.5)
    return np.clip((y - lo) / (hi - lo), 0.0, 1.0)


def _glide(x: np.ndarray, sr: int, tau_s: float) -> np.ndarray:
    """One-pole smoothing of the frequency track, so sweeps do not step."""
    if tau_s <= 1e-6 or x.size < 2:
        return x
    a = float(np.exp(-1.0 / max(tau_s * sr, 1.0)))
    return sg.lfilter([1.0 - a], [1.0, -a], x)


def _fade(n: int, sr: int, ms: float = 12.0) -> np.ndarray:
    """Equal-power-ish fade in/out, so a buffer never starts or ends on a step."""
    env = np.ones(n, dtype=float)
    k = int(min(max(sr * ms / 1000.0, 1), max(n // 2, 1)))
    ramp = np.sin(np.linspace(0, np.pi / 2, k)) ** 2
    env[:k] *= ramp
    env[-k:] *= ramp[::-1]
    return env


def _pan(index: int, total: int, spread: float) -> tuple:
    """Constant-power pan for trace ``index`` of ``total``."""
    if total <= 1:
        pos = 0.0
    else:
        pos = (index / (total - 1.0)) * 2.0 - 1.0
    pos *= float(np.clip(spread, 0.0, 1.0))
    ang = (pos + 1.0) * 0.25 * np.pi
    return float(np.cos(ang)), float(np.sin(ang))


def strike(freq: float, timbre: str, decay_s: float, sr: int = SAMPLE_RATE,
           gain: float = 1.0) -> np.ndarray:
    """One struck/sustained note. Returns a mono buffer."""
    spec = _TIMBRES.get(timbre, _TIMBRES["bell"])
    longest = max(spec["decay"]) * decay_s
    n = int(max(sr * longest, 8))
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=float)
    for ratio, g, dmul in zip(spec["partials"], spec["gains"], spec["decay"]):
        f = float(freq) * float(ratio)
        if f <= 0 or f >= sr * 0.48:          # above Nyquist: silently skip
            continue
        env = np.exp(-t / max(dmul * decay_s / 3.0, 1e-3))
        out += g * env * np.sin(2 * np.pi * f * t)
    atk = int(max(sr * spec["attack"], 1))
    if atk < n:
        out[:atk] *= np.linspace(0.0, 1.0, atk) ** 2
    peak = float(np.abs(out).max())
    if peak > 1e-9:
        out /= peak
    return out * float(gain)


# ----------------------------------------------------------------- the voices
def _is_oscillating(y: np.ndarray) -> bool:
    """True for a measured record, false for a curve that only rises.

    Tested about the signal's OWN MEAN, not about zero: a gyro channel sits on
    a DC bias (the test rig's gz rests near -1.65 deg/s) and oscillates happily
    without its samples ever changing sign. Judging by zero crossings called
    that curve monotone and gave it the wobbling pitch this exists to avoid.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 8:
        return False
    centred = y - float(np.mean(y))
    scale = float(np.std(centred))
    if scale <= 1e-15:
        return False
    neg = float(np.mean(centred < 0.0))
    return 0.2 < neg < 0.8


def _shape(y: np.ndarray, mode: str, sr: int, dur_s: float,
           smooth_frac: float, kind: str = "") -> np.ndarray:
    """What the pitch should follow.

    A time history swings symmetrically through zero, so feeding it straight to
    the pitch map reads the minus half as "low" and the plus half as "high" —
    the pitch wobbles at the structural frequency instead of describing the
    motion. Taking the magnitude removes the sign but doubles the rate; taking
    a smoothed envelope leaves a slow contour, which is exactly the shape a
    mode shape already has and why that view is the one that sounds continuous.
    """
    y = np.nan_to_num(np.asarray(y, dtype=float))
    if mode == "auto":
        # ONLY a time-domain record gets the contour treatment. Everything else
        # is left signed on purpose:
        #   * a mode shape's sign IS the information — rectifying it would fold
        #     the negative lobe up and erase the node, and mode shapes are the
        #     one view that already sounds right;
        #   * a spectrum rises from a floor and never needs it.
        # A custom view falls back to the oscillation test.
        if kind == "time":
            mode = "envelope"
        elif kind in ("spectrum", "response", "shape"):
            mode = "signed"
        else:
            mode = "envelope" if _is_oscillating(y) else "signed"
    if mode == "signed" or y.size < 4:
        return y
    y = y - float(np.mean(y)) if kind == "time" else y
    mag = np.abs(y)
    if mode == "magnitude":
        return mag
    # Envelope: smooth |y| over a window that is at least a couple of the
    # curve's OWN oscillations. A fixed fraction of the sweep is not enough on
    # a short record — at 2 Hz over 20 s one cycle already occupies 2.5 % of
    # the sweep, so a 2 % window smooths less than a single cycle and the pitch
    # keeps rippling at the structural rate. Measure the rate from sign changes
    # and widen the window to cover it.
    want = float(smooth_frac)
    crossings = int(np.count_nonzero(np.diff(np.signbit(y - np.mean(y)))))
    if crossings >= 2:
        cycle_frac = 2.0 / crossings          # one oscillation, as a fraction
        want = max(want, 2.5 * cycle_frac)    # ~2.5 cycles per window
    want = min(want, 0.5)                     # never swallow the whole shape
    n = int(max(round(y.size * want), 3))
    if n >= y.size:
        return mag
    # Two box passes, not a windowed convolution. ``np.convolve`` is O(n*m),
    # and m here is a fraction of the SWEEP: at 20 s and 44.1 kHz an adaptive
    # window reached 441 000 samples, which is ~6e11 operations — it did not
    # finish, and the live worker hung inside it. A running-mean pass is O(n)
    # whatever the window, and two of them give a triangular response that is
    # smooth enough for a pitch contour.
    return _box2(mag, n)


def _box2(y: np.ndarray, n: int) -> np.ndarray:
    """Two running means of width ``n``. O(len(y)), independent of ``n``."""
    half = max(int(n) // 2, 1)
    out = y
    for _ in range(2):
        pad = np.concatenate([np.full(half, out[0]), out, np.full(half, out[-1])])
        c = np.cumsum(np.concatenate([[0.0], pad]))
        width = 2 * half + 1
        out = (c[width:] - c[:-width]) / float(width)
        out = out[:y.size]
        if out.size < y.size:                 # guard against an off-by-one
            out = np.pad(out, (0, y.size - out.size), mode="edge")
    return out



def _arc_voice(y_t: np.ndarray, cfg: PulseConfig, sr: int,
               amplitude_follows_y: bool, dur_s: float = 1.0,
               kind: str = "") -> np.ndarray:
    """UPIC: height becomes frequency, read left to right."""
    norm = _norm01(_shape(y_t, cfg.scale, sr, dur_s, cfg.envelope_smooth, kind))
    f = cfg.f_lo * (cfg.f_hi / cfg.f_lo) ** norm
    f = _glide(f, sr, cfg.glide)
    phase = np.cumsum(2 * np.pi * f / sr)
    out = np.zeros_like(phase)
    for h in range(1, int(cfg.harmonics) + 1):
        if float(f.max()) * h >= sr * 0.48:
            break
        out += np.sin(phase * h) / h
    if amplitude_follows_y:
        amp = 0.12 + 0.88 * norm ** 1.4
        out *= amp
    return out


def _pulsar_voice(y_t: np.ndarray, cfg: PulseConfig, sr: int,
                  fundamental_hz: float, dur_s: float = 1.0,
                  kind: str = "") -> np.ndarray:
    """Roads' pulsar train: the structure's own rate, heard as a rate.

    Pulse repetition = ``fundamental_hz`` (the eigenfrequency, 1-20 Hz, felt as
    rhythm). Each pulsaret is a windowed burst whose formant frequency comes
    from the curve. The silence between pulsarets is what makes the rate
    audible as a rate rather than a tone.
    """
    n = y_t.size
    out = np.zeros(n, dtype=float)
    fp = float(fundamental_hz)
    if not np.isfinite(fp) or fp <= 0.05:
        fp = 2.0
    norm = _norm01(_shape(y_t, cfg.scale, sr, dur_s, cfg.envelope_smooth, kind))
    formant = cfg.f_lo * (cfg.f_hi / cfg.f_lo) ** norm
    period = sr / fp
    duty = float(np.clip(cfg.pulsar_duty, 0.02, 1.0))
    width = max(int(period * duty), 8)
    w = np.hanning(width)
    starts = np.arange(0, n, period).astype(int)
    for s in starts:
        e = min(s + width, n)
        if e - s < 4:
            continue
        fd = float(formant[s])
        if fd <= 0 or fd >= sr * 0.48:
            continue
        k = e - s
        tt = np.arange(k) / sr
        out[s:e] += w[:k] * np.sin(2 * np.pi * fd * tt)
    return out


def _audify(y: np.ndarray, fs_data: float, sr: int, target_hz: float,
            source_hz: float) -> tuple:
    """Speed the record up until the structure lands in the audible range.

    Returns ``(audio, speed)``. Speed is chosen so ``source_hz`` arrives at
    ``target_hz``; at 2 Hz -> 220 Hz that is a factor of 110, which is the same
    move seismologists make to listen to an earthquake.
    """
    y = np.nan_to_num(np.asarray(y, dtype=float).ravel())
    if y.size < 4 or not np.isfinite(fs_data) or fs_data <= 0:
        return np.zeros(0), 1.0
    src = float(source_hz) if np.isfinite(source_hz) and source_hz > 0.05 else 2.0
    speed = float(np.clip(target_hz / src, 1.0, 2000.0))
    n_out = int(max(y.size * (sr / (fs_data * speed)), 8))
    out = np.interp(np.linspace(0, y.size - 1, n_out), np.arange(y.size), y)
    peak = float(np.abs(out).max())
    return (out / peak if peak > 1e-12 else out), speed


# ------------------------------------------------------------------ the render
def render_view(view: PulseView, cfg: PulseConfig, *,
                modal_frequencies=None, sample_rate: int = SAMPLE_RATE,
                data_fs: float = float("nan")) -> RenderResult:
    """Turn one plotted view into a stereo buffer plus playhead track."""
    cfg = cfg.clamped()
    sr = int(sample_rate)
    if view is None or not view.is_usable:
        return RenderResult(sample_rate=sr, message="nothing to play")

    curves = [c for c in view.curves if c.is_usable]
    x0, x1 = view.x_range()
    freqs = np.asarray(modal_frequencies if modal_frequencies is not None
                       else [], dtype=float).ravel()
    f1 = float(freqs[0]) if freqs.size and np.isfinite(freqs[0]) else 2.0

    # --- audification runs on its own clock ---------------------------------
    if cfg.voice == "audify":
        # Audification reads the curve AS a waveform, so it is only meaningful
        # where x is time. A spectrum or a mode shape is not a signal; speeding
        # it up would produce a confident-sounding noise that means nothing.
        if view.kind != "time":
            return RenderResult(
                sample_rate=sr,
                message=f"audify needs a time-domain view; "
                        f"'{view.name}' is {view.kind} — use arc or pulsar")
        mono, speed = _audify(curves[0].y, data_fs, sr,
                              target_hz=max(cfg.f_lo * 2.0, 180.0), source_hz=f1)
        if mono.size == 0:
            return RenderResult(sample_rate=sr, message="cannot audify this view")
        n = mono.size
        dur = n / sr
        mono = mono * _fade(n, sr)
        stereo = np.column_stack([mono, mono]).astype(np.float32)
        t_grid = np.arange(n) / sr
        return _finish(stereo, sr, dur, t_grid, np.linspace(x0, x1, n), [], cfg,
                       f"audified ×{speed:.0f}")

    # --- arc / pulsar sweep the plot at the chosen duration ------------------
    dur = float(cfg.duration_s)
    n = int(max(sr * dur, 16))
    t_grid = np.arange(n) / sr
    frac = t_grid / max(dur, 1e-9)
    # The playhead must cross the SCREEN evenly. On a log axis a linear sweep
    # in x covers most of the plot in the first instant and then crawls through
    # the last decade for the rest of the time — on a 0.02-2 s response
    # spectrum the line reached 39 % of the width in the first 5 % of the
    # sweep. Match the sweep to the axis instead.
    log_sweep = bool(view.x_log) and x0 > 0 and x1 > x0
    if log_sweep:
        x_of_t = x0 * (x1 / x0) ** frac
    else:
        x_of_t = x0 + (x1 - x0) * frac

    mix = np.zeros((n, 2), dtype=float)
    for i, c in enumerate(curves):
        y_t = np.interp(x_of_t, c.x, c.y, left=c.y[0], right=c.y[-1])
        if cfg.voice == "pulsar":
            fp = cfg.pulsar_fundamental_hz if cfg.pulsar_fundamental_hz > 0 else f1
            mono = _pulsar_voice(y_t, cfg, sr, fp, dur, view.kind)
        else:
            mono = _arc_voice(y_t, cfg, sr, view.amplitude_follows_y, dur,
                              view.kind)
        gl, gr = _pan(i, len(curves), cfg.stereo_spread)
        scale = cfg.arc_gain / np.sqrt(max(len(curves), 1))
        mix[:, 0] += mono * gl * scale
        mix[:, 1] += mono * gr * scale

    mix *= _fade(n, sr)[:, None]

    # --- bells at the eigenfrequency markers --------------------------------
    strikes = []
    if cfg.bells and view.markers:
        span = max(x1 - x0, 1e-12)
        log_span = np.log(x1 / x0) if log_sweep else 1.0
        for mk in view.markers:
            if mk.timbre == "off":
                continue
            if log_sweep and float(mk.x) > 0:
                frac = float(np.log(float(mk.x) / x0) / log_span)
            else:
                frac = (float(mk.x) - x0) / span
            if not (0.0 <= frac <= 1.0):
                continue
            t_hit = frac * dur
            # the bell agrees with the sweep: it rings at the pitch the arc is
            # singing where it stands
            pitch = cfg.f_lo * (cfg.f_hi / cfg.f_lo) ** float(np.clip(frac, 0, 1))
            note = strike(pitch, mk.timbre, cfg.bell_decay_s, sr, cfg.bell_gain)
            s = int(t_hit * sr)
            e = min(s + note.size, n)
            if e > s:
                mix[s:e, 0] += note[:e - s] * 0.7
                mix[s:e, 1] += note[:e - s] * 0.7
            strikes.append((float(t_hit), mk.label))

    return _finish(mix, sr, dur, t_grid, x_of_t, strikes, cfg, "")


def _finish(mix: np.ndarray, sr: int, dur: float, t_grid, x_of_t, strikes,
            cfg: PulseConfig, message: str) -> RenderResult:
    """Soft-limit, scale and package."""
    mix = np.nan_to_num(np.asarray(mix, dtype=float))
    if mix.ndim == 1:
        mix = np.column_stack([mix, mix])
    peak = float(np.abs(mix).max())
    if peak > 1e-9:
        mix = mix / peak * 0.92          # normalise, then soft-clip for safety
    mix = np.tanh(mix * 1.05) * 0.95
    mix *= cfg.master
    return RenderResult(
        audio=mix.astype(np.float32), sample_rate=sr, duration_s=float(dur),
        x_of_t=np.asarray(x_of_t, dtype=float),
        t_grid=np.asarray(t_grid, dtype=float),
        strikes=list(strikes), peak=float(np.abs(mix).max()), message=message)


def write_wav(path, result: RenderResult) -> None:
    """Save a render. Kept here so the GUI never touches audio formatting."""
    import wave
    from pathlib import Path

    pcm = (np.clip(result.audio, -1.0, 1.0) * 32767.0).astype("<i2")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(result.sample_rate))
        w.writeframes(pcm.tobytes())
