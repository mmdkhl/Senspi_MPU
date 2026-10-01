"""Data contracts for Structure Pulse.

Pure dataclasses + numpy. No Qt, no audio device, no GUI imports (guardrail G7).

The model, in one line: **a plotted curve is a control signal, and playing it
means sweeping left to right while its height drives a synthesiser.** That is
Xenakis' UPIC (1977) — x is time, y is frequency — with the curve coming from a
measured building instead of a stylus.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

SAMPLE_RATE = 44100

#: How a view's curve is turned into sound.
VOICE_MODES = ("arc", "pulsar", "audify")

#: What part of the curve drives the pitch.
#:
#: ``signed``     the value itself. A signal that swings through zero makes the
#:                pitch swing with it — you hear the waveform, which on a 2 Hz
#:                record is a wobble, not a reading.
#: ``magnitude``  |y|. Direction is discarded; bigger motion is higher.
#: ``envelope``   |y| smoothed to a slow contour. This is what a mode shape
#:                already looks like, and why mode shapes are the one view that
#:                sounds continuous. The default for time-domain views.
SCALE_MODES = ("envelope", "magnitude", "signed")

#: Timbres available for the per-eigenfrequency markers. The user picks one per
#: mode, so f1/f2/f3 are told apart by *what kind of thing is struck*, not only
#: by pitch.
BELL_TIMBRES = ("bell", "organ", "triangle", "woodblock", "string", "off")


@dataclass
class Curve:
    """One trace on a plot: ``y`` sampled at ``x``.

    ``x`` must be increasing. It is NOT required to be uniformly spaced — a
    response spectrum is log-spaced in period — because the playhead sweeps
    linearly across the *plot*, and the renderer resamples onto that sweep.
    """

    x: np.ndarray
    y: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()
        n = min(self.x.size, self.y.size)
        self.x, self.y = self.x[:n], self.y[:n]

    @property
    def is_usable(self) -> bool:
        return (self.x.size >= 2 and np.isfinite(self.x).any()
                and np.isfinite(self.y).any())


@dataclass
class Marker:
    """A labelled vertical line on the plot — an eigenfrequency, usually.

    When the playhead crosses it, its ``timbre`` is struck.
    """

    x: float
    label: str = ""
    mode: int = -1          # 0-based eigenfrequency index, -1 = not a mode
    timbre: str = "bell"


@dataclass
class PulseView:
    """One selectable plot: what is drawn, and therefore what is heard."""

    name: str
    kind: str                       # time | spectrum | response | shape | custom
    x_label: str = ""
    y_label: str = ""
    curves: list = field(default_factory=list)      # list[Curve]
    markers: list = field(default_factory=list)     # list[Marker]
    x_log: bool = False
    #: Views whose y is an amplitude (spectra) sound better when loudness
    #: follows y as well as pitch; a mode shape or a raw time history does not.
    amplitude_follows_y: bool = False
    note: str = ""

    @property
    def is_usable(self) -> bool:
        return any(c.is_usable for c in self.curves)

    def x_range(self) -> tuple:
        lo = min(float(np.nanmin(c.x)) for c in self.curves if c.is_usable)
        hi = max(float(np.nanmax(c.x)) for c in self.curves if c.is_usable)
        return lo, (hi if hi > lo else lo + 1.0)


@dataclass
class ModalSummary:
    """What the identification found, kept beside the views."""

    frequencies_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    damping_ratios: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shapes: np.ndarray | None = None        # (n_sensors, n_modes)
    ok: bool = False
    message: str = ""


@dataclass
class PulseDataset:
    """Everything derived from ONE recording.

    Built once, then played as many times as the user likes — the point of
    storing it rather than re-analysing per playback.
    """

    source: str = ""
    fs: float = float("nan")
    duration_s: float = 0.0
    sensor_ids: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    views: dict = field(default_factory=dict)       # name -> PulseView
    modal: ModalSummary = field(default_factory=ModalSummary)
    created_utc: str = ""
    #: The measurement this was derived from, ``{axis: (n_sensors, n_samples)}``.
    #: Carried so a saved session can be RE-analysed later with different
    #: settings, not merely replayed. Empty when a dataset was loaded from disk
    #: without its raw half.
    raw: dict = field(default_factory=dict)
    #: Where this was saved, once it has been.
    saved_path: str = ""

    @property
    def has_raw(self) -> bool:
        return bool(self.raw)

    def view_names(self) -> list:
        return [n for n, v in self.views.items() if v.is_usable]

    def view(self, name: str):
        v = self.views.get(name)
        if v is None and self.views:
            v = next(iter(self.views.values()))
        return v


@dataclass
class PulseConfig:
    """Everything the UI can steer about the sound."""

    # --- the sweep -----------------------------------------------------------
    duration_s: float = 20.0        # how long the playhead takes to cross
    voice: str = "pulsar"           # see VOICE_MODES; pulsar is the default
                                    # because it is the only one that lets a 2 Hz
                                    # building be heard at 2 Hz

    # --- the arc: y -> oscillator frequency (the UPIC mapping) ---------------
    # A log map, so equal vertical distances are equal musical intervals.
    f_lo: float = 110.0
    f_hi: float = 3520.0            # five octaves, A2..A7
    #: Which part of the curve drives pitch — see SCALE_MODES. "auto" picks
    #: envelope for a signal that swings through zero and signed for a curve
    #: that does not (a spectrum, a response spectrum).
    scale: str = "auto"
    #: Envelope smoothing, as a fraction of the sweep. 2 % turns a 2 Hz
    #: oscillation into a contour without flattening the shape.
    envelope_smooth: float = 0.02
    #: Oscillator timbre for the arc voice: how many harmonics ride the sweep.
    harmonics: int = 3
    glide: float = 0.004            # s, smoothing on the frequency track

    # --- the pulsar voice (after Roads / Pietruszewski) ----------------------
    # The building's own eigenfrequency becomes the pulse REPETITION rate, which
    # is felt as rhythm; the curve drives the formant, which is heard as pitch.
    # The two are independent — that is what makes 0-20 Hz audible as itself
    # rather than transposed.
    pulsar_fundamental_hz: float = 0.0   # 0 = use mode 1 from the dataset
    pulsar_duty: float = 0.35            # d:p ratio, 0<duty<=1

    # --- audification --------------------------------------------------------
    audify_speed: float = 0.0       # 0 = choose automatically from duration

    # --- the bells at f1, f2, f3 --------------------------------------------
    bells: bool = True
    bell_gain: float = 0.9
    bell_decay_s: float = 2.2

    # --- mix -----------------------------------------------------------------
    master: float = 0.8
    arc_gain: float = 0.7
    stereo_spread: float = 0.5      # traces fanned across the stereo field

    def clamped(self) -> "PulseConfig":
        """Return a CLAMPED COPY (never mutate in place: the audio thread may
        be reading the live config while the GUI edits it)."""
        c = replace(self)
        c.duration_s = float(np.clip(c.duration_s, 1.0, 600.0))
        c.f_lo = float(np.clip(c.f_lo, 20.0, 2000.0))
        c.f_hi = float(np.clip(c.f_hi, c.f_lo * 1.5, 16000.0))
        c.harmonics = int(np.clip(c.harmonics, 1, 12))
        c.glide = float(np.clip(c.glide, 0.0, 0.2))
        c.pulsar_duty = float(np.clip(c.pulsar_duty, 0.02, 1.0))
        c.bell_gain = float(np.clip(c.bell_gain, 0.0, 2.0))
        c.bell_decay_s = float(np.clip(c.bell_decay_s, 0.05, 12.0))
        c.master = float(np.clip(c.master, 0.0, 1.0))
        c.arc_gain = float(np.clip(c.arc_gain, 0.0, 2.0))
        c.stereo_spread = float(np.clip(c.stereo_spread, 0.0, 1.0))
        if c.voice not in VOICE_MODES:
            c.voice = "arc"
        if c.scale not in SCALE_MODES and c.scale != "auto":
            c.scale = "auto"
        c.envelope_smooth = float(np.clip(c.envelope_smooth, 0.0005, 0.25))
        return c


@dataclass
class RenderResult:
    """Audio plus what the GUI needs to draw the playhead in sync."""

    audio: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    sample_rate: int = SAMPLE_RATE
    duration_s: float = 0.0
    #: Plot-x as a function of time, so the playhead is exact rather than assumed.
    x_of_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t_grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: (time_s, label) for every bell that was struck.
    strikes: list = field(default_factory=list)
    peak: float = 0.0
    message: str = ""

    def x_at(self, t: float) -> float:
        """Where the playhead sits on the plot at time ``t``."""
        if self.t_grid.size == 0:
            return 0.0
        return float(np.interp(float(t), self.t_grid, self.x_of_t))
