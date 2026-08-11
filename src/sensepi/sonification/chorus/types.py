"""Data contracts for the Bioacoustic Chorus sonification model.

Pure dataclasses + numpy. No Qt, no audio device, no GUI imports (guardrail G7).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

SAMPLE_RATE = 44100
CONTROL_HZ = 20.0
BLOCK_SIZE = 1024


@dataclass
class SpeciesInfo:
    """One catalogued species, measured from a clean field recording."""

    species: str
    group: str
    carrier: float          # Hz, the audible pitch of this voice
    echeme_rate: float      # Hz, how often it naturally repeats a chirp
    pulse_rate: float       # Hz, tooth-strikes inside one chirp
    unit_s: float           # s, length of one call unit
    snr: float
    flat: float
    license: str = ""
    observer: str = ""
    observation: str = ""
    common: str = ""

    @property
    def is_train(self) -> bool:
        """True when this species builds a chirp from a fast train of strikes."""
        return self.pulse_rate >= 28.0 and self.unit_s < 0.018


@dataclass
class CastEntry:
    """A species assigned to a structural role."""

    info: SpeciesInfo
    mode: int               # structural mode index, or -1 ambient / -2 torsion
    role: str               # lead | chorus | resonance | torsion | ambient
    target_carrier: float   # the carrier the structure asked for
    mode_freq: float        # the modal frequency driving its chirp rate


@dataclass
class ModalState:
    """Latest modal identification driving the cast."""

    frequencies_hz: np.ndarray = field(default_factory=lambda: np.array([1.0, 5.0, 10.0]))
    damping: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.01, 0.01]))
    shapes: np.ndarray | None = None          # (n_sensors, n_modes)
    fs: float = float("nan")
    t_identified: float = 0.0
    ok: bool = False
    message: str = ""


# Structural cases the model can hear, beyond plain chorusing.
CASES = ("approach", "lock", "beating", "torsion", "drift", "impact", "dropout")


@dataclass
class ControlFrame:
    """One control-rate observation of the structure."""

    t: float = 0.0
    band_energy: np.ndarray = field(default_factory=lambda: np.zeros(3))   # 0..1 per mode
    env_floor: np.ndarray = field(default_factory=lambda: np.zeros(3))
    env_global: float = 0.0
    exc_freq_hz: float = 0.0
    exc_conf: float = 0.0
    sync: np.ndarray = field(default_factory=lambda: np.zeros(3))          # 0..1 per mode
    torsion: float = 0.0
    torsion_pan: float = 0.0
    rate_hz: float = 0.0
    nan_ratio: float = 0.0
    psd_freqs: np.ndarray | None = None
    psd: np.ndarray | None = None
    # --- structural cases -------------------------------------------------
    approach: np.ndarray = field(default_factory=lambda: np.zeros(3))  # per mode 0..1
    impact: float = 0.0            # edge-triggered, decays
    impact_floor: int = -1
    beating: float = 0.0           # strength 0..1
    beat_hz: float = 0.0           # true beat frequency
    drift: np.ndarray = field(default_factory=lambda: np.zeros(2))     # adjacent pairs
    dropout: float = 0.0           # data-quality severity 0..1
    dead_sensors: tuple = ()
    # instantaneous per-sensor displacement proxy, for the animated structure
    motion: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class VizFrame:
    """What the GUI draws. Produced in the worker, consumed on the GUI thread."""

    t: float = 0.0
    frame: ControlFrame = field(default_factory=ControlFrame)
    modal: ModalState = field(default_factory=ModalState)
    cast: list[CastEntry] = field(default_factory=list)
    singing: dict = field(default_factory=dict)      # species -> individuals singing
    audio_rms: float = 0.0
    state_text: str = ""
    spectro_col: np.ndarray | None = None            # one spectrogram column
    spectro_freqs: np.ndarray | None = None


@dataclass
class ChorusConfig:
    """Everything the UI can steer. Defaults match the approved prototype."""

    # sound
    naturalism: float = 0.90
    chorus_size: float = 1.0
    density: float = 1.0
    brightness: float = 1.0
    master: float = 0.85
    # structure -> sound
    sync_strength: float = 1.0
    pitch_rise: float = 0.06
    torsion_voice: float = 1.0
    resonance_voice: float = 1.0
    damping_expression: float = 1.0
    # space
    space: float = 0.50
    depth: float = 1.0
    ambient_bed: float = 0.45
    # casting map. Structural frequency -> audible carrier, log-log.
    # autofit rescales BOTH ends to match: the frequency ends are fitted to the
    # structure's own identified modes, the carrier ends to the catalog's actual
    # range, so the modes spread across the whole species palette instead of
    # bunching in the middle of it. The fit is taken ONCE, at baseline, and then
    # held — otherwise the map would rescale along with any frequency drift and
    # the recasting-on-damage diagnostic would cancel itself out.
    autofit: bool = True
    f_lo: float = 0.25
    f_hi: float = 20.0
    c_lo: float = 320.0
    c_hi: float = 13000.0
    # identification
    n_modes: int = 3
    # Real buildings live in 0-20 Hz and mode 1 is commonly 0.3-2 Hz, so the
    # identification band must reach well below 1 Hz and up to 20.
    f_min: float = 0.25
    f_max: float = 20.0
    axis: str = "ax"
    reid_interval_s: float = 8.0
    id_window_s: float = 30.0
    fast_window_s: float = 6.0
    # Hysteresis on species selection. Identified frequencies scatter 1-2% per
    # cycle and the catalog is ~0.109 octaves apart, so without this a mode near
    # a boundary swaps species every re-identification. 0.06 oct means a rival
    # must beat the incumbent by half a slot, which suppresses jitter while
    # leaving real drift (>~7%) free to recast.
    cast_hysteresis_oct: float = 0.06
    # roles. One supporting voice, not three: same-register supporters were the
    # main source of mud and blurred the lead they were meant to support.
    n_chorus: int = 1
    n_ambient: int = 1
    family_per_mode: bool = True   # mode 1 frogs, mode 2 crickets, mode 3 katydids
    # --- clarity ----------------------------------------------------------
    duck_db: float = 9.0           # how hard a locked mode ducks the others
    lock_gesture: bool = True      # unison burst at the moment of locking
    solo_mode: int = -1            # -1 = all; 0/1/2 = solo that mode
    # --- case levels ------------------------------------------------------
    impact_voice: float = 1.0
    approach_voice: float = 1.0
    beating_voice: float = 1.0
    drift_voice: float = 1.0
    dropout_voice: float = 1.0
    enabled_roles: tuple = ("lead", "chorus", "resonance", "torsion",
                            "drift", "alarm", "ambient")

    def clamped(self) -> "ChorusConfig":
        """Return a CLAMPED COPY.

        Deliberately not in-place: the config is read by the audio thread while
        the GUI thread edits knobs, so handing out the same mutable object
        across threads would be shared mutable state (guardrail G1).
        """
        return replace(self)._clamp_in_place()

    def _clamp_in_place(self) -> "ChorusConfig":
        self.naturalism = float(np.clip(self.naturalism, 0.0, 1.0))
        self.chorus_size = float(np.clip(self.chorus_size, 0.1, 2.0))
        self.density = float(np.clip(self.density, 0.1, 3.0))
        self.brightness = float(np.clip(self.brightness, 0.3, 1.6))
        self.master = float(np.clip(self.master, 0.0, 1.0))
        self.sync_strength = float(np.clip(self.sync_strength, 0.0, 1.5))
        self.pitch_rise = float(np.clip(self.pitch_rise, 0.0, 0.25))
        self.space = float(np.clip(self.space, 0.0, 1.0))
        self.depth = float(np.clip(self.depth, 0.0, 1.0))
        self.ambient_bed = float(np.clip(self.ambient_bed, 0.0, 1.0))
        self.duck_db = float(np.clip(self.duck_db, 0.0, 24.0))
        self.damping_expression = float(np.clip(self.damping_expression, 0.0, 2.0))
        return self


# depth tiers: (gain, air-absorption lowpass Hz, reverb send)
DEPTH_TIERS = (
    (1.00, 17000.0, 0.05),
    (0.60, 11000.0, 0.20),
    (0.33, 6800.0, 0.42),
    (0.15, 4200.0, 0.68),
)

ROLE_COLORS = {
    "lead": "#5ac8fa",
    "chorus": "#7ee787",
    "resonance": "#ff9f43",
    "torsion": "#c792ea",
    "drift": "#e05c5c",
    "alarm": "#ffd93d",
    "ambient": "#6b7280",
}

# which animal family sings each mode. Register order matches mode order, so a
# mode is recognised by WHAT KIND of animal it is, not merely by pitch.
FAMILY_BY_MODE = ("anura", "gryllidae", "tettigoniidae", "oecanthinae")
FAMILY_LABEL = {
    "anura": "frogs", "gryllidae": "crickets", "tettigoniidae": "katydids",
    "oecanthinae": "tree crickets", "cicadidae": "cicadas", "acrididae": "grasshoppers",
}

ROLE_MEANING = {
    "lead": "the mode's own species; chirp rate = its natural frequency, 1:1",
    "chorus": "one supporting voice from the same family",
    "resonance": "sustained cicada buzz = that mode is LOCKED",
    "torsion": "fast chirps that pan with the sign of rotation (gz)",
    "drift": "dry rasping between two floors moving against each other",
    "alarm": "a startled call after an impact — the meadow hushes first",
    "ambient": "background meadow; never goes dead",
}

CASE_MEANING = {
    "approach": "chorus grows agitated as excitation nears a mode",
    "lock": "chorus phase-locks, pitch rises, cicada joins, other modes duck",
    "beating": "two close frequencies — the chorus pulses at the beat rate",
    "torsion": "twisting — fast chirps ping-ponging across the stereo field",
    "drift": "adjacent floors moving apart — rasping between them",
    "impact": "a knock — the whole meadow falls silent, then one startled call",
    "dropout": "lost samples or a dead sensor — the scene stutters and thins",
}
