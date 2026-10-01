"""Structure Pulse — the measured building as a graphical score.

One recording is analysed once into a set of **views** (time history, spectrum,
response spectrum, mode shapes, displacement, torsion). Any view can then be
*played*: a playhead sweeps left to right and the curve's height drives a
synthesiser, so the plot on screen and the sound in the room are the same
object.

The lineage is explicit. The sweep is Iannis Xenakis' UPIC (1977), where x is
time and y is frequency and a drawn shape becomes sound. The ``pulsar`` voice is
Curtis Roads' pulsar synthesis as revived in Marcin Pietruszewski's nuPG, whose
useful property here is that a pulsar's repetition rate and its formant are
independent — so a 2 Hz eigenfrequency can be heard at 2 Hz, as rhythm, while
the pitch stays audible.

Qt-free throughout (guardrail G7); the GUI tab supplies the plot, the audio
device and the playhead.
"""
from .types import (BELL_TIMBRES, SAMPLE_RATE, SCALE_MODES,  # noqa: F401
                    VOICE_MODES,
                    PulseConfig, PulseDataset, PulseView, Curve,
                    Marker, ModalSummary, RenderResult)
from .spectra import peak_of_spectrum, response_spectrum, sdof_response  # noqa: F401
from .analysis import (build_dataset, build_dataset_from_arrays,  # noqa: F401
                       build_dataset_from_capture, load_channels)
from .store import (describe, list_sessions, load_dataset,  # noqa: F401
                    reanalyse, save_dataset)
from .render import render_view, strike, write_wav  # noqa: F401
from .player import BufferPlayer  # noqa: F401

__all__ = [
    "SAMPLE_RATE", "VOICE_MODES", "BELL_TIMBRES", "SCALE_MODES",
    "Curve", "Marker", "PulseView", "ModalSummary", "PulseDataset",
    "PulseConfig", "RenderResult",
    "response_spectrum", "sdof_response", "peak_of_spectrum",
    "build_dataset", "build_dataset_from_arrays",
    "build_dataset_from_capture", "load_channels",
    "save_dataset", "load_dataset", "list_sessions", "describe", "reanalyse",
    "render_view", "strike", "write_wav", "BufferPlayer",
]
