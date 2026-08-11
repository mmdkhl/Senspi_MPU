"""Bioacoustic Chorus — a sonification model for SensePi.

Live sensor data drives a chorus of real recorded insect and amphibian voices.
Each structural mode casts its own species by mapping its measured natural
frequency onto the audible carrier range; the mode's frequency then becomes the
chirp repetition rate one-to-one, with no transposition. Excitation controls how
many individuals sing, resonance phase-locks the chorus and adds a sustained
cicada layer, and torsion casts a fast rattling voice that pans with rotation.

Only :mod:`live_worker` imports Qt (QtCore). Everything else is pure numpy so it
stays testable and GUI-agnostic (guardrail G7).
"""
from .types import (CastEntry, ChorusConfig, ControlFrame, ModalState,  # noqa: F401
                    ROLE_COLORS, ROLE_MEANING, SpeciesInfo, VizFrame)
from .catalog import (carrier_for_mode, cast_meadow, catalog_available,  # noqa: F401
                      load_catalog, load_grain_banks)
from .engine import ChorusEngine, run_offline  # noqa: F401

__all__ = [
    "CastEntry", "ChorusConfig", "ControlFrame", "ModalState", "SpeciesInfo",
    "VizFrame", "ROLE_COLORS", "ROLE_MEANING", "carrier_for_mode", "cast_meadow",
    "catalog_available", "load_catalog", "load_grain_banks", "ChorusEngine",
    "run_offline",
]
