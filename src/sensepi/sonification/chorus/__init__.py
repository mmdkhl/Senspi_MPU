"""Bioacoustic Chorus — a sonification model for SensePi.

Live sensor data drives a chorus of real recorded animal voices. The user
assigns an animal TYPE (frogs, crickets, katydids, cicadas, woodpeckers, owls,
bats, doves, squirrels, grasshoppers) to each structural mode and to each
structural case; within that type the mode's measured natural frequency, mapped
onto the type's own carrier range, picks the species. The mode's frequency then
becomes the chirp repetition rate one-to-one, with no transposition. Excitation controls how
many individuals sing, resonance phase-locks the chorus and adds a sustained
cicada layer, and torsion casts a fast rattling voice that pans with rotation.

Only :mod:`live_worker` imports Qt (QtCore). Everything else is pure numpy so it
stays testable and GUI-agnostic (guardrail G7).
"""
from .types import (CastEntry, ChorusConfig, ControlFrame, ModalState,  # noqa: F401
                    ROLE_COLORS, ROLE_MEANING, SpeciesInfo, VizFrame)
from .catalog import (available_types, carrier_for_mode, cast_meadow,  # noqa: F401
                      catalog_available, load_catalog, load_grain_banks, type_span)
from .types import TYPES, TYPE_ORDER, type_label  # noqa: F401
from .engine import ChorusEngine, run_offline  # noqa: F401

__all__ = [
    "CastEntry", "ChorusConfig", "ControlFrame", "ModalState", "SpeciesInfo",
    "VizFrame", "ROLE_COLORS", "ROLE_MEANING", "carrier_for_mode", "cast_meadow",
    "catalog_available", "load_catalog", "load_grain_banks", "ChorusEngine",
    "run_offline", "available_types", "type_span", "TYPES", "TYPE_ORDER", "type_label",
]
