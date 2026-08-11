"""Species catalog + casting.

The casting rule is the heart of the model: a structural natural frequency is
mapped logarithmically onto the audible carrier range, and the catalogued
species nearest that carrier is cast in the role.

Consequence: if the structure softens, its frequencies drop, the target carriers
drop, and the meadow changes species. The cast is a diagnostic readout.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from .types import FAMILY_BY_MODE, CastEntry, ChorusConfig, SpeciesInfo

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).with_name("data")
CATALOG_JSON = DATA_DIR / "species_catalog.json"
GRAIN_NPZ = DATA_DIR / "grain_banks.npz"


def carrier_for_mode(f_hz: float, cfg: ChorusConfig) -> float:
    """Map a structural natural frequency onto an audible carrier (log-log)."""
    f_lo = max(cfg.f_lo, 1e-6)
    f_hi = max(cfg.f_hi, f_lo * 1.001)
    span = np.log2(f_hi / f_lo)
    frac = np.log2(max(float(f_hz), f_lo) / f_lo) / span
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(cfg.c_lo * (cfg.c_hi / cfg.c_lo) ** frac)


@lru_cache(maxsize=1)
def load_catalog() -> tuple[SpeciesInfo, ...]:
    """Load the measured species catalog shipped with the package."""
    if not CATALOG_JSON.is_file():
        logger.warning("chorus: species catalog missing at %s", CATALOG_JSON)
        return ()
    try:
        raw = json.loads(CATALOG_JSON.read_text())
    except Exception:
        logger.exception("chorus: could not parse species catalog")
        return ()
    out = []
    for r in raw:
        try:
            out.append(SpeciesInfo(
                species=r["species"], group=r.get("group", ""),
                carrier=float(r["carrier"]),
                echeme_rate=float(r.get("echeme_rate", r.get("rate", 0.0)) or 0.0),
                pulse_rate=float(r.get("pulse_rate", 0.0) or 0.0),
                unit_s=float(r.get("unit_s", 0.003) or 0.003),
                snr=float(r.get("snr", 0.0) or 0.0),
                flat=float(r.get("flat", 0.0) or 0.0),
                license=str(r.get("license", "")), observer=str(r.get("observer", "")),
                observation=str(r.get("observation", "")), common=str(r.get("common", "")),
            ))
        except Exception:
            continue
    out.sort(key=lambda s: s.carrier)
    return tuple(out)


@lru_cache(maxsize=1)
def load_grain_banks() -> dict:
    """Load the call-unit banks. Keys are species names, values lists of float32."""
    if not GRAIN_NPZ.is_file():
        logger.warning("chorus: grain banks missing at %s", GRAIN_NPZ)
        return {}
    try:
        z = np.load(GRAIN_NPZ)
    except Exception:
        logger.exception("chorus: could not load grain banks")
        return {}
    banks: dict[str, list] = {}
    for key in z.files:
        sp = key.rsplit("|", 1)[0]
        arr = np.asarray(z[key], dtype=np.float32)
        if arr.size >= 8:
            banks.setdefault(sp, []).append(arr)
    return banks


def carrier_span() -> tuple[float, float]:
    """Actual carrier range present in the catalog, for autofitting the map."""
    cat = load_catalog()
    if not cat:
        return 320.0, 13000.0
    lo = min(s.carrier for s in cat)
    hi = max(s.carrier for s in cat)
    return float(lo), float(max(hi, lo * 2))


def fit_casting_map(frequencies, cfg: ChorusConfig) -> tuple[float, float, float, float]:
    """Match the two ranges to each other.

    The structure's modes rarely fill 0.5-20 Hz, and the catalog rarely fills
    320 Hz-13 kHz. Fitting both ends spreads the cast across every available
    voice rather than crowding the middle.
    """
    freqs = np.asarray(frequencies, dtype=float).ravel()
    freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
    c_lo, c_hi = carrier_span()
    if freqs.size == 0:
        return cfg.f_lo, cfg.f_hi, c_lo, c_hi
    f_lo = float(max(0.1, freqs.min() * 0.75))
    f_hi = float(max(f_lo * 2.0, freqs.max() * 1.25))
    return f_lo, f_hi, c_lo, c_hi


def catalog_available() -> bool:
    return bool(load_catalog()) and bool(load_grain_banks())


CAST_HYSTERESIS_OCT = 0.25


def cast_meadow(frequencies: np.ndarray, cfg: ChorusConfig,
                previous: list | None = None) -> list[CastEntry]:
    """THE STRUCTURE CASTS THE MEADOW.

    Two rules, in order:

    1. **One animal family per mode** (``family_per_mode``). Mode 1 is frogs,
       mode 2 crickets, mode 3 katydids. A mode is then recognised by WHAT KIND
       of animal it is, not merely by pitch — which is what was missing when
       every mode could independently land on a frog.
    2. **Within that family, nearest carrier wins.** The damage readout
       survives: a softening mode moves to a different member of the same
       family, so it stays recognisably that mode while audibly changing.

    ``previous`` supplies the outgoing cast so a species is only swapped when
    the new candidate is meaningfully closer (``CAST_HYSTERESIS_OCT``). Without
    it an unstable mode -- one wandering between 9 and 16 Hz, as observed live
    -- reshuffles the whole meadow every few seconds and nothing is learnable.
    """
    catalog = load_catalog()
    banks = load_grain_banks()
    if not catalog or not banks:
        return []
    avail = [s for s in catalog if s.species in banks and s.snr >= 18.0]
    if not avail:
        avail = [s for s in catalog if s.species in banks]
    if not avail:
        return []

    used: set[str] = set()
    out: list[CastEntry] = []
    freqs = np.asarray(frequencies, dtype=float).ravel()

    def nearest(pool, target, n=1):
        return sorted(pool, key=lambda s: abs(np.log2(max(s.carrier, 1e-6) / target)))[:n]

    prev_by_role: dict = {}
    for entry in (previous or []):
        prev_by_role[(entry.mode, entry.role)] = entry.info

    def sticky(pool, target, mode, role):
        """Keep the incumbent unless a challenger is clearly closer."""
        best = nearest(pool, target, 1)
        if not best:
            return None
        held = prev_by_role.get((mode, role))
        if held is None or held.species in used:
            return best[0]
        if not any(s.species == held.species for s in pool):
            return best[0]
        d_held = abs(np.log2(max(held.carrier, 1e-6) / target))
        d_new = abs(np.log2(max(best[0].carrier, 1e-6) / target))
        return best[0] if (d_held - d_new) > CAST_HYSTERESIS_OCT else held

    for m, fm in enumerate(freqs):
        if not np.isfinite(fm) or fm <= 0:
            continue
        target = carrier_for_mode(float(fm), cfg)

        if cfg.family_per_mode:
            family = FAMILY_BY_MODE[min(m, len(FAMILY_BY_MODE) - 1)]
            pool = [s for s in avail if s.group == family and s.species not in used]
            if not pool:                      # family exhausted: fall back gracefully
                pool = [s for s in avail
                        if s.species not in used and s.group != "cicadidae"]
        else:
            pool = [s for s in avail
                    if s.species not in used and s.group != "cicadidae"]
        if not pool:
            continue

        lead = sticky(pool, target, m, "lead")
        if lead is None:
            continue
        used.add(lead.species)
        if "lead" in cfg.enabled_roles:
            out.append(CastEntry(lead, m, "lead", target, float(fm)))

        if "chorus" in cfg.enabled_roles and cfg.n_chorus > 0:
            rest = [s for s in pool if s.species not in used]
            for s in nearest(rest, target, cfg.n_chorus):
                used.add(s.species)
                out.append(CastEntry(s, m, "chorus", target, float(fm)))

        if "resonance" in cfg.enabled_roles:
            cic = [s for s in avail if s.group == "cicadidae" and s.species not in used]
            for s in nearest(cic, target, 1):
                used.add(s.species)
                out.append(CastEntry(s, m, "resonance", target, float(fm)))

    # --- non-modal roles ---------------------------------------------------
    if "torsion" in cfg.enabled_roles:
        tors = sorted((s for s in avail if s.species not in used
                       and s.pulse_rate > 80.0 and s.unit_s < 0.004),
                      key=lambda s: -s.pulse_rate)
        if tors:
            used.add(tors[0].species)
            out.append(CastEntry(tors[0], -2, "torsion", tors[0].carrier, 6.0))

    if "drift" in cfg.enabled_roles:
        # grasshoppers rasp — the right texture for two floors grinding apart
        rasp = [s for s in avail if s.group == "acrididae" and s.species not in used]
        rasp = rasp or [s for s in avail if s.species not in used]
        if rasp:
            pick = max(rasp, key=lambda s: s.snr)
            used.add(pick.species)
            out.append(CastEntry(pick, -3, "drift", pick.carrier, 9.0))

    if "alarm" in cfg.enabled_roles:
        # the startled call after an impact: loud, short, unmistakable
        cand = [s for s in avail if s.species not in used and s.unit_s < 0.02]
        cand = cand or [s for s in avail if s.species not in used]
        if cand:
            pick = max(cand, key=lambda s: s.snr)
            used.add(pick.species)
            out.append(CastEntry(pick, -4, "alarm", pick.carrier, 1.0))

    if "ambient" in cfg.enabled_roles:
        amb = sorted((s for s in avail if s.species not in used and s.carrier > 3000.0),
                     key=lambda s: -s.snr)[: max(0, cfg.n_ambient)]
        for s in amb:
            used.add(s.species)
            out.append(CastEntry(s, -1, "ambient", s.carrier, 2.6))

    return out


def cast_signature(cast: list[CastEntry]) -> tuple:
    """Stable identity of a cast, so the engine only re-casts when it changes."""
    return tuple((c.info.species, c.mode, c.role) for c in cast)
