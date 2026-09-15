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

from .types import TYPES, CastEntry, ChorusConfig, SpeciesInfo, type_of_group

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).with_name("data")
CATALOG_JSON = DATA_DIR / "species_catalog.json"
GRAIN_NPZ = DATA_DIR / "grain_banks.npz"


def carrier_for_mode(f_hz: float, cfg: ChorusConfig, type_key: str | None = None) -> float:
    """Map a structural natural frequency onto an audible carrier (log-log).

    The frequency's position inside ``f_lo..f_hi`` is projected onto the
    carrier span of ``type_key`` (the animal type's own lowest..highest
    catalogued carrier), or onto the whole catalog's span when no type is
    given. Each type therefore uses its entire species palette, so an owl-only
    register (0.3-1.5 kHz) and a katydid register (5-15 kHz) both map the full
    0-20 Hz building range.
    """
    f_lo = max(cfg.f_lo, 1e-6)
    f_hi = max(cfg.f_hi, f_lo * 1.001)
    span = np.log2(f_hi / f_lo)
    frac = np.log2(max(float(f_hz), f_lo) / f_lo) / span
    frac = float(np.clip(frac, 0.0, 1.0))
    c_lo, c_hi = type_span(type_key) if type_key else carrier_span()
    return float(c_lo * (c_hi / c_lo) ** frac)


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
                type=str(r.get("type") or type_of_group(r.get("group", ""))),
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
    """Actual carrier range present in the catalog (all types together)."""
    cat = load_catalog()
    if not cat:
        return 320.0, 13000.0
    lo = min(s.carrier for s in cat)
    hi = max(s.carrier for s in cat)
    return float(lo), float(max(hi, lo * 2))


_MIN_TYPE_SPAN_OCT = 1.0


@lru_cache(maxsize=64)
def type_span(type_key: str) -> tuple[float, float]:
    """The carrier range of one animal type, widened to at least an octave.

    Taken from the catalog, not declared: it is whatever the shipped species
    of that type actually sing at. A type with a single species gets an octave
    around it so the map still has somewhere to go.
    """
    cat = [s for s in load_catalog() if s.type == type_key]
    if not cat:
        return carrier_span()
    lo = min(s.carrier for s in cat)
    hi = max(s.carrier for s in cat)
    if np.log2(max(hi, 1e-6) / max(lo, 1e-6)) < _MIN_TYPE_SPAN_OCT:
        mid = float(np.sqrt(lo * hi))
        lo, hi = mid / np.sqrt(2.0), mid * np.sqrt(2.0)
    return float(lo), float(hi)


def available_types() -> tuple[str, ...]:
    """Type keys that have at least one species with a grain bank, in menu order."""
    banks = load_grain_banks()
    have = {s.type for s in load_catalog() if s.species in banks}
    return tuple(k for k in TYPES if k in have)


def fit_casting_map(frequencies, cfg: ChorusConfig) -> tuple[float, float]:
    """Fit the frequency ends of the map to the structure's own modes.

    The structure's modes rarely fill 0.25-20 Hz; fitting the ends spreads the
    cast across every available voice of each type rather than crowding the
    middle. The carrier ends are per type (``type_span``) and not fitted.
    """
    freqs = np.asarray(frequencies, dtype=float).ravel()
    freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
    if freqs.size == 0:
        return cfg.f_lo, cfg.f_hi
    f_lo = float(max(0.1, freqs.min() * 0.75))
    f_hi = float(max(f_lo * 2.0, freqs.max() * 1.25))
    return f_lo, f_hi


def catalog_available() -> bool:
    return bool(load_catalog()) and bool(load_grain_banks())


def _type_for_mode(cfg: ChorusConfig, m: int) -> str:
    types = tuple(getattr(cfg, "type_of_mode", ()) or ())
    if not types:
        return ""
    return str(types[min(m, len(types) - 1)])


def cast_meadow(frequencies: np.ndarray, cfg: ChorusConfig,
                previous: list | None = None) -> list[CastEntry]:
    """THE STRUCTURE CASTS THE MEADOW.

    Two rules, in order:

    1. **One animal type per mode**, chosen by the user (``cfg.type_of_mode``).
       A mode is recognised by WHAT KIND of animal it is, not merely by pitch.
    2. **Within that type, the mode's frequency picks the species**: the
       frequency is mapped onto the type's own carrier span and the nearest
       carrier wins. The damage readout survives: a softening mode moves to a
       neighbouring species of the same type, so it stays recognisably that
       mode while audibly changing.

    The case voices (resonance, torsion, drift, alarm, ambient) each come
    from their own user-chosen type, and are matched to the structure the same
    way where a frequency is meaningful.

    ``previous`` supplies the outgoing cast so a species is only swapped when
    the new candidate is meaningfully closer (``cfg.cast_hysteresis_oct``).
    Without it an unstable mode -- one wandering between 9 and 16 Hz, as
    observed live -- reshuffles the whole meadow every few seconds.
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
    good = freqs[np.isfinite(freqs) & (freqs > 0)]
    hyst = float(getattr(cfg, "cast_hysteresis_oct", 0.06))

    def pool_of(type_key: str, exclude_types: tuple = (), reuse: bool = True) -> list:
        """Unused species of the chosen type; degrade gracefully.

        A small type asked to fill every role runs out of species. Hearing
        the same owl twice is still "owls", which is what the user asked
        for, so supporting roles may reuse a species of the chosen type
        before anything reaches for another type. Leads never reuse (two
        modes singing as the same species at two rates would be unreadable)
        and never take the resonance type, which must stay distinct.
        """
        p = [s for s in avail if s.type == type_key and s.species not in used]
        if not p and reuse:
            p = [s for s in avail if s.type == type_key]
        if not p:                        # type missing entirely: fall back
            p = [s for s in avail if s.species not in used
                 and s.type not in exclude_types]
        return p

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
        return best[0] if (d_held - d_new) > hyst else held

    res_type = str(getattr(cfg, "resonance_type", "cicadas"))

    for m, fm in enumerate(freqs):
        if not np.isfinite(fm) or fm <= 0:
            continue
        tkey = _type_for_mode(cfg, m)
        pool = pool_of(tkey, exclude_types=(res_type,), reuse=False)
        if not pool:
            continue
        target = carrier_for_mode(float(fm), cfg, tkey if any(
            s.type == tkey for s in pool) else None)

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
            rpool = [s for s in avail if s.type == res_type and s.species not in used] \
                or [s for s in avail if s.type == res_type]
            if rpool:
                rtarget = carrier_for_mode(float(fm), cfg, res_type)
                s = sticky(rpool, rtarget, m, "resonance")
                if s is not None:
                    used.add(s.species)
                    out.append(CastEntry(s, m, "resonance", rtarget, float(fm)))

    # --- non-modal roles ---------------------------------------------------
    # Each matched to the structure where a frequency means something:
    # torsion follows the HIGHEST identified mode (torsional modes sit above
    # the first sway mode), drift the geometric middle of the modes.
    f_top = float(good.max()) if good.size else 6.0
    f_mid = float(np.exp(np.mean(np.log(good)))) if good.size else 4.0

    if "torsion" in cfg.enabled_roles:
        tkey = str(getattr(cfg, "torsion_type", "bats"))
        pool = pool_of(tkey)
        if pool:
            in_type = any(s.type == tkey for s in pool)
            target = carrier_for_mode(f_top, cfg, tkey if in_type else None)
            # a torsion voice wants to rattle: prefer pulsed callers, but never
            # at the cost of leaving the chosen type
            pulsed = [s for s in pool if s.pulse_rate > 28.0 and s.unit_s < 0.02]
            pick = sticky(pulsed or pool, target, -2, "torsion")
            if pick is not None:
                used.add(pick.species)
                out.append(CastEntry(pick, -2, "torsion", target, f_top))

    if "drift" in cfg.enabled_roles:
        tkey = str(getattr(cfg, "drift_type", "grasshoppers"))
        pool = pool_of(tkey)
        if pool:
            in_type = any(s.type == tkey for s in pool)
            target = carrier_for_mode(f_mid, cfg, tkey if in_type else None)
            pick = sticky(pool, target, -3, "drift")
            if pick is not None:
                used.add(pick.species)
                out.append(CastEntry(pick, -3, "drift", target, f_mid))

    if "alarm" in cfg.enabled_roles:
        # the startled call after an impact: loud, short, unmistakable. No
        # frequency to match, so the best short caller of the type.
        tkey = str(getattr(cfg, "alarm_type", "squirrels"))
        pool = pool_of(tkey)
        if pool:
            cand = [s for s in pool if s.unit_s < 0.05] or pool
            pick = max(cand, key=lambda s: s.snr)
            used.add(pick.species)
            out.append(CastEntry(pick, -4, "alarm", pick.carrier, 1.0))

    if "ambient" in cfg.enabled_roles:
        tkey = str(getattr(cfg, "ambient_type", "auto"))
        if tkey == "auto":
            amb = sorted((s for s in avail if s.species not in used and s.carrier > 3000.0),
                         key=lambda s: -s.snr)
        else:
            amb = sorted(pool_of(tkey), key=lambda s: -s.snr)
        for s in amb[: max(0, cfg.n_ambient)]:
            used.add(s.species)
            out.append(CastEntry(s, -1, "ambient", s.carrier, 2.6))

    return out


def cast_signature(cast: list[CastEntry]) -> tuple:
    """Stable identity of a cast, so the engine only re-casts when it changes."""
    return tuple((c.info.species, c.mode, c.role) for c in cast)
