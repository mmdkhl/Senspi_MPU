"""Merge a fresh build (data/processed/sonification_chorus) into the preserved
full extraction beside this script, so the archive stays complete and the raw
audio stays disposable.

Run after ``build_catalog.py``. Species already present in the full extraction
are replaced only when the new measurement has the better SNR. The download
manifest and the full attribution are extended the same way.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
NEW = REPO / "data/processed/sonification_chorus"
RAW = REPO / "data/raw/bioacoustics"

FULL_CAT = HERE / "measured_catalog_full.json"
FULL_NPZ = HERE / "measured_grains_full.npz"
FULL_MANIFEST = HERE / "download_manifest.json"
FULL_ATTR = HERE / "ATTRIBUTION_full.md"


def main() -> None:
    new_cat = json.loads((NEW / "species_catalog.json").read_text())
    new_z = np.load(NEW / "grain_banks.npz")
    old_cat = json.loads(FULL_CAT.read_text()) if FULL_CAT.is_file() else []
    old_z = np.load(FULL_NPZ) if FULL_NPZ.is_file() else None

    by_species = {c["species"]: c for c in old_cat}
    grains: dict[str, dict[str, np.ndarray]] = {}
    if old_z is not None:
        for k in old_z.files:
            grains.setdefault(k.rsplit("|", 1)[0], {})[k] = old_z[k]

    replaced = added = 0
    for c in new_cat:
        sp = c["species"]
        keys = [k for k in new_z.files if k.rsplit("|", 1)[0] == sp]
        if not keys:
            continue
        old = by_species.get(sp)
        if old is not None and float(old.get("snr", 0)) >= float(c.get("snr", 0)):
            continue
        by_species[sp] = c
        grains[sp] = {k: new_z[k] for k in keys}
        replaced += old is not None
        added += old is None

    merged = sorted(by_species.values(), key=lambda c: c["carrier"])
    FULL_CAT.write_text(json.dumps(merged, indent=2))
    # float16 halves the archive; the loader upcasts and 16 bits sits far
    # below a field recording's noise floor
    np.savez_compressed(FULL_NPZ, **{k: np.asarray(v, dtype=np.float16)
                                     for sp in by_species
                                     for k, v in grains.get(sp, {}).items()})
    print(f"catalog: {len(old_cat)} -> {len(merged)} species (+{added} new, "
          f"{replaced} re-measured)")

    # manifest + attribution
    raw_manifest = json.loads((RAW / "manifest.json").read_text()) \
        if (RAW / "manifest.json").is_file() else []
    old_manifest = json.loads(FULL_MANIFEST.read_text()) if FULL_MANIFEST.is_file() else []
    have = {m.get("file") for m in old_manifest}
    for m in raw_manifest:
        if m.get("file") not in have:
            old_manifest.append(m)
            have.add(m.get("file"))
    FULL_MANIFEST.write_text(json.dumps(old_manifest, indent=2))
    total = sum(m.get("bytes", 0) for m in old_manifest)
    species = sorted({m["species"] for m in old_manifest})
    lines = ["# Bioacoustic sample dataset — attribution (all downloads)", "",
             "Source: iNaturalist (https://www.inaturalist.org), public API.",
             "Timbre source material for the SensePi bioacoustic sonification model.",
             "**Non-commercial research use.** CC-BY family licences require that the",
             "attribution below be preserved wherever this material is redistributed.", "",
             f"{len(old_manifest)} recordings, {len(species)} species, {total/1e6:.1f} MB.",
             "", "| Species | Common name | Licence | Observer | Observation |",
             "|---|---|---|---|---|"]
    for m in sorted(old_manifest, key=lambda x: x["species"]):
        lines.append(f"| _{m['species']}_ | {m.get('common', '')} | "
                     f"{str(m.get('license', '')).upper()} | {m.get('observer', '')} | "
                     f"{m.get('observation', '')} |")
    FULL_ATTR.write_text("\n".join(lines) + "\n")
    print(f"manifest: {len(old_manifest)} recordings, {len(species)} species")


if __name__ == "__main__":
    main()
