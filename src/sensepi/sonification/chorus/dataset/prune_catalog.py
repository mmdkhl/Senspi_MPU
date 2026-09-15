"""Prune the full extraction to the shipped working set, per ANIMAL TYPE.

The user assigns one animal type to each mode and to each structural case
(resonance, torsion, impact, drift, ambient). Inside a type the structure's
frequency picks the species, so every type needs enough species, spread
evenly across its own carrier range, that a MEANINGFUL stiffness loss
(~10-15 %) moves the pick to a neighbour while identification noise does not.
With ~12 species per type over ~2 octaves that is ~0.17-octave spacing.

Selection is by even carrier coverage WITHIN the type (best SNR per
log-spaced bin), not best SNR overall, which would clump wherever recordings
happen to be good.

Defaults read the preserved full extraction beside this script, so the shipped
catalog can be regenerated with NO raw audio and NO re-download.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent     # .../chorus/dataset
SRC = HERE
DST = HERE.parent / "data"                 # .../chorus/data  (the shipped set)
CATALOG_IN = "measured_catalog_full.json"
GRAINS_IN = "measured_grains_full.npz"
C_LO, C_HI = 180.0, 16000.0
MIN_SNR = 18.0
LONG_UNIT_CAP = 24

sys.path.insert(0, str(HERE.parents[2]))   # .../sensepi/sonification -> importable? no:
# import the type registry from the package without installing anything
sys.path.insert(0, str(HERE.parents[4]))   # repo/src
from sensepi.sonification.chorus.types import TYPES, type_of_group  # noqa: E402

# species per type. Cicadas stay a little smaller (they are mostly the
# resonance layer); the four original types keep their register.
QUOTAS = {
    "frogs": 15, "crickets": 15, "katydids": 15, "cicadas": 10,
    "woodpeckers": 12, "owls": 12, "bats": 12, "doves": 10,
    "squirrels": 10, "grasshoppers": 10,
}


def _write_attribution(chosen: list) -> None:
    """The shipped ATTRIBUTION.md: every download of every shipped species."""
    mp = SRC / "download_manifest.json"
    if not mp.is_file():
        return
    manifest = json.loads(mp.read_text())
    keep = {c["species"] for c in chosen}
    rows = [m for m in manifest if m.get("species") in keep]
    total = sum(m.get("bytes", 0) for m in rows)
    lines = ["# Bioacoustic sample dataset — attribution", "",
             "Source: iNaturalist (https://www.inaturalist.org), public API.",
             "Timbre source material for the SensePi bioacoustic sonification model.",
             "**Non-commercial research use.** CC-BY family licences require that the",
             "attribution below be preserved wherever this material is redistributed.",
             "", f"{len(rows)} recordings, {len(keep)} shipped species, {total/1e6:.1f} MB "
             f"of source audio. The full download list is in "
             f"`dataset/ATTRIBUTION_full.md`.", "",
             "| Species | Type | Common name | Licence | Observer | Observation |",
             "|---|---|---|---|---|---|"]
    type_of = {c["species"]: c["type"] for c in chosen}
    for m in sorted(rows, key=lambda x: (type_of.get(x["species"], ""), x["species"])):
        lines.append(f"| _{m['species']}_ | {type_of.get(m['species'], '')} | "
                     f"{m.get('common', '')} | {str(m.get('license', '')).upper()} | "
                     f"{m.get('observer', '')} | {m.get('observation', '')} |")
    (DST / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    cat = json.loads((SRC / CATALOG_IN).read_text())
    z = np.load(SRC / GRAINS_IN)
    banks: dict[str, list] = {}
    for k in z.files:
        banks.setdefault(k.rsplit("|", 1)[0], []).append(k)

    for c in cat:
        c["type"] = c.get("type") or type_of_group(c["group"])
    pool = [c for c in cat if C_LO <= c["carrier"] <= C_HI and c["species"] in banks
            and c["snr"] >= MIN_SNR and c["type"]]
    print(f"{len(cat)} catalogued -> {len(pool)} inside {C_LO:.0f}-{C_HI:.0f} Hz, "
          f"SNR>={MIN_SNR:.0f}, typed")

    chosen: list = []
    for tkey, quota in QUOTAS.items():
        fam = [c for c in pool if c["type"] == tkey]
        if not fam:
            print(f"  {tkey:<13} -- no species measured")
            continue
        lo = min(c["carrier"] for c in fam)
        hi = max(c["carrier"] for c in fam) * 1.001
        picked: list = []
        edges = np.geomspace(lo, hi, quota + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            names = {x["species"] for x in picked}
            binned = [c for c in fam if a <= c["carrier"] < b and c["species"] not in names]
            if binned:
                picked.append(max(binned, key=lambda c: c["snr"]))
        for c in sorted(fam, key=lambda c: -c["snr"]):          # top up
            if len(picked) >= quota:
                break
            if c["species"] in {x["species"] for x in picked}:
                continue
            if all(abs(np.log2(c["carrier"] / o["carrier"])) > 0.04 for o in picked):
                picked.append(c)
        car = sorted(x["carrier"] for x in picked)
        gaps = np.diff(np.log2(car)) if len(car) > 1 else np.array([0.0])
        print(f"  {tkey:<13} {len(picked):2d}/{quota}  {min(car):6.0f}-{max(car):6.0f} Hz"
              f"  ({np.log2(max(car) / min(car)):.2f} oct, median gap {np.median(gaps):.3f})")
        chosen.extend(picked)

    chosen.sort(key=lambda c: c["carrier"])
    DST.mkdir(exist_ok=True)
    (DST / "species_catalog.json").write_text(json.dumps(chosen, indent=2))
    # Package-data budget: a 300 ms hoot is 200x a katydid tooth strike, and 64
    # of them per owl would ship 25 MB. Long-unit species keep 24 units (the
    # renderer draws them at random; 24 is plenty of variety for a note that
    # long) and everything is stored as float16 — the loader upcasts to
    # float32, and 16 bits is far below the noise floor of a field recording.
    out = {}
    for c in chosen:
        keys = sorted(banks[c["species"]], key=lambda k: int(k.rsplit("|", 1)[1]))
        if c["unit_s"] > 0.05:
            keys = keys[:LONG_UNIT_CAP]
        for k in keys:
            out[k] = z[k].astype(np.float16)
        c["n_units"] = len(keys)
    (DST / "species_catalog.json").write_text(json.dumps(chosen, indent=2))
    np.savez_compressed(DST / "grain_banks.npz", **out)

    _write_attribution(chosen)

    print(f"\nkept {len(chosen)} species  "
          f"{chosen[0]['carrier']:.0f}-{chosen[-1]['carrier']:.0f} Hz")
    print(collections.Counter(c["type"] for c in chosen))
    print(f"npz: {(DST / 'grain_banks.npz').stat().st_size / 1e6:.1f} MB")
    missing = [k for k in TYPES if k not in {c["type"] for c in chosen}]
    if missing:
        print(f"WARNING types with no species: {missing}")


if __name__ == "__main__":
    main()
