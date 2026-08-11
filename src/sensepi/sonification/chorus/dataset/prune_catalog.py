"""Prune the catalog to an optimal working set.

Buildings occupy 0-20 Hz, which the casting map spreads over ~5.3 octaves of
carrier. A 3-mode cast consumes 19 species at once, and we want a MEANINGFUL
stiffness loss (~10-15%) to change the lead species, not sensor noise. That
implies roughly 0.15-octave spacing -> ~40 species, selected for EVEN carrier
coverage (best SNR within each bin) rather than best SNR overall, which would
clump wherever recordings happen to be good.
"""
import json, shutil
from pathlib import Path
import numpy as np

# Defaults read the preserved full extraction that lives beside this script, so
# the shipped catalog can be regenerated with NO raw audio and NO re-download.
HERE = Path(__file__).resolve().parent     # .../chorus/dataset
SRC = HERE
DST = HERE.parent / "data"                 # .../chorus/data  (the shipped set)
CATALOG_IN = "measured_catalog_full.json"
GRAINS_IN = "measured_grains_full.npz"
C_LO, C_HI = 250.0, 14000.0

# ONE FAMILY PER MODE. Each family owns a register and a character, so a mode is
# recognised by what KIND of animal it is, not just by pitch. Quotas give each
# family enough species that a meaningful stiffness loss still moves the pick
# within that family (the damage readout), without piling up voices.
QUOTAS = {
    "anura":         10,   # mode 1 - frogs: croaks/barks, low
    "gryllidae":     10,   # mode 2 - crickets: clean rhythmic chirps, mid
    "tettigoniidae": 10,   # mode 3 - katydids: buzzy, harsh, high
    "cicadidae":      8,   # resonance voices - sustained buzz
    "oecanthinae":    4,   # mode 4 / ambient - pure trills
    "acrididae":      4,   # drift voices - dry rasping
}

cat = json.loads((SRC / CATALOG_IN).read_text())
z = np.load(SRC / GRAINS_IN)
banks = {}
for k in z.files:
    banks.setdefault(k.rsplit("|", 1)[0], []).append(k)

pool = [c for c in cat if C_LO <= c["carrier"] <= C_HI and c["species"] in banks
        and c["snr"] >= 18]
print(f"{len(cat)} catalogued -> {len(pool)} inside {C_LO:.0f}-{C_HI:.0f} Hz, SNR>=18")

chosen = []
for family, quota in QUOTAS.items():
    fam = [c for c in pool if c["group"] == family]
    if not fam:
        continue
    lo = min(c["carrier"] for c in fam)
    hi = max(c["carrier"] for c in fam) * 1.001
    picked: list = []
    # even carrier coverage WITHIN the family: best SNR per log-spaced bin
    for a, b in zip(*(lambda e: (e[:-1], e[1:]))(np.geomspace(lo, hi, quota + 1))):
        names = {x["species"] for x in picked}
        binned = [c for c in fam if a <= c["carrier"] < b and c["species"] not in names]
        if binned:
            picked.append(max(binned, key=lambda c: c["snr"]))
    # top up from the rest of the family, keeping carriers apart
    for c in sorted(fam, key=lambda c: -c["snr"]):
        if len(picked) >= quota:
            break
        if c["species"] in {x["species"] for x in picked}:
            continue
        if all(abs(np.log2(c["carrier"] / o["carrier"])) > 0.05 for o in picked):
            picked.append(c)
    car = [x["carrier"] for x in picked]
    print(f"  {family:<15} {len(picked):2d}/{quota}  {min(car):6.0f}-{max(car):6.0f} Hz"
          f"  ({np.log2(max(car)/min(car)):.2f} oct)")
    chosen.extend(picked)

used = {c["species"] for c in chosen}

# guarantee a fast-pulsing torsion candidate survives
if not any(c.get("pulse_rate", 0) > 80 and c["unit_s"] < 0.004 for c in chosen):
    t = sorted((c for c in pool if c.get("pulse_rate", 0) > 80 and c["unit_s"] < 0.004),
               key=lambda c: -c["pulse_rate"])
    if t:
        chosen.append(t[0])

chosen.sort(key=lambda c: c["carrier"])
(DST / "species_catalog.json").write_text(json.dumps(chosen, indent=2))
np.savez_compressed(DST / "grain_banks.npz",
                    **{k: z[k] for c in chosen for k in banks[c["species"]]})

import collections
print(f"\nkept {len(chosen)} species  "
      f"{chosen[0]['carrier']:.0f}-{chosen[-1]['carrier']:.0f} Hz")
print(collections.Counter(c["group"] for c in chosen))
gaps = np.diff(np.log2([c["carrier"] for c in chosen]))
print(f"spacing: median {np.median(gaps):.3f} oct, max {gaps.max():.3f} oct")
print(f"npz: {(DST/'grain_banks.npz').stat().st_size/1e6:.1f} MB")
