"""Expand the bioacoustic dataset: many more species, merged into the existing manifest."""
from __future__ import annotations

import json
import time
from pathlib import Path

import fetch_dataset as F

OUT = F.OUT

# taxon_id, n_species, recordings per species
GROUPS = {
    "tettigoniidae": (48124, 34, 3),    # katydids / bush crickets
    "gryllidae":     (52884, 30, 3),    # true crickets
    "oecanthinae":   (148912, 10, 3),   # tree crickets
    "cicadidae":     (50186, 26, 3),    # cicadas
    "anura":         (20979, 34, 3),    # frogs
    "acrididae":     (47651, 20, 2),    # grasshoppers (order-level fallback)
}

existing = []
mp = OUT / "manifest.json"
if mp.exists():
    try:
        existing = json.loads(mp.read_text())
    except Exception:
        existing = []
have = {m.get("file") for m in existing}
manifest = list(existing)

for group, (tid, nsp, per) in GROUPS.items():
    gdir = OUT / group
    gdir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{group}] taxon {tid} — collecting up to {nsp} species", flush=True)
    F.PAGES = 10
    F.PER_PAGE = 100 if group != "cicadidae" else 50
    try:
        recs = F.collect(tid, nsp, per)
    except Exception as e:
        print(f"  collect failed: {e}", flush=True)
        continue
    print(f"  {len(recs)} candidates / {len({r['species'] for r in recs})} species", flush=True)
    got = 0
    for i, rec in enumerate(recs):
        ext = ".m4a" if ".m4a" in rec["url"] else ".mp3"
        safe = rec["species"].replace(" ", "_").replace("/", "-")
        dest = gdir / f"{safe}_{i:03d}{ext}"
        rel = str(dest.relative_to(OUT))
        if dest.exists():
            if rel not in have:
                rec["file"] = rel
                rec["bytes"] = dest.stat().st_size
                manifest.append(rec)
                have.add(rel)
            continue
        try:
            n = F.download(rec, dest)
        except Exception:
            continue
        if n:
            rec["file"] = rel
            rec["bytes"] = n
            manifest.append(rec)
            have.add(rel)
            got += 1
        time.sleep(0.3)
    print(f"  +{got} new files", flush=True)

mp.write_text(json.dumps(manifest, indent=2))
total = sum(m.get("bytes", 0) for m in manifest)
species = sorted({m["species"] for m in manifest})
lines = ["# Bioacoustic sample dataset — attribution", "",
         "Source: iNaturalist (https://www.inaturalist.org), public API.",
         "Timbre source material for the SensePi bioacoustic sonification model.",
         "**Non-commercial research use.** CC-BY family licences require that the",
         "attribution below be preserved wherever this material is redistributed.", "",
         f"{len(manifest)} recordings, {len(species)} species, {total/1e6:.1f} MB.", "",
         "| Species | Common name | Licence | Observer | Observation |", "|---|---|---|---|---|"]
for m in sorted(manifest, key=lambda x: x["species"]):
    lines.append(f"| _{m['species']}_ | {m.get('common','')} | {str(m.get('license','')).upper()} | "
                 f"{m.get('observer','')} | {m.get('observation','')} |")
(OUT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")
print(f"\n=== {len(manifest)} files, {len(species)} species, {total/1e6:.1f} MB", flush=True)
