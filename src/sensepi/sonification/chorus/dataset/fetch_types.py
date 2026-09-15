"""Fetch the recordings for the NEW animal types (2026-09-15) from iNaturalist.

Adds woodpeckers, owls, bats, doves, squirrels and true grasshoppers to the
dataset, merging into the same manifest / attribution files as
``fetch_dataset.py`` and ``fetch_more.py``. Same licence filter, same politeness.

Bats: iNaturalist transcodes every upload to MP3/M4A (<= 48 kHz), so nothing
ultrasonic survives. What is there is already audible - heterodyne or
time-expansion detector output, or audible social calls - and is measured like
any other audible recording. No pitch division is needed or possible.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_dataset as F  # noqa: E402

OUT = F.OUT

# group -> (iNaturalist taxon_id, n_species, recordings per species)
GROUPS = {
    "picidae":     (17599, 24, 3),   # woodpeckers: drums + calls
    "strigidae":   (19728, 24, 3),   # owls: low hoots
    "chiroptera":  (40268, 24, 3),   # bats: detector recordings / social calls
    "columbidae":  (2715, 20, 3),    # doves and pigeons: coos
    "sciuridae":   (45933, 20, 3),   # squirrels and chipmunks: chip trains (family)
    "acrididae":   (47649, 20, 3),   # true grasshoppers (family, not order)
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mp = OUT / "manifest.json"
    existing = json.loads(mp.read_text()) if mp.exists() else []
    have = {m.get("file") for m in existing}
    manifest = list(existing)
    for group, (tid, nsp, per) in GROUPS.items():
        gdir = OUT / group
        gdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{group}] taxon {tid} - collecting up to {nsp} species", flush=True)
        F.PAGES = 12
        F.PER_PAGE = 100
        try:
            recs = F.collect(tid, nsp, per)
        except Exception as e:
            print(f"  collect failed: {e}", flush=True)
            continue
        print(f"  {len(recs)} candidates / {len({r['species'] for r in recs})} species",
              flush=True)
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
            except Exception as e:
                print(f"    skip {rec['species']}: {e}", flush=True)
                continue
            if n:
                rec["file"] = rel
                rec["bytes"] = n
                manifest.append(rec)
                have.add(rel)
                got += 1
            time.sleep(0.35)
        print(f"  +{got} new files", flush=True)
        mp.write_text(json.dumps(manifest, indent=2))

    total = sum(m.get("bytes", 0) for m in manifest)
    species = sorted({m["species"] for m in manifest})
    lines = ["# Bioacoustic sample dataset - attribution", "",
             "Source: iNaturalist (https://www.inaturalist.org), public API.",
             "Timbre source material for the SensePi bioacoustic sonification model.",
             "**Non-commercial research use.** CC-BY family licences require that the",
             "attribution below be preserved wherever this material is redistributed.", "",
             f"{len(manifest)} recordings, {len(species)} species, {total/1e6:.1f} MB.", "",
             "| Species | Common name | Licence | Observer | Observation |",
             "|---|---|---|---|---|"]
    for m in sorted(manifest, key=lambda x: x["species"]):
        lines.append(f"| _{m['species']}_ | {m.get('common', '')} | "
                     f"{str(m.get('license', '')).upper()} | {m.get('observer', '')} | "
                     f"{m.get('observation', '')} |")
    (OUT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")
    print(f"\n=== {len(manifest)} files, {len(species)} species, {total/1e6:.1f} MB",
          flush=True)


if __name__ == "__main__":
    main()
