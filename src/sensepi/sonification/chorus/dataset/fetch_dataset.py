"""
Download a small, species-diverse, CC-licensed bioacoustic dataset from iNaturalist.

Licensing: only CC0 / CC-BY / CC-BY-SA / CC-BY-NC / CC-BY-NC-SA are requested.
Every file's licence + observer is recorded in manifest.json and ATTRIBUTION.md,
which is what CC-BY family licences require. Non-commercial use only.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# repo root is four levels up: chorus/dataset -> chorus -> sonification -> sensepi -> src
OUT = Path(__file__).resolve().parents[4].parent / "data/raw/bioacoustics"
UA = "SensePi-structural-monitoring-research/1.0 (academic; contact peshawa93@gmail.com)"
OK_LIC = {"cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa"}

# group -> (taxon_id, how many species, how many recordings per species)
GROUPS = {
    "tettigoniidae": (48124, 14, 2),   # katydids / bush crickets  (the Schonblick voice)
    "gryllidae":     (52884, 12, 2),   # true crickets
    "oecanthinae":   (148912, 5, 2),   # tree crickets (pure-tone trillers)
    "cicadidae":     (50186, 10, 2),   # cicadas (sustained resonance drones)
    "anura":         (20979, 14, 2),   # frogs (low-carrier chorusers)
}
MAX_BYTES = 3_500_000     # skip anything unusually large
PAGES = 6
PER_PAGE = 100


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def collect(taxon_id, n_species, per_species):
    """Gather candidate recordings, spread across as many species as possible."""
    by_species: dict[str, list] = {}
    for page in range(1, PAGES + 1):
        q = urllib.parse.urlencode({
            "taxon_id": taxon_id, "sounds": "true",
            "license": "cc-by-nc,cc-by,cc0,cc-by-sa,cc-by-nc-sa",
            "quality_grade": "research", "per_page": PER_PAGE, "page": page,
            "order_by": "votes",
        })
        try:
            d = get(f"https://api.inaturalist.org/v1/observations?{q}")
        except Exception as e:
            print(f"    page {page} failed: {e}")
            break
        res = d.get("results", [])
        if not res:
            break
        for r in res:
            taxon = r.get("taxon") or {}
            name = taxon.get("name")
            if not name or (taxon.get("rank") not in ("species", "subspecies")):
                continue
            for s in (r.get("sounds") or []):
                lic = (s.get("license_code") or "").lower()
                url = s.get("file_url")
                if lic not in OK_LIC or not url or ".m4a" not in url and ".mp3" not in url:
                    continue
                bucket = by_species.setdefault(name, [])
                if len(bucket) >= per_species:
                    continue
                user = (r.get("user") or {})
                bucket.append({
                    "species": name,
                    "common": taxon.get("preferred_common_name") or "",
                    "url": url,
                    "license": lic,
                    "observer": user.get("name") or user.get("login") or "unknown",
                    "observation": f"https://www.inaturalist.org/observations/{r.get('id')}",
                })
        time.sleep(1.1)                      # be polite to the API
        if len(by_species) >= n_species * 3:
            break
    # prefer species with the most material, cap at n_species
    ranked = sorted(by_species.items(), key=lambda kv: -len(kv[1]))[:n_species]
    return [rec for _, recs in ranked for rec in recs]


def download(rec, dest: Path):
    req = urllib.request.Request(rec["url"], headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        n = int(r.headers.get("Content-Length") or 0)
        if n > MAX_BYTES:
            return 0
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or len(data) < 4000:
        return 0
    dest.write_bytes(data)
    return len(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, total = [], 0
    for group, (tid, nsp, per) in GROUPS.items():
        gdir = OUT / group
        gdir.mkdir(exist_ok=True)
        print(f"\n[{group}] taxon {tid} — collecting…")
        recs = collect(tid, nsp, per)
        print(f"  {len(recs)} candidates across {len({r['species'] for r in recs})} species")
        got = 0
        for i, rec in enumerate(recs):
            ext = ".m4a" if ".m4a" in rec["url"] else ".mp3"
            safe = rec["species"].replace(" ", "_").replace("/", "-")
            dest = gdir / f"{safe}_{i:03d}{ext}"
            if dest.exists():
                got += 1
                continue
            try:
                n = download(rec, dest)
            except Exception as e:
                print(f"    skip {rec['species']}: {e}")
                continue
            if n:
                rec["file"] = str(dest.relative_to(OUT))
                rec["bytes"] = n
                manifest.append(rec)
                total += n
                got += 1
            time.sleep(0.35)
        print(f"  downloaded {got} files")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = ["# Bioacoustic sample dataset — attribution",
             "",
             "Source: iNaturalist (https://www.inaturalist.org), downloaded via the public API.",
             "Used as timbre source material for the SensePi bioacoustic sonification model.",
             "**Non-commercial research use.** Licences are per-file below; CC-BY family",
             "licences require the attribution recorded here to be preserved.",
             "", f"{len(manifest)} recordings, "
             f"{len({m['species'] for m in manifest})} species, "
             f"{total/1e6:.1f} MB.", "",
             "| Species | Common name | Licence | Observer | Observation |",
             "|---|---|---|---|---|"]
    for m in sorted(manifest, key=lambda x: (x["species"])):
        lines.append(f"| _{m['species']}_ | {m['common']} | {m['license'].upper()} | "
                     f"{m['observer']} | {m['observation']} |")
    (OUT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")

    print(f"\n=== {len(manifest)} files, "
          f"{len({m['species'] for m in manifest})} species, {total/1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
