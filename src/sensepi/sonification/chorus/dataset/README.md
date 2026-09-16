# Bioacoustic dataset — archive and tooling

Lives with the sonification code it belongs to, not in a separate top-level
folder. Everything needed to reproduce the species material used by the
**Bioacoustic Chorus** model.

The raw audio downloads (~1.1 GB) are **not** required and were deliberately
deleted on 2026-07-30. This folder is what replaced them. Regeneration of the
shipped catalog was verified byte-identical *after* that deletion, with no
`data/` folders present at all.

```
chorus/
├── data/        runtime, shipped, read by catalog.py       (12 MB)
└── dataset/     this folder: archive + build tooling        (30 MB)
```

## What the app actually needs at runtime

Only two files, and they are committed inside the package:

```
src/sensepi/sonification/chorus/data/
├── species_catalog.json     121 species, measured, typed (10 animal types)
├── grain_banks.npz          6,157 real call units, float16   (12 MB)
└── ATTRIBUTION.md            licences + observers of every shipped species
```

Nothing under `data/` is read at runtime. `catalog.py` resolves its data
directory as `Path(__file__).with_name("data")`, i.e. inside the installed
package. Verified by hiding `data/` entirely and running the suite.

## What is in this folder

| File | Purpose | Re-obtainable? |
|---|---|---|
| `download_manifest.json` | **The re-download recipe.** 855 recordings (473 insects/frogs from July + 382 for the 2026-09 types): species, licence, observer, observation URL, direct audio URL | No — iNaturalist result ordering changes over time, so a fresh query returns a *different* set. This pins the exact one used. |
| `measured_grains_full.npz` | Extracted call units for all **264** usable species (30 MB, float16) — **this is what makes the raw audio disposable** | Yes, but only by re-downloading ~180 MB and re-extracting |
| `measured_catalog_full.json` | Measurements for all 264 species (carrier, chirp rate, pulse rate, unit length, SNR) | Derived from the above |
| `ATTRIBUTION_full.md` | Attribution for all 855 downloads | No, without the manifest |
| `fetch_dataset.py` | First download pass (insects + frogs) | — |
| `fetch_more.py` | Expansion pass, merges into the manifest | — |
| `fetch_types.py` | **2026-09-15:** woodpeckers, owls, bats, doves, squirrels, true grasshoppers | — |
| `build_catalog.py` | Measures recordings and cuts call units (taxon-limited carrier windows per group) | — |
| `merge_measured.py` | Folds a fresh build into the full extraction beside this script, so the raw audio stays disposable | — |
| `prune_catalog.py` | Selects the shipped subset **per animal type** and writes the package data | — |

## The key insight (why the raw audio was disposable)

The raw MP3/M4A files were only ever needed to **extract grains from**. Once
`build_catalog.py` has measured each recording and cut its call units, the source
audio contributes nothing further. That extraction is
`measured_grains_full.npz` — 30 MB standing in for ~1.2 GB of audio plus
WAV cache, covering 264 species rather than the 121 that ship.

A further 950 MB of the deleted folder was just a decompressed WAV cache that
`build_catalog.py` regenerates from the originals in seconds.

## Regenerating the shipped catalog (no download needed)

`prune_catalog.py` defaults to reading the preserved full extraction beside it:

```bash
python src/sensepi/sonification/chorus/dataset/prune_catalog.py
```

This rewrites `src/sensepi/sonification/chorus/data/`. Verified byte-identical to
the committed files. Change `TARGET`, `C_LO` or `C_HI` at the top to ship a
different subset — for example more species for finer damage resolution, or a
narrower carrier range for small speakers.

### Animal types (since 2026-09-15)

The user assigns one **animal type** to each mode and to each structural case
(resonance, torsion, impact, drift, ambient) in the tab; inside a type the
structure's frequency picks the species. Types and their taxonomic groups are
declared once, in `chorus/types.py` (`TYPES`): frogs, crickets (Gryllidae +
Oecanthinae), katydids, cicadas, woodpeckers, owls, bats, doves, squirrels,
grasshoppers. `prune_catalog.py` writes the `type` field into every catalog
entry and fills a per-type quota.

### Choosing how many species

The count is a resolution decision, not a taste one. Each type maps the whole
0–20 Hz building range onto **its own** catalogued carrier span (an owl register
is 0.3–1.5 kHz, a katydid register 5–15 kHz), so every type needs enough species,
evenly spread over that span, that a *meaningful* stiffness loss (~10–15 %)
moves the pick to a neighbour while identification noise does not. 12–15
species over ~2 octaves gives ~0.15–0.2-octave spacing, which is the quota.

Selection is by **even carrier coverage within the type** (best SNR per
log-spaced bin), not best SNR overall — the latter clumps wherever recordings
happen to be good and leaves octave-wide gaps.

**Bats:** iNaturalist transcodes every upload to MP3/M4A (≤ 48 kHz), so nothing
ultrasonic survives. The bat recordings are heterodyne / time-expansion detector
output or audible social calls, and are measured like any other audible
recording in a 1–16 kHz window. No pitch division is applied.

## Re-downloading the raw audio (only if you must)

Needed only to add species beyond the 139 already extracted, or to re-cut units
with different parameters (longer grains, different onset threshold).

```bash
python src/sensepi/sonification/chorus/dataset/fetch_dataset.py     # writes data/raw/bioacoustics/
python src/sensepi/sonification/chorus/dataset/fetch_more.py        # expansion pass
python src/sensepi/sonification/chorus/dataset/fetch_types.py       # the 2026-09 animal types
python src/sensepi/sonification/chorus/dataset/build_catalog.py     # measure + cut units
python src/sensepi/sonification/chorus/dataset/merge_measured.py    # fold into the full extraction
python src/sensepi/sonification/chorus/dataset/prune_catalog.py     # select the shipped subset
```

To add a new animal type: add it to `TYPES` in `chorus/types.py`, add its
iNaturalist taxon id to `fetch_types.GROUPS` and its carrier window to
`build_catalog.CARRIER_BAND`, then run the four steps above. The tab's menus
are built from the types that actually have species in the shipped catalog.

`fetch_*.py` write to `data/raw/bioacoustics/` (gitignored) and expect ~180 MB of
originals plus a ~950 MB WAV cache, which can be deleted afterwards:

```bash
rm -rf data/raw/bioacoustics/_wav
```

The manifest URLs were spot-checked live and resolve, but iNaturalist may
eventually retire an asset. `download_manifest.json` records the observation URL
alongside each audio URL, so an individual recording can be located by hand.

## Licensing — non-negotiable

All material is Creative Commons, filtered at download time to CC0 / CC-BY /
CC-BY-SA / CC-BY-NC / CC-BY-NC-SA. Most is **CC-BY-NC: non-commercial use only**,
which is more restrictive than the rest of this repository.

`ATTRIBUTION.md` ships inside the package and **must travel with the grain
bank** wherever it is redistributed. `ATTRIBUTION_full.md` here covers all 473
downloads. About a dozen manifest rows read "see iNaturalist" for the observer —
those lost their metadata when an early script overwrote the manifest; re-fetch
them from the observation URL before any redistribution.

## Three measurement traps (do not reintroduce)

`build_catalog.py` had all of these and they are worth remembering:

1. **The carrier search must be taxon-limited.** Unrestricted, wind and handling
   rumble win the spectral peak and a cicada gets recorded as a 183 Hz species.
   Insects are searched above 1.5 kHz, frogs above 250 Hz.
2. **The two timescales must be measured separately.** The slow envelope
   modulation peak is the *chirp* rate; the fast one is the *tooth-strike* rate.
   Conflating them makes every species fire one lonely tick per chirp instead of
   a trill, and the whole model sounds thin.
3. **The call-unit length must be measured on the note's own timescale.** The
   insect rule (threshold the raw envelope, take the median run) finds a 1.5 ms
   tooth strike. Applied to an owl it finds the 3 ms wiggles riding on a 250 ms
   hoot, and the owl ships as a click. `UNIT_SHAPE` smooths the envelope to the
   note timescale per taxon and floors the unit length (owls 120 ms, doves
   100 ms, squirrels 20 ms, woodpeckers 15 ms, bats 3 ms).

## Hearing it without the rig

```bash
python -m sensepi.sonification.chorus.preview                      # newest recording
python -m sensepi.sonification.chorus.preview <session_dir> --map 3f-base --only owls birds
```

writes `output/sonification/previews/<recording>_<stamp>/preview_<config>.wav` and a
report of which species each eigenfrequency picked per animal type.
