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
├── data/        runtime, shipped, read by catalog.py       (3.6 MB)
└── dataset/     this folder: archive + build tooling        (11 MB)
```

## What the app actually needs at runtime

Only two files, and they are committed inside the package:

```
src/sensepi/sonification/chorus/data/
├── species_catalog.json     37 species, measured
├── grain_banks.npz          2,346 real call units   (3.5 MB)
└── ATTRIBUTION.md            licences + observers
```

Nothing under `data/` is read at runtime. `catalog.py` resolves its data
directory as `Path(__file__).with_name("data")`, i.e. inside the installed
package. Verified by hiding `data/` entirely and running the suite.

## What is in this folder

| File | Purpose | Re-obtainable? |
|---|---|---|
| `download_manifest.json` | **The re-download recipe.** 473 recordings: species, licence, observer, observation URL, direct audio URL | No — iNaturalist result ordering changes over time, so a fresh query returns a *different* set. This pins the exact one used. |
| `measured_grains_full.npz` | Extracted call units for all **139** usable species (10.7 MB) — **this is what made the 1.1 GB disposable** | Yes, but only by re-downloading ~180 MB and re-extracting |
| `measured_catalog_full.json` | Measurements for all 139 species (carrier, chirp rate, pulse rate, unit length, SNR) | Derived from the above |
| `ATTRIBUTION_full.md` | Attribution for all 473 downloads | No, without the manifest |
| `fetch_dataset.py` | First download pass | — |
| `fetch_more.py` | Expansion pass, merges into the manifest | — |
| `build_catalog.py` | Measures recordings and cuts call units | — |
| `prune_catalog.py` | Selects the shipped subset and writes the package data | — |

## The key insight (why the raw audio was disposable)

The raw MP3/M4A files were only ever needed to **extract grains from**. Once
`build_catalog.py` has measured each recording and cut its call units, the source
audio contributes nothing further. That extraction is
`measured_grains_full.npz` — 10.7 MB standing in for 1.1 GB, covering 139
species rather than the 37 that ship.

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

### Choosing how many species

The count is a resolution decision, not a taste one. Buildings occupy 0–20 Hz,
which the casting map spreads over ~5.3 octaves of carrier. A 3-mode cast
consumes 19 species at once (3 × [lead + 3 chorus + cicada] + torsion + 3
ambient). For the recasting diagnostic to fire on a *meaningful* stiffness loss
(~10–15 %) rather than on identification noise, leads want roughly 0.15-octave
spacing → about 35 species. 37 ship, at median 0.109-octave spacing.

Selection is by **even carrier coverage** (best SNR per log-spaced bin), not best
SNR overall — the latter clumps wherever recordings happen to be good and leaves
octave-wide gaps.

## Re-downloading the raw audio (only if you must)

Needed only to add species beyond the 139 already extracted, or to re-cut units
with different parameters (longer grains, different onset threshold).

```bash
python src/sensepi/sonification/chorus/dataset/fetch_dataset.py     # writes data/raw/bioacoustics/
python src/sensepi/sonification/chorus/dataset/fetch_more.py        # expansion pass
python src/sensepi/sonification/chorus/dataset/build_catalog.py     # measure + cut units
python src/sensepi/sonification/chorus/dataset/prune_catalog.py     # select the shipped subset
```

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

## Two measurement traps (do not reintroduce)

`build_catalog.py` had both of these and they are worth remembering:

1. **The carrier search must be taxon-limited.** Unrestricted, wind and handling
   rumble win the spectral peak and a cicada gets recorded as a 183 Hz species.
   Insects are searched above 1.5 kHz, frogs above 250 Hz.
2. **The two timescales must be measured separately.** The slow envelope
   modulation peak is the *chirp* rate; the fast one is the *tooth-strike* rate.
   Conflating them makes every species fire one lonely tick per chirp instead of
   a trill, and the whole model sounds thin.
