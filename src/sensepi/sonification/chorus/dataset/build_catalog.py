"""
Convert the downloaded dataset to WAV, MEASURE every species, and cut a grain bank.

The point: species -> structural-mode assignment should be driven by measured
acoustics (carrier frequency, natural call rate, tonality), not by my guesses.

Each species contributes "call units" sized to its own biology:
  a katydid tooth-strike is ~3 ms, a frog note is ~200 ms. Both are stored as
  units; the renderer just fires whatever unit that species makes at the rate the
  structure demands.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal as sg
from scipy.io import wavfile

_REPO = Path(__file__).resolve().parents[4].parent
SRC = _REPO / "data/raw/bioacoustics"
WAV = SRC / "_wav"
OUT = _REPO / "data/processed/sonification_chorus"
SR = 44100
RNG = np.random.default_rng(3)


def convert_all():
    WAV.mkdir(exist_ok=True)
    files = [p for g in SRC.iterdir() if g.is_dir() and g.name != "_wav"
             for p in sorted(g.glob("*.m4a")) + sorted(g.glob("*.mp3"))]
    out = []
    for p in files:
        w = WAV / f"{p.parent.name}__{p.stem}.wav"
        if not w.exists():
            r = subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1",
                                str(p), str(w)], capture_output=True)
            if r.returncode != 0 or not w.exists():
                continue
        out.append(w)
    return out


# insects never call this low; a sub-kHz "peak" is wind/handling/traffic rumble
CARRIER_BAND = {
    "anura":         (250, 9000),      # frogs genuinely go low
    "gryllidae":     (1500, 20000),
    "tettigoniidae": (1500, 20000),
    "oecanthinae":   (1500, 20000),
    "cicadidae":     (1500, 20000),
}


def measure(path: Path, group: str):
    """Acoustic fingerprint of one recording."""
    fs, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < fs * 1.5:
        return None
    x -= x.mean()
    m = np.abs(x).max()
    if m < 1e-4:
        return None
    x /= m

    # search for the carrier only where this taxon can actually call
    clo, chi = CARRIER_BAND.get(group, (1000, 20000))
    f, P = sg.welch(x, fs, nperseg=8192)
    band = (f >= clo) & (f <= min(chi, fs / 2 * 0.95))
    if band.sum() < 8:
        return None
    fb, Pb = f[band], P[band]
    carrier = float(fb[np.argmax(Pb)])

    # quality: how far the call stands above the noise floor
    snr = float(10 * np.log10(Pb.max() / (np.median(Pb) + 1e-20)))

    # tonality around the carrier
    near = (fb > carrier * 0.55) & (fb < carrier * 1.8)
    flat = float(np.exp(np.mean(np.log(Pb[near] + 1e-20))) / (np.mean(Pb[near]) + 1e-20))

    # envelope in the species' own band -> call unit length + repetition rate
    lo, hi = max(120, carrier * 0.55), min(carrier * 1.9, fs / 2 * 0.95)
    sos = sg.butter(4, [lo, hi], "bandpass", fs=fs, output="sos")
    xb = sg.sosfilt(sos, x)
    env = np.abs(sg.hilbert(xb))
    env = sg.decimate(env, 4, ftype="fir")
    fse = fs / 4
    env = np.maximum(env, 0)

    # TWO timescales, measured separately (this is the whole point):
    #   echeme_rate = how often a chirp/call repeats   (slow, the phrase rhythm)
    #   pulse_rate  = tooth-strikes inside one chirp   (fast, the buzzy timbre)
    fe, Pe = sg.welch(env - env.mean(), fse, nperseg=min(len(env), int(fse * 4)))
    slow = (fe > 0.6) & (fe < 28)
    fast = (fe >= 28) & (fe < 320)
    echeme_rate = float(fe[slow][np.argmax(Pe[slow])]) if slow.sum() else 0.0
    pulse_rate = float(fe[fast][np.argmax(Pe[fast])]) if fast.sum() else 0.0
    # is the fast layer actually present, or is this a single-note caller?
    pulse_strength = float(Pe[fast].max() / (Pe[slow].max() + 1e-20)) \
        if (fast.sum() and slow.sum()) else 0.0
    rate = echeme_rate

    thr = np.percentile(env, 88)
    ab = env > thr
    on = np.flatnonzero(np.diff(ab.astype(int)) == 1)
    off = np.flatnonzero(np.diff(ab.astype(int)) == -1)
    if len(on) < 3 or len(off) < 3:
        return None
    if off[0] < on[0]:
        off = off[1:]
    n = min(len(on), len(off))
    dur = (off[:n] - on[:n]) / fse
    dur = dur[dur > 0.0008]
    if not len(dur):
        return None
    unit_s = float(np.median(dur))
    duty = float(np.mean(ab))
    return dict(carrier=carrier, snr=snr, flat=flat, rate=rate,
                echeme_rate=echeme_rate, pulse_rate=pulse_rate,
                pulse_strength=pulse_strength,
                unit_s=unit_s, duty=duty, fs=fs, n=len(x))


def cut_units(path, info, n_units=64):
    """Cut this species' own call units out of a clean recording."""
    fs, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    x -= x.mean()
    x /= max(np.abs(x).max(), 1e-9)
    c = info["carrier"]
    lo, hi = max(120, c * 0.5), min(c * 2.1, fs / 2 * 0.95)
    sos = sg.butter(4, [lo, hi], "bandpass", fs=fs, output="sos")
    xb = sg.sosfilt(sos, x)
    env = np.abs(sg.hilbert(xb))
    envs = sg.savgol_filter(env, max(5, int(fs * 0.0006) | 1), 2)

    # unit length from this species' own biology, padded for the ring-out
    L = int(np.clip(info["unit_s"] * 1.8, 0.002, 0.45) * fs)
    thr = np.percentile(envs, 90)
    on = np.flatnonzero((envs[1:] > thr) & (envs[:-1] <= thr))
    RNG.shuffle(on)
    units, used = [], []
    fade = int(min(L * 0.15, fs * 0.004))
    win = np.ones(L)
    win[:max(2, fade // 3)] = np.linspace(0, 1, max(2, fade // 3))
    win[-fade:] = np.linspace(1, 0, fade) ** 0.6
    for s in on:
        s0 = max(0, s - int(fs * 0.0006))
        if s0 + L >= len(xb) or any(abs(s0 - u) < L * 0.6 for u in used[-60:]):
            continue
        g = xb[s0:s0 + L] * win
        pk = np.abs(g).max()
        if pk < 0.02:
            continue
        units.append((g / pk).astype(np.float32))
        used.append(s0)
        if len(units) >= n_units:
            break
    if len(units) < 4:
        return None
    if fs != SR:
        units = [sg.resample_poly(u, SR, fs).astype(np.float32) for u in units]
    return units


def main():
    print("converting…")
    wavs = convert_all()
    print(f"  {len(wavs)} wav files")

    manifest = json.loads((SRC / "manifest.json").read_text())
    lic = {}
    for m in manifest:
        key = f"{Path(m['file']).parent.name}__{Path(m['file']).stem}"
        lic[key] = m

    per_species: dict[str, dict] = {}
    print("\nmeasuring…")
    for w in wavs:
        group_g = w.stem.split("__", 1)[0]
        try:
            info = measure(w, group_g)
        except Exception:
            info = None
        if not info or info["snr"] < 12:
            continue
        group, stem = w.stem.split("__", 1)
        species = stem.rsplit("_", 1)[0].replace("_", " ")
        rec = per_species.get(species)
        if rec is None or info["snr"] > rec["info"]["snr"]:
            per_species[species] = {"info": info, "wav": w, "group": group,
                                    "meta": lic.get(w.stem, {})}

    print(f"  {len(per_species)} species passed the quality gate (SNR >= 12 dB)")

    print("\ncutting call units…")
    catalog, banks = [], {}
    for sp, rec in sorted(per_species.items()):
        try:
            units = cut_units(rec["wav"], rec["info"])
        except Exception:
            units = None
        if not units:
            continue
        i = rec["info"]
        catalog.append(dict(species=sp, group=rec["group"], carrier=i["carrier"],
                            rate=i["rate"], echeme_rate=i["echeme_rate"],
                            pulse_rate=i["pulse_rate"],
                            pulse_strength=i["pulse_strength"],
                            unit_s=i["unit_s"], flat=i["flat"],
                            snr=i["snr"], duty=i["duty"], n_units=len(units),
                            license=rec["meta"].get("license", ""),
                            observer=rec["meta"].get("observer", ""),
                            observation=rec["meta"].get("observation", "")))
        banks[sp] = units

    catalog.sort(key=lambda c: c["carrier"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "species_catalog.json").write_text(json.dumps(catalog, indent=2))
    np.savez_compressed(OUT / "grain_banks.npz",
                        **{f"{s}|{k}": u for s, us in banks.items()
                           for k, u in enumerate(us)})

    print(f"\n{'species':<32}{'group':<14}{'carrier':>9}{'echeme':>8}{'pulse':>8}"
          f"{'pstr':>7}{'unit':>8}{'snr':>7}{'n':>4}")
    for c in catalog:
        print(f"{c['species']:<32}{c['group']:<14}{c['carrier']:8.0f}H"
              f"{c['echeme_rate']:7.1f}H{c['pulse_rate']:7.0f}H"
              f"{c['pulse_strength']:7.2f}{c['unit_s']*1000:7.1f}m"
              f"{c['snr']:7.1f}{c['n_units']:4d}")
    print(f"\n{len(catalog)} usable species -> species_catalog.json + grain_banks.npz")


if __name__ == "__main__":
    main()
