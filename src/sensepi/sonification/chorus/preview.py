"""Render a sensor recording through the Bioacoustic Chorus engine, offline.

For when a live test is not possible: takes a Smart Recording session folder
(``output/sensor_recordings/<host>/mpu/<stamp>``), runs the identical
identify → cast → render path the live tab uses, and writes one WAV per cast
configuration plus a report saying which eigenfrequency was identified and
which species each animal type picked for it.

Qt-free (guardrail G7); uses the same loader as the tabs.

    python -m sensepi.sonification.chorus.preview                 # newest recording
    python -m sensepi.sonification.chorus.preview <session_dir> --map 3f-torsion
"""
from __future__ import annotations

import argparse
import json
import time
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np

from ...analysis import sensor_layout as slayout
from ...config.app_config import AppPaths
from ...dataio import modal_session_loader as msl
from .catalog import (available_types, carrier_for_mode, cast_meadow, load_catalog,
                      type_span)
from .engine import run_offline
from .types import SAMPLE_RATE, TYPES, ChorusConfig, type_label

AXES = ("ax", "ay", "az", "gx", "gy", "gz")

#: Placement maps that can be named on the command line (the tab reads the
#: Settings map; a recording made before placement was stored in the meta
#: needs it stated).
MAPS = {
    "3f-torsion": {"n_floors": 3, "axis": "x", "placements": [
        {"sensor_id": 1, "floor": 1, "cell": "B2"}, {"sensor_id": 2, "floor": 2, "cell": "B2"},
        {"sensor_id": 3, "floor": 3, "cell": "A3"}, {"sensor_id": 4, "floor": 3, "cell": "C1"}]},
    "3f-base": {"n_floors": 3, "axis": "x", "placements": [
        {"sensor_id": 1, "floor": 0, "cell": "B2"}, {"sensor_id": 2, "floor": 1, "cell": "B2"},
        {"sensor_id": 3, "floor": 2, "cell": "B2"}, {"sensor_id": 4, "floor": 3, "cell": "B2"}]},
    "none": None,
}

#: The cast configurations rendered by default: name -> overrides.
CONFIGS = {
    "default": dict(type_of_mode=("frogs", "crickets", "katydids")),
    "owls": dict(type_of_mode=("owls", "owls", "owls")),
    "woodpeckers": dict(type_of_mode=("woodpeckers", "woodpeckers", "woodpeckers")),
    "birds": dict(type_of_mode=("owls", "woodpeckers", "doves"), alarm_type="doves"),
    "night": dict(type_of_mode=("bats", "katydids", "cicadas"), torsion_type="bats",
                  ambient_type="bats"),
    "squirrels_grasshoppers": dict(type_of_mode=("squirrels", "grasshoppers", "crickets"),
                                   drift_type="grasshoppers"),
}


def load_recording(session_dir: Path) -> dict:
    """Every channel the recording has, as ModalSessions keyed by axis."""
    out = {}
    for axis in AXES:
        try:
            sess = msl.load_session(session_dir, axis=axis)
        except Exception:
            continue
        if getattr(sess, "success", False) and sess.data.size and np.isfinite(sess.fs):
            if np.any(np.abs(np.nan_to_num(sess.data)) > 1e-9):
                out[axis] = sess
    return out


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def species_by_frequency(freqs, cfg: ChorusConfig) -> dict:
    """For every type: which species each eigenfrequency would pick as lead."""
    table = {}
    for key in available_types():
        c = replace(cfg, type_of_mode=(key,) * max(1, len(freqs)))
        leads = {e.mode: e for e in cast_meadow(np.asarray(freqs, float), c) if e.role == "lead"}
        table[key] = [
            (float(f), leads[i].info.species if i in leads else "",
             float(leads[i].info.carrier) if i in leads else float("nan"),
             float(carrier_for_mode(float(f), c, key)))
            for i, f in enumerate(freqs)]
    return table


def sweep_table(cfg: ChorusConfig, points=(0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 15, 20)) -> dict:
    """Species per type across the whole building range, one frequency at a time."""
    table = {}
    for key in available_types():
        row = []
        for f in points:
            c = replace(cfg, type_of_mode=(key,))
            leads = [e for e in cast_meadow(np.array([f]), c) if e.role == "lead"]
            row.append((f, leads[0].info.species if leads else ""))
        table[key] = row
    return table


def render_previews(session_dir: Path, out_dir: Path, mapping: dict | None,
                    configs: dict = CONFIGS, seconds: float | None = None,
                    log=print) -> Path:
    chans = load_recording(session_dir)
    if "ax" not in chans and "ay" not in chans:
        raise SystemExit(f"no acceleration channel found in {session_dir}")
    layout = slayout.layout_from_mapping(mapping)
    axis = layout.channel if layout.is_valid else "ax"
    main = chans.get(axis) or chans.get("ax") or chans["ay"]
    dur = seconds or float(main.data.shape[1] / main.fs)
    gz = chans.get("gz")
    extra = {k: v for k, v in chans.items() if k not in (axis, "gz")}
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"recording {session_dir.name}: sensors {main.sensor_ids}, fs {main.fs:.1f} Hz, "
        f"{main.data.shape[1] / main.fs:.0f} s, channels {sorted(chans)}; "
        f"driven axis {axis}; map {'none' if mapping is None else layout.describe()}")

    base = ChorusConfig()
    base.sensor_map = mapping
    base.axis = axis
    report = {"recording": str(session_dir), "sensors": list(main.sensor_ids),
              "fs": float(main.fs), "seconds": dur, "channels": sorted(chans),
              "axis": axis, "map": mapping, "renders": []}
    lines = [f"# Chorus preview — {session_dir.name}", "",
             f"Recording: `{session_dir}`  ", f"Sensors: {list(main.sensor_ids)} · fs {main.fs:.1f} Hz · "
             f"{dur:.0f} s · channels {', '.join(sorted(chans))}  ",
             f"Driven axis: `{axis}` · placement: {'none' if mapping is None else layout.describe()}",
             "", "Rendered offline through the same identify → cast → render path as the "
             "live tab. Species are chosen by the identified eigenfrequency inside the "
             "chosen animal type; the tables below show exactly which.", ""]

    freqs_seen = None
    for name, over in configs.items():
        cfg = replace(base, **over)
        t0 = time.time()
        audio, frames, engine = run_offline(main, gz, cfg, duration_s=dur,
                                            channels=extra or None)
        wav = out_dir / f"preview_{name}.wav"
        write_wav(wav, audio)
        modal = engine.modal
        freqs = [float(f) for f in np.asarray(modal.frequencies_hz).ravel()]
        damp = [float(d) for d in np.asarray(modal.damping).ravel()]
        if freqs_seen is None and modal.ok:
            freqs_seen = freqs
        cast = [{"role": e.role, "f": e.mode_freq, "type": e.info.type,
                 "species": e.info.species, "common": e.info.common,
                 "carrier": e.info.carrier, "target": e.target_carrier}
                for e in engine.cast]
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        ch_seen = sorted({c for fr in frames for c in fr.frame.channels})
        report["renders"].append({"name": name, "wav": str(wav), "config": over,
                                  "identified_hz": freqs, "damping": damp,
                                  "cast": cast, "peak": peak, "rms": rms,
                                  "channels_seen": ch_seen})
        log(f"  {name:<24} {wav.name:<32} f = {', '.join(f'{f:.2f}' for f in freqs)} Hz  "
            f"peak {peak:.2f} rms {rms:.3f}  ({time.time() - t0:.0f} s)")
        lines += [f"## `{wav.name}` — {name}", "",
                  f"f1..f{len(freqs)} = {', '.join(f'{f:.2f}' for f in freqs)} Hz · "
                  f"ζ = {', '.join(f'{d * 100:.1f}%' for d in damp)} · extra channels heard: "
                  f"{', '.join(ch_seen) or 'none'} · peak {peak:.2f} · rms {rms:.3f}", "",
                  "| role | f (Hz) | type | species | carrier (Hz) | target (Hz) |",
                  "|---|---|---|---|---|---|"]
        for c in sorted(cast, key=lambda c: (c["role"] != "lead", c["f"])):
            lines.append(f"| {c['role']} | {c['f']:.2f} | {c['type']} | _{c['species']}_"
                         f"{' (' + c['common'] + ')' if c['common'] else ''} | "
                         f"{c['carrier']:.0f} | {c['target']:.0f} |")
        lines.append("")

    # the demonstration the user asked for: each type, each eigenfrequency, its species
    freqs_demo = freqs_seen or [1.9, 8.4, 12.8]
    fitted = replace(base)
    from .catalog import fit_casting_map
    fitted.f_lo, fitted.f_hi = fit_casting_map(np.array(freqs_demo), fitted)
    byf = species_by_frequency(freqs_demo, fitted)
    lines += ["## Which species each eigenfrequency picks, per animal type", "",
              f"Identified f = {', '.join(f'{f:.2f}' for f in freqs_demo)} Hz; map fitted to "
              f"{fitted.f_lo:.2f}–{fitted.f_hi:.2f} Hz (autofit). Each type maps that range "
              f"onto its own carrier span, so every type answers with a different species "
              f"for f1, f2 and f3.", "",
              "| type | span (Hz) | " + " | ".join(f"f{i + 1} = {f:.2f} Hz" for i, f in enumerate(freqs_demo)) + " |",
              "|---|---|" + "---|" * len(freqs_demo)]
    for key, rows in byf.items():
        lo, hi = type_span(key)
        cells = " | ".join(f"_{sp}_ ({car:.0f} Hz ← {tgt:.0f})" if sp else "—"
                           for _f, sp, car, tgt in rows)
        lines.append(f"| {type_label(key)} | {lo:.0f}–{hi:.0f} | {cells} |")
    report["species_by_frequency"] = byf

    sweep = sweep_table(fitted)
    pts = next(iter(sweep.values()))
    lines += ["", "## Across the whole building range", "",
              "One eigenfrequency at a time, default map 0.25–20 Hz: the species each type "
              "would sing it as. A softening structure walks left along a row.", "",
              "| type | " + " | ".join(f"{f:g} Hz" for f, _ in pts) + " |",
              "|---|" + "---|" * len(pts)]
    for key, row in sweep.items():
        lines.append(f"| {type_label(key)} | " + " | ".join(
            f"_{sp.split()[0][:12]}_" if sp else "—" for _f, sp in row) + " |")
    report["sweep"] = sweep

    n_types = {k: sum(1 for s in load_catalog() if s.type == k) for k in TYPES}
    lines += ["", f"Catalog: {sum(n_types.values())} species — " +
              ", ".join(f"{type_label(k)} {n}" for k, n in n_types.items()), ""]
    (out_dir / "preview_report.md").write_text("\n".join(lines))
    (out_dir / "preview_report.json").write_text(json.dumps(report, indent=2, default=str))
    return out_dir


def _newest_recording(root: Path) -> Path | None:
    try:
        sessions = msl.list_sessions(root)
    except Exception:
        sessions = []
    return sessions[0] if sessions else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session", nargs="?", help="recording folder (default: newest)")
    ap.add_argument("--out", help="output folder (default: output/sonification/previews/<stamp>)")
    ap.add_argument("--map", default="3f-torsion", choices=sorted(MAPS),
                    help="placement map to assume (default 3f-torsion)")
    ap.add_argument("--seconds", type=float, default=None, help="render only the first N s")
    ap.add_argument("--only", nargs="*", help="subset of configurations to render")
    args = ap.parse_args(argv)

    paths = AppPaths()
    session = Path(args.session) if args.session else _newest_recording(paths.sensor_recordings)
    if session is None or not session.exists():
        print("no recording found; pass a session folder")
        return 2
    out = Path(args.out) if args.out else (
        paths.sonification_output / "previews" / f"{session.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    configs = CONFIGS if not args.only else {k: CONFIGS[k] for k in args.only if k in CONFIGS}
    render_previews(session, out, MAPS[args.map], configs, args.seconds)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
