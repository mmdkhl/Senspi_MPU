"""Saving and reloading a Structure Pulse session.

A saved session is a folder holding three things:

``pulse.json``
    Everything small and human-readable: provenance, sampling, the modal
    result, and the *shape* of each view — its labels, its markers, and the
    array keys its curves live under.

``curves.npz``
    The plotted arrays themselves, float32. These are the score: reload them
    and the session can be played again without re-running any analysis.

``raw.npz`` *(when the raw half was kept)*
    The measurement the views were derived from, one array per channel. This
    is what makes a saved session **re-analysable** rather than merely
    replayable — change the damping, the mode count or the placement later and
    rebuild from the same data.

JSON plus npz rather than a pickle: a pickle of these dataclasses would break
the first time a field is renamed, and would be unreadable by anything but this
version of this program. A folder of plain arrays and plain text survives both.

Qt-free (guardrail G7).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np

from .types import (Curve, Marker, ModalSummary, PulseDataset, PulseView)

logger = logging.getLogger(__name__)

FORMAT = "sensepi.structure_pulse/1"
MANIFEST = "pulse.json"
CURVES = "curves.npz"
RAW = "raw.npz"


def _safe(name: str) -> str:
    """A folder-safe fragment of a user-supplied name."""
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "").strip())
    return out.strip("-._")[:48]


def _jsonable(value):
    """numpy -> plain Python, so json never chokes on a float32."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None                     # JSON has no NaN; None round-trips
    return value


def _as_array(seq, dtype=float) -> np.ndarray:
    arr = np.asarray([np.nan if v is None else v for v in (seq or [])],
                     dtype=dtype)
    return arr.ravel()


# ---------------------------------------------------------------------- save
def save_dataset(ds: PulseDataset, base_dir, *, name: str = "",
                 include_raw: bool = True) -> Path:
    """Write ``ds`` into a new folder under ``base_dir``. Returns that folder."""
    base = Path(base_dir)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    tag = _safe(name)
    out = base / (f"{stamp}_{tag}" if tag else stamp)
    out.mkdir(parents=True, exist_ok=True)

    arrays: dict = {}
    views_json = []
    for vi, (vname, view) in enumerate(ds.views.items()):
        curves = []
        for ci, c in enumerate(view.curves):
            kx, ky = f"v{vi}_c{ci}_x", f"v{vi}_c{ci}_y"
            arrays[kx] = np.asarray(c.x, dtype=np.float32)
            arrays[ky] = np.asarray(c.y, dtype=np.float32)
            curves.append({"label": c.label, "x": kx, "y": ky})
        views_json.append({
            "name": view.name, "kind": view.kind,
            "x_label": view.x_label, "y_label": view.y_label,
            "x_log": bool(view.x_log),
            "amplitude_follows_y": bool(view.amplitude_follows_y),
            "note": view.note, "curves": curves,
            "markers": [{"x": float(m.x), "label": m.label,
                         "mode": int(m.mode), "timbre": m.timbre}
                        for m in view.markers],
            "key": vname,
        })

    shapes = ds.modal.shapes
    manifest = {
        "format": FORMAT,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "name": name,
        "source": ds.source,
        "fs": float(ds.fs) if np.isfinite(ds.fs) else None,
        "duration_s": float(ds.duration_s),
        "sensor_ids": _jsonable(ds.sensor_ids),
        "channels": list(ds.channels),
        "created_utc": ds.created_utc,
        "modal": {
            "ok": bool(ds.modal.ok),
            "message": ds.modal.message,
            "frequencies_hz": _jsonable(ds.modal.frequencies_hz),
            "damping_ratios": _jsonable(ds.modal.damping_ratios),
            "shapes": (_jsonable(np.asarray(shapes, dtype=float))
                       if shapes is not None else None),
        },
        "views": views_json,
        "raw": None,
    }

    if include_raw and ds.raw:
        raw_arrays = {str(k): np.asarray(v, dtype=np.float32)
                      for k, v in ds.raw.items() if np.size(v)}
        if raw_arrays:
            np.savez_compressed(out / RAW, **raw_arrays)
            manifest["raw"] = {"axes": sorted(raw_arrays),
                               "fs": float(ds.fs) if np.isfinite(ds.fs) else None,
                               "sensor_ids": _jsonable(ds.sensor_ids)}

    np.savez_compressed(out / CURVES, **arrays)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    ds.saved_path = str(out)
    logger.info("structure pulse: saved session to %s", out)
    return out


# ---------------------------------------------------------------------- load
def load_dataset(folder) -> PulseDataset:
    """Rebuild a dataset written by :func:`save_dataset`."""
    folder = Path(folder)
    manifest = json.loads((folder / MANIFEST).read_text())
    if str(manifest.get("format", "")).split("/")[0] != FORMAT.split("/")[0]:
        raise ValueError(f"not a Structure Pulse session: {folder}")

    arrays = {}
    cpath = folder / CURVES
    if cpath.is_file():
        with np.load(cpath) as z:
            arrays = {k: np.asarray(z[k], dtype=float) for k in z.files}

    views: dict = {}
    for v in manifest.get("views", []):
        curves = [Curve(arrays.get(c["x"], np.zeros(0)),
                        arrays.get(c["y"], np.zeros(0)), c.get("label", ""))
                  for c in v.get("curves", [])]
        markers = [Marker(x=float(m.get("x", 0.0)), label=m.get("label", ""),
                          mode=int(m.get("mode", -1)),
                          timbre=m.get("timbre", "bell"))
                   for m in v.get("markers", [])]
        view = PulseView(
            name=v.get("name", ""), kind=v.get("kind", "custom"),
            x_label=v.get("x_label", ""), y_label=v.get("y_label", ""),
            curves=curves, markers=markers, x_log=bool(v.get("x_log", False)),
            amplitude_follows_y=bool(v.get("amplitude_follows_y", False)),
            note=v.get("note", ""))
        views[v.get("key") or view.name] = view

    m = manifest.get("modal", {}) or {}
    shapes = m.get("shapes")
    modal = ModalSummary(
        frequencies_hz=_as_array(m.get("frequencies_hz")),
        damping_ratios=_as_array(m.get("damping_ratios")),
        shapes=(np.asarray(shapes, dtype=float) if shapes else None),
        ok=bool(m.get("ok", False)), message=m.get("message", ""))

    raw = {}
    rpath = folder / RAW
    if rpath.is_file():
        try:
            with np.load(rpath) as z:
                raw = {k: np.asarray(z[k], dtype=float) for k in z.files}
        except Exception:
            logger.debug("structure pulse: raw half unreadable", exc_info=True)

    fs = manifest.get("fs")
    return PulseDataset(
        source=manifest.get("source", str(folder)),
        fs=float(fs) if fs else float("nan"),
        duration_s=float(manifest.get("duration_s", 0.0)),
        sensor_ids=list(manifest.get("sensor_ids", []) or []),
        channels=list(manifest.get("channels", []) or []),
        views=views, modal=modal,
        created_utc=manifest.get("created_utc", ""),
        raw=raw, saved_path=str(folder))


def list_sessions(base_dir) -> list:
    """Saved sessions under ``base_dir``, newest first."""
    base = Path(base_dir)
    if not base.is_dir():
        return []
    found = [p.parent for p in base.glob(f"*/{MANIFEST}")]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def describe(folder) -> str:
    """A one-line label for a session folder, without loading its arrays."""
    folder = Path(folder)
    try:
        m = json.loads((folder / MANIFEST).read_text())
    except Exception:
        return folder.name
    f = [v for v in (m.get("modal", {}) or {}).get("frequencies_hz", []) or []
         if v is not None]
    bits = [folder.name]
    if m.get("name"):
        bits.append(str(m["name"]))
    if m.get("duration_s"):
        bits.append(f"{float(m['duration_s']):.0f}s")
    if f:
        bits.append("f=" + "/".join(f"{float(v):.2f}" for v in f[:3]))
    if m.get("raw"):
        bits.append("raw")
    return " · ".join(bits)


def reanalyse(folder, *, mapping: dict | None = None, n_modes: int = 3,
              damping_for_spectrum: float = 0.05) -> PulseDataset:
    """Rebuild the views from a saved session's RAW half.

    This is the reason the raw arrays are kept: change the placement map, the
    mode count or the spectrum damping and the same measurement yields a new
    set of views.
    """
    from .analysis import build_dataset_from_arrays

    ds = load_dataset(folder)
    if not ds.raw:
        raise ValueError(f"{Path(folder).name} was saved without its raw data; "
                         f"it can be replayed but not re-analysed")
    out = build_dataset_from_arrays(
        ds.raw, fs=ds.fs, sensor_ids=ds.sensor_ids, mapping=mapping,
        n_modes=n_modes, damping_for_spectrum=damping_for_spectrum,
        source=f"re-analysed · {Path(folder).name}")
    out.saved_path = str(folder)
    return out
