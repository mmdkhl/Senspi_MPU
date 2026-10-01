"""Recording -> :class:`PulseDataset`.

Runs the analyses the app already owns over one finished recording and packages
every result as a *plottable curve*, because in Structure Pulse the plot is the
score: whatever can be drawn can be played.

Pure numpy/scipy, no Qt (guardrail G7). The modal identification is the same
``sensepi.analysis.modal.identify_modes`` the Spectrum tab uses, so the
frequencies heard here are the frequencies shown there.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ...analysis import sensor_layout as slayout
from .spectra import response_spectrum
from .types import (BELL_TIMBRES, PulseDataset, PulseView, Curve, Marker,
                    ModalSummary)

logger = logging.getLogger(__name__)

AXES = ("ax", "ay", "az", "gx", "gy", "gz")

#: The views a recording normally yields, in the order they are built. Offered
#: by the UI before the first live cycle has landed, so the user can choose what
#: to sonify instead of waiting a whole window for an empty list to fill.
PREVIEW_VIEWS = ("Time history", "Spectrum", "Response spectrum", "Mode shapes",
                 "Displacement", "Torsion (gz)")

#: Which timbre each eigenfrequency is struck with by default. f1 is the slow,
#: heavy mode, so it gets the heavy voice.
DEFAULT_TIMBRES = ("bell", "organ", "triangle", "woodblock")

#: Identification band. Matches the rest of the app (buildings live in 0-20 Hz).
F_MIN = 0.25
F_MAX = 20.0


def load_channels(session_dir: Path) -> dict:
    """Every channel a Smart Recording holds, as ModalSessions keyed by axis."""
    from ...dataio import modal_session_loader as msl

    out = {}
    for axis in AXES:
        try:
            sess = msl.load_session(Path(session_dir), axis=axis)
        except Exception:
            continue
        if (getattr(sess, "success", False) and sess.data.size
                and np.isfinite(sess.fs)
                and np.any(np.abs(np.nan_to_num(sess.data)) > 1e-12)):
            out[axis] = sess
    return out


def _identify(data: np.ndarray, fs: float, sensor_ids, mapping,
              n_modes: int = 3) -> ModalSummary:
    """Modal identification, with the shaker row excluded if the map says so."""
    from ...analysis.modal import identify_modes

    rows = list(range(data.shape[0]))
    layout = slayout.layout_from_mapping(mapping)
    if layout.is_valid and sensor_ids:
        try:
            resp = list(layout.response_rows(list(sensor_ids)))
            if resp and len(resp) < len(sensor_ids):
                rows = resp
        except Exception:
            pass
        n_modes = layout.max_modes(n_modes)
    try:
        res = identify_modes(np.nan_to_num(data[rows, :]), float(fs),
                             f_min=F_MIN, f_max=min(F_MAX, fs / 2 - 0.5),
                             n_modes=int(max(1, n_modes)))
    except Exception as exc:
        logger.debug("structure_pulse: identification failed: %s", exc)
        return ModalSummary(ok=False, message=f"identification failed: {exc}")
    if not getattr(res, "success", False):
        return ModalSummary(ok=False, message=getattr(res, "message", "no modes"))
    shapes = np.asarray(res.mode_shapes_sensor, dtype=float)
    if shapes.ndim == 2 and shapes.shape[0] != len(rows):
        shapes = shapes.T
    return ModalSummary(
        frequencies_hz=np.asarray(res.frequencies_hz, dtype=float).ravel(),
        damping_ratios=np.asarray(res.damping_ratios, dtype=float).ravel(),
        shapes=shapes if shapes.ndim == 2 else None,
        ok=True, message=getattr(res, "message", "ok"),

    )


def _mode_markers(modal: ModalSummary, as_period: bool = False) -> list:
    """f1, f2, f3 as plot markers, each with its own timbre."""
    out = []
    for i, f in enumerate(np.asarray(modal.frequencies_hz, dtype=float).ravel()):
        if not np.isfinite(f) or f <= 0:
            continue
        x = (1.0 / f) if as_period else float(f)
        out.append(Marker(x=float(x), label=f"f{i + 1}",
                          mode=i,
                          timbre=DEFAULT_TIMBRES[min(i, len(DEFAULT_TIMBRES) - 1)]))
    return out


def build_dataset(session_dir, *, mapping: dict | None = None,
                  n_modes: int = 3, damping_for_spectrum: float = 0.05,
                  max_seconds: float | None = None) -> PulseDataset:
    """Analyse one Smart Recording from disk and return every view it offers."""
    session_dir = Path(session_dir)
    sessions = load_channels(session_dir)
    if not sessions:
        return PulseDataset(source=str(session_dir),
                            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()))
    first = next(iter(sessions.values()))
    fs = float(first.fs)
    ids = list(getattr(first, "sensor_ids", []) or [])
    chans = {}
    for axis, sess in sessions.items():
        arr = np.nan_to_num(np.asarray(sess.data, dtype=float))
        if max_seconds and max_seconds > 0:
            arr = arr[:, -int(min(arr.shape[1], max_seconds * fs)):]
        chans[axis] = arr
    return build_dataset_from_arrays(
        chans, fs=fs, sensor_ids=ids, mapping=mapping, n_modes=n_modes,
        damping_for_spectrum=damping_for_spectrum, source=str(session_dir))


def build_dataset_from_arrays(chans: dict, *, fs: float, sensor_ids=None,
                              mapping: dict | None = None, n_modes: int = 3,
                              damping_for_spectrum: float = 0.05,
                              source: str = "", keep_raw: bool = True
                              ) -> PulseDataset:
    """The core: plain per-axis arrays -> every view.

    Every other entry point funnels through here, so a recording read from
    disk, a live capture and a re-analysed saved session all take an identical
    path and cannot drift apart.
    """
    chans = {str(k): np.nan_to_num(np.asarray(v, dtype=float))
             for k, v in (chans or {}).items() if np.size(v)}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not chans or not np.isfinite(fs) or fs <= 0:
        return PulseDataset(source=source, created_utc=stamp)

    layout = slayout.layout_from_mapping(mapping)
    axis = layout.channel if layout.is_valid else "ax"
    data = chans.get(axis)
    if data is None:
        data = chans.get("ax", next(iter(chans.values())))
    if data.ndim == 1:
        data = data[None, :]
    ids = list(sensor_ids or range(1, data.shape[0] + 1))[:data.shape[0]]

    ds = PulseDataset(
        source=source, fs=float(fs), duration_s=data.shape[1] / float(fs),
        sensor_ids=ids, channels=sorted(chans), created_utc=stamp,
        raw=(dict(chans) if keep_raw else {}))
    ds.modal = _identify(data, float(fs), ids, mapping, n_modes=n_modes)
    ds.views = _build_views(ds, data, float(fs), ids, chans, layout,
                            damping_for_spectrum)
    return ds


def build_dataset_from_capture(capture, *, seconds: float,
                               mapping: dict | None = None, n_modes: int = 3,
                               damping_for_spectrum: float = 0.05) -> PulseDataset:
    """Same as :func:`build_dataset`, but from the LIVE accumulator.

    ``capture`` is ``RecorderController.snapshot_modal_capture``, which is
    documented thread-safe, so this runs on a worker thread. The raw arrays are
    kept on the dataset so the capture can be saved and re-analysed later.
    """
    chans, fs, ids = {}, float("nan"), []
    for axis in AXES:
        try:
            sess = capture(axis=axis, last_seconds=seconds)
        except Exception:
            continue
        data = getattr(sess, "data", None)
        if (data is None or not np.size(data)
                or not np.isfinite(getattr(sess, "fs", float("nan")))
                or not np.any(np.abs(np.nan_to_num(data)) > 1e-12)):
            continue
        chans[axis] = np.nan_to_num(np.asarray(data, dtype=float))
        if not np.isfinite(fs):
            fs = float(sess.fs)
            ids = list(getattr(sess, "sensor_ids", []) or [])
    return build_dataset_from_arrays(
        chans, fs=fs, sensor_ids=ids, mapping=mapping, n_modes=n_modes,
        damping_for_spectrum=damping_for_spectrum,
        source=f"live capture · {seconds:.0f} s")


def _build_views(ds: PulseDataset, data: np.ndarray, fs: float, ids: list,
                 chans: dict, layout, damping: float) -> dict:
    """``chans`` maps axis -> plain (n_sensors, n_samples) array."""
    views: dict = {}
    t = np.arange(data.shape[1]) / fs
    floor_of = {}
    if layout.is_valid:
        floor_of = {int(s): int(layout.story_map.get(int(s), 0)) for s in ids}

    def row_label(i: int) -> str:
        sid = int(ids[i]) if i < len(ids) else i + 1
        fl = floor_of.get(sid)
        return f"S{sid}" + (f" · floor {fl}" if fl is not None else "")

    # --- 1. time history -----------------------------------------------------
    views["Time history"] = PulseView(
        name="Time history", kind="time", x_label="time (s)",
        y_label="acceleration (m/s²)",
        curves=[Curve(t, data[i], row_label(i)) for i in range(data.shape[0])],
        note="The record as measured. Sweeping it reads the motion itself.")

    # --- 2. spectrum (the identification spectrum, as the Spectrum tab shows) -
    spec_f, spec_y = _identification_spectrum(data, fs)
    if spec_f.size:
        views["Spectrum"] = PulseView(
            name="Spectrum", kind="spectrum", x_label="frequency (Hz)",
            y_label="amplitude",
            curves=[Curve(spec_f, spec_y, "identification spectrum")],
            markers=_mode_markers(ds.modal),
            amplitude_follows_y=True,
            note="Peaks are the building's eigenfrequencies; each is struck as "
                 "the playhead crosses it.")

    # --- 3. response spectrum (Sa vs period) ---------------------------------
    try:
        ref = data.mean(axis=0) if data.shape[0] > 1 else data[0]
        rs = response_spectrum(ref, fs, damping=damping)
        views["Response spectrum"] = PulseView(
            name="Response spectrum", kind="response", x_label="period (s)",
            y_label="Sa (m/s²)",
            curves=[Curve(rs["periods"], rs["Sa"], f"Sa, ζ={damping:.0%}")],
            x_log=True, amplitude_follows_y=True,
            note="Peak response of a single-degree-of-freedom oscillator at "
                 "each period, swept from stiff to soft. No bells: the response "
                 "itself is the sound.")
    except Exception:
        logger.debug("structure_pulse: response spectrum failed", exc_info=True)

    # --- 4. mode shapes ------------------------------------------------------
    shapes = ds.modal.shapes
    if shapes is not None and getattr(shapes, "ndim", 0) == 2:
        curves = []
        n_rows = shapes.shape[0]
        floors = [floor_of.get(int(ids[i]), i + 1) if i < len(ids) else i + 1
                  for i in range(n_rows)]
        order = np.argsort(floors)
        x = np.asarray([0] + [floors[i] for i in order], dtype=float)
        for m in range(shapes.shape[1]):
            col = np.asarray(shapes[:, m], dtype=float)[order]
            denom = max(float(np.abs(col).max()), 1e-12)
            curves.append(Curve(x, np.concatenate([[0.0], col / denom]),
                                f"mode {m + 1}"))
        views["Mode shapes"] = PulseView(
            name="Mode shapes", kind="shape", x_label="floor",
            y_label="normalised ordinate",
            curves=curves,
            note="Swept from the base upward: you hear the deflected shape "
                 "climb the building.")

    # --- 5. displacement -----------------------------------------------------
    try:
        from ...digital_twin import motion as twin_motion

        disp = np.vstack([twin_motion.integrate_twice(data[i], fs)
                          for i in range(data.shape[0])])
        views["Displacement"] = PulseView(
            name="Displacement", kind="time", x_label="time (s)",
            y_label="displacement (m)",
            curves=[Curve(t, disp[i], row_label(i)) for i in range(disp.shape[0])],
            note="Acceleration integrated twice, band-limited. Relative motion, "
                 "not absolute position.")
    except Exception:
        logger.debug("structure_pulse: displacement failed", exc_info=True)

    # --- 6. torsion, when the gyro is there ----------------------------------
    gz = chans.get("gz")
    if gz is not None and np.size(gz):
        g = np.nan_to_num(np.asarray(gz, dtype=float))
        n = min(g.shape[1], t.size)
        views["Torsion (gz)"] = PulseView(
            name="Torsion (gz)", kind="time", x_label="time (s)",
            y_label="yaw rate (deg/s)",
            curves=[Curve(t[:n], g[i, :n], row_label(i))
                    for i in range(g.shape[0])],
            note="Rotation measured directly by each gyro.")
    return views


#: Views that are a function of time and can therefore be shown on any channel.
TIME_KINDS = ("Time history", "Displacement", "Torsion (gz)")


def rebuild_time_view(ds: PulseDataset, base_name: str, axis: str,
                      mapping: dict | None = None) -> PulseView | None:
    """A time-domain view on a channel the user picked.

    Possible only because the dataset keeps its raw half: swapping ax for ay
    costs an interpolation, not a re-analysis.
    """
    arr = (ds.raw or {}).get(str(axis))
    if arr is None or not np.size(arr) or not np.isfinite(ds.fs) or ds.fs <= 0:
        return None
    arr = np.nan_to_num(np.asarray(arr, dtype=float))
    if arr.ndim == 1:
        arr = arr[None, :]
    fs = float(ds.fs)
    t = np.arange(arr.shape[1]) / fs
    layout = slayout.layout_from_mapping(mapping)
    floors = ({int(s): int(layout.story_map.get(int(s), 0))
               for s in ds.sensor_ids} if layout.is_valid else {})

    def label(i: int) -> str:
        sid = int(ds.sensor_ids[i]) if i < len(ds.sensor_ids) else i + 1
        fl = floors.get(sid)
        return f"S{sid}" + (f" · floor {fl}" if fl is not None else "")

    gyro = str(axis).startswith("g")
    if base_name == "Displacement":
        if gyro:
            return None
        from ...digital_twin import motion as twin_motion

        rows = np.vstack([twin_motion.integrate_twice(arr[i], fs)
                          for i in range(arr.shape[0])])
        y_label, note = "displacement (m)", ("Acceleration integrated twice, "
                                             "band-limited. Relative motion.")
    else:
        rows = arr
        y_label = ("yaw rate (deg/s)" if gyro else "acceleration (m/s²)")
        note = f"The {axis} record as measured."
    return PulseView(
        name=f"{base_name} · {axis}", kind="time", x_label="time (s)",
        y_label=y_label,
        curves=[Curve(t, rows[i], label(i)) for i in range(rows.shape[0])],
        note=note)


def _identification_spectrum(data: np.ndarray, fs: float) -> tuple:
    """The averaged amplitude spectrum in the structural band."""
    try:
        from ...analysis.fft import compute_fft

        freqs, mags = compute_fft(data, fs, axis=-1)
        freqs = np.asarray(freqs, dtype=float).ravel()
        mags = np.asarray(mags, dtype=float)
        if mags.ndim == 2:
            mags = mags.mean(axis=0)
        sel = (freqs >= F_MIN) & (freqs <= min(F_MAX, fs / 2))
        if sel.sum() >= 4:
            return freqs[sel], mags[sel]
    except Exception:
        logger.debug("structure_pulse: spectrum failed", exc_info=True)
    return np.zeros(0), np.zeros(0)
