"""PC-side Smart Recording writer orchestration (Front C, SF-2/3/4/5).

Writes the live MPU stream to byte-compatible per-sensor CSVs **off the GUI thread**:
``RecorderController`` enqueues sample batches with a non-blocking :meth:`submit`
(cheap, GUI-thread-safe, G4) and a dedicated writer thread drains the queue to disk
via :class:`StreamingCsvWriter`. It is **additive** — it does NOT modify
``SensorIngestWorker`` or the live read loop, so live streaming is unaffected.

Audit add-on (2026-06-23)
-------------------------
The recording behaviour (probe -> cap/decimate -> main CSV) is unchanged. On top of
it, every *incoming* sample (BEFORE decimation) is measured so the ``.csv.meta.json``
sidecar can answer the **frequency question** (PRE-3: requested vs actually delivered
Hz) and general sensor-health questions:

- per-sensor **true incoming rate** from the sample timestamps (independent of the
  probe's ``actual_hz``), plus the recorded rate after decimation;
- inter-sample dt statistics (mean/median/std/min/max/p05/p95/p99), a dt histogram,
  and gap/dropout detection — overruns show up as a long-tailed / multi-modal dt;
- per-channel min/max/mean/std (ax, ay, az, gx, gy, gz) to spot clipping / dead axes;
- PC-wall vs Pi-timestamp duration (clock drift).

Optionally (``save_raw``) the FULL un-decimated stream (all six channels) is written
to a per-recording ``audit_<stamp>/`` subfolder so the data can be audited offline.
The main recording files and the locked CSV schema are untouched.

Pure (threading + dataio + config + numpy); no Qt, no SSH — unit-testable headless.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..config.log_paths import build_log_file_paths
from .csv_writer import SENSOR_CSV_HEADER, StreamingCsvWriter, write_recording_meta

logger = logging.getLogger(__name__)

# Audit-only raw capture schema (SUPERSET of the locked main schema, with all six
# channels). It lives in its own subfolder and is never read by the loaders, so it is
# free to differ from SENSOR_CSV_HEADER.
_RAW_CSV_HEADER: tuple[str, ...] = (
    "timestamp_ns", "t_s", "sensor_id", "ax", "ay", "az", "gx", "gy", "gz")
_CHANNELS: tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")
AUDIT_SCHEMA_VERSION = 1


class _ChannelAccum:
    """Running min/max/mean/std for one channel (no full-array storage).

    Only **finite** values feed the statistics, so a channel the Pi does not deliver
    (e.g. ``az`` arrives as NaN — the stream is ax/ay/gz only) reports null stats with
    ``n_finite=0`` instead of emitting ``Infinity``/``NaN`` (which is invalid JSON).
    """

    __slots__ = ("n", "nf", "s", "ss", "mn", "mx")

    def __init__(self) -> None:
        self.n = 0          # total samples seen
        self.nf = 0         # finite samples (the ones that count)
        self.s = 0.0
        self.ss = 0.0
        self.mn = math.inf
        self.mx = -math.inf

    def add(self, v: float) -> None:
        self.n += 1
        v = float(v)
        if not math.isfinite(v):
            return
        self.nf += 1
        self.s += v
        self.ss += v * v
        if v < self.mn:
            self.mn = v
        if v > self.mx:
            self.mx = v

    def summary(self) -> dict:
        if self.nf == 0:
            return {"min": None, "max": None, "mean": None, "std": None,
                    "n": self.n, "n_finite": 0}
        mean = self.s / self.nf
        var = max(0.0, self.ss / self.nf - mean * mean)
        return {"min": round(self.mn, 6), "max": round(self.mx, 6),
                "mean": round(mean, 6), "std": round(math.sqrt(var), 6),
                "n": self.n, "n_finite": self.nf}


class SmartRecorder:
    """Drain live sample batches to per-sensor CSVs with a PC clock + true rate.

    Parameters mirror the SF design: ``start_dt`` is the PC wall-clock at probe-end
    (SF-3), ``actual_hz`` the probe-measured rate written to meta (SF-2 fixes the
    PRE-3 "lie"), and ``decimate`` (>=1) drops every Nth sample on the WRITER path
    only (SF-2, when the device delivers more than requested). The live display is
    never decimated — only the recording.

    Audit knobs (additive; do not change the recording itself):
    - ``audit`` (default True): measure the incoming stream and write the rich meta.
    - ``save_raw`` (default True): also persist the full un-decimated stream (all six
      channels) under ``audit_<stamp>/`` for offline analysis.
    - ``host_name`` / ``probe_seconds``: provenance copied into the meta.
    """

    def __init__(
        self,
        *,
        out_dir: Path,
        session_name: Optional[str],
        sensor_ids: Sequence[int],
        start_dt: datetime,
        requested_hz: float,
        actual_hz: float,
        decimate: int = 1,
        host_name: Optional[str] = None,
        probe_seconds: Optional[float] = None,
        probe_combined_hz: Optional[float] = None,
        audit: bool = True,
        save_raw: bool = True,
        sensor_map: Optional[dict] = None,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._session_name = session_name
        self._sensor_ids = [int(s) for s in sensor_ids]
        self._start_dt = start_dt
        self._requested_hz = float(requested_hz)
        # actual_hz is now the PER-SENSOR probe estimate (the fix); probe_combined_hz
        # is the raw all-sensors-interleaved number kept for transparency/audit trail.
        self._actual_hz = float(actual_hz)
        self._probe_combined_hz = (float(probe_combined_hz)
                                   if probe_combined_hz is not None else None)
        self._decimate = max(1, int(decimate))
        self._host_name = host_name
        self._probe_seconds = probe_seconds
        self._audit = bool(audit)
        self._save_raw = bool(save_raw)
        # Where each sensor physically sat during this recording. Written
        # into the sidecars so a session is self-describing: a CSV alone
        # cannot say which floor sensor 3 was on.
        self._sensor_map = dict(sensor_map) if sensor_map else None
        self._queue: "queue.SimpleQueue" = queue.SimpleQueue()
        self._writers: dict[int, StreamingCsvWriter] = {}
        self._data_paths: dict[int, Path] = {}
        self._meta_paths: dict[int, Path] = {}
        self._counts: dict[int, int] = {}
        self._written: dict[int, int] = {}
        self._thread: Optional[threading.Thread] = None
        self._active = False
        # Audit accumulators (per sensor), captured for EVERY incoming sample.
        self._ts_ns: dict[int, list[int]] = {}
        self._t_s_first: dict[int, float] = {}
        self._t_s_last: dict[int, float] = {}
        self._chan: dict[int, dict[str, _ChannelAccum]] = {}
        self._raw_writers: dict[int, StreamingCsvWriter] = {}
        self._audit_dir: Optional[Path] = None

    @property
    def data_paths(self) -> dict[int, Path]:
        return dict(self._data_paths)

    @property
    def audit_dir(self) -> Optional[Path]:
        return self._audit_dir

    def start(self) -> None:
        stamp = self._start_dt.strftime("%Y-%m-%d_%H-%M-%S")
        if self._audit and self._save_raw:
            self._audit_dir = self._out_dir / f"audit_{stamp}"
        for sid in self._sensor_ids:
            paths = build_log_file_paths(
                "mpu", self._session_name, sid, self._start_dt, "csv", self._out_dir)
            self._writers[sid] = StreamingCsvWriter(paths.data_path).open()
            self._data_paths[sid] = paths.data_path
            self._meta_paths[sid] = paths.meta_path
            self._counts[sid] = 0
            self._written[sid] = 0
            if self._audit:
                self._ts_ns[sid] = []
                self._chan[sid] = {c: _ChannelAccum() for c in _CHANNELS}
            if self._audit_dir is not None:
                raw_path = self._audit_dir / f"raw_S{sid}_{stamp}.csv"
                self._raw_writers[sid] = StreamingCsvWriter(raw_path, header=_RAW_CSV_HEADER).open()
        self._active = True
        self._thread = threading.Thread(
            target=self._drain, name="smart-recorder", daemon=True)
        self._thread.start()

    def submit(self, batch: Iterable) -> None:
        """Non-blocking enqueue from the GUI thread (G4-safe; O(1))."""
        if self._active:
            self._queue.put_nowait(list(batch))

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # stop sentinel
                break
            for sample in item:
                try:
                    self._write_sample(sample)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("SmartRecorder: failed to write a sample")

    def _write_sample(self, s) -> None:
        sid = getattr(s, "sensor_id", None)
        if sid is None:
            return
        sid = int(sid)
        writer = self._writers.get(sid)
        if writer is None:
            return
        ts_ns = int(getattr(s, "timestamp_ns", 0))
        t_s = (float(s.t_s) if getattr(s, "t_s", None) is not None
               else ts_ns * 1e-9)

        # ── Audit: measure EVERY incoming sample (before decimation). ───────────
        if self._audit:
            self._ts_ns[sid].append(ts_ns)
            if sid not in self._t_s_first:
                self._t_s_first[sid] = t_s
            self._t_s_last[sid] = t_s
            acc = self._chan[sid]
            for c in _CHANNELS:
                acc[c].add(getattr(s, c, 0.0))
        raw = self._raw_writers.get(sid)
        if raw is not None:
            raw.write_row([ts_ns, round(t_s, 6), sid,
                           float(getattr(s, "ax", 0.0)), float(getattr(s, "ay", 0.0)),
                           float(getattr(s, "az", 0.0)), float(getattr(s, "gx", 0.0)),
                           float(getattr(s, "gy", 0.0)), float(getattr(s, "gz", 0.0))])

        # ── Main recording (UNCHANGED): decimate on the writer path only. ───────
        c = self._counts.get(sid, 0)
        self._counts[sid] = c + 1
        if self._decimate > 1 and (c % self._decimate) != 0:
            return  # SF-2: decimate on the recording path only
        writer.write_sample(
            ts_ns, round(t_s, 6), sid,
            float(getattr(s, "ax", 0.0)), float(getattr(s, "ay", 0.0)),
            float(getattr(s, "gz", 0.0)))
        self._written[sid] = self._written.get(sid, 0) + 1

    # ── audit computation ───────────────────────────────────────────────────────
    def _timing_audit(self, sid: int) -> dict:
        """Per-sensor timing + rate + channel stats from the captured incoming stream."""
        ts = np.asarray(self._ts_ns.get(sid, []), dtype=np.int64)
        recv = int(ts.size)
        written = int(self._written.get(sid, 0))
        out: dict = {
            "samples_received": recv,       # incoming (pre-decimation) — the truth
            "samples_written": written,     # after decimation (== rows in the CSV)
            "decimate": self._decimate,
            "rate_requested_hz": round(self._requested_hz, 3) if self._requested_hz else None,
            "rate_probe_hz": round(self._actual_hz, 3) if self._actual_hz else None,  # per-sensor
            "rate_probe_combined_hz": (round(self._probe_combined_hz, 3)
                                       if self._probe_combined_hz else None),
        }
        if recv >= 2:
            span_s = float(ts[-1] - ts[0]) / 1e9
            out["duration_pi_s"] = round(span_s, 4)
            t_first = self._t_s_first.get(sid)
            t_last = self._t_s_last.get(sid)
            if t_first is not None and t_last is not None:
                out["duration_t_s"] = round(t_last - t_first, 4)
            rate_in = (recv - 1) / span_s if span_s > 0 else None
            out["rate_incoming_hz"] = round(rate_in, 3) if rate_in else None
            out["rate_written_hz"] = (round((written - 1) / span_s, 3)
                                      if (span_s > 0 and written >= 2) else None)
            if rate_in and self._requested_hz:
                # < 1.0 => the device under-delivers vs the request (PRE-3).
                out["delivered_vs_requested"] = round(rate_in / self._requested_hz, 4)
            if rate_in and self._actual_hz:
                # How far the 5 s probe's actual_hz was from the real per-sensor rate.
                out["probe_vs_incoming"] = round(self._actual_hz / rate_in, 4)

            dt_ms = np.diff(ts) / 1e6
            med = float(np.median(dt_ms))
            out["dt_ms"] = {
                "mean": round(float(np.mean(dt_ms)), 4),
                "median": round(med, 4),
                "std": round(float(np.std(dt_ms)), 4),
                "min": round(float(np.min(dt_ms)), 4),
                "max": round(float(np.max(dt_ms)), 4),
                "p05": round(float(np.percentile(dt_ms, 5)), 4),
                "p95": round(float(np.percentile(dt_ms, 95)), 4),
                "p99": round(float(np.percentile(dt_ms, 99)), 4),
            }
            thr = max(3.0 * med, med + 1e-9)
            gaps = dt_ms[dt_ms > thr]
            out["gaps"] = {"threshold_ms": round(thr, 4), "count": int(gaps.size),
                           "max_gap_ms": round(float(np.max(dt_ms)), 4)}
            counts, edges = np.histogram(dt_ms, bins=12)
            out["dt_histogram_ms"] = {"edges": [round(float(e), 3) for e in edges],
                                      "counts": [int(c) for c in counts]}
        out["channels"] = {c: self._chan[sid][c].summary() for c in _CHANNELS}
        return out

    def stop(self) -> dict:
        """Stop the writer thread, close files, and write the meta sidecars.

        Returns ``{sensor_id: rows_written}``.
        """
        if not self._active and self._thread is None:
            return {}
        self._active = False
        self._queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        pc_stop = datetime.now()
        pc_wall_s = round((pc_stop - self._start_dt).total_seconds(), 3)

        for sid, writer in self._writers.items():
            try:
                writer.flush()
                writer.close()
            except Exception:  # pragma: no cover - defensive
                logger.exception("SmartRecorder: failed to close writer for sensor %s", sid)
            raw = self._raw_writers.get(sid)
            if raw is not None:
                try:
                    raw.flush()
                    raw.close()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("SmartRecorder: failed to close raw writer for sensor %s", sid)

            meta: dict = dict(
                pc_start_utc=self._start_dt.isoformat(),
                requested_hz=self._requested_hz,
                actual_hz=self._actual_hz,
                decimate=self._decimate,
                sensor_id=sid,
                rows=self._written.get(sid, 0),
                pi_clock_note="Pi clock unreliable - use pc_start_utc",
            )
            if self._audit:
                audit = self._timing_audit(sid)
                audit["clock"] = {
                    "pc_wall_s": pc_wall_s,
                    "pi_ts_s": audit.get("duration_pi_s"),
                    "drift_s": (round(pc_wall_s - audit["duration_pi_s"], 3)
                                if audit.get("duration_pi_s") is not None else None),
                }
                meta.update(
                    pc_stop_utc=pc_stop.isoformat(),
                    host=self._host_name,
                    session=self._session_name,
                    probe_seconds=self._probe_seconds,
                    schema=list(SENSOR_CSV_HEADER),
                    audit_schema_version=AUDIT_SCHEMA_VERSION,
                    audit=audit,
                )
                if self._raw_writers.get(sid) is not None:
                    meta["raw_audit_file"] = self._raw_writers[sid].path.name
            placement = self._placement_for(sid)
            if placement is not None:
                meta["placement"] = placement
            write_recording_meta(self._meta_paths[sid], **meta)

        if self._audit and self._audit_dir is not None:
            self._write_audit_summary(pc_stop, pc_wall_s)
        return dict(self._written)

    def _placement_for(self, sid: int) -> Optional[dict]:
        """This sensor's floor/cell/role from the placement map, if there is one."""
        if not self._sensor_map:
            return None
        for row in self._sensor_map.get("placements") or []:
            if int(row.get("sensor_id", -1)) == int(sid):
                out = dict(row)
                out["axis"] = self._sensor_map.get("axis")
                out["n_floors"] = self._sensor_map.get("n_floors")
                return out
        return None

    def _write_audit_summary(self, pc_stop: datetime, pc_wall_s: float) -> None:
        """One bundled JSON per recording (global + per-sensor) for offline auditing."""
        summary = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "host": self._host_name,
            "session": self._session_name,
            "pc_start_utc": self._start_dt.isoformat(),
            "pc_stop_utc": pc_stop.isoformat(),
            "pc_wall_s": pc_wall_s,
            "probe_seconds": self._probe_seconds,
            "rate_requested_hz": self._requested_hz,
            "rate_probe_hz": self._actual_hz,                       # per-sensor probe estimate
            "rate_probe_combined_hz": self._probe_combined_hz,      # raw all-sensors (diagnostic)
            "decimate": self._decimate,
            "sensors": {str(sid): self._timing_audit(sid) for sid in self._sensor_ids},
            "sensor_map": self._sensor_map,
            "note": ("rate_incoming_hz is the TRUE per-sensor delivered rate measured "
                     "from sample timestamps; compare to rate_requested_hz to audit "
                     "PRE-3 under-sampling. rate_probe_hz is the PER-SENSOR 5 s probe "
                     "estimate (median dt); rate_probe_combined_hz is the raw "
                     "all-sensors-interleaved number (~N_sensors x per-sensor)."),
        }
        write_recording_meta(self._audit_dir / "audit_summary.json", **summary)
