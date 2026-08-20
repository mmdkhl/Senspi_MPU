from __future__ import annotations

import logging
import math
import queue
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Deque, Dict, Optional, Tuple

from datetime import datetime

from PySide6.QtCore import QObject, QMetaObject, QThread, QTimer, Qt, Signal, Slot

from .config.acquisition_state import GuiAcquisitionConfig, SensorSelectionConfig
from ..analysis.rate import RateController
from ..config.app_config import (
    AppPaths,
    HostConfig,
    HostInventory,
    SensorDefaults,
    normalize_remote_path,
)
from ..dataio.smart_recorder import SmartRecorder
from ..config.pi_logger_config import PiLoggerConfig
from ..config.sampling import GuiSamplingDisplay, SamplingConfig
from ..core.live_stream import select_parser
from ..data import BufferConfig, StreamingDataBuffer
from ..remote.pi_recorder import PiRecorder
from ..remote.sensor_ingest_worker import SensorIngestWorker
from ..remote.ssh_client import Host
from ..sensors.mpu6050 import MpuSample

logger = logging.getLogger(__name__)


def _per_sensor_hz_from_times(probe_times: dict[int, list[float]]) -> tuple[float, dict[int, float]]:
    """Per-sensor rate from probe-window timestamps via each sensor's MEDIAN dt.

    The shared RateController counts all sensors interleaved (~N x per-sensor) and is
    inflated by the startup burst — the audit (2026-06-24) showed it mislabels ~41 Hz as
    ~160 Hz and triggers spurious decimation. The median inter-sample dt of each sensor's
    OWN stream is robust to the burst and to occasional gaps. Returns
    ``(representative_per_sensor_hz, {sensor_id: hz})``. Pure — unit-testable without Qt.
    """
    import statistics
    rates: dict[int, float] = {}
    for sid, ts in (probe_times or {}).items():
        ordered = sorted(t for t in ts if t is not None)
        if len(ordered) < 5:
            continue
        dts = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
        if not dts:
            continue
        med = statistics.median(dts)
        if med > 0:
            rates[int(sid)] = 1.0 / med
    rep = statistics.median(rates.values()) if rates else 0.0
    return float(rep), rates


def _decimate_for(measured_hz: float, requested_hz: float) -> int:
    """Decimation factor that NEVER drops the written rate below the request (audit fix).

    Uses a FLOOR (not round): decimate only when the device genuinely exceeds the request
    by >=2x, and ``written = measured / decimate`` then stays >= requested. The device
    usually under-delivers (PRE-3), so this is normally 1 (record everything).
    """
    if requested_hz <= 0 or measured_hz <= 0:
        return 1
    return max(1, int(measured_hz // requested_hz))


@dataclass
class MpuGuiConfig:
    enabled: bool = True
    rate_hz: float = 100.0
    sensors: str = "1,2,3,4"
    channels: str = "default"
    include_temp: bool = False
    limit_duration: bool = False
    duration_s: float = 0.0


class ModalCaptureBuffer:
    """Long-window, lock-protected per-sensor sample store for modal capture.

    Separate from the GUI's 6 s ``StreamingDataBuffer`` (guardrail G3/G5 keep
    that one small). Written on the GUI thread in ``_on_samples_batch`` and read
    by the Mode B worker thread via :meth:`snapshot`; a ``threading.Lock``
    serializes the two so the deques are never mutated mid-read. Stores only
    ``(t_seconds, ax, ay, az)`` — enough for X/Y modal identification.
    """

    def __init__(self, window_seconds: float = 120.0) -> None:
        self._window_s = max(1.0, float(window_seconds))
        self._lock = threading.Lock()
        self._buf: Dict[int, Deque[Tuple[float, float, float, float]]] = {}

    def set_window(self, window_seconds: float) -> None:
        with self._lock:
            self._window_s = max(1.0, float(window_seconds))

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def add_batch(self, batch: Iterable[MpuSample]) -> None:
        with self._lock:
            for s in batch:
                if not isinstance(s, MpuSample):
                    continue
                t = float(s.t_s) if s.t_s is not None else float(s.timestamp_ns) * 1e-9
                sid = int(s.sensor_id) if s.sensor_id is not None else 1
                dq = self._buf.get(sid)
                if dq is None:
                    dq = deque()
                    self._buf[sid] = dq
                dq.append((t, float(s.ax), float(s.ay), float(s.az), float(s.gz)))
            self._trim_locked()

    def _trim_locked(self) -> None:
        latest = 0.0
        for dq in self._buf.values():
            if dq:
                latest = max(latest, dq[-1][0])
        if latest <= 0.0:
            return
        threshold = latest - self._window_s
        for dq in self._buf.values():
            while dq and dq[0][0] < threshold:
                dq.popleft()

    def available_seconds(self) -> float:
        with self._lock:
            spans = [dq[-1][0] - dq[0][0] for dq in self._buf.values() if len(dq) > 1]
        return min(spans) if spans else 0.0

    _AXIS_COLUMNS = {"ax": 1, "ay": 2, "az": 3, "gz": 4}

    def snapshot_series(self, axis: str) -> Dict[int, list[Tuple[float, float]]]:
        """Return a copied {sensor_id: [(t, axis_value), ...]} (thread-safe).

        An unknown axis used to fall back to ``ax`` silently, which let callers
        believe they were reading a channel they were not. Unknown names now
        warn loudly before falling back.
        """
        key = axis.lower()
        col = self._AXIS_COLUMNS.get(key)
        if col is None:
            logger.warning(
                "ModalCaptureBuffer: unknown axis %r, falling back to 'ax'", axis)
            col = 1
        with self._lock:
            return {
                sid: [(row[0], row[col]) for row in dq]
                for sid, dq in self._buf.items()
                if dq
            }


class RecorderController(QObject):
    """Non-visual controller that manages remote acquisition sessions."""

    sample_received = Signal(object)
    streaming_started = Signal()
    streaming_stopped = Signal()
    stream_started = Signal()
    stream_stopped = Signal()
    error_reported = Signal(str)
    rate_updated = Signal(str, float)
    stream_rate_updated = Signal(str, float)
    sampling_config_changed = Signal(object)
    recording_started = Signal()
    recording_stopped = Signal()
    sensorSelectionChanged = Signal(SensorSelectionConfig)
    recording_error = Signal(str)
    # Smart Recording (Front C): human-readable rate warning + phase narration.
    rate_warning = Signal(str)
    recording_status = Signal(str)

    def __init__(
        self,
        host_inventory: HostInventory | None = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._host_inventory = host_inventory or HostInventory()
        self._sensor_defaults = SensorDefaults()
        self._sampling_config = self._load_sampling_config()

        self._hosts: Dict[str, Dict[str, object]] = {}
        self._pi_recorder: Optional[PiRecorder] = None

        self._ingest_thread: Optional[QThread] = None
        self._ingest_worker: Optional[SensorIngestWorker] = None
        self._rate_controllers: Dict[str, RateController] = {
            "mpu6050": RateController(window_size=500, default_hz=0.0),
        }
        self._recording_mode: bool = False
        self._recording_preference: bool = False
        self._stop_requested: bool = False
        self._ingest_batch_size = 50
        self._ingest_max_latency_ms = 100
        self._ingest_had_error = False
        self._last_session_name: str = ""
        self._expected_remote_end: bool = False

        decimation = self._sampling_config.compute_decimation()
        self._data_buffer = StreamingDataBuffer(
            BufferConfig(
                max_seconds=6.0,
                sample_rate_hz=decimation["stream_rate_hz"],
            )
        )
        self._active_stream: Iterable[str] | None = None
        self._sample_queue: queue.Queue[object] = queue.Queue(maxsize=10_000)
        self._current_sensor_selection = SensorSelectionConfig()
        self._current_gui_acquisition_config: GuiAcquisitionConfig | None = None

        # Smart Recording (Front C): PC-side probe -> writer -> fixed duration.
        self._smart_recorder: SmartRecorder | None = None
        self._smart_ctx: dict | None = None
        self._probe_timer: QTimer | None = None
        self._record_duration_timer: QTimer | None = None
        self._probe_seconds: float = 5.0
        # Per-sensor timestamp collector active only during the probe window. The shared
        # RateController counts all sensors interleaved (~N x per-sensor), so we measure
        # each sensor's own stream here for an accurate per-sensor probe (audit 2026-06-24).
        self._probe_times: dict[int, list[float]] | None = None

        # Long-window store for modal capture (Mode B). Independent of the 6 s
        # GUI buffer above; written on the GUI thread, read by the modal worker.
        self._modal_buffer = ModalCaptureBuffer(window_seconds=120.0)

    # --------------------------------------------------------------- helpers
    def _load_sampling_config(self) -> SamplingConfig:
        try:
            config = self._sensor_defaults.load()
            sampling = SamplingConfig.from_mapping(config)
        except Exception:
            sampling = SamplingConfig(device_rate_hz=100.0)
        self._sampling_config = sampling
        return sampling

    def _get_default_mpu_dlpf(self) -> int | None:
        try:
            config = self._sensor_defaults.load()
        except Exception:
            return 3
        sensors = config.get("sensors", {}) if isinstance(config, dict) else {}
        mpu_cfg = dict(sensors.get("mpu6050", {}) or {})
        dlpf = mpu_cfg.get("dlpf")
        try:
            return int(dlpf)
        except (TypeError, ValueError):
            return None

    def _current_sampling(self) -> SamplingConfig:
        if not isinstance(self._sampling_config, SamplingConfig):
            return self._load_sampling_config()
        return self._sampling_config

    def sampling_config(self) -> SamplingConfig:
        return self._current_sampling()

    def set_sampling_config(self, sampling: SamplingConfig) -> None:
        self._apply_sampling_config(sampling, notify=True)

    def _apply_sampling_config(self, sampling: SamplingConfig, *, notify: bool = True) -> None:
        previous = self._current_sampling()
        normalized = SamplingConfig(
            device_rate_hz=float(sampling.device_rate_hz),
            mode_key=str(sampling.mode_key),
        )
        changed = (
            not math.isclose(previous.device_rate_hz, normalized.device_rate_hz, rel_tol=1e-6, abs_tol=1e-6)
            or previous.mode_key != normalized.mode_key
        )
        self._sampling_config = normalized
        if hasattr(self._data_buffer, "config"):
            try:
                self._data_buffer.config.sample_rate_hz = GuiSamplingDisplay.from_sampling(normalized).stream_rate_hz
            except Exception:
                pass
        if notify and changed:
            self.sampling_config_changed.emit(normalized)

    def target_stream_rate_hz(self) -> float:
        display = GuiSamplingDisplay.from_sampling(self._current_sampling())
        return float(display.stream_rate_hz)

    def compute_stream_every(self, *_, **__) -> int:
        return 1

    def data_buffer(self) -> StreamingDataBuffer | None:
        return self._data_buffer

    # --------------------------------------------------------------- modal capture
    def is_streaming(self) -> bool:
        """True when a live stream is active (Mode B prerequisite)."""
        return self._ingest_worker is not None

    def set_modal_window_seconds(self, seconds: float) -> None:
        """Resize the long-window modal accumulator (Mode B cycle length)."""
        self._modal_buffer.set_window(seconds)

    def modal_window_seconds(self) -> float:
        """Current accumulator window, so callers can grow it without shrinking it."""
        return float(getattr(self._modal_buffer, "_window_s", 0.0))

    def require_modal_window_seconds(self, seconds: float) -> None:
        """Grow the accumulator window if needed, never shrink it.

        Several features share this one buffer (Mode B continuous update and the
        Sonification tab). Last-writer-wins would let one silently truncate the
        other's data mid-run, so requesters may only raise the floor.
        """
        want = float(seconds)
        if want > self.modal_window_seconds():
            self._modal_buffer.set_window(want)

    def modal_available_seconds(self) -> float:
        """Seconds of data currently buffered across all sensors (min span)."""
        return self._modal_buffer.available_seconds()

    def snapshot_modal_capture(
        self,
        *,
        axis: str = "ax",
        last_seconds: float | None = None,
        target_fs: float | None = None,
    ):
        """Snapshot the live accumulator into an aligned ``ModalSession``.

        Thread-safe: the underlying buffer copies its deques under a lock, so
        this can be called from the Mode B worker thread. Returns the same
        ``ModalSession`` type the disk loader produces, so downstream code is
        identical for live and recorded data.
        """
        from ..dataio.modal_session_loader import align_per_sensor_series

        series = self._modal_buffer.snapshot_series(axis)
        return align_per_sensor_series(
            series, last_seconds=last_seconds, target_fs=target_fs, source="live capture"
        )

    def recording_requested(self) -> bool:
        return bool(self._recording_preference)

    def set_recording_requested(self, enabled: bool) -> None:
        self._recording_preference = bool(enabled)

    def current_remote_data_dir(self) -> Path | None:
        cfg = getattr(self._pi_recorder, "config", None)
        if cfg is None:
            return None
        return cfg.data_dir

    def report_error(self, message: str) -> None:
        logger.error("RecorderController error: %s", message)
        self.error_reported.emit(str(message))

    # --------------------------------------------------------------- wiring helpers
    def apply_sensor_selection(self, cfg: SensorSelectionConfig) -> None:
        self._current_sensor_selection = cfg

    def apply_gui_acquisition_config(self, cfg: GuiAcquisitionConfig) -> None:
        self._current_gui_acquisition_config = cfg
        self._apply_sampling_config(cfg.sampling)

    # --------------------------------------------------------------- start/stop
    def start_live_stream(
        self,
        *,
        recording_enabled: bool,
        gui_config: GuiAcquisitionConfig,
        host_cfg: HostConfig,
        session_name: str | None = None,
    ) -> None:
        self._current_gui_acquisition_config = gui_config
        logger.info("RecorderController received GuiAcquisitionConfig: %s", gui_config.summary())

        session_name = (session_name or "").strip() or None
        self._recording_preference = bool(recording_enabled)
        record_only = bool(gui_config.record_only)
        self._recording_mode = bool(recording_enabled or record_only)
        self._current_sensor_selection = gui_config.sensor_selection
        self._expected_remote_end = bool(gui_config.limit_duration and gui_config.duration_s > 0)

        self._apply_sampling_config(gui_config.sampling, notify=True)
        self._last_session_name = session_name or (
            gui_config.sampling.mode_key if hasattr(gui_config, "sampling") else ""
        )

        extra_cli: dict[str, object] = {}
        sel = gui_config.sensor_selection

        if session_name:
            extra_cli["session_name"] = session_name

        if gui_config.limit_duration and gui_config.duration_s > 0:
            extra_cli["duration"] = max(1, min(600, int(round(gui_config.duration_s))))  # Rec-length max

        if sel.active_sensors:
            extra_cli["sensors"] = ",".join(str(s) for s in sel.active_sensors)

        ch = set(sel.active_channels or [])
        if ch == {"ax", "ay", "az"}:
            extra_cli["channels"] = "acc"
        elif ch == {"gx", "gy", "gz"}:
            extra_cli["channels"] = "gyro"
        elif ch == {"ax", "ay", "az", "gx", "gy", "gz"}:
            extra_cli["channels"] = "both"

        dlpf = self._get_default_mpu_dlpf()
        if dlpf is not None:
            extra_cli["dlpf"] = dlpf

        pi_logger_cfg = PiLoggerConfig.from_sampling(gui_config.sampling, extra_cli=extra_cli)

        self._start_mpu_stream(
            host_cfg=host_cfg,
            pi_logger_cfg=pi_logger_cfg,
            selection=gui_config.sensor_selection,
            recording_enabled=recording_enabled,
            record_only=record_only,
            session_name=session_name,
        )

    def stop_live_stream(
        self,
        *,
        wait: bool = False,
        wait_timeout_ms: int | None = 5000,
    ) -> None:
        thread = self._ingest_thread
        self._stop_stream()

        if wait and thread is not None:
            if wait_timeout_ms is None:
                thread.wait()
            else:
                thread.wait(max(0, int(wait_timeout_ms)))

    def _stop_stream(self) -> None:
        self._finalize_smart_recorder()  # close any active PC recording first
        worker = self._ingest_worker
        if worker is not None:
            self._stop_requested = True
            QMetaObject.invokeMethod(worker, "stop", Qt.QueuedConnection)
            self.streaming_stopped.emit()
            self.recording_stopped.emit()
            self._close_active_stream()
        else:
            self._stop_requested = False
            self._close_active_stream()
            if self._pi_recorder is not None:
                try:
                    self._pi_recorder.close()
                except Exception:
                    logger.exception("Failed to close recorder")
            self.streaming_stopped.emit()
            self.stream_stopped.emit()
            self.recording_stopped.emit()

    # --------------------------------------------------------------- smart recording
    def start_smart_recording(
        self,
        *,
        gui_config: GuiAcquisitionConfig,
        host_cfg: HostConfig,
        session_name: str | None,
        duration_s: float,
    ) -> None:
        """Front C: start a live stream, probe the real rate for ~5 s, then write a
        PC-authoritative, PC-clock, fixed-duration recording (Pi-side recording off).

        Additive: reuses the normal live-stream machinery; the PC writer is fed from
        ``_on_samples_batch`` via a non-blocking queue (no read-loop changes).
        """
        if self._ingest_worker is not None:
            self.report_error("A stream is already running; stop it before recording.")
            return
        # Live stream only — the PC writes the authoritative file; the Pi just streams.
        self.start_live_stream(
            recording_enabled=False, gui_config=gui_config,
            host_cfg=host_cfg, session_name=session_name)
        requested_hz = float(getattr(gui_config.sampling, "device_rate_hz", 0.0) or 0.0)
        self._smart_ctx = {
            "host": host_cfg,
            "session": session_name,
            "duration_s": max(1.0, float(duration_s)),
            "requested_hz": requested_hz,
            "selection": gui_config.sensor_selection,
        }
        self._probe_times = {}   # begin per-sensor probe collection
        probe_msg = f"Probing sample rate for {self._probe_seconds:.0f} s"
        if requested_hz > 0:
            probe_msg += f" (requested {requested_hz:.0f} Hz)"
        self.recording_status.emit(probe_msg + "...")
        self._probe_timer = QTimer(self)
        self._probe_timer.setSingleShot(True)
        self._probe_timer.timeout.connect(self._on_probe_complete)
        self._probe_timer.start(int(self._probe_seconds * 1000))

    @Slot()
    def _on_probe_complete(self) -> None:
        ctx = self._smart_ctx
        if ctx is None or self._ingest_worker is None:
            return
        sel = ctx["selection"]
        sensor_ids = (list(sel.active_sensors) if sel and sel.active_sensors
                      else (self._data_buffer.get_sensor_ids()
                            if self._data_buffer is not None else []))
        if not sensor_ids:
            sensor_ids = [1, 2, 3, 4]

        # Accurate PER-SENSOR rate (audit fix). Keep the combined estimate as a
        # diagnostic; fall back to combined/N only if the per-sensor measure was sparse.
        per_sensor_hz, _per = _per_sensor_hz_from_times(self._probe_times or {})
        combined = float(self._rate_controllers["mpu6050"].estimate().hz_effective)
        measured = per_sensor_hz if per_sensor_hz > 0 else combined / max(1, len(sensor_ids))
        self._probe_times = None   # stop collecting
        requested = float(ctx["requested_hz"]) or measured

        # Decimate ONLY when the device genuinely exceeds the request, by a FLOOR factor
        # so the WRITTEN rate never drops below the requested rate (audit fix / IMP-1).
        # The device usually under-delivers here, so this is normally decimate=1.
        decimate = _decimate_for(measured, requested)
        if decimate > 1:
            msg = (f"Device {measured:.1f} Hz/sensor > requested {requested:.1f} Hz "
                   f"-> decimating x{decimate} (writing ~{measured / decimate:.1f} Hz).")
        elif measured < requested * 0.90:
            msg = (f"Requested {requested:.1f} Hz - device delivers {measured:.1f} Hz/sensor "
                   f"(hardware ceiling). Recording all of it at {measured:.1f} Hz.")
        else:
            msg = f"Recording at {measured:.1f} Hz/sensor (~ requested {requested:.1f} Hz)."
        self.rate_warning.emit(msg)

        host = ctx["host"]
        out_dir = AppPaths().raw_data / host.name / "mpu"
        self._smart_recorder = SmartRecorder(
            out_dir=out_dir, session_name=ctx["session"], sensor_ids=sensor_ids,
            start_dt=datetime.now(), requested_hz=requested, actual_hz=measured,
            decimate=decimate, host_name=host.name, probe_seconds=self._probe_seconds,
            probe_combined_hz=combined)
        self._smart_recorder.start()
        self.recording_started.emit()
        # SF-4: PC-controlled fixed duration.
        self._record_duration_timer = QTimer(self)
        self._record_duration_timer.setSingleShot(True)
        self._record_duration_timer.timeout.connect(self.stop_smart_recording)
        self._record_duration_timer.start(int(ctx["duration_s"] * 1000))
        # Comprehensive recording feed: requested vs detected vs effective (written)
        # rate, decimation, sensor count and duration. The single-line status label
        # is overwritten by each message, so this persistent "Recording..." line
        # carries the full rate story (the earlier rate_warning would be hidden).
        effective_hz = measured / max(1, decimate)
        detail = f"requested {requested:.0f} Hz · detected {measured:.0f} Hz/sensor"
        if decimate > 1:
            detail += f" ÷{decimate} → ~{effective_hz:.0f} Hz written"
        self._smart_summary = (
            f"~{effective_hz:.0f} Hz, {ctx['duration_s']:.0f} s")
        self.recording_status.emit(
            f"Recording {len(sensor_ids)} sensor(s) "
            f"({detail}) for {ctx['duration_s']:.0f} s...")

    @Slot()
    def stop_smart_recording(self) -> None:
        self._finalize_smart_recorder()
        self.stop_live_stream()

    def _finalize_smart_recorder(self) -> None:
        for attr in ("_probe_timer", "_record_duration_timer"):
            timer = getattr(self, attr, None)
            if timer is not None:
                timer.stop()
                setattr(self, attr, None)
        rec = self._smart_recorder
        self._smart_recorder = None
        self._smart_ctx = None
        self._probe_times = None
        if rec is not None:
            try:
                written = rec.stop()
                total = sum(written.values()) if written else 0
                summary = getattr(self, "_smart_summary", "")
                suffix = f" at {summary}" if summary else ""
                self.recording_status.emit(f"Saved recording ({total} samples{suffix}).")
            except Exception:
                logger.exception("Failed to finalize smart recorder")
            finally:
                self._smart_summary = ""

    # --------------------------------------------------------------- start helpers
    def _create_streaming_buffer(
        self, selection: SensorSelectionConfig, stream_rate_hz: float
    ) -> StreamingDataBuffer:
        rate = (
            float(stream_rate_hz)
            if stream_rate_hz > 0
            else self._sampling_config.compute_decimation().get("stream_rate_hz", 0.0)
        )
        return StreamingDataBuffer(BufferConfig(max_seconds=6.0, sample_rate_hz=rate))

    def _create_pi_recorder_for_host(self, host_cfg: HostConfig) -> PiRecorder:
        host = Host(
            name=host_cfg.name,
            host=host_cfg.host,
            user=host_cfg.user,
            password=host_cfg.password,
            port=host_cfg.port,
        )
        recorder = PiRecorder(host, host_cfg.base_path)
        recorder.connect()
        self._pi_recorder = recorder
        return recorder

    def _start_mpu_stream(
        self,
        *,
        host_cfg: HostConfig,
        pi_logger_cfg: PiLoggerConfig,
        selection: SensorSelectionConfig,
        recording_enabled: bool,
        record_only: bool,
        session_name: str | None = None,
    ) -> None:
        if self._ingest_worker is not None:
            raise RuntimeError("MPU6050 streaming is already running.")

        self._close_active_stream()
        self._clear_sample_queue()

        recorder = self._create_pi_recorder_for_host(host_cfg)
        if recording_enabled or record_only:
            output_dir = PurePosixPath(
                normalize_remote_path(host_cfg.data_dir, host_cfg.user)
            ) / "mpu"
            logger.info("Clearing previous Pi recordings in %s", output_dir)
            recorder.clear_recording_output(output_dir.as_posix())

        if record_only:
            logger.info("Starting record-only capture on %s", host_cfg.name)
            stream = recorder.start_record_only(pi_logger_cfg)
            self._active_stream = stream
            self._data_buffer = None
            self.recording_started.emit()
            return

        logger.info(
            "Starting streaming capture on %s (recording_enabled=%s)",
            host_cfg.name,
            recording_enabled,
        )
        stream = recorder.stream_mpu6050(
            cfg=pi_logger_cfg,
            recording_enabled=recording_enabled,
            session_name=session_name,
        )

        self._data_buffer = self._create_streaming_buffer(selection, pi_logger_cfg.stream_rate_hz)
        self._active_stream = stream
        self._stop_requested = False
        rc = self._rate_controllers["mpu6050"]
        rc.reset()

        parser = select_parser("mpu6050")

        def _stream_factory():
            return stream

        thread = QThread(self)
        worker = SensorIngestWorker(
            recorder=recorder,
            stream_factory=_stream_factory,
            parser=parser,
            batch_size=self._ingest_batch_size,
            max_latency_ms=self._ingest_max_latency_ms,
            stream_label="mpu6050",
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.start)
        worker.samples_batch.connect(self._on_samples_batch)
        worker.error.connect(self._on_ingest_error)
        worker.finished.connect(self._on_ingest_finished)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.start()

        self._ingest_thread = thread
        self._ingest_worker = worker

        self.recording_started.emit()
        self.streaming_started.emit()
        self.stream_started.emit()

    def _close_active_stream(self) -> None:
        stream = self._active_stream
        self._active_stream = None
        self._ingest_worker = None
        self._ingest_thread = None
        if stream is None:
            return
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close active stream")
        # Closing the local SSH channel above does not stop the remote
        # logger process by itself (see PiRecorder.stop_remote_logger) --
        # without this it keeps sampling and refreshing the OLED heartbeat
        # indefinitely after the GUI thinks the stream has stopped.
        if self._pi_recorder is not None:
            try:
                self._pi_recorder.stop_remote_logger()
            except Exception:
                logger.exception("Failed to stop remote logger")

    def _clear_sample_queue(self) -> None:
        try:
            while True:
                self._sample_queue.get_nowait()
        except queue.Empty:
            pass

    # --------------------------------------------------------------- ingest callbacks
    @Slot(list)
    def _on_samples_batch(self, batch: list[object]) -> None:
        if not batch:
            return

        rc = self._rate_controllers["mpu6050"]
        first = batch[0]
        if isinstance(first, MpuSample):
            def _to_seconds(s: MpuSample) -> float:
                # Prefer explicit seconds field if present; otherwise convert ns -> s.
                if getattr(s, "t_s", None) is not None:
                    return float(s.t_s)  # type: ignore[arg-type]
                return float(s.timestamp_ns) * 1e-9

            # Feed ALL sample times so estimated_hz reflects samples/sec, not batches/sec.
            rc.feed_times(_to_seconds(s) for s in batch if isinstance(s, MpuSample))
            # During a Smart-Recording probe, also collect PER-SENSOR times for an
            # accurate per-sensor rate estimate (the RateController above is combined).
            if self._probe_times is not None:
                for s in batch:
                    if isinstance(s, MpuSample) and s.sensor_id is not None:
                        self._probe_times.setdefault(int(s.sensor_id), []).append(_to_seconds(s))
            stream_rate_hz = float(rc.estimate().hz_effective)
            self.stream_rate_updated.emit("mpu6050", stream_rate_hz)
            if self._data_buffer is not None:
                try:
                    self._data_buffer.add_samples(batch)  # type: ignore[arg-type]
                except Exception:
                    logger.exception("RecorderController: failed to add samples to buffer")
            try:
                self._modal_buffer.add_batch(batch)  # type: ignore[arg-type]
            except Exception:
                logger.exception("RecorderController: failed to add samples to modal buffer")
            # Smart Recording (Front C): non-blocking enqueue to the PC writer thread
            # (G4-safe — the disk I/O happens off the GUI thread inside SmartRecorder).
            rec = self._smart_recorder
            if rec is not None:
                rec.submit(batch)
        for sample in batch:
            self.sample_received.emit(sample)

    @Slot(str)
    def _on_ingest_error(self, message: str) -> None:
        self._ingest_had_error = True
        self._emit_error(message)
        self.recording_error.emit(message)

    @Slot()
    def _on_ingest_finished(self) -> None:
        if (
            not self._stop_requested
            and not self._ingest_had_error
            and not self._expected_remote_end
        ):
            self._emit_error("Live stream stopped unexpectedly (no stop request)")
        self._stop_requested = False
        self._expected_remote_end = False
        self._ingest_worker = None
        self._ingest_thread = None
        self.streaming_stopped.emit()
        self.stream_stopped.emit()
        self.recording_stopped.emit()

    def _emit_error(self, message: str) -> None:
        logger.error("RecorderController error: %s", message)
        self.error_reported.emit(str(message))

    # --------------------------------------------------------------- legacy helpers
    def set_control_panel_enabled(self, enabled: bool) -> None:
        logger.debug("RecorderController set_control_panel_enabled(%s)", enabled)

    def current_host_details(self):
        return None

    def current_host_config(self) -> HostConfig | None:
        return None

    def last_session_name(self) -> str:
        return self._last_session_name
