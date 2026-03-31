from __future__ import annotations

import logging
import math
import queue
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QObject, QMetaObject, QThread, Qt, Signal, Slot

from .config.acquisition_state import GuiAcquisitionConfig, SensorSelectionConfig
from ..analysis.rate import RateController
from ..config.app_config import HostConfig, HostInventory, SensorDefaults
from ..config.pi_logger_config import PiLoggerConfig
from ..config.sampling import GuiSamplingDisplay, SamplingConfig
from ..core.live_stream import select_parser
from ..data import BufferConfig, StreamingDataBuffer
from ..remote.pi_recorder import PiRecorder
from ..remote.sensor_ingest_worker import SensorIngestWorker
from ..remote.ssh_client import Host
from ..sensors.mpu6050 import MpuSample

logger = logging.getLogger(__name__)


@dataclass
class MpuGuiConfig:
    enabled: bool = True
    rate_hz: float = 100.0
    sensors: str = "1,2,3"
    channels: str = "default"
    include_temp: bool = False
    limit_duration: bool = False
    duration_s: float = 0.0


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

    # --------------------------------------------------------------- helpers
    def _load_sampling_config(self) -> SamplingConfig:
        try:
            config = self._sensor_defaults.load()
            sampling = SamplingConfig.from_mapping(config)
        except Exception:
            sampling = SamplingConfig(device_rate_hz=200.0)
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

        self._apply_sampling_config(gui_config.sampling, notify=True)
        self._last_session_name = session_name or (
            gui_config.sampling.mode_key if hasattr(gui_config, "sampling") else ""
        )

        extra_cli: dict[str, object] = {}
        sel = gui_config.sensor_selection

        if session_name:
            extra_cli["session_name"] = session_name

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
            stream_rate_hz = float(rc.estimate().hz_effective)
            self.stream_rate_updated.emit("mpu6050", stream_rate_hz)
            if self._data_buffer is not None:
                try:
                    self._data_buffer.add_samples(batch)  # type: ignore[arg-type]
                except Exception:
                    logger.exception("RecorderController: failed to add samples to buffer")
        for sample in batch:
            self.sample_received.emit(sample)

    @Slot(str)
    def _on_ingest_error(self, message: str) -> None:
        self._ingest_had_error = True
        self._emit_error(message)
        self.recording_error.emit(message)

    @Slot()
    def _on_ingest_finished(self) -> None:
        if not self._stop_requested and not self._ingest_had_error:
            self._emit_error("Live stream stopped unexpectedly (no stop request)")
        self._stop_requested = False
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

