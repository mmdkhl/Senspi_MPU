"""Main window for the SensePi GUI."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget, QVBoxLayout, QWidget

from ..config.app_config import AppConfig, HostInventory
from ..config.sampling import SamplingConfig
from ..remote.log_sync import SyncReport, sync_logs_from_pi
from .config.acquisition_state import (
    CalibrationOffsets,
    GuiAcquisitionConfig,
    SensorSelectionConfig,
)
from .recorder_controller import RecorderController
from .tabs.tab_fft import FftTab
from .tabs.tab_digital_twin import DigitalTwinExperimentTab
from .tabs.tab_model_updating import ModelUpdatingTab
from .tabs.tab_settings import SettingsTab
from .tabs.tab_signals import SignalsTab
from .tabs.tab_sonification import SonificationTab


class _LogSyncTask(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, host_cfg, session_name: str | None) -> None:
        super().__init__()
        self._host_cfg = host_cfg
        self._session_name = session_name

    @Slot()
    def run(self) -> None:
        try:
            report = sync_logs_from_pi(self._host_cfg, session_name=self._session_name)
        except Exception as exc:  # pragma: no cover - GUI surface
            self.error.emit(str(exc))
        else:
            self.finished.emit(report)


class MainWindow(QMainWindow):
    """Main window for the SensePi GUI."""

    def __init__(self, app_config: AppConfig | None = None) -> None:
        super().__init__()
        self.setWindowTitle("SensePi Recorder")

        self._app_config = app_config or AppConfig()
        self._host_inventory = HostInventory()
        self._tabs = QTabWidget()
        self._logger = logging.getLogger(__name__)

        self._current_sensor_selection = SensorSelectionConfig()
        self._current_gui_acquisition_config: GuiAcquisitionConfig | None = None
        self._current_calibration_offsets: CalibrationOffsets | None = None
        self._current_host: dict | None = None
        self._log_sync_thread: QThread | None = None
        self._log_sync_worker: _LogSyncTask | None = None
        self._auto_stop_timer = QTimer(self)
        self._auto_stop_timer.setSingleShot(True)
        # How long a close waits for Model Updating before postponing itself,
        # and the poll that completes a postponed close once the job is done.
        self._shutdown_wait_ms = 10000
        self._close_retry_timer = QTimer(self)
        self._close_retry_timer.setInterval(250)
        self._close_retry_timer.timeout.connect(self._retry_close_when_idle)

        # Authoritative sensor placement, owned by Settings. Consumers read it
        # from here rather than each keeping their own picker.
        self.sensor_map = None

        self._build_tabs()
        self._wire_signals()

        if isinstance(self._app_config.sampling_config, SamplingConfig):
            self._on_sampling_changed(self._app_config.sampling_config)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Stop the Digital Twin worker before shutting down the stream/OpenSees state.
        try:
            self.digital_twin_tab.shutdown()
        except Exception as exc:  # pragma: no cover - best-effort shutdown
            self.recorder_tab.report_error(
                f"Failed to stop Digital Twin experiment on close: {exc!r}"
            )

        # Stop the sonification workers FIRST. stop_live_stream(wait=True) blocks
        # the GUI thread, so the queued stream_stopped that would otherwise stop
        # a tab can never be delivered during shutdown — leaving a running
        # QThread to be destroyed under us.
        try:
            self.sonification_tab.shutdown()
        except Exception as exc:  # pragma: no cover - best-effort shutdown
            self.recorder_tab.report_error(
                f"Failed to stop sonification on close: {exc!r}"
            )
        # Model Updating's workers (Continuous Update, Calibrate, Run Analysis,
        # Identify) run on their own threads; closing without waiting for them
        # destroyed a running QThread and crashed the process. A one-shot job
        # cannot be interrupted mid-OpenSees, so if it outlasts the wait the
        # close is postponed and completes by itself when the job is done.
        try:
            idle = self.model_updating_tab.shutdown(wait_ms=self._shutdown_wait_ms)
        except Exception as exc:  # pragma: no cover - best-effort shutdown
            idle = True
            self.recorder_tab.report_error(
                f"Failed to stop Model Updating on close: {exc!r}"
            )
        if not idle:
            event.ignore()
            self.statusBar().showMessage(
                "Waiting for the running Model Updating job to finish; "
                "SensePi will close by itself when it is done.")
            self._close_retry_timer.start()
            return
        self._close_retry_timer.stop()
        try:
            self.recorder_tab.stop_live_stream(wait=True)
        except Exception as exc:  # pragma: no cover - best-effort shutdown
            self.recorder_tab.report_error(
                f"Failed to stop stream on close: {exc!r}"
            )
        super().closeEvent(event)

    @Slot()
    def _retry_close_when_idle(self) -> None:
        """Complete a close that was postponed for a running Model Updating job."""
        if not self.model_updating_tab.is_busy():
            self._close_retry_timer.stop()
            self.close()

    def _build_tabs(self) -> None:
        """Create and register all main workflow tabs."""
        self.recorder_tab = RecorderController()
        self.settings_tab = SettingsTab()
        self.signals_tab = SignalsTab(
            recorder_tab=self.recorder_tab, parent=self, app_config=self._app_config
        )
        self.fft_tab = FftTab(
            recorder_tab=self.recorder_tab,
            signals_tab=self.signals_tab,
            parent=self,
            app_config=self._app_config,
        )
        self.model_updating_tab = ModelUpdatingTab(parent=self)
        # Give the Model Updating tab access to the live capture buffer for
        # sensor-driven Continuous Update (Mode B). The tab never touches SSH;
        # it only reads thread-safe snapshots from the controller (guardrail G2).
        self.model_updating_tab.set_recorder_controller(self.recorder_tab)
        # The Spectrum tab reports measured torsion as a real rotation, which
        # needs the slab's plan dimensions. Those live in the model definition,
        # not in the sensor placement, and the Spectrum tab is constructed
        # before this one -- so it is handed over here rather than at build time.
        self.fft_tab.set_model_updating_tab(self.model_updating_tab)
        # The Sonification tab hosts the sonification models as sub-tabs:
        # "Bioacoustic Chorus" (built) and "Team Model" (held for the
        # sonification team). Models pull thread-safe modal snapshots from the
        # controller inside their own workers; none touches SSH (guardrail G2).
        self.sonification_tab = SonificationTab(
            recorder_controller=self.recorder_tab, parent=self
        )
        self.digital_twin_tab = DigitalTwinExperimentTab(
            recorder_controller=self.recorder_tab,
            model_updating_tab=self.model_updating_tab,
            sonification_tab=self.sonification_tab,
            parent=self,
        )

        self._tabs.addTab(self.signals_tab, self.tr("Live Signals"))
        self._tabs.addTab(self.fft_tab, self.tr("Spectrum"))
        self._tabs.addTab(self.model_updating_tab, self.tr("Model Updating"))
        self._tabs.addTab(self.sonification_tab, self.tr("Sonification"))
        self._tabs.addTab(self.digital_twin_tab, self.tr("Digital Twin Experiment"))
        self._tabs.addTab(self.settings_tab, self.tr("Settings"))

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._tabs)

        self.setCentralWidget(container)

    def _wire_signals(self) -> None:
        """Connect signals across tabs and seed initial state."""

        self.signals_tab.start_stream_requested.connect(
            self._on_start_stream_requested
        )
        self.signals_tab.stop_stream_requested.connect(self._on_stop_stream_requested)
        self.signals_tab.record_requested.connect(self._on_record_requested)
        self.recorder_tab.recording_status.connect(self.signals_tab._set_manual_status)
        self.recorder_tab.rate_warning.connect(self.signals_tab._set_manual_status)
        self.signals_tab.sync_logs_requested.connect(self._on_sync_logs_requested)
        self.recorder_tab.stream_started.connect(self.signals_tab.on_stream_started)
        self.recorder_tab.stream_stopped.connect(self.signals_tab.on_stream_stopped)
        self.recorder_tab.stream_started.connect(self.fft_tab.on_stream_started)
        self.recorder_tab.stream_stopped.connect(self.fft_tab.on_stream_stopped)
        self.recorder_tab.stream_started.connect(self.sonification_tab.on_stream_started)
        self.recorder_tab.stream_stopped.connect(self.sonification_tab.on_stream_stopped)
        self.recorder_tab.stream_started.connect(self.digital_twin_tab.on_stream_started)
        self.recorder_tab.stream_stopped.connect(self.digital_twin_tab.on_stream_stopped)
        self.recorder_tab.stream_stopped.connect(self._cancel_auto_stop_timer)
        self.recorder_tab.recording_stopped.connect(self._cancel_auto_stop_timer)
        self._auto_stop_timer.timeout.connect(self._on_auto_stop_timeout)
        # SettingsTab is the canonical source of sensor / channel selection.
        self.settings_tab.sensorSelectionChanged.connect(
            self._on_sensor_selection_changed
        )
        self.settings_tab.sensorsUpdated.connect(self._on_sensors_updated)
        # Settings owns the placement map; keep the app-wide copy in step.
        self.settings_tab.sensorMapChanged.connect(self._on_sensor_map_changed)
        self.sensor_map = self.settings_tab.current_sensor_map()
        self.recorder_tab.apply_sensor_map(self.sensor_map)
        self.fft_tab.apply_sensor_map(self.sensor_map)
        self.model_updating_tab.apply_sensor_map(self.sensor_map)
        self.sonification_tab.apply_sensor_map(self.sensor_map)
        self.digital_twin_tab.apply_sensor_map(self.sensor_map)
        # Keep the recorder controller in sync with the canonical selection.
        self.settings_tab.sensorSelectionChanged.connect(
            self.recorder_tab.apply_sensor_selection
        )
        self.signals_tab.acquisitionConfigChanged.connect(
            self._on_acquisition_config_changed
        )
        self.signals_tab.calibrationChanged.connect(self._on_calibration_changed)
        self.recorder_tab.stream_rate_updated.connect(
            self.signals_tab.update_stream_rate
        )
        self.recorder_tab.stream_rate_updated.connect(self.fft_tab.update_stream_rate)
        self.fft_tab.final_values_ready_for_model_updating.connect(
            self.model_updating_tab.apply_spectrum_final_values
        )
        if hasattr(self.settings_tab, "acquisitionConfigChanged"):
            self.settings_tab.acquisitionConfigChanged.connect(
                self.fft_tab.update_acquisition_config
            )
        try:
            self._on_sensor_selection_changed(self.settings_tab.current_sensor_selection())
        except Exception:
            self._logger.exception(
                "Failed to seed initial sensor selection from SettingsTab"
            )

    @Slot(str)
    def _on_start_stream_requested(self, session_name: str) -> None:
        acquisition_settings = self.signals_tab.current_acquisition_settings()
        acquisition_widget = getattr(self.signals_tab, "_acquisition_widget", None)

        sensor_selection = getattr(self, "_current_sensor_selection", None)
        if sensor_selection is None:
            sensor_selection = SensorSelectionConfig(active_sensors=[], active_channels=[])

        if acquisition_widget is not None:
            gui_cfg = acquisition_widget.current_gui_acquisition_config(
                sensor_selection=sensor_selection
            )
        else:
            stream_rate_hz = float(acquisition_settings.effective_stream_rate_hz)
            gui_cfg = GuiAcquisitionConfig(
                sampling=acquisition_settings.sampling,
                stream_rate_hz=stream_rate_hz,
                record_only=False,
                sensor_selection=sensor_selection,
            )

        # Pipeline kept by request; the checkbox is hidden. Ask the tab's own
        # accessor rather than an attribute that no longer exists.
        gui_cfg.record_only = bool(self.signals_tab._get_record_only_checked())
        gui_cfg.limit_duration = self.signals_tab.duration_limit_enabled()
        gui_cfg.duration_s = float(self.signals_tab.duration_limit_seconds())

        gui_cfg.calibration = self._current_calibration_offsets
        self._current_gui_acquisition_config = gui_cfg
        self._logger.info(
            "Starting stream with GuiAcquisitionConfig: %s", gui_cfg.summary()
        )

        host_cfg_raw = self.settings_tab.current_host_config()
        if host_cfg_raw is None:
            self.recorder_tab.report_error("No Raspberry Pi host selected.")
            return

        self._current_host = host_cfg_raw
        host_cfg = self._host_inventory.to_host_config(host_cfg_raw)

        self.recorder_tab.apply_sensor_selection(gui_cfg.sensor_selection)
        self.recorder_tab.apply_gui_acquisition_config(gui_cfg)

        self.signals_tab.set_sensor_selection(gui_cfg.sensor_selection)
        self.signals_tab.apply_gui_acquisition_config(gui_cfg)

        self.fft_tab.update_sensor_selection(gui_cfg.sensor_selection)
        self.fft_tab.update_acquisition_config(gui_cfg)

        device_rate = float(gui_cfg.sampling.device_rate_hz)
        self.signals_tab.set_sampling_rate_hz(device_rate)
        self.fft_tab.set_sampling_rate_hz(device_rate)

        self.fft_tab.set_refresh_interval_ms(acquisition_settings.fft_refresh_ms)

        record_only = gui_cfg.record_only
        recording_flag = bool(record_only or self.recorder_tab.recording_requested())

        if recording_flag:
            self._log_recording_calibration("starting")

        self.signals_tab.set_record_only_mode(record_only)
        self.fft_tab.set_record_only_mode(record_only)
        if record_only:
            self._logger.info("Record-only mode active: live streaming disabled.")

        self.recorder_tab.start_live_stream(
            recording_enabled=recording_flag,
            gui_config=gui_cfg,
            host_cfg=host_cfg,
            session_name=session_name,
        )
        self._arm_auto_stop_timer(gui_cfg)

    @Slot(str, int)
    def _on_record_requested(self, session_name: str, duration_s: int) -> None:
        """Smart Recording (Front C): probe-measured, PC-clock, fixed-duration record.

        Reuses the same config/host prep as Start, but routes to
        ``start_smart_recording`` (which owns the probe + PC writer + duration timer),
        and forces ``record_only=False`` (Smart Recording is a stream-based PC path).
        """
        acquisition_settings = self.signals_tab.current_acquisition_settings()
        acquisition_widget = getattr(self.signals_tab, "_acquisition_widget", None)
        sensor_selection = getattr(self, "_current_sensor_selection", None)
        if sensor_selection is None:
            sensor_selection = SensorSelectionConfig(active_sensors=[], active_channels=[])
        if acquisition_widget is not None:
            gui_cfg = acquisition_widget.current_gui_acquisition_config(
                sensor_selection=sensor_selection)
        else:
            gui_cfg = GuiAcquisitionConfig(
                sampling=acquisition_settings.sampling,
                stream_rate_hz=float(acquisition_settings.effective_stream_rate_hz),
                record_only=False, sensor_selection=sensor_selection)
        gui_cfg.record_only = False
        gui_cfg.calibration = self._current_calibration_offsets
        self._current_gui_acquisition_config = gui_cfg

        host_cfg_raw = self.settings_tab.current_host_config()
        if host_cfg_raw is None:
            self.recorder_tab.report_error("No Raspberry Pi host selected.")
            return
        self._current_host = host_cfg_raw
        host_cfg = self._host_inventory.to_host_config(host_cfg_raw)

        self.recorder_tab.apply_sensor_selection(gui_cfg.sensor_selection)
        self.recorder_tab.apply_gui_acquisition_config(gui_cfg)
        self.signals_tab.set_sensor_selection(gui_cfg.sensor_selection)
        self.signals_tab.apply_gui_acquisition_config(gui_cfg)
        self.fft_tab.update_sensor_selection(gui_cfg.sensor_selection)
        self.fft_tab.update_acquisition_config(gui_cfg)
        device_rate = float(gui_cfg.sampling.device_rate_hz)
        self.signals_tab.set_sampling_rate_hz(device_rate)
        self.fft_tab.set_sampling_rate_hz(device_rate)
        self.fft_tab.set_refresh_interval_ms(acquisition_settings.fft_refresh_ms)

        # PC controls the stop (SF-4) inside the controller; no GUI auto-stop timer here.
        self.recorder_tab.start_smart_recording(
            gui_config=gui_cfg, host_cfg=host_cfg,
            session_name=session_name, duration_s=float(duration_s))

    @Slot()
    def _on_stop_stream_requested(self) -> None:
        if getattr(self.recorder_tab, "_recording_mode", False):
            self._log_recording_calibration("stopping")
        self._cancel_auto_stop_timer()
        # Ensure the ingest worker thread has fully stopped before allowing a new Start.
        self.recorder_tab.stop_live_stream(wait=True)

    @Slot()
    def _on_auto_stop_timeout(self) -> None:
        self.recorder_tab.stop_live_stream(wait=True)

    @Slot()
    def _cancel_auto_stop_timer(self) -> None:
        if self._auto_stop_timer.isActive():
            self._auto_stop_timer.stop()

    def _arm_auto_stop_timer(self, cfg: GuiAcquisitionConfig) -> None:
        self._cancel_auto_stop_timer()
        if not cfg.limit_duration:
            return
        duration_s = max(1.0, min(600.0, float(cfg.duration_s)))  # match Rec-length max
        self._auto_stop_timer.start(int(duration_s * 1000))

    @Slot()
    def _on_sync_logs_requested(self) -> None:
        host_cfg_raw = self.settings_tab.current_host_config()
        if host_cfg_raw is None:
            QMessageBox.warning(
                self, "No host selected", "Select a host in the Settings tab first."
            )
            return

        self._current_host = host_cfg_raw
        session_name = self.signals_tab.current_session_name()

        host_cfg = self._host_inventory.to_host_config(host_cfg_raw)
        worker = _LogSyncTask(host_cfg, session_name)
        thread = QThread(self)
        worker.moveToThread(thread)

        worker.finished.connect(self._on_log_sync_finished)
        worker.error.connect(self._on_log_sync_error)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_log_sync_worker)

        self._log_sync_thread = thread
        self._log_sync_worker = worker

        try:
            self.signals_tab.sync_logs_button.setEnabled(False)
        except Exception:
            pass
        self.statusBar().showMessage("Starting log sync …", 3000)
        thread.start()

    @Slot(object)
    def _on_log_sync_finished(self, report: SyncReport) -> None:
        try:
            self.signals_tab.sync_logs_button.setEnabled(True)
        except Exception:
            pass
        self.statusBar().showMessage("Log sync complete.", 5000)
        QMessageBox.information(
            self,
            "Log sync complete",
            f"Remote: {report.remote_root}\n"
            f"Local: {report.local_root}\n"
            f"Downloaded: {len(report.downloaded)}\n"
            f"Skipped: {report.skipped}",
        )

    @Slot(str)
    def _on_log_sync_error(self, message: str) -> None:
        try:
            self.signals_tab.sync_logs_button.setEnabled(True)
        except Exception:
            pass
        self.statusBar().showMessage("Log sync failed.", 5000)
        QMessageBox.critical(self, "Sync failed", message)

    def _clear_log_sync_worker(self) -> None:
        self._log_sync_thread = None
        self._log_sync_worker = None

    @Slot(SensorSelectionConfig)
    def _on_sensor_selection_changed(self, cfg: SensorSelectionConfig) -> None:
        self._current_sensor_selection = cfg
        self._logger.info("Updated sensor selection: %s", cfg)
        self.signals_tab.set_sensor_selection(cfg)
        self.fft_tab.update_sensor_selection(cfg)

    @Slot(GuiAcquisitionConfig)
    def _on_acquisition_config_changed(self, cfg: GuiAcquisitionConfig) -> None:
        self._current_gui_acquisition_config = cfg
        if getattr(cfg, "calibration", None) is not None:
            self._current_calibration_offsets = cfg.calibration
        self._logger.info("GuiAcquisitionConfig updated: %s", cfg.summary())
        self.signals_tab.apply_gui_acquisition_config(cfg)
        self.recorder_tab.apply_gui_acquisition_config(cfg)
        self.fft_tab.update_acquisition_config(cfg)
        self.signals_tab.set_record_only_mode(cfg.record_only)
        self.fft_tab.set_record_only_mode(cfg.record_only)

    @Slot(CalibrationOffsets)
    def _on_calibration_changed(self, offsets: CalibrationOffsets) -> None:
        self._current_calibration_offsets = offsets
        self.fft_tab.set_calibration_offsets(offsets)

    @Slot(dict)
    @Slot(object)
    def _on_sensor_map_changed(self, smap) -> None:
        """Fan the placement map out. Settings is its only source.

        Every tab that consumes placement now reads it from here. Digital Twin
        inherits it through Model Updating's calibrated snapshot.
        """
        self.sensor_map = smap
        # recordings embed the placement, so a session is self-describing
        self.recorder_tab.apply_sensor_map(smap)
        # Spectrum: decides the channel, the response sensors and the mode cap.
        self.fft_tab.apply_sensor_map(smap)
        # Model Updating: same placement feeds identification and, through the
        # calibrated snapshot, the Digital Twin.
        self.model_updating_tab.apply_sensor_map(smap)
        # Sonification: the shaker is excluded, and the chorus is laid out in
        # stereo and depth the way the sensors are laid out on the rig.
        self.sonification_tab.apply_sensor_map(smap)
        # Digital Twin: calibrates from the same placement, and excludes the
        # shaker row from identification the way every other tab now does.
        self.digital_twin_tab.apply_sensor_map(smap)

    def _on_sensors_updated(self, data: dict) -> None:
        """Apply updated sampling settings emitted from the Settings tab."""

        try:
            sampling = SamplingConfig.from_mapping(data)
        except Exception:
            self._logger.exception("Failed to parse sampling config from sensorsUpdated")
            return

        self._app_config.sensor_defaults = dict(data or {})
        self._on_sampling_changed(sampling)

    @Slot(SamplingConfig)
    def _on_sampling_changed(self, sampling: SamplingConfig) -> None:
        """Propagate sampling configuration updates across tabs."""

        normalized = SamplingConfig(
            device_rate_hz=float(sampling.device_rate_hz),
            mode_key=str(sampling.mode_key),
        )
        self._app_config.sampling_config = normalized

        try:
            self.signals_tab.set_sampling_config(normalized)
        except Exception:
            self._logger.exception("Failed to update Signals tab sampling config")

        try:
            self.recorder_tab.set_sampling_config(normalized)
        except Exception:
            self._logger.exception("Failed to update Recorder tab sampling config")

    def _log_recording_calibration(self, action: str) -> None:
        if not self.signals_tab.apply_calibration_to_recording():
            return

        offsets = self._current_calibration_offsets
        if offsets is None or offsets.is_empty():
            self._logger.info(
                "Recording %s: calibration requested but no offsets present.", action
            )
            return

        self._logger.info(
            "Recording %s with calibration (%d channels) at %s: %s",
            action,
            len(offsets.per_sensor_channel_offset),
            offsets.timestamp,
            offsets.description or "no description",
        )
        # TODO: apply calibration offsets to recorded samples before saving.

    def get_current_calibration(self) -> CalibrationOffsets | None:
        """Return the most recent calibration collected from the Signals tab."""

        return self._current_calibration_offsets
