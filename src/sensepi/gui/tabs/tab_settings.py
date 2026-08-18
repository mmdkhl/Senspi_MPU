"""Settings tab for SSH hosts and sensor defaults.

This widget is backed by the YAML files in :mod:`sensepi.config`:

* ``hosts.yaml`` / ``hosts.local.yaml`` via :class:`sensepi.config.app_config.HostInventory`
* ``sensors.yaml``    via :class:`sensepi.config.app_config.SensorDefaults`

It exposes two signals so other tabs (e.g. RecorderTab) can keep in sync:

* ``hostsUpdated(list[dict])``   – emitted after saving host configuration
* ``sensorsUpdated(dict)``       – emitted after saving sensors.yaml

RecorderTab can either:

* Call :meth:`current_host_config` / :meth:`all_hosts` /
  :meth:`sensor_defaults` directly, or
* Connect to the signals to be notified when the user changes settings.

Example usage in RecorderTab (pseudo-code)::

    settings_tab: SettingsTab = ...

    def on_hosts_updated(hosts: list[dict]) -> None:
        self.populate_host_combo(hosts)

    settings_tab.hostsUpdated.connect(on_hosts_updated)

    # When starting a recording:
    host_cfg = settings_tab.current_host_config()
    sensor_cfg = settings_tab.sensor_defaults()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import QSignalBlocker, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QLineEdit,
    QWidget,
)

from ...config.app_config import (
    AppConfig,
    HostConfig,
    HostInventory,
    SensorDefaults,
    build_pi_config_for_host,
    normalize_remote_path,
)
from ...config.sampling import RECORDING_MODES, SamplingConfig
from ...remote.ssh_client import SSHClient
from ..config.acquisition_state import SensorSelectionConfig

# Fixed device sampling rate used by recording and streaming.
FIXED_DEVICE_RATE_HZ = 50.0


class SettingsTab(QWidget):
    """
    Configuration tab for SSH hosts and default sampling values.

    Responsibilities:
    - Edit host configuration and ``sensors.yaml`` so :class:`RecorderTab` can
      launch loggers with consistent network + sampling defaults.
    - Notify device/plotting tabs when hosts or sensor presets change, keeping
      UI choices in sync with disk-backed configuration.
    - Focused on setup (no live data), complementing the live-streaming
      ``Signals`` and ``Spectrum`` tabs.
    """

    # Emitted after a successful save of the corresponding YAML file
    hostsUpdated = Signal(list)   # list[dict] – entries from the active host config
    sensorsUpdated = Signal(dict) # dict      – full sensors.yaml mapping
    # New signal emitted whenever the sensor selection changes.
    sensorSelectionChanged = Signal(SensorSelectionConfig)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._host_inventory = HostInventory()
        self._sensor_defaults = SensorDefaults()

        self._DLPF_DESCRIPTIONS = {
            0: "DLPF 0: Accel BW 260 Hz, Gyro BW 256 Hz, typical delay ~1 ms",
            1: "DLPF 1: Accel BW 184 Hz, Gyro BW 188 Hz, typical delay ~2 ms",
            2: "DLPF 2: Accel BW 94 Hz, Gyro BW 98 Hz, typical delay ~3 ms",
            3: "DLPF 3: Accel BW 44 Hz, Gyro BW 42 Hz, typical delay ~5 ms",
            4: "DLPF 4: Accel BW 21 Hz, Gyro BW 20 Hz, typical delay ~8–9 ms",
            5: "DLPF 5: Accel BW 10 Hz, Gyro BW 10 Hz, typical delay ~14 ms",
            6: "DLPF 6: Accel BW 5 Hz, Gyro BW 5 Hz, typical delay ~19 ms",
        }

        # In-memory models mirroring the YAML content
        self._hosts: List[Dict[str, Any]] = []
        self._sensors: Dict[str, Any] = {}

        self._current_host_index: Optional[int] = None
        self._hosts_dirty: bool = False

        # Last sampling config loaded from sensors.yaml (used to preserve rate)
        self._sampling_config: SamplingConfig | None = None

        self._build_ui()
        self._load_from_disk()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ----- Host configuration --------------------------------------
        hosts_group = QGroupBox("Raspberry Pi hosts", self)
        hosts_layout = QHBoxLayout(hosts_group)

        # Left column: host list + add/remove buttons
        host_list_col = QVBoxLayout()
        host_list_col.addWidget(QLabel("Configured Pis:"))

        self.host_list = QListWidget(hosts_group)
        host_list_col.addWidget(self.host_list)

        host_btn_row = QHBoxLayout()
        self.btn_add_host = QPushButton("Add")
        self.btn_remove_host = QPushButton("Remove")
        host_btn_row.addWidget(self.btn_add_host)
        host_btn_row.addWidget(self.btn_remove_host)
        host_list_col.addLayout(host_btn_row)

        hosts_layout.addLayout(host_list_col, 1)

        # Right column: host detail form
        host_form_col = QVBoxLayout()
        form = QFormLayout()

        self.edit_host_name = QLineEdit(hosts_group)
        self.edit_host_address = QLineEdit(hosts_group)
        self.edit_host_user = QLineEdit(hosts_group)

        self.edit_host_port = QSpinBox(hosts_group)
        self.edit_host_port.setRange(1, 65535)
        self.edit_host_port.setValue(22)

        self.edit_password = QLineEdit(hosts_group)
        self.edit_password.setEchoMode(QLineEdit.Password)

        # Base path row: line edit + "Browse..."
        base_row = QHBoxLayout()
        self.edit_base_path = QLineEdit(hosts_group)
        self.btn_browse_base = QPushButton("Browse…", hosts_group)
        base_row.addWidget(self.edit_base_path)
        base_row.addWidget(self.btn_browse_base)

        self.edit_data_dir = QLineEdit(hosts_group)
        self.edit_pi_config = QLineEdit(hosts_group)

        form.addRow("Name:", self.edit_host_name)
        form.addRow("Host / IP:", self.edit_host_address)
        form.addRow("User:", self.edit_host_user)
        form.addRow("SSH port:", self.edit_host_port)
        form.addRow("Password:", self.edit_password)
        form.addRow("Scripts base path:", base_row)
        form.addRow("Data directory:", self.edit_data_dir)
        form.addRow("Pi config path:", self.edit_pi_config)

        host_form_col.addLayout(form)

        self.btn_save_hosts = QPushButton("Save host config", hosts_group)

        buttons_row = QHBoxLayout()
        buttons_row.addStretch()
        buttons_row.addWidget(self.btn_save_hosts)
        host_form_col.addLayout(buttons_row)
        self._hosts_dirty_label = QLabel("Unsaved changes", hosts_group)
        self._hosts_dirty_label.setStyleSheet("color: red; font-size: 11px;")
        self._hosts_dirty_label.setVisible(False)
        host_form_col.addWidget(self._hosts_dirty_label, alignment=Qt.AlignRight)

        hosts_layout.addLayout(host_form_col, 2)
        root.addWidget(hosts_group)

        # ----- Sensor defaults ----------------------------------------
        sensors_group = QGroupBox("Sensor defaults", self)
        sensors_layout = QVBoxLayout(sensors_group)

        # High-level sensor selection defaults (used as app-wide defaults)
        sensor_selection_form = QFormLayout()

        # 1/2/3/4 sensors instead of free-text "1,2,3,4"
        self.mpu_sensor_count_combo = QComboBox(sensors_group)
        self.mpu_sensor_count_combo.addItem("1 sensor", 1)
        self.mpu_sensor_count_combo.addItem("2 sensors", 2)
        self.mpu_sensor_count_combo.addItem("3 sensors", 3)
        self.mpu_sensor_count_combo.addItem("4 sensors", 4)
        # Old default was "1,2,3" -> 3 sensors
        self.mpu_sensor_count_combo.setCurrentIndex(2)
        sensor_selection_form.addRow("Number of sensors:", self.mpu_sensor_count_combo)

        sensors_layout.addLayout(sensor_selection_form)

        # Sampling (single source of truth)
        sampling_group = QGroupBox("Sampling (single source of truth)", sensors_group)
        sampling_form = QFormLayout(sampling_group)

        self.device_rate_label = QLabel(f"{FIXED_DEVICE_RATE_HZ:.0f} Hz", sampling_group)

        self.mode_combo = QComboBox(sampling_group)
        for key, mode in RECORDING_MODES.items():
            self.mode_combo.addItem(mode.label, userData=key)

        sampling_form.addRow("Sampling (device) rate:", self.device_rate_label)
        sampling_form.addRow("Mode:", self.mode_combo)

        # MPU6050 defaults (unchanged)
        mpu_group = QGroupBox("MPU6050", sensors_group)
        mpu_form = QFormLayout(mpu_group)

        self.mpu_channels = QComboBox(mpu_group)
        self.mpu_channels.addItem(
            "AX / AY / GZ (default 3 channels)", userData="default"
        )
        self.mpu_channels.addItem("AX / AY / AZ (accel only)", userData="acc")
        self.mpu_channels.addItem("GX / GY / GZ (gyro only)", userData="gyro")
        self.mpu_channels.addItem(
            "AX / AY / AZ / GX / GY / GZ (all 6 channels)", userData="both"
        )

        self.mpu_dlpf = QSpinBox(mpu_group)
        self.mpu_dlpf.setRange(0, 6)
        self.mpu_dlpf.setValue(3)

        self.mpu_include_temp = QCheckBox("Include on-die temperature", mpu_group)

        mpu_form.addRow("Channels:", self.mpu_channels)
        mpu_form.addRow("DLPF:", self.mpu_dlpf)
        mpu_form.addRow("", self.mpu_include_temp)

        sensors_layout.addWidget(sampling_group)
        sensors_layout.addWidget(mpu_group)
        self.btn_sync_pi = QPushButton("Sync Pi defaults (pi_config.yaml)", sensors_group)
        self.btn_sync_pi.setToolTip("Uploads generated pi_config.yaml to the selected Pi host")
        sync_note = QLabel(
            "Sync writes pi_config.yaml on the selected Pi (output_dir, sampling rate, "
            "channels, DLPF). Only needed if you run the Pi scripts manually or want Pi "
            "defaults updated.",
            sensors_group,
        )
        sync_note.setWordWrap(True)
        sensors_layout.addWidget(sync_note)

        self.btn_save_sensors = QPushButton("Save sensors.yaml", sensors_group)
        buttons_sync_row = QHBoxLayout()
        buttons_sync_row.addStretch()
        buttons_sync_row.addWidget(self.btn_sync_pi)
        buttons_sync_row.addWidget(self.btn_save_sensors)
        sensors_layout.addLayout(buttons_sync_row)

        root.addWidget(sensors_group)

        # ----- signal wiring ------------------------------------------
        for edit in (
            self.edit_host_name,
            self.edit_host_address,
            self.edit_host_user,
            self.edit_password,
            self.edit_base_path,
            self.edit_data_dir,
            self.edit_pi_config,
        ):
            edit.textEdited.connect(lambda _text=None: self._set_hosts_dirty(True))
        self.edit_host_port.valueChanged.connect(lambda _value=None: self._set_hosts_dirty(True))
        self.host_list.currentRowChanged.connect(self._on_host_row_changed)
        self.btn_add_host.clicked.connect(self._on_add_host)
        self.btn_remove_host.clicked.connect(self._on_remove_host)
        self.btn_browse_base.clicked.connect(self._on_browse_base)
        self.btn_sync_pi.clicked.connect(self._on_sync_to_pi)
        self.btn_save_hosts.clicked.connect(self._on_save_hosts_clicked)
        self.btn_save_sensors.clicked.connect(self._on_save_sensors_clicked)

        # Sensor selection (UI-level)
        self.mpu_sensor_count_combo.currentIndexChanged.connect(
            self._on_sensor_ui_changed
        )

        # Sampling rate is fixed; sensor/channel changes still update selection state.
        self.mpu_sensor_count_combo.currentIndexChanged.connect(
            self._refresh_sampling_rate_choices
        )
        self.mpu_channels.currentIndexChanged.connect(self._refresh_sampling_rate_choices)
        self.mpu_channels.currentIndexChanged.connect(self._on_sensor_ui_changed)

        # Populate initial fixed sampling display.
        self._refresh_sampling_rate_choices()

        self.mpu_dlpf.valueChanged.connect(self._update_mpu_dlpf_info)
        self.mpu_dlpf.valueChanged.connect(self._on_sensor_ui_changed)

        self._set_host_fields_enabled(False)
        self._on_sensor_ui_changed()
        self._update_mpu_dlpf_info()
        self._update_save_hosts_ui()

    # ------------------------------------------------------------------
    # Host helpers
    # ------------------------------------------------------------------
    def _set_host_fields_enabled(self, enabled: bool) -> None:
        widgets = [
            self.edit_host_name,
            self.edit_host_address,
            self.edit_host_user,
            self.edit_host_port,
            self.edit_password,
            self.edit_base_path,
            self.btn_browse_base,
            self.btn_remove_host,
            self.edit_data_dir,
            self.edit_pi_config,
            self.btn_sync_pi,
        ]
        for w in widgets:
            w.setEnabled(enabled)

    def _update_save_hosts_ui(self) -> None:
        base_text = "Save host config"
        if getattr(self, "_hosts_dirty", False):
            self.btn_save_hosts.setText(f"{base_text} *")
            if hasattr(self, "_hosts_dirty_label"):
                self._hosts_dirty_label.setVisible(True)
        else:
            self.btn_save_hosts.setText(base_text)
            if hasattr(self, "_hosts_dirty_label"):
                self._hosts_dirty_label.setVisible(False)

    def _set_hosts_dirty(self, dirty: bool) -> None:
        self._hosts_dirty = bool(dirty)
        self._update_save_hosts_ui()

    def _clear_host_fields(self) -> None:
        self.edit_host_name.clear()
        self.edit_host_address.clear()
        self.edit_host_user.clear()
        self.edit_password.clear()
        self.edit_base_path.clear()
        self.edit_data_dir.clear()
        self.edit_pi_config.clear()
        with QSignalBlocker(self.edit_host_port):
            self.edit_host_port.setValue(22)

    def _load_from_disk(self) -> None:
        # --- Hosts -----------------------------------------------------
        try:
            data = self._host_inventory.load()
        except Exception as exc:
            QMessageBox.warning(self, "Config error", f"Could not load host config:\n{exc}")
            data = {}

        self._hosts = list(data.get("pis", [])) if isinstance(data, dict) else []
        self._refresh_host_list()
        self._set_hosts_dirty(False)

        # --- Sensors ---------------------------------------------------
        try:
            self._sensors = self._sensor_defaults.load()
        except Exception as exc:
            QMessageBox.warning(self, "Config error", f"Could not load sensors.yaml:\n{exc}")
            self._sensors = {}

        self._load_sensor_widgets_from_model()

    def _refresh_host_list(self) -> None:
        self.host_list.blockSignals(True)
        self.host_list.clear()
        for host in self._hosts:
            label = host.get("name") or host.get("host") or "<unnamed>"
            self.host_list.addItem(label)
        self.host_list.blockSignals(False)

        if self._hosts:
            self.host_list.setCurrentRow(0)
        else:
            self._current_host_index = None
            self._set_host_fields_enabled(False)
            self._clear_host_fields()

    @Slot(int)
    def _on_host_row_changed(self, row: int) -> None:
        # Persist edits from previous host
        if self._current_host_index is not None:
            self._update_model_from_host_fields(self._current_host_index)

        if row < 0 or row >= len(self._hosts):
            self._current_host_index = None
            self._set_host_fields_enabled(False)
            self._clear_host_fields()
            return

        self._current_host_index = row
        self._set_host_fields_enabled(True)

        host = self._hosts[row]
        self.edit_host_name.setText(str(host.get("name", "")))
        self.edit_host_address.setText(str(host.get("host", "")))
        self.edit_host_user.setText(str(host.get("user", "")))
        self.edit_password.setText(str(host.get("password", "")))
        self.edit_base_path.setText(str(host.get("base_path", host.get("scripts_dir", ""))))
        self.edit_data_dir.setText(str(host.get("data_dir", "")))
        self.edit_pi_config.setText(str(host.get("pi_config_path", "")))
        with QSignalBlocker(self.edit_host_port):
            self.edit_host_port.setValue(int(host.get("port", 22)))

    def _update_model_from_host_fields(self, index: int) -> None:
        if index < 0 or index >= len(self._hosts):
            return

        previous = dict(self._hosts[index])  # keep unknown keys
        original = dict(self._hosts[index])
        name = self.edit_host_name.text().strip()
        host = self.edit_host_address.text().strip()
        user = self.edit_host_user.text().strip()
        password = self.edit_password.text()
        base_path = self.edit_base_path.text().strip()
        data_dir = self.edit_data_dir.text().strip()
        pi_config = self.edit_pi_config.text().strip()
        port = int(self.edit_host_port.value())

        if name:
            original["name"] = name
        else:
            original.pop("name", None)

        if host:
            original["host"] = host
        else:
            original.pop("host", None)

        if user:
            original["user"] = user
        else:
            original.pop("user", None)

        if password:
            # TODO: This stores the password in plain text. Consider a keyring.
            original["password"] = password
        else:
            original.pop("password", None)

        if base_path:
            original["base_path"] = base_path
        else:
            original.pop("base_path", None)

        if data_dir:
            original["data_dir"] = data_dir
        else:
            original.pop("data_dir", None)

        if pi_config:
            original["pi_config_path"] = pi_config
        else:
            original.pop("pi_config_path", None)

        # store port even if default
        original["port"] = port

        changed = original != previous
        self._hosts[index] = original

        item = self.host_list.item(index)
        if item is not None:
            label = original.get("name") or original.get("host") or "<unnamed>"
            item.setText(label)
        if changed:
            self._set_hosts_dirty(True)

    @Slot()
    def _on_add_host(self) -> None:
        if self._current_host_index is not None:
            self._update_model_from_host_fields(self._current_host_index)

        new = {
            "name": f"pi-{len(self._hosts) + 1}",
            "host": "raspberrypi.local",
            "user": "pi",
            "password": "",
            "base_path": "~/sensor",
            "data_dir": "~/logs",
            "pi_config_path": "~/sensor/pi_config.yaml",
            "port": 22,
        }
        self._hosts.append(new)
        self._refresh_host_list()
        self.host_list.setCurrentRow(len(self._hosts) - 1)
        self._set_hosts_dirty(True)

    @Slot()
    def _on_remove_host(self) -> None:
        row = self.host_list.currentRow()
        if row < 0 or row >= len(self._hosts):
            return
        del self._hosts[row]
        self._refresh_host_list()
        self._set_hosts_dirty(True)

    @Slot()
    def _on_browse_base(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select remote scripts directory",
            self.edit_base_path.text() or "",
        )
        if path:
            self.edit_base_path.setText(path)
            self._set_hosts_dirty(True)

    @Slot()
    def _on_save_hosts_clicked(self) -> None:
        if self._current_host_index is not None:
            self._update_model_from_host_fields(self._current_host_index)

        try:
            existing = self._host_inventory.load()
        except Exception as exc:
            QMessageBox.warning(self, "Config error", f"Could not reload host config:\n{exc}")
            existing = {}

        if not isinstance(existing, dict):
            existing = {}
        existing["pis"] = self._hosts

        try:
            self._host_inventory.save(existing)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", f"Failed to write host config:\n{exc}")
            return

        QMessageBox.information(
            self,
            "Saved",
            "Host configuration saved to hosts.local.yaml.",
        )
        self.hostsUpdated.emit([dict(h) for h in self._hosts])
        self._set_hosts_dirty(False)

    @Slot()
    def _on_sync_to_pi(self) -> None:
        host_dict = self.current_host_config()
        if host_dict is None:
            QMessageBox.information(self, "No host", "Select a host to sync.")
            return

        host_cfg = self._host_inventory.to_host_config(host_dict)
        sensor_defaults, sampling_cfg = self._build_sensor_defaults_payload()
        app_cfg = AppConfig(
            sensor_defaults=sensor_defaults,
            sampling_config=sampling_cfg,
        )
        pi_cfg = build_pi_config_for_host(host_cfg, app_cfg)
        contents = pi_cfg.render_pi_config_yaml()

        remote_host = self._host_inventory.to_remote_host(host_dict)
        client = SSHClient(remote_host)
        try:
            client.connect()
        except Exception as exc:
            QMessageBox.critical(self, "SSH error", f"Could not connect: {exc}")
            return

        try:
            # Normalize remote paths to POSIX-style, independent of Windows host
            remote_data_dir = normalize_remote_path(host_cfg.data_dir, host_cfg.user)
            remote_scripts_dir = normalize_remote_path(host_cfg.base_path, host_cfg.user)
            remote_pi_config_path = normalize_remote_path(
                host_cfg.pi_config_path, host_cfg.user
            )

            if not client.path_exists(remote_data_dir):
                QMessageBox.critical(
                    self,
                    "Validation failed",
                    f"Remote data directory does not exist: {remote_data_dir}",
                )
                return
            if not client.path_exists(remote_scripts_dir):
                QMessageBox.critical(
                    self,
                    "Validation failed",
                    f"Remote scripts directory does not exist: {remote_scripts_dir}",
                )
                return

            with client.sftp() as sftp:
                with sftp.open(remote_pi_config_path, "w") as fh:
                    fh.write(contents)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Sync error",
                f"Failed to upload config to {host_cfg.name}:\n{exc}",
            )
            return
        finally:
            client.close()

        QMessageBox.information(
            self,
            "Config synced",
            f"Uploaded configuration to {remote_pi_config_path}.",
        )

    # ------------------------------------------------------------------
    # Sensor defaults helpers
    # ------------------------------------------------------------------
    def _update_mpu_dlpf_info(self) -> None:
        try:
            dlpf_value = int(self.mpu_dlpf.value())
        except (TypeError, ValueError):
            dlpf_value = 3
        text = self._DLPF_DESCRIPTIONS.get(
            dlpf_value,
            f"DLPF {dlpf_value}: see datasheet for bandwidth and delay details.",
        )
        self.mpu_dlpf.setToolTip(text)

    def _load_sensor_widgets_from_model(self) -> None:
        """
        Populate sampling + MPU6050 widgets from the in-memory sensors.yaml mapping.
        """
        # Sampling config from sensors.yaml (single source of truth)
        sampling_cfg = SamplingConfig.from_mapping(self._sensors)
        self._sampling_config = sampling_cfg

        # Mode combo: pick the saved mode or fall back to high_fidelity
        idx_mode = self.mode_combo.findData(sampling_cfg.mode_key)
        if idx_mode < 0:
            idx_mode = self.mode_combo.findData("high_fidelity")
        self.mode_combo.setCurrentIndex(max(0, idx_mode))

        # Per-sensor MPU6050 defaults (YAML-backed)
        sensors = self._sensors.get("sensors", {}) if isinstance(self._sensors, dict) else {}
        mpu_cfg = dict(sensors.get("mpu6050", {}) or {})

        mpu_ch_raw = str(mpu_cfg.get("channels", "default"))
        if mpu_ch_raw in {"default", "both", "acc", "gyro"}:
            mpu_ch = mpu_ch_raw
        else:
            mpu_ch = "default"

        idx = self.mpu_channels.findData(mpu_ch)
        if idx < 0:
            idx = self.mpu_channels.findData("default")
        if idx < 0:
            idx = 0
        self.mpu_channels.setCurrentIndex(idx)

        self.mpu_dlpf.setValue(int(mpu_cfg.get("dlpf", 3)))
        self.mpu_include_temp.setChecked(bool(mpu_cfg.get("include_temperature", False)))

        # Keep the displayed sampling rate at the fixed configured value.
        self._refresh_sampling_rate_choices()

    def _current_sampling_from_widgets(self) -> SamplingConfig:
        """
        Build a SamplingConfig from the current Settings tab widgets.
        """
        mode_key = self.mode_combo.currentData()

        return SamplingConfig(
            device_rate_hz=FIXED_DEVICE_RATE_HZ,
            mode_key=str(mode_key or "high_fidelity"),
        )

    def _refresh_sampling_rate_choices(self) -> None:
        """
        Update the fixed device-rate display and cached sampling config.
        """
        self.device_rate_label.setText(f"{FIXED_DEVICE_RATE_HZ:.0f} Hz")
        if isinstance(self._sampling_config, SamplingConfig):
            mode_key = self._sampling_config.mode_key
        else:
            mode_key = "high_fidelity"
        self._sampling_config = SamplingConfig(
            device_rate_hz=FIXED_DEVICE_RATE_HZ,
            mode_key=mode_key,
        )

    def _build_sensor_defaults_payload(self) -> tuple[Dict[str, Any], SamplingConfig]:
        sensors_model = dict(self._sensors) if isinstance(self._sensors, dict) else {}
        sampling_cfg = self._current_sampling_from_widgets()

        sensors_block = dict(sensors_model.get("sensors", {}) or {})
        mpu_cfg = dict(sensors_block.get("mpu6050", {}) or {})
        try:
            dlpf_int = int(self.mpu_dlpf.value())
        except (TypeError, ValueError):
            dlpf_int = 3

        mpu_cfg.update(
            {
                "channels": str(self.mpu_channels.currentData() or "default"),
                "dlpf": dlpf_int,
                "include_temperature": bool(self.mpu_include_temp.isChecked()),
            }
        )
        mpu_cfg.pop("sample_rate_hz", None)
        sensors_block["mpu6050"] = mpu_cfg

        sensors_model["sampling"] = sampling_cfg.to_mapping()["sampling"]
        sensors_model["sensors"] = sensors_block
        sensors_model.pop("mpu6050", None)
        sensors_model.pop("adxl203_ads1115", None)
        return sensors_model, sampling_cfg

    def current_sensor_selection(self) -> SensorSelectionConfig:
        """
        Build a SensorSelectionConfig from the current UI state.

        - Uses the sensor-count combo (1/2/3/4 sensors).
        - Chooses active_channels based on the channels combo.
        """
        try:
            count = int(self.mpu_sensor_count_combo.currentData() or 3)
        except (TypeError, ValueError):
            count = 3

        if count < 1:
            count = 1
        elif count > 4:
            count = 4

        active_sensors = list(range(1, count + 1))

        # Map preset selection to channels; you can adjust later if needed.
        preset = str(self.mpu_channels.currentData() or "default")

        if preset == "both":
            active_channels = ["ax", "ay", "az", "gx", "gy", "gz"]
        elif preset in {"acc", "accel_only"}:
            active_channels = ["ax", "ay", "az"]
        elif preset in {"gyro", "gyro_only"}:
            active_channels = ["gx", "gy", "gz"]
        else:
            active_channels = ["ax", "ay", "gz"]

        return SensorSelectionConfig(
            active_sensors=active_sensors,
            active_channels=active_channels,
        )

    @Slot()
    def _on_save_sensors_clicked(self) -> None:
        sensors, _ = self._build_sensor_defaults_payload()

        try:
            self._sensor_defaults.save(sensors)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", f"Failed to write sensors.yaml:\n{exc}")
            return

        self._sensors = self._sensor_defaults.load()
        QMessageBox.information(self, "Saved", "Sensor defaults saved to sensors.yaml.")
        self.sensorsUpdated.emit(dict(self._sensors))

    @Slot()
    def _on_sensor_ui_changed(self) -> None:
        """
        Handle any change to the sensors/channels widgets and emit
        the updated SensorSelectionConfig.
        """
        sel = self.current_sensor_selection()
        self.sensorSelectionChanged.emit(sel)

    # ------------------------------------------------------------------
    # Public helpers for RecorderTab (API points)
    # ------------------------------------------------------------------
    def current_host_config(self) -> Optional[Dict[str, Any]]:
        """
        Return the currently selected host dictionary (as in ``hosts.yaml``).

        A shallow copy is returned so callers can mutate it freely.
        """
        row = self.host_list.currentRow()
        if row < 0 or row >= len(self._hosts):
            return None
        if row == self._current_host_index:
            self._update_model_from_host_fields(row)
        return dict(self._hosts[row])

    def all_hosts(self) -> List[Dict[str, Any]]:
        """Return a list of host dictionaries (copied from the in-memory model)."""
        if self._current_host_index is not None:
            self._update_model_from_host_fields(self._current_host_index)
        return [dict(h) for h in self._hosts]

    def sensor_defaults(self) -> Dict[str, Any]:
        """Return the full sensor-defaults mapping."""
        sensors, _ = self._build_sensor_defaults_payload()
        return dict(sensors)


# Developer notes about what the SettingsTab currently provides and how to
# integrate it with other components. Kept as a module-level constant so the
# information remains close to the implementation without affecting runtime
# behavior.
SETTINGS_TAB_NOTES = """
What this gives you:

A full host editor (list/add/remove, edit name/host/user/password/base_path/port).

Per-sensor defaults UI that directly mirrors sensors.yaml.

Safe load/save that preserves unknown keys in both YAML files.

API surface + signals for RecorderTab:

current_host_config(), all_hosts(), sensor_defaults()

hostsUpdated and sensorsUpdated signals.

Using HostInventory + SensorDefaults in RecorderTab

Once you wire your RecorderTab to know about the SettingsTab, you can:

from ...config.app_config import HostInventory, SensorDefaults

# build Pi host + scripts dir without re-parsing YAML schema
inventory = HostInventory()
hosts = inventory.list_hosts()
host_cfg = hosts[0]
remote_host = inventory.to_remote_host(host_cfg)
scripts_dir = inventory.scripts_dir_for(host_cfg)

# build CLI args from sensor defaults
sensors = SensorDefaults()
mpu_args = sensors.build_mpu6050_cli_args()          # ['--rate', '200', '--channels', 'both', '--dlpf', '3']


You can also combine this with the live SettingsTab:

settings_tab: SettingsTab = ...

host_cfg = settings_tab.current_host_config()
sensor_cfg = settings_tab.sensor_defaults()

inventory = HostInventory()
remote_host = inventory.to_remote_host(host_cfg)
scripts_dir = inventory.scripts_dir_for(host_cfg)

mpu_defaults = sensor_cfg.get("mpu6050", {})

from ...config.app_config import build_mpu6050_cli_args

mpu_args = build_mpu6050_cli_args(mpu_defaults)


All ~ expansion happens right before use, not in the YAML itself.
"""
