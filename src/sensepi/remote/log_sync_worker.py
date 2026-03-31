"""Background worker to sync log files from a Raspberry Pi over SFTP."""

from __future__ import annotations

import logging
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from PySide6.QtCore import QObject, Signal, Slot

from sensepi.config.app_config import AppPaths, HostInventory, normalize_remote_path
from sensepi.config.log_paths import LOG_SUBDIR_MPU, slugify_session_name
from sensepi.remote.ssh_client import SSHClient

logger = logging.getLogger(__name__)


def _is_log_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".csv") or lower.endswith(".jsonl") or lower.endswith(".meta.json")


def _download_tree(sftp, remote_dir: PurePosixPath, local_dir: Path, progress_cb=None) -> int:
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for attr in sftp.listdir_attr(str(remote_dir)):
        remote_path = remote_dir / attr.filename
        local_path = local_dir / attr.filename

        if stat.S_ISDIR(attr.st_mode):
            downloaded += _download_tree(sftp, remote_path, local_path, progress_cb=progress_cb)
            continue

        if not _is_log_file(attr.filename):
            continue

        if local_path.exists() and local_path.stat().st_size == attr.st_size:
            continue

        if progress_cb:
            progress_cb(f"Downloading {remote_path} …")

        sftp.get(str(remote_path), str(local_path))
        downloaded += 1

    return downloaded


class LogSyncWorker(QObject):
    progress = Signal(str)
    finished = Signal(str, int)  # (local_dir, files_downloaded)
    error = Signal(str)

    def __init__(
        self,
        host_inventory: HostInventory,
        host_dict: Mapping[str, Any],
        session_name: str | None,
        *,
        sensor_prefix: str = LOG_SUBDIR_MPU,
    ) -> None:
        super().__init__()
        self._inv = host_inventory
        self._host_dict = dict(host_dict)
        self._session_name = (session_name or "").strip() or None
        self._sensor_prefix = sensor_prefix

    @Slot()
    def run(self) -> None:
        try:
            app_paths = AppPaths()
            app_paths.ensure()

            host_cfg = self._inv.to_host_config(self._host_dict)
            remote_host = self._inv.to_remote_host(self._host_dict)

            client = SSHClient(remote_host)
            self.progress.emit(f"Connecting to {host_cfg.name} …")
            client.connect()

            try:
                remote_data_dir = normalize_remote_path(host_cfg.data_dir, host_cfg.user)
                remote_root = PurePosixPath(remote_data_dir) / self._sensor_prefix

                if not client.path_exists(str(remote_root)):
                    raise RuntimeError(f"Remote log directory does not exist: {remote_root}")

                if self._session_name:
                    remote_target = remote_root / slugify_session_name(self._session_name)
                else:
                    remote_target = remote_root

                if not client.path_exists(str(remote_target)):
                    raise RuntimeError(f"No logs found at: {remote_target}")

                if self._session_name:
                    local_target = app_paths.raw_data / slugify_session_name(self._session_name)
                else:
                    local_target = app_paths.raw_data / str(host_cfg.name) / self._sensor_prefix

                self.progress.emit(f"Syncing {remote_target} → {local_target} …")

                with client.sftp() as sftp:
                    n = _download_tree(
                        sftp,
                        remote_target,
                        local_target,
                        progress_cb=self.progress.emit,
                    )

                self.finished.emit(str(local_target), int(n))

            finally:
                client.close()

        except Exception as exc:
            logger.exception("Log sync failed")
            self.error.emit(str(exc))
