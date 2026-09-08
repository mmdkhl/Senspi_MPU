"""Upload a generated ``pi_config.yaml`` to a Pi — off the GUI thread.

The Settings tab used to do this inline: build the YAML, open an SSH session,
check two remote paths, write the file over SFTP, close — all in the button's
slot, on the GUI thread (G2 and G4 both broken). An unreachable Pi froze the
whole application for the SSH timeout.

This worker does the network part. The tab still builds the YAML and the
remote paths (pure work, no I/O) and only hands strings in; what comes back is a
signal. It follows :class:`LogSyncWorker` in shape.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal, Slot

from .ssh_client import SSHClient

logger = logging.getLogger(__name__)


class PiConfigSyncWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)       # remote path written
    error = Signal(str)

    def __init__(self, remote_host, *, data_dir: str, scripts_dir: str,
                 pi_config_path: str, contents: str, parent=None) -> None:
        super().__init__(parent)
        self._remote_host = remote_host
        self._data_dir = str(data_dir)
        self._scripts_dir = str(scripts_dir)
        self._pi_config_path = str(pi_config_path)
        self._contents = str(contents)

    @Slot()
    def run(self) -> None:
        client = SSHClient(self._remote_host)
        try:
            self.progress.emit("Connecting…")
            client.connect()
        except Exception as exc:
            self.error.emit(f"Could not connect: {exc}")
            return
        try:
            for label, path in (("data directory", self._data_dir),
                                ("scripts directory", self._scripts_dir)):
                self.progress.emit(f"Checking {label}…")
                if not client.path_exists(path):
                    self.error.emit(f"Remote {label} does not exist: {path}")
                    return
            self.progress.emit("Uploading pi_config.yaml…")
            with client.sftp() as sftp:
                with sftp.open(self._pi_config_path, "w") as fh:
                    fh.write(self._contents)
        except Exception as exc:
            self.error.emit(f"Failed to upload config: {exc}")
            return
        finally:
            try:
                client.close()
            except Exception:                        # pragma: no cover - defensive
                logger.debug("pi config sync: close failed", exc_info=True)
        self.finished.emit(self._pi_config_path)
