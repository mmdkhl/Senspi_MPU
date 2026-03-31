"""Remote communication helpers for controlling Raspberry Pi loggers.

Classes such as :class:`SSHClient` and :class:`PiRecorder` establish SSH/SFTP
connections, start sensor scripts on the Pi, and stream stdout/stderr back to
the desktop application.
"""

from .ssh_client import Host, SSHClient, SSHConfig
from .pi_recorder import PiRecorder
from .log_sync import SyncReport, sync_logs_from_pi
from .log_sync_worker import LogSyncWorker

__all__ = [
    "Host",
    "SSHClient",
    "SSHConfig",
    "PiRecorder",
    "LogSyncWorker",
    "SyncReport",
    "sync_logs_from_pi",
]
