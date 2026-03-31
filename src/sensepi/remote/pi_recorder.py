"""High-level helpers for starting and streaming logger scripts on the Pi."""

from __future__ import annotations

import shlex
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Optional

from ..config.app_config import DEFAULT_BASE_PATH
from ..config.pi_logger_config import PiLoggerConfig
from .ssh_client import Host, SSHClient


class PiRecorder:
    """Launches Raspberry Pi logger scripts over SSH."""

    def __init__(self, host: Host, base_path: Optional[Path] = None) -> None:
        self.host = host
        if base_path is None:
            base_path = DEFAULT_BASE_PATH

        # Ensure the remote path is always POSIX-style, even on Windows hosts.
        base_path = Path(base_path).expanduser()
        self.base_path = PurePosixPath(base_path.as_posix())
        self.client = SSHClient(host)

    # ------------------------------------------------------------------ connection
    def connect(self) -> None:
        """Ensure an SSH connection is open."""
        self.client.connect()

    def close(self) -> None:
        """Close the SSH connection (does not kill remote loggers)."""
        self.client.close()

    # ------------------------------------------------------------------ simple runner
    def start_logger(
        self, script_name: str, args: Optional[Iterable[str]] = None
    ):
        """
        Run a logger script and return (stdout, stderr) file-like objects.

        This is mainly useful for short-lived commands or testing; for live
        streaming use :meth:`stream_mpu6050`.
        """
        cmd_parts: list[str] = [
            "python3",
            str(self.base_path / script_name),
        ]
        if args:
            cmd_parts.extend(args)

        # Safely quote each part for the remote shell
        command = " ".join(shlex.quote(part) for part in cmd_parts)

        self.connect()
        _, stdout, stderr = self.client.run(command)
        return stdout, stderr

    # ------------------------------------------------------------------ streaming
    def _stream_logger(
        self,
        script_name: str,
        extra_args: str = "",
        on_stderr: Optional[Callable[[str], None]] = None,
        *,
        recording: bool = False,
    ) -> Iterable[str]:
        """
        Internal helper: start a logger in ``--stream-stdout`` mode.

        ``extra_args`` is a free-form CLI string; this helper will append
        ``--stream-stdout`` and optionally ``--no-record`` depending on ``recording``.
        """
        self.connect()

        extra_args = extra_args.strip()
        parts = shlex.split(extra_args) if extra_args else []

        if "--stream-stdout" not in parts:
            parts.append("--stream-stdout")

        wants_recording = recording or ("--record" in parts)

        # Ensure we don't accidentally send both flags
        parts = [p for p in parts if p != "--no-record"]

        if not wants_recording and "--no-record" not in parts:
            parts.append("--no-record")

        cmd_parts = ["python3", script_name, *parts]
        cmd = " ".join(shlex.quote(part) for part in cmd_parts)

        # Use cwd so the script can rely on relative paths.
        cwd = self.base_path.as_posix()
        return self.client.exec_stream(cmd, cwd=cwd, stderr_callback=on_stderr)

    def stream_mpu6050(
        self,
        cfg: PiLoggerConfig,
        recording_enabled: bool,
        session_name: Optional[str] = None,
    ) -> Iterable[str]:
        """
        Start the mpu logger on the Pi and stream samples via stdout.

        If ``recording_enabled`` is False, ``--no-record`` is appended. The
        stream always includes ``--stream-stdout``.
        """

        extra = []
        if not recording_enabled:
            extra.append("--no-record")
        extra.append("--stream-stdout")
        has_session_flag = False
        try:
            has_session_flag = bool(getattr(cfg, "extra_cli", {}).get("session_name"))
        except Exception:
            has_session_flag = False

        if session_name and not has_session_flag:
            extra.extend(["--session-name", session_name])

        cmd_parts = cfg.build_command(extra_cli=" ".join(extra))
        cmd = " ".join(shlex.quote(part) for part in cmd_parts)
        return self.client.exec_stream(cmd, cwd=self.base_path.as_posix())

    def start_record_only(self, cfg: PiLoggerConfig) -> Iterable[str]:
        """
        Start the logger on the Pi in record-only mode (no stdout streaming).
        """

        cmd_parts = cfg.build_command()
        cmd = " ".join(shlex.quote(part) for part in cmd_parts)
        return self.client.exec_stream(cmd, cwd=self.base_path.as_posix())

    # ------------------------------------------------------------------ convenience
    def stop(self) -> None:
        """Alias for :meth:`close` to match older code."""
        self.close()
