"""High-level helpers for starting and streaming logger scripts on the Pi."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Optional

from ..config.app_config import DEFAULT_BASE_PATH
from ..config.pi_logger_config import PiLoggerConfig
from .ssh_client import Host, SSHClient

logger = logging.getLogger(__name__)


class PiRecorder:
    """Launches Raspberry Pi logger scripts over SSH."""

    # Standalone OLED status display, if present alongside the logger scripts
    # (see raspberrypi_scripts_4_sensor/oled_status.py). It reads the
    # heartbeat file the logger writes and is intentionally decoupled from
    # it -- a crash in either process can't take down the other -- so it's
    # started independently here rather than bundled into the logger command.
    OLED_SCRIPT = "oled_status.py"

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

    def clear_recording_output(self, output_dir: str) -> None:
        """Delete previous recording files from the configured Pi output folder."""

        target = PurePosixPath(str(output_dir))
        target_text = target.as_posix()
        if target_text in {"", "/", "/home"} or target.name != "mpu":
            raise ValueError(f"Refusing to clear unsafe recording folder: {target_text!r}")

        quoted_target = shlex.quote(target_text)
        cmd = (
            f"mkdir -p {quoted_target} && "
            f"find {quoted_target} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +"
        )
        _stdin, stdout, stderr = self.client.run(cmd)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            message = stderr.read()
            if isinstance(message, bytes):
                message = message.decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Failed to clear recording folder {target_text}: {message}"
            )

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

    # ------------------------------------------------------------------ OLED display
    def _ensure_oled_running(self) -> None:
        """
        Start the Pi-side OLED status display if it isn't already running.

        Best-effort and idempotent: if it's already running this is a no-op;
        if the Pi has no OLED script deployed or no display wired up, the
        remote command fails silently and recording proceeds unaffected.
        Deliberately left running after recording stops (rather than killed
        in :meth:`close`/`stop`) so the display keeps showing status between
        sessions instead of going blank.

        The check and the launch are two *separate* SSH commands, not one
        combined ``pgrep ... || nohup ...`` shell line. ``pgrep -f`` matches
        against every process's full command line, including its own parent
        shell's -- so if the check and the launch shared one shell
        invocation, the parent shell's own command text (which necessarily
        contains "oled_status.py" from the launch half) would always match
        the check, and the launch half would silently never run. Splitting
        them avoids that self-match. The ``[o]`` bracket in the check
        pattern is the standard trick to also keep the check itself from
        matching its own argv (same idea as ``ps aux | grep '[o]led'``).
        """
        try:
            self.connect()
            _, stdout, _ = self.client.run("pgrep -f '[o]led_status.py'")
            already_running = stdout.channel.recv_exit_status() == 0
            if already_running:
                return

            base = self.base_path.as_posix()
            launch_cmd = (
                f"cd {shlex.quote(base)} && "
                f"nohup python3 {shlex.quote(self.OLED_SCRIPT)} --interval 1.0 "
                f">/tmp/sensepi_oled.log 2>&1 &"
            )
            self.client.run(launch_cmd)
        except Exception:
            logger.warning("Could not start OLED status display on %s", self.host.host, exc_info=True)

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

        self._ensure_oled_running()

        cmd_parts = cfg.build_command(extra_cli=" ".join(extra))
        cmd = " ".join(shlex.quote(part) for part in cmd_parts)
        return self.client.exec_stream(cmd, cwd=self.base_path.as_posix())

    def start_record_only(self, cfg: PiLoggerConfig) -> Iterable[str]:
        """
        Start the logger on the Pi in record-only mode (no stdout streaming).
        """

        self._ensure_oled_running()

        cmd_parts = cfg.build_command()
        cmd = " ".join(shlex.quote(part) for part in cmd_parts)
        return self.client.exec_stream(cmd, cwd=self.base_path.as_posix())

    # ------------------------------------------------------------------ convenience
    def stop_remote_logger(self) -> None:
        """
        Explicitly terminate the remote ``mpu6050_multi_logger.py`` process.

        Closing the local SSH channel/stream is not enough to stop it: the
        script's per-sample loop catches the broken-pipe error that results
        from a closed channel internally (so its stdout stream can survive a
        dropped GUI connection) and just keeps looping, still sampling
        sensors and refreshing the "publishing" OLED heartbeat forever. This
        sends SIGTERM instead, which the script's own signal handler turns
        into a clean shutdown -- heartbeat flips to "not publishing", per
        sensor writers flushed and closed. Best-effort: failures here are
        logged, not raised, so a stop action never gets stuck on this.

        Note this matches by script name (``pkill -f``), so it will also
        stop any other ``mpu6050_multi_logger.py`` instance running on the
        same Pi outside the GUI (e.g. a manually started
        ``run_all_sensors.sh``) -- deliberate, since a stray instance is
        exactly what would otherwise keep the OLED stuck on "publishing".
        """
        try:
            self.connect()
            self.client.run("pkill -f mpu6050_multi_logger.py")
        except Exception:
            logger.warning(
                "Could not stop remote logger on %s", self.host.host, exc_info=True
            )

    def stop(self) -> None:
        """Alias for :meth:`close` to match older code."""
        self.close()
