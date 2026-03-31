from __future__ import annotations

"""Coordinator for running remote loggers and writing data locally."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ..dataio import csv_writer, file_paths
from ..remote.pi_recorder import PiRecorder
from .models import SessionInfo


class RecorderSession:
    def __init__(
        self,
        pi_recorder: PiRecorder,
        session_name: str,
        sensor_type: str,
        sample_rate_hz: float,
    ) -> None:
        self.pi_recorder = pi_recorder
        self.session_name = session_name
        self.sensor_type = sensor_type
        self.sample_rate_hz = sample_rate_hz
        self.session_dir: Path = file_paths.session_directory(session_name)
        self.meta = SessionInfo(
            name=session_name,
            sensor_type=sensor_type,
            sample_rate_hz=sample_rate_hz,
            started_at=datetime.now(timezone.utc),
            output_path=self.session_dir,
        )

    def start(self, script: str, args: Iterable[str] | None = None) -> int:
        """
        Start a remote logger script via :class:`PiRecorder`.

        Returns the PID of the started remote process.
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)
        return self.pi_recorder.start_logger(script, args)

    def save_rows(
        self,
        filename: str,
        headers: Sequence[str],
        rows: Iterable[Sequence[float]],
    ) -> None:
        csv_writer.write_rows(self.session_dir / filename, headers, rows)
