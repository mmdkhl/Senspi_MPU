"""Synthetic benchmark helpers for the SensePi GUI."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import QApplication

from ..perf_system import get_process_cpu_percent
from .main_window import MainWindow

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkOptions:
    rate_hz: float = 200.0
    duration_s: float = 30.0
    refresh_hz: float = 20.0
    channel_count: int = 18
    log_interval_s: float = 1.0
    csv_path: Optional[Path] = Path("benchmark_results.csv")
    sensor_count: int = 3
    keep_open: bool = False


class BenchmarkDriver(QObject):
    """Drive the GUI in synthetic benchmark mode."""

    def __init__(
        self,
        app: QApplication,
        window: MainWindow,
        options: BenchmarkOptions,
    ) -> None:
        super().__init__(window)
        self._app = app
        self._window = window
        self._options = options
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        interval_ms = int(max(100, round(options.log_interval_s * 1000.0)))
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._on_tick)
        self._start_monotonic: float = 0.0
        self._rows: List[dict[str, float]] = []
        self._started = False
        self._finished = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        signals_tab = self._window.signals_tab
        refresh_hz = max(0.5, float(self._options.refresh_hz))
        refresh_interval_ms = max(1, int(round(1000.0 / refresh_hz)))
        signals_tab.fixed_interval_ms = refresh_interval_ms
        signals_tab.set_refresh_mode("fixed")
        signals_tab.set_view_mode_by_channels(self._options.channel_count)
        signals_tab.set_perf_hud_visible(True)

        sensor_count = max(1, int(self._options.sensor_count))
        sensor_ids = list(range(1, sensor_count + 1))
        signals_tab.on_stream_started()
        signals_tab.start_synthetic_stream(
            rate_hz=self._options.rate_hz,
            sensor_ids=sensor_ids,
        )
        signals_tab.update_stream_rate("mpu6050", self._options.rate_hz)

        self._start_monotonic = time.perf_counter()
        self._timer.start()
        logger.info(
            "[benchmark] running at %.1f Hz for %.1f s, GUI refresh %.1f Hz, sensors=%d",
            self._options.rate_hz,
            self._options.duration_s,
            refresh_hz,
            sensor_count,
        )
        if self._options.duration_s <= 0.0:
            QTimer.singleShot(0, self._finish)

    def _on_tick(self) -> None:
        elapsed = time.perf_counter() - self._start_monotonic
        snap = self._window.signals_tab.get_perf_snapshot()
        fps = float(snap.get("fps", 0.0) or 0.0)
        target_fps = float(snap.get("target_fps", 0.0) or 0.0)
        timer_hz = float(snap.get("timer_hz", 0.0) or 0.0)
        avg_frame_ms = float(snap.get("avg_frame_ms", 0.0) or 0.0)
        avg_latency_ms = float(snap.get("avg_latency_ms", 0.0) or 0.0)
        max_latency_ms = float(snap.get("max_latency_ms", 0.0) or 0.0)
        approx_drop = snap.get("approx_dropped_fps")
        if approx_drop is None:
            approx_drop = snap.get("approx_dropped_frames_per_sec", 0.0)
        approx_drop = float(approx_drop or 0.0)

        try:
            cpu_percent = get_process_cpu_percent()
        except Exception as exc:  # pragma: no cover - metrics are best-effort
            logger.warning("Failed to read process CPU percent: %r", exc)
            cpu_percent = 0.0

        row = {
            "t": elapsed,
            "fps": fps,
            "target_fps": target_fps,
            "timer_hz": timer_hz,
            "avg_frame_ms": avg_frame_ms,
            "avg_latency_ms": avg_latency_ms,
            "max_latency_ms": max_latency_ms,
            "approx_dropped_fps": approx_drop,
            "cpu_percent": cpu_percent,
        }
        self._rows.append(row)

        logger.info(
            (
                "[benchmark] t=%5.1fs fps=%5.1f/%5.1f timer=%4.1fHz frame=%5.2fms "
                "lat=%5.2f/%5.2fms drop=%5.2f cpu=%5.1f%%"
            ),
            elapsed,
            fps,
            target_fps,
            timer_hz,
            avg_frame_ms,
            avg_latency_ms,
            max_latency_ms,
            approx_drop,
            cpu_percent,
        )

        if elapsed >= self._options.duration_s:
            self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._timer.stop()
        signals_tab = self._window.signals_tab
        signals_tab.stop_synthetic_stream()
        signals_tab.on_stream_stopped()
        if self._options.csv_path and self._rows:
            self._write_csv(self._options.csv_path)
        logger.info("[benchmark] completed; duration %.1f s", self._options.duration_s)
        if not self._options.keep_open:
            self._app.quit()

    def _write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "t",
            "fps",
            "target_fps",
            "timer_hz",
            "avg_frame_ms",
            "avg_latency_ms",
            "max_latency_ms",
            "approx_dropped_fps",
            "cpu_percent",
        ]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({k: row.get(k, 0.0) for k in fieldnames} for row in self._rows)
        logger.info("[benchmark] metrics written to %s", path)
