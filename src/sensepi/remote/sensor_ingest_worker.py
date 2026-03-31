"""Threaded worker that reads sensor samples and emits batched Qt signals."""

from __future__ import annotations

from typing import Callable, Iterable, Optional
import time

from PySide6.QtCore import QObject, Signal, Slot

from .pi_recorder import PiRecorder
from ..sensors.mpu6050 import MpuSample
from ..tools.debug import debug_enabled


class SensorIngestWorker(QObject):
    """QObject-based worker that pulls data from a PiRecorder stream and batches samples.

    It is meant to live in its own QThread: lines are parsed in the worker
    thread and emitted as small batches to the GUI via the samples_batch signal.
    """

    samples_batch = Signal(list)  # list[MpuSample]
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        recorder: PiRecorder,
        stream_factory: Callable[[], Iterable[str]],
        parser: Callable[[str], Optional[MpuSample]],
        *,
        batch_size: int = 50,
        max_latency_ms: int = 100,
        parent: QObject | None = None,
        stream_label: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._recorder = recorder
        self._stream_factory = stream_factory
        self._parser = parser
        self._batch_size = max(1, int(batch_size))
        self._max_latency_ms = max(0.0, float(max_latency_ms))
        self._running = False
        self._stream_label = stream_label or "stream"
        self._lines: Iterable[str] | None = None  # track current stream, if any

    @Slot()
    def start(self) -> None:
        """Entry point for the QThread: consume the remote stream and emit batches."""
        self._running = True
        buffer: list[MpuSample] = []
        last_emit = time.monotonic()
        debug_on = debug_enabled()
        debug_total_samples = 0
        debug_window_samples = 0
        debug_start = time.perf_counter()
        debug_last_log = debug_start

        try:
            try:
                lines = self._stream_factory()
                self._lines = lines  # track the current stream iterator
            except Exception as exc:  # pragma: no cover - best-effort guard
                self.error.emit(f"Failed to start sensor stream: {exc}")
                return

            for line in lines:
                if not self._running:
                    break
                if not line:
                    continue

                try:
                    sample = self._parser(line)
                except Exception as exc:  # pragma: no cover - parser errors
                    self.error.emit(f"Failed to parse sensor line: {exc}")
                    continue

                if sample is None:
                    continue

                buffer.append(sample)
                if debug_on:
                    debug_total_samples += 1
                    debug_window_samples += 1

                now = time.monotonic()
                latency_elapsed = (now - last_emit) * 1000.0
                # Emit a batch either when we have enough samples or when the
                # oldest one has been waiting longer than max_latency_ms.
                should_emit = len(buffer) >= self._batch_size
                if not should_emit and self._max_latency_ms > 0.0:
                    should_emit = latency_elapsed >= self._max_latency_ms

                if should_emit and buffer:
                    self.samples_batch.emit(list(buffer))
                    buffer.clear()
                    last_emit = now

                if debug_on:
                    perf_now = time.perf_counter()
                    if perf_now - debug_last_log >= 5.0:
                        elapsed_total = max(1e-9, perf_now - debug_start)
                        elapsed_window = max(1e-9, perf_now - debug_last_log)
                        avg_rate = debug_total_samples / elapsed_total
                        window_rate = debug_window_samples / elapsed_window
                        print(
                            f"[DEBUG] stream={self._stream_label} samples={debug_total_samples} "
                            f"avg≈{avg_rate:.1f} Hz recent≈{window_rate:.1f} Hz",
                            flush=True,
                        )
                        debug_last_log = perf_now
                        debug_window_samples = 0

            if buffer:
                self.samples_batch.emit(list(buffer))
        except Exception as exc:  # pragma: no cover - safety net for stream errors
            self.error.emit(str(exc))
        finally:
            self._running = False
            self._lines = None
            try:
                self._recorder.close()
            except Exception:
                pass
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        """Request the reading loop to terminate."""
        self._running = False
        
        # Try to actively close the current stream iterator, if it supports it.
        lines = self._lines
        close_fn = getattr(lines, "close", None) if lines is not None else None
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

        try:
            self._recorder.close()
        except Exception:
            pass
