"""QThread worker driving the Bioacoustic Chorus from the live stream.

QtCore only — no widgets are touched here (guardrail G1). All analysis and
synthesis happen on this thread (G4); the GUI receives bounded queue frames
plus a low-rate status signal.

Data arrives by PULL: ``RecorderController.snapshot_modal_capture`` is
documented thread-safe, so there is no per-sample signal traffic.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from queue import Queue

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ...core.pipeline import _offer_queue
from .audio_out import AudioOutput, WavCapture, is_audio_available
from .engine import ChorusEngine
from .types import BLOCK_SIZE, ChorusConfig

logger = logging.getLogger(__name__)

TICK_MS = 50
STATUS_MIN_INTERVAL = 0.2          # <=5 Hz status digests


class ChorusWorker(QObject):
    """Owns the engine, the audio device and the tick timer."""

    started_ok = Signal()
    stopped = Signal()
    error = Signal(str)
    status = Signal(dict)
    captured = Signal(str)

    def __init__(self, controller, viz_queue: Queue,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._queue = viz_queue
        self._engine: ChorusEngine | None = None
        self._audio: AudioOutput | None = None
        self._capture = WavCapture()
        self._timer: QTimer | None = None
        self._t0 = 0.0
        self._last_status = 0.0
        self._rate_hz = 0.0
        self._silent = False
        self._running = False
        # the accumulator only carries gyro on builds that store it; without it
        # the torsion voice would silently be driven by ax (it used to be)
        buf = getattr(controller, "_modal_buffer", None)
        self._has_gyro = "gz" in getattr(type(buf), "_AXIS_COLUMNS", {})

    # ------------------------------------------------------------------ start
    @Slot(object)
    def start(self, cfg: ChorusConfig) -> None:
        if self._running:
            return
        try:
            self._engine = ChorusEngine(cfg)
        except Exception as exc:
            logger.exception("chorus: engine construction failed")
            self.error.emit(f"Could not start the chorus engine: {exc}")
            return
        if not self._engine.available:
            self.error.emit(
                "Species catalog or grain bank missing — reinstall the package data.")
            return

        # Grow the shared accumulator if needed. Never shrink it: Mode B
        # continuous update uses the same buffer and a last-writer-wins resize
        # would truncate its data mid-run.
        want = max(120.0, cfg.id_window_s * 2)
        try:
            if hasattr(self._controller, "require_modal_window_seconds"):
                self._controller.require_modal_window_seconds(want)
            else:
                self._controller.set_modal_window_seconds(want)
        except Exception:
            logger.debug("chorus: could not resize the modal window", exc_info=True)

        self._silent = not is_audio_available()
        if not self._silent:
            self._audio = AudioOutput(block_size=BLOCK_SIZE)
            if not self._audio.start(self._pull_audio):
                self._audio = None
                self._silent = True

        self._t0 = time.monotonic()
        self._running = True
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()
        self.started_ok.emit()
        self._emit_status(force=True)

    # ------------------------------------------------------------------- stop
    @Slot()
    def stop(self) -> None:
        """Shut down on the worker thread. Idempotent."""
        if self._timer is None and self._engine is None and not self._running:
            self.stopped.emit()
            return
        self._running = False
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        if self._audio is not None:
            self._audio.stop()
            self._audio = None
        if self._capture.active:
            self._save_capture()
        self._engine = None
        self.stopped.emit()

    # ---------------------------------------------------------------- options
    @Slot(str, object)
    def set_option(self, key: str, value) -> None:
        if self._engine is None:
            return
        try:
            self._engine.set_option(key, value)
        except Exception:
            logger.debug("chorus: set_option(%r) failed", key, exc_info=True)

    @Slot()
    def refit_map(self) -> None:
        if self._engine is not None:
            try:
                self._engine.refit_casting_map()
            except Exception:
                logger.debug("chorus: refit failed", exc_info=True)

    @Slot(bool)
    def set_capture(self, enabled: bool) -> None:
        if enabled:
            self._capture.start()
        elif self._capture.active:
            self._save_capture()

    def _save_capture(self) -> None:
        try:
            from ...config.app_config import AppPaths
            base = Path(AppPaths().processed_data) / "sonification_chorus"
        except Exception:
            base = Path("data/processed/sonification_chorus")
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        try:
            path = self._capture.save(base / f"chorus_{stamp}.wav")
        except Exception as exc:
            logger.exception("chorus: capture save failed")
            self.error.emit(f"Could not save the capture: {exc}")
            return
        if path is not None:
            self.captured.emit(str(path))

    @Slot(str, float)
    def on_stream_rate(self, _sensor: str, rate_hz: float) -> None:
        self._rate_hz = float(rate_hz)

    # ------------------------------------------------------------------- tick
    def _pull_audio(self, frames: int):
        engine = self._engine
        if engine is None:
            return None
        block = engine.render_block(frames)
        self._capture.add(block)
        return block

    def _on_tick(self) -> None:
        engine = self._engine
        if engine is None or not self._running:
            return
        t = time.monotonic() - self._t0
        cfg = engine.cfg
        try:
            if engine.wants_reid(t):
                long_snap = self._controller.snapshot_modal_capture(
                    axis=cfg.axis, last_seconds=cfg.id_window_s)
                engine.reidentify(long_snap, t)
            ax_snap = self._controller.snapshot_modal_capture(
                axis=cfg.axis, last_seconds=cfg.fast_window_s)
            gz_snap = None
            if self._has_gyro:
                try:
                    gz_snap = self._controller.snapshot_modal_capture(
                        axis="gz", last_seconds=cfg.fast_window_s)
                except Exception:
                    gz_snap = None
            viz = engine.tick(t, ax_snap, gz_snap, rate_hz=self._rate_hz)
            engine.schedule_ahead()
            if self._silent:
                # no device: still advance the renderer so views and capture work
                target = engine.renderer.play_seconds + TICK_MS / 1000.0
                guard = 0
                while engine.renderer.play_seconds < target and guard < 200:
                    self._capture.add(engine.render_block())
                    guard += 1
            _offer_queue(self._queue, viz)
            self._emit_status()
        except Exception as exc:
            logger.exception("chorus: tick failed")
            # stop ourselves here; the tab's error handler must not also tear
            # us down or the teardown runs twice
            self.stop()
            self.error.emit(f"Chorus tick failed: {exc}")

    def _emit_status(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_status) < STATUS_MIN_INTERVAL:
            return
        self._last_status = now
        engine = self._engine
        if engine is None:
            return
        cfg = engine.cfg
        self.status.emit({
            # the fitted casting map, so the tab's sliders can stop lying
            "map": {"f_lo": cfg.f_lo, "f_hi": cfg.f_hi,
                    "c_lo": cfg.c_lo, "c_hi": cfg.c_hi, "autofit": cfg.autofit},
            "silent": self._silent,
            "buffer_s": round(engine.buffered_seconds, 3),
            "underruns": self._audio.underruns if self._audio else 0,
            "rate_hz": self._rate_hz,
            "capture_s": round(self._capture.seconds, 1) if self._capture.active else 0.0,
            "capturing": self._capture.active,
            "grains": engine.renderer.grains_written,
            "thinned": engine.renderer.thinned,
            "modal_ok": engine.modal.ok,
        })
