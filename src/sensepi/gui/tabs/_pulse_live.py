"""Live mode for Structure Pulse: a rolling record / analyse / play loop.

One worker thread runs the cycle forever:

    capture N s  ->  analyse  ->  render the chosen view  ->  hand it over

while the tab plays the *previous* render. Preparing the next cycle during
playback is what keeps the sound continuous: by the time the current buffer
ends, the next one is already waiting, so the only gap is the swap itself.

The loop is deliberately **one cycle ahead, not more**. A deeper queue would
drift further and further behind the structure, and the point of live mode is
to hear what the building is doing now.

QtCore only — no widgets are touched here (G1). No SSH (G2): the worker pulls
thread-safe ``snapshot_modal_capture`` windows, the same as every other model.
"""
from __future__ import annotations

import logging
import time

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)

#: Never cycle faster than this; analysis plus render needs room.
MIN_WINDOW_S = 3.0

#: Modal identification refuses anything at or under 10 s, and a "10 s" request
#: actually yields 9.99 s of samples, so it failed by a hair and the spectrum
#: came back with no eigenfrequencies at all. The CYCLE may be as short as the
#: user likes; the window ANALYSED is widened to at least this, overlapping
#: previous cycles the way rolling analysis normally does.
MIN_ANALYSIS_S = 16.0


class LivePulseWorker(QObject):
    """Rolls capture -> analyse -> render on its own thread."""

    #: dataset, render, cycle number
    ready = Signal(object, object, int)
    progress = Signal(float, int)          # 0..1 through the current capture
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, controller, mapping, *, window_s: float = 10.0,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._mapping = mapping
        self._window_s = max(float(window_s), MIN_WINDOW_S)
        self._view_name = ""
        self._config = None
        self._running = False
        self._cycle = 0

    # ------------------------------------------------------------- settings
    _UNSET = object()

    def configure(self, *, view_name: str = None, config=None,
                  window_s: float = None, mapping=_UNSET) -> None:
        """Change what the next cycle renders. Safe to call from the GUI thread:
        these are plain assignments read once per cycle.

        ``mapping`` uses a sentinel rather than ``None`` as its default, because
        *no placement map* is a real state the user can be in — treating None as
        "leave it alone" would make clearing the map impossible.
        """
        if view_name is not None:
            self._view_name = str(view_name)
        if config is not None:
            self._config = config
        if window_s is not None:
            self._window_s = max(float(window_s), MIN_WINDOW_S)
        if mapping is not self._UNSET:
            self._mapping = mapping

    def stop(self) -> None:
        self._running = False

    # ----------------------------------------------------------------- loop
    def run(self) -> None:
        from ...sonification.structure_pulse import (build_dataset_from_capture,
                                                     render_view)

        ctrl = self._controller
        if ctrl is None or not hasattr(ctrl, "snapshot_modal_capture"):
            self.failed.emit("live mode needs a running stream")
            self.stopped.emit()
            return
        self._running = True
        try:
            if hasattr(ctrl, "require_modal_window_seconds"):
                ctrl.require_modal_window_seconds(
                    max(self._window_s * 3, MIN_ANALYSIS_S * 2, 120.0))
        except Exception:
            logger.debug("live: could not grow the modal window", exc_info=True)

        while self._running:
            window = self._window_s
            # 1. let the accumulator fill, reporting progress so the tab can
            #    show the cycle advancing rather than appearing to hang
            t0 = time.monotonic()
            while self._running:
                done = (time.monotonic() - t0) / max(window, 1e-6)
                if done >= 1.0:
                    break
                self.progress.emit(min(done, 1.0), self._cycle)
                QThread.msleep(100)
            if not self._running:
                break

            # 2. analyse, then render whatever view is currently selected.
            #    The analysed window is widened so identification can actually
            #    succeed — the cycle rate is unaffected.
            analysis_s = max(window, MIN_ANALYSIS_S)
            try:
                ds = build_dataset_from_capture(
                    ctrl.snapshot_modal_capture, seconds=analysis_s,
                    mapping=self._mapping)
            except Exception as exc:
                logger.exception("live: analysis failed")
                self.failed.emit(f"live analysis failed: {exc}")
                QThread.msleep(500)
                continue
            if not ds.view_names():
                self.failed.emit("live: no data in that window")
                continue

            view = ds.view(self._view_name) if self._view_name else None
            if view is None or not view.is_usable:
                view = ds.views[ds.view_names()[0]]
            cfg = self._config
            if cfg is None:
                from ...sonification.structure_pulse import PulseConfig

                cfg = PulseConfig()
            try:
                render = render_view(view, cfg,
                                     modal_frequencies=ds.modal.frequencies_hz,
                                     data_fs=ds.fs)
            except Exception as exc:
                logger.exception("live: render failed")
                self.failed.emit(f"live render failed: {exc}")
                continue

            self._cycle += 1
            self.ready.emit(ds, render, self._cycle)
        self.stopped.emit()
