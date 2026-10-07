"""Sonification — the container tab that hosts sonification models.

One top-level tab, with a sub-tab per model:

* **Bioacoustic Chorus** — each eigenfrequency casts an animal and chirps at
  its own rate (``tab_bioacoustic_chorus.py``)
* **Structure Pulse** — the measured building as a graphical score: a playhead
  sweeps a plot and the curve becomes sound (``tab_structure_pulse.py``)

Sibling sub-tabs rather than one merged model, so each can change without
disturbing the other. Both follow the same rules: analysis and synthesis run off
the GUI thread (G1/G4), no SSH lives in a tab — they hold a
``RecorderController`` reference and pull thread-safe ``snapshot_modal_capture``
calls (G2) — and both engines import no Qt, so they stay testable (G7).
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from .tab_bioacoustic_chorus import BioacousticChorusTab
from .tab_structure_pulse import StructurePulseTab

logger = logging.getLogger(__name__)

# Chrome follows the operating system, like every other tab: BG is a Qt
# palette role so the window colour is the native one, and the remaining
# values are tuned for that native (light) chrome. The plot canvases stay
# dark to match the Spectrum tab, so anything drawn ON a plot uses the
# *_ON_DARK / PLOT_* values instead.
from .. import theme
BG = "palette(window)"
PANEL = "palette(base)"
FG = "palette(text)"


class SonificationTab(QWidget):
    """Hosts the sonification models as sub-tabs."""

    def __init__(self, recorder_controller=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = recorder_controller
        self.setStyleSheet(f"background:{BG};color:{FG};")

        self.chorus_tab = BioacousticChorusTab(recorder_controller=recorder_controller)
        self.pulse_tab = StructurePulseTab(recorder_controller=recorder_controller)

        self._tabs = QTabWidget()
        self._tabs.addTab(self.chorus_tab, self.tr("Bioacoustic Chorus"))
        self._tabs.addTab(self.pulse_tab, self.tr("Structure Pulse"))
        # Two different models of the same measurement, so say which is which
        # before the user has to open both to find out.
        self._tabs.setTabToolTip(0, self.tr(
            "The structure sings as a meadow of animals. Each eigenfrequency "
            "becomes a species, and its chirp rate is that frequency."))
        self._tabs.setTabToolTip(1, self.tr(
            "A plotted result becomes a score. A playhead sweeps it left to "
            "right and the height of the curve becomes sound."))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(self._tabs)

    @property
    def models(self) -> tuple:
        return (self.chorus_tab, self.pulse_tab)

    def apply_sensor_map(self, mapping) -> None:
        """Forward the Settings placement map to every model that wants it.

        The team's model may never want it, so this asks rather than requires.
        """
        for model in self.models:
            fn = getattr(model, "apply_sensor_map", None)
            if fn is None:
                continue
            try:
                fn(mapping)
            except Exception:                      # pragma: no cover - defensive
                logger.debug("sonification: apply_sensor_map failed", exc_info=True)

    # --- forwarded so MainWindow keeps a single connection per signal -----
    @Slot()
    def on_stream_started(self) -> None:
        for model in self.models:
            try:
                model.on_stream_started()
            except Exception:                      # pragma: no cover - defensive
                logger.debug("sonification: on_stream_started failed", exc_info=True)

    @Slot()
    def on_stream_stopped(self) -> None:
        for model in self.models:
            try:
                model.on_stream_stopped()
            except Exception:                      # pragma: no cover - defensive
                logger.debug("sonification: on_stream_stopped failed", exc_info=True)

    def shutdown(self) -> None:
        """Stop every model. Called by MainWindow on application close."""
        for model in self.models:
            try:
                model.shutdown()
            except Exception:                      # pragma: no cover - defensive
                logger.debug("sonification: shutdown failed", exc_info=True)
