"""Sonification — the container tab that hosts sonification models.

One top-level tab, with a sub-tab per model:

* **Bioacoustic Chorus** — built and working (``tab_bioacoustic_chorus.py``)
* **Team Model** — the space held for the sonification team's own method

The team is still developing its approach. Keeping the models as sibling
sub-tabs means neither blocks the other: the team's work lands in its own sub-tab
without touching, competing with, or having to be merged into the chorus.

To fill the placeholder in, replace :class:`TeamModelTab`'s body with the real
UI. The chorus tab is a working reference for the pattern that keeps the
guardrails satisfied: the tab owns a ``QThread`` worker so no analysis or
synthesis runs on the GUI thread (G1/G4); frames cross back through a bounded
``_offer_queue`` drained by a ``QTimer`` (G5); no SSH lives in the tab — it holds
a ``RecorderController`` reference and pulls thread-safe
``snapshot_modal_capture`` calls from inside the worker (G2); and the engine
stays importable without Qt so it remains testable (G7).
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from .tab_bioacoustic_chorus import BioacousticChorusTab

logger = logging.getLogger(__name__)

BG = "#12151a"
PANEL = "#1a1f27"
FG = "#e8eaed"
DIM = "#8b93a1"
EDGE = "#2b3440"


class TeamModelTab(QWidget):
    """Placeholder held for the sonification team's implementation."""

    def __init__(self, recorder_controller=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # kept so the team can start from live data without re-wiring MainWindow
        self._controller = recorder_controller
        self.setStyleSheet(f"background:{BG};color:{FG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.addStretch(1)

        card = QLabel()
        card.setTextFormat(Qt.RichText)
        card.setWordWrap(True)
        card.setAlignment(Qt.AlignCenter)
        # Qt rich text ignores div padding, so vertical rhythm uses paragraph
        # margins rather than CSS padding.
        card.setText(
            f"<p style='font-size:21px;font-weight:bold;color:{FG};margin:0;'>"
            f"Tab under progress</p>"
            f"<p style='font-size:14px;color:#5ac8fa;margin:10px 0 0 0;'>"
            f"Placeholder for the sonification team</p>"
            f"<p style='margin:22px 0 0 0;'>&nbsp;</p>"
            f"<p style='font-size:12px;color:{DIM};margin:0;line-height:160%;'>"
            f"This space is reserved for the sonification team's own method.</p>"
            f"<p style='font-size:12px;color:{DIM};margin:14px 0 0 0;line-height:160%;'>"
            f"<b style='color:{FG};'>Bioacoustic Chorus</b> runs in the sibling tab.</p>"
        )
        card.setStyleSheet(
            f"background:{PANEL};border:1px solid {EDGE};border-radius:8px;padding:38px;")
        card.setMaximumWidth(660)

        row = QVBoxLayout()
        row.addWidget(card, alignment=Qt.AlignHCenter)
        root.addLayout(row)
        root.addStretch(2)

    @Slot()
    def on_stream_started(self) -> None:
        """No-op until the team's implementation lands."""

    @Slot()
    def on_stream_stopped(self) -> None:
        """No-op until the team's implementation lands."""

    def shutdown(self) -> None:
        """No-op. Present so application close can treat every model alike."""


class SonificationTab(QWidget):
    """Hosts the sonification models as sub-tabs."""

    def __init__(self, recorder_controller=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = recorder_controller
        self.setStyleSheet(f"background:{BG};color:{FG};")

        self.chorus_tab = BioacousticChorusTab(recorder_controller=recorder_controller)
        self.team_tab = TeamModelTab(recorder_controller=recorder_controller)

        self._tabs = QTabWidget()
        self._tabs.addTab(self.chorus_tab, self.tr("Bioacoustic Chorus"))
        self._tabs.addTab(self.team_tab, self.tr("Team Model"))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(self._tabs)

    @property
    def models(self) -> tuple:
        return (self.chorus_tab, self.team_tab)

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
