"""Decision lights: what to change on the PHYSICAL structure.

Shared by the Digital Twin experiment (one calibration) and Model Updating's
Continuous Update output (per cycle / rolling average). The reasoning — the
rebalance alternative and the caveats — lives in the summary's tooltip; on the
rig, four lights and one line are what get read.
"""
from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from ...digital_twin import decisions as twin_decisions

_LIGHT = {
    twin_decisions.NOTHING: "#2E7D32",
    twin_decisions.ADD: "#E4572E",
    twin_decisions.REMOVE: "#E4572E",
    twin_decisions.STIFFEN: "#1565C0",
    twin_decisions.SOFTEN: "#1565C0",
}


class DecisionPanel(QGroupBox):
    def __init__(self, title: str = "Decisions, change the structure to match the design",
                 parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(3)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(2)
        col.addWidget(self._body)
        self.summary = QLabel("Calibrate to see what the structure needs.")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color:#3A4A5C;font-size:11px;")
        col.addWidget(self.summary)
        col.addStretch(1)

    def set_decisions(self, decisions, *, source: str = "") -> None:
        layout = self._body_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if decisions is None or not getattr(decisions, "ok", False):
            self.summary.setText((getattr(decisions, "message", "") if decisions else "")
                                 or "Calibrate to see what the structure needs.")
            self.summary.setToolTip("")
            return
        rows = list(reversed(decisions.masses))          # top storey first
        if decisions.stiffness is not None:
            rows.append(decisions.stiffness)
        for d in rows:
            colour = _LIGHT.get(d.action, "#7A8794")
            name = "E" if d.target == "stiffness" else f"F{d.story}"
            label = QLabel(f"<span style='color:{colour};font-size:14px'>\u25cf</span> "
                           f"<b>{name}</b>&nbsp;&nbsp;{d.headline()}")
            label.setToolTip(d.detail())
            label.setStyleSheet("font-size:11px;")
            layout.addWidget(label)
        prefix = f"{source} · " if source else ""
        self.summary.setText(prefix + decisions.summary() + "  (hover for details)")
        bits = []
        plan = decisions.rebalance
        if plan:
            moves = ", ".join(f"F{k} {v:+.3f}" for k, v in sorted(plan.items()) if k)
            bits.append(f"Rebalance instead: {moves} (net {plan.get(0, 0.0):+.3f}).")
        bits.extend(decisions.notes)
        self.summary.setToolTip("\n\n".join(bits))
