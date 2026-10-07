"""Sensor placement map — where each sensor physically sits on the structure.

The map is the one authoritative answer to "which sensor is where", so that
every consumer (Spectrum, Model Updating, Sonification) stops guessing. Each
sensor gets:

* a **floor** — ``0`` is the shaker/base, ``1..n`` are structural storeys
* a **plan cell** — ``A1``..``C3`` on a 3x3 grid, ``B2`` being the centre
* a **role**, derived rather than typed:
    ``base``      floor 0, measures the INPUT, not the response
    ``structure`` floor >= 1, a response DOF
    ``unused``    not mounted

Presets are the supported path: they encode placements that are known to give
correct results. Free placement is allowed but is the user's risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import shiboken6
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QSizePolicy, QSpinBox,
                               QVBoxLayout, QWidget)

# Matches the storey range Model Updating allows (1-20). At 6 a 7-storey model
# could never be fully instrumented and the coverage warning fired forever.
MAX_FLOORS = 20
MAX_SENSORS = 4
COLS = ("A", "B", "C")
ROWS = ("1", "2", "3")
CELLS = tuple(f"{c}{r}" for r in ROWS for c in COLS)   # A1 B1 C1 A2 ... C3
CENTRE = "B2"
UNASSIGNED = "—"

SENSOR_COLORS = ("#5ac8fa", "#7ee787", "#ff9f43", "#c792ea")
BASE_COLOR = "#e05c5c"


@dataclass
class SensorPlacement:
    """Where one sensor sits. ``floor`` 0 = shaker/base, ``None`` = unused."""

    sensor_id: int
    floor: int | None = None
    cell: str = CENTRE

    @property
    def role(self) -> str:
        if self.floor is None:
            return "unused"
        return "base" if self.floor == 0 else "structure"


@dataclass
class SensorMap:
    """The full placement, plus the shaking direction."""

    n_floors: int = 3
    axis: str = "x"
    placements: list = field(default_factory=list)
    #: Name of the preset this equals, or "Custom" for a hand-edited placement.
    #: Carried into recordings so a session says which supported layout it used.
    preset: str = ""

    def structural(self) -> list:
        return [p for p in self.placements if p.role == "structure"]

    def base(self):
        return next((p for p in self.placements if p.role == "base"), None)

    def story_map(self) -> dict:
        """``{sensor_id: floor}`` for structural sensors only."""
        return {p.sensor_id: p.floor for p in self.structural()}

    def covered_floors(self) -> list:
        return sorted({p.floor for p in self.structural()})

    def to_mapping(self) -> dict:
        """Plain dict for sensors.yaml."""
        return {
            "n_floors": int(self.n_floors),
            "axis": str(self.axis),
            "preset": str(self.preset or ""),
            "placements": [
                {"sensor_id": p.sensor_id, "floor": p.floor, "cell": p.cell}
                for p in self.placements
            ],
        }

    @classmethod
    def from_mapping(cls, data) -> "SensorMap":
        if not isinstance(data, dict):
            return cls()
        placements = []
        for row in data.get("placements") or []:
            try:
                placements.append(SensorPlacement(
                    sensor_id=int(row["sensor_id"]),
                    floor=None if row.get("floor") is None else int(row["floor"]),
                    cell=str(row.get("cell") or CENTRE),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return cls(n_floors=int(data.get("n_floors", 3) or 3),
                   axis=str(data.get("axis", "x") or "x"),
                   placements=placements,
                   preset=str(data.get("preset") or ""))


# --- presets -------------------------------------------------------------
# (label, n_floors, {sensor_id: (floor, cell)})
PRESETS: dict = {
    "3 storeys — base + one per floor (centre)": (
        3, {1: (0, CENTRE), 2: (1, CENTRE), 3: (2, CENTRE), 4: (3, CENTRE)}),
    # Torsion pairs sit on the A3 / C1 diagonal — the corners the rig actually
    # uses. Any two different cells work for the maths (the lever arm is the
    # perpendicular separation, 2 cells on either axis here); the preset must
    # simply say where the sensors really are.
    "3 storeys — one per floor + torsion pair on top": (
        3, {1: (1, CENTRE), 2: (2, CENTRE), 3: (3, "A3"), 4: (3, "C1")}),
    "4 storeys — one per floor (centre)": (
        4, {1: (1, CENTRE), 2: (2, CENTRE), 3: (3, CENTRE), 4: (4, CENTRE)}),
    "4 storeys — two floors + torsion pair on top": (
        4, {1: (1, CENTRE), 2: (2, CENTRE), 3: (4, "A3"), 4: (4, "C1")}),
    "5 storeys — floors 1, 2, 4, 5 (centre)": (
        5, {1: (1, CENTRE), 2: (2, CENTRE), 3: (4, CENTRE), 4: (5, CENTRE)}),
    "5 storeys — floors 1, 3 (centre) + torsion pair on top": (
        5, {1: (1, CENTRE), 2: (3, CENTRE), 3: (5, "A3"), 4: (5, "C1")}),
    "2 storeys — base + 2 floors + torsion pair on top": (
        2, {1: (0, CENTRE), 2: (1, CENTRE), 3: (2, "A3"), 4: (2, "C1")}),
    "6 storeys — base + floors 1, 3, 6 (centre)": (
        6, {1: (0, CENTRE), 2: (1, CENTRE), 3: (3, CENTRE), 4: (6, CENTRE)}),
    "Custom": (0, {}),
}
CUSTOM = "Custom"


class _MapFigure(QWidget):
    """Elevation + plan reference figure, drawn from the current placement."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._map = SensorMap()

    def set_map(self, smap: SensorMap) -> None:
        self._map = smap
        self.update()

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event) -> None:          # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#12151a"))
        w, h = self.width(), self.height()
        split = int(w * 0.46)
        self._draw_elevation(p, QRectF(6, 6, split - 12, h - 12))
        self._draw_plan(p, QRectF(split + 6, 6, w - split - 12, h - 12))
        p.end()

    def _draw_elevation(self, p: QPainter, r: QRectF) -> None:
        m = self._map
        p.setPen(QPen(QColor("#8b93a1")))
        p.setFont(QFont("", 8, QFont.Bold))
        p.drawText(QRectF(r.x(), r.y(), r.width(), 14), Qt.AlignLeft, "ELEVATION")

        top, bottom = r.y() + 22, r.bottom() - 34
        n = max(m.n_floors, 1)
        step = (bottom - top) / max(n, 1)
        x0, x1 = r.x() + 26, r.right() - 14

        # ground / shaker
        p.setPen(QPen(QColor("#8b93a1"), 3))
        p.drawLine(QPointF(x0 - 8, bottom), QPointF(x1 + 4, bottom))
        for i in range(7):
            gx = x0 - 6 + i * (x1 - x0 + 8) / 6
            p.drawLine(QPointF(gx, bottom), QPointF(gx - 6, bottom + 8))
        p.setFont(QFont("", 7))
        p.setPen(QPen(QColor("#8b93a1")))
        p.drawText(QRectF(r.x(), bottom + 14, r.width(), 14),
                   Qt.AlignHCenter, "floor 0  ·  base / shaker")

        # columns + floor slabs
        for f in range(1, n + 1):
            y = bottom - f * step
            p.setPen(QPen(QColor("#3d4756"), 2))
            p.drawLine(QPointF(x0, bottom - (f - 1) * step), QPointF(x0, y))
            p.drawLine(QPointF(x1, bottom - (f - 1) * step), QPointF(x1, y))
            p.setPen(QPen(QColor("#e8eaed"), 4))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
            p.setPen(QPen(QColor("#8b93a1")))
            p.setFont(QFont("", 7))
            p.drawText(QRectF(r.x(), y - 7, 22, 14), Qt.AlignRight, str(f))

        # sensor dots
        for pl in m.placements:
            if pl.floor is None:
                continue
            y = bottom - pl.floor * step
            col = COLS.index(pl.cell[0]) if pl.cell[0] in COLS else 1
            x = x0 + (x1 - x0) * (0.5 if pl.floor == 0 else (0.18 + 0.32 * col))
            colour = QColor(BASE_COLOR if pl.floor == 0
                            else SENSOR_COLORS[(pl.sensor_id - 1) % len(SENSOR_COLORS)])
            p.setBrush(colour)
            p.setPen(QPen(QColor("#12151a"), 1))
            p.drawEllipse(QPointF(x, y), 6, 6)
            p.setPen(QPen(QColor("#12151a")))
            p.setFont(QFont("", 6, QFont.Bold))
            p.drawText(QRectF(x - 6, y - 6, 12, 12), Qt.AlignCenter, str(pl.sensor_id))

    def _draw_plan(self, p: QPainter, r: QRectF) -> None:
        m = self._map
        p.setPen(QPen(QColor("#8b93a1")))
        p.setFont(QFont("", 8, QFont.Bold))
        p.drawText(QRectF(r.x(), r.y(), r.width(), 14), Qt.AlignLeft, "PLAN (looking down)")

        pad = 26
        side = min(r.width() - pad - 10, r.height() - pad - 78)
        gx, gy = r.x() + pad, r.y() + 24
        cell = side / 3.0

        # The elevation left a brush set for the sensor dots. Without clearing it
        # drawRect() fills EVERY grid cell with that colour.
        p.setBrush(Qt.NoBrush)

        occupied: dict = {}
        for pl in m.placements:
            if pl.floor is not None:
                occupied.setdefault(pl.cell, []).append(pl)

        for ri in range(3):
            for ci in range(3):
                name = f"{COLS[ci]}{ROWS[ri]}"
                rect = QRectF(gx + ci * cell, gy + ri * cell, cell, cell)
                here = occupied.get(name, [])
                if here:
                    c = QColor(BASE_COLOR if here[0].floor == 0
                               else SENSOR_COLORS[(here[0].sensor_id - 1) % len(SENSOR_COLORS)])
                    c.setAlpha(70)
                    p.fillRect(rect, c)
                p.setPen(QPen(QColor("#3d4756")))
                p.drawRect(rect)
                p.setPen(QPen(QColor("#e8eaed" if here else "#6b7280")))
                if here:
                    p.setFont(QFont("", 8, QFont.Bold))
                    p.drawText(QRectF(rect.x(), rect.y() + 3, rect.width(), rect.height() * 0.45),
                               Qt.AlignCenter, name)
                    tags = " ".join(f"S{x.sensor_id}" for x in here)
                    p.setFont(QFont("", 5 if len(here) > 2 else (6 if len(here) > 1 else 7),
                                    QFont.Bold))
                    p.drawText(QRectF(rect.x() + 1, rect.y() + rect.height() * 0.45,
                                      rect.width() - 2, rect.height() * 0.5),
                               Qt.AlignCenter, tags)
                else:
                    p.setFont(QFont("", 8))
                    p.drawText(rect, Qt.AlignCenter, name)
                p.setBrush(Qt.NoBrush)

        # axes: a plain L below the grid. Per-arrow captions were cramped, so the
        # arrows carry only "x"/"y" and one caption line states the shaking axis.
        shake = m.axis.lower()
        ox, oy = gx + 14, gy + side + 52
        for label, is_x in (("x", True), ("y", False)):
            hot = (label == shake)
            colour = QColor("#ffd93d" if hot else "#6b7280")
            p.setPen(QPen(colour, 2 if hot else 1))
            p.setBrush(colour)
            if is_x:
                tip = QPointF(ox + 40, oy)
                head = QPolygonF([QPointF(tip.x() + 8, tip.y()),
                                  QPointF(tip.x(), tip.y() - 4),
                                  QPointF(tip.x(), tip.y() + 4)])
                lab = QRectF(tip.x() + 11, oy - 9, 16, 18)
            else:
                tip = QPointF(ox, oy - 30)
                head = QPolygonF([QPointF(tip.x(), tip.y() - 8),
                                  QPointF(tip.x() - 4, tip.y()),
                                  QPointF(tip.x() + 4, tip.y())])
                lab = QRectF(tip.x() - 20, tip.y() - 12, 16, 18)
            p.drawLine(QPointF(ox, oy), tip)
            p.drawPolygon(head)
            p.setFont(QFont("", 9, QFont.Bold))
            p.drawText(lab, Qt.AlignCenter, label)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#ffd93d")))
        p.setFont(QFont("", 8, QFont.Bold))
        p.drawText(QRectF(ox - 14, oy + 9, r.right() - ox - 4, 18),
                   Qt.AlignLeft | Qt.AlignVCenter, f"shaking along {shake.upper()}")

class SensorMapWidget(QGroupBox):
    """Settings panel: number of floors, preset, axis, and per-sensor placement."""

    mapChanged = Signal(object)          # emits SensorMap; MainWindow fans it out

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Sensor placement map", parent)
        self._loading = False
        self._emit_pending = False
        self._n_sensors = MAX_SENSORS
        self._build()
        self._apply_preset(next(iter(PRESETS)))

    # -- construction -----------------------------------------------------
    def _build(self) -> None:
        outer = QHBoxLayout(self)
        left = QVBoxLayout()

        form = QFormLayout()
        self.spin_floors = QSpinBox(self)
        self.spin_floors.setRange(1, MAX_FLOORS)
        self.spin_floors.setValue(3)
        self.spin_floors.valueChanged.connect(self._on_floors_changed)
        form.addRow("Number of floors:", self.spin_floors)

        self.combo_preset = QComboBox(self)
        for name in PRESETS:
            self.combo_preset.addItem(name)
        self.combo_preset.currentTextChanged.connect(self._on_preset_changed)
        form.addRow("Preset:", self.combo_preset)

        self.combo_axis = QComboBox(self)
        self.combo_axis.addItem("X — shaking along X", "x")
        self.combo_axis.addItem("Y — shaking along Y", "y")
        self.combo_axis.currentIndexChanged.connect(self._emit)
        form.addRow("Excitation axis:", self.combo_axis)
        left.addLayout(form)

        grid = QGridLayout()
        grid.addWidget(QLabel("<b>Sensor</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Floor</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Location</b>"), 0, 2)
        grid.addWidget(QLabel("<b>Role</b>"), 0, 3)
        self._floor_combos: dict = {}
        self._cell_combos: dict = {}
        self._role_labels: dict = {}
        for sid in range(1, MAX_SENSORS + 1):
            colour = SENSOR_COLORS[(sid - 1) % len(SENSOR_COLORS)]
            tag = QLabel(f"<span style='color:{colour};font-weight:bold'>Sensor {sid}</span>")
            fc = QComboBox(self)
            fc.currentIndexChanged.connect(self._on_manual_edit)
            cc = QComboBox(self)
            for c in CELLS:
                cc.addItem(c)
            cc.setCurrentText(CENTRE)
            cc.currentIndexChanged.connect(self._on_manual_edit)
            rl = QLabel("—")
            grid.addWidget(tag, sid, 0)
            grid.addWidget(fc, sid, 1)
            grid.addWidget(cc, sid, 2)
            grid.addWidget(rl, sid, 3)
            self._floor_combos[sid] = fc
            self._cell_combos[sid] = cc
            self._role_labels[sid] = rl
        left.addLayout(grid)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet("color:#8b93a1;font-size:11px;")
        left.addWidget(self.lbl_summary)

        note = QLabel("Used by every tab: Smart Recording, Spectrum, Model Updating, "
                      "Sonification and the Digital Twin all read this map.")
        note.setStyleSheet("color:#ffd93d;font-size:11px;font-style:italic;")
        left.addWidget(note)
        left.addStretch(1)

        outer.addLayout(left, 3)
        self.figure = _MapFigure(self)
        outer.addWidget(self.figure, 2)
        self._rebuild_floor_choices()

    # -- state ------------------------------------------------------------
    def set_sensor_count(self, n: int) -> None:
        """Follow the 'Number of sensors' combo; extra rows grey out."""
        self._n_sensors = max(1, min(int(n), MAX_SENSORS))
        for sid in range(1, MAX_SENSORS + 1):
            on = sid <= self._n_sensors
            self._floor_combos[sid].setEnabled(on)
            self._cell_combos[sid].setEnabled(on)
            self._role_labels[sid].setEnabled(on)
        # Fewer sensors can turn a "torsion pair" preset into a single sensor
        # on one floor; the label must say what is actually placed.
        self._loading = True
        self.combo_preset.setCurrentText(self._match_preset(self.current_map()))
        self._loading = False
        self._emit()

    def _rebuild_floor_choices(self) -> None:
        n = self.spin_floors.value()
        # Save and RESTORE the flag. This used to set it True and then False,
        # clobbering an outer _apply_preset() that had set it True — so the
        # per-sensor edits that followed counted as manual edits and one preset
        # change emitted seven maps.
        was_loading = self._loading
        self._loading = True
        for sid, fc in self._floor_combos.items():
            keep = fc.currentData()
            fc.clear()
            fc.addItem(UNASSIGNED, None)
            fc.addItem("0 · base / shaker", 0)
            for f in range(1, n + 1):
                fc.addItem(f"{f}", f)
            idx = fc.findData(keep)
            fc.setCurrentIndex(idx if idx >= 0 else 0)
        self._loading = was_loading

    def apply_map(self, smap: SensorMap) -> None:
        """Load a stored placement back into the widgets."""
        self._loading = True
        self.spin_floors.setValue(max(1, min(int(smap.n_floors or 3), MAX_FLOORS)))
        self._rebuild_floor_choices()
        i = self.combo_axis.findData(smap.axis)
        self.combo_axis.setCurrentIndex(i if i >= 0 else 0)
        by_id = {p.sensor_id: p for p in smap.placements}
        for sid in range(1, MAX_SENSORS + 1):
            pl = by_id.get(sid)
            idx = self._floor_combos[sid].findData(pl.floor if pl else None)
            self._floor_combos[sid].setCurrentIndex(max(idx, 0))
            if pl:
                self._cell_combos[sid].setCurrentText(pl.cell)
        self.combo_preset.setCurrentText(self._match_preset(smap))
        self._loading = False
        self._emit()

    def _match_preset(self, smap: SensorMap) -> str:
        """Name the preset this placement equals, else Custom."""
        want = {p.sensor_id: (p.floor, p.cell) for p in smap.placements
                if p.floor is not None}
        for name, (floors, layout) in PRESETS.items():
            if name == CUSTOM:
                continue
            if floors == smap.n_floors and layout == want:
                return name
        return CUSTOM

    def current_map(self) -> SensorMap:
        placements = []
        for sid in range(1, self._n_sensors + 1):
            placements.append(SensorPlacement(
                sensor_id=sid,
                floor=self._floor_combos[sid].currentData(),
                cell=self._cell_combos[sid].currentText(),
            ))
        return SensorMap(n_floors=self.spin_floors.value(),
                         axis=self.combo_axis.currentData() or "x",
                         placements=placements,
                         preset=self.combo_preset.currentText())

    # -- events -----------------------------------------------------------
    def _on_floors_changed(self, _v: int) -> None:
        self._rebuild_floor_choices()
        self._emit()

    def _on_preset_changed(self, name: str) -> None:
        if self._loading or name == CUSTOM:
            self._emit()
            return
        self._apply_preset(name)

    def _apply_preset(self, name: str) -> None:
        floors, layout = PRESETS.get(name, (0, {}))
        if not layout:
            return
        self._loading = True
        self.spin_floors.setValue(floors)
        self._rebuild_floor_choices()
        for sid in range(1, MAX_SENSORS + 1):
            if sid in layout:
                floor, cell = layout[sid]
                i = self._floor_combos[sid].findData(floor)
                self._floor_combos[sid].setCurrentIndex(max(i, 0))
                self._cell_combos[sid].setCurrentText(cell)
            else:
                self._floor_combos[sid].setCurrentIndex(0)
        self.combo_preset.setCurrentText(name)
        self._loading = False
        self._emit()

    def _on_manual_edit(self, *_a) -> None:
        if self._loading:
            return
        self._loading = True
        self.combo_preset.setCurrentText(CUSTOM)   # hand edits leave the preset
        self._loading = False
        self._emit()

    def _emit(self, *_a) -> None:
        """Coalesce: one map per user action, delivered on the next event-loop turn.

        Five consumers hang off this signal and Spectrum launches a worker on
        each delivery; a preset change used to deliver seven.
        """
        if self._loading or self._emit_pending:
            return
        self._emit_pending = True
        QTimer.singleShot(0, self._emit_now)

    def _emit_now(self) -> None:
        self._emit_pending = False
        # _emit queues this through QTimer.singleShot, which holds the bound
        # method alive. If the widget is torn down before the queued call
        # arrives — closing the window with a map edit still pending — the Python
        # wrapper survives while the child combo boxes are already deleted on the
        # C++ side, and reading them raises RuntimeError mid-teardown.
        if not shiboken6.isValid(self):
            return
        try:
            smap = self.current_map()
        except RuntimeError:
            return
        for sid in range(1, MAX_SENSORS + 1):
            pl = next((p for p in smap.placements if p.sensor_id == sid), None)
            role = pl.role if pl else "unused"
            colour = {"base": BASE_COLOR, "structure": "#7ee787"}.get(role, "#6b7280")
            self._role_labels[sid].setText(
                f"<span style='color:{colour}'>{role}</span>")
        self.figure.set_map(smap)
        self.lbl_summary.setText(self._summarise(smap))
        self.mapChanged.emit(smap)

    @staticmethod
    def _summarise(m: SensorMap) -> str:
        struct = m.structural()
        covered = m.covered_floors()
        base = m.base()
        bits = [f"<b>{len(struct)}</b> structural sensor(s) on floor(s) "
                f"{', '.join(str(f) for f in covered) if covered else '—'}"]
        bits.append("base sensor present (measures input)" if base
                    else "no base sensor — no direct excitation reading")
        missing = [f for f in range(1, m.n_floors + 1) if f not in covered]
        if missing:
            bits.append(f"unmeasured floor(s): {', '.join(map(str, missing))} "
                        f"(mode shape omits them, never interpolates)")
        pairs = {}
        for p in struct:
            pairs.setdefault(p.floor, []).append(p)
        multi = [f for f, v in pairs.items() if len(v) > 1]
        if multi:
            bits.append(f"floor(s) {', '.join(map(str, multi))} carry a pair "
                        f"→ torsion indicator available")
        return " · ".join(bits)
