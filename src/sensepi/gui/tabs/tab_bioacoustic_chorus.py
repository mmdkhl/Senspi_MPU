"""Bioacoustic Chorus — a live sonification workflow.

The structure's own modal frequencies cast a chorus of real recorded animal
voices. The user chooses WHAT KIND of animal sings each mode and each
structural case (frogs, crickets, katydids, cicadas, woodpeckers, owls, bats,
doves, squirrels, grasshoppers); inside that type the mode's frequency picks
the species, and every voice chirps at exactly its mode's natural frequency
with no transposition.

This is ONE sonification workflow, on its own tab. The separate "Sonification"
tab is reserved for the sonification team's own method, so neither blocks the
other and nothing has to be merged into anything else.

Threading: the tab owns a :class:`ChorusWorker` moved onto its own ``QThread``.
No analysis or synthesis runs on the GUI thread (G1/G4). Frames arrive through
a bounded queue drained by a ``QTimer`` (G5). No SSH lives here (G2) — the tab
only holds a reference to ``RecorderController`` and pulls thread-safe
snapshots from inside the worker.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from queue import Empty, Queue

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QScrollArea, QSlider,
                               QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ...analysis import sensor_layout as slayout
from ..widgets.wireframe import LiveStructureView
from ...sonification.chorus.types import (ROLE_COLORS, ROLE_MEANING,
                                          TYPE_ORDER, TYPES, ChorusConfig, VizFrame,
                                          type_label)

logger = logging.getLogger(__name__)

BG = "#12151a"
PANEL = "#1a1f27"
FG = "#e8eaed"
DIM = "#8b93a1"
EDGE = "#2b3440"
MODE_COLORS = ("#5ac8fa", "#7ee787", "#ff9f43", "#c792ea")

#: Eigenfrequencies the tab works with: f1, f2, f3 (a three-storey frame).
N_FREQS = 3

VIZ_QUEUE_MAX = 8
DRAIN_MS = 40
WATERFALL_SLICES = 26
SCORE_SECONDS = 60.0

# knob spec: (attribute, label, min, max, decimals)
# The panel used to present 23 sliders at once, which is 23 decisions before the
# first sound. These six cover everything an ordinary listening session needs;
# the rest still exist, one click away under "Advanced".
_KNOBS_MAIN = {
    "Listen": [
        ("master", "Master", 0.0, 1.0, 2),
        ("chorus_size", "Chorus size", 0.1, 2.0, 2),
        ("density", "Density", 0.1, 3.0, 2),
        ("naturalism", "Naturalism", 0.0, 1.0, 2),
        ("space", "Reverb / space", 0.0, 1.0, 2),
    ],
}

#: The seven case voices and their defaults. The "Structure expression" macro
#: scales all of them together, relative to these — one slider for "how strongly
#: should the structure's behaviour colour the sound", with the individual voices
#: still available under Advanced for anyone who wants to isolate one.
_CASE_VOICES = {
    "resonance_voice": 1.0, "approach_voice": 1.0, "beating_voice": 1.0,
    "torsion_voice": 1.0, "drift_voice": 1.0, "impact_voice": 1.0,
    "dropout_voice": 1.0,
}

_KNOBS_ADVANCED = {
    "Sound": [
        ("brightness", "Brightness", 0.3, 1.6, 2),
    ],
    "Structure → sound": [
        ("sync_strength", "Sync strength", 0.0, 1.5, 2),
        ("pitch_rise", "Pitch rise (resonance)", 0.0, 0.25, 3),
        ("duck_db", "Duck others on lock (dB)", 0.0, 24.0, 1),
        ("damping_expression", "Damping expression", 0.0, 2.0, 2),
    ],
    "Cases": [
        ("resonance_voice", "Resonance", 0.0, 1.5, 2),
        ("approach_voice", "Approach", 0.0, 1.5, 2),
        ("beating_voice", "Beating", 0.0, 1.5, 2),
        ("torsion_voice", "Torsion", 0.0, 1.5, 2),
        ("drift_voice", "Inter-storey drift", 0.0, 1.5, 2),
        ("impact_voice", "Impact", 0.0, 1.5, 2),
        ("dropout_voice", "Data dropout", 0.0, 1.5, 2),
    ],
    "Space": [
        ("depth", "Depth", 0.0, 1.0, 2),
        ("ambient_bed", "Ambient bed", 0.0, 1.0, 2),
    ],
    # The carrier ends are no longer knobs: each animal type maps the building
    # range onto its OWN catalogued carrier span, so only the frequency ends
    # remain to be fitted or set.
    "Frequency map": [
        ("f_lo", "f low (Hz)", 0.1, 5.0, 2),
        ("f_hi", "f high (Hz)", 5.0, 40.0, 1),
    ],
}

#: The case voices the user can assign an animal type to, in panel order:
#: (config attribute, label, tooltip). Modes are handled separately.
_CASE_SLOTS = (
    ("resonance_type", "Resonance", "Sustained layer that joins while a mode is "
                                    "locked to the excitation"),
    ("torsion_type", "Torsion", "Fast voice driven by the gyro (gz); pans with the "
                                "direction of rotation"),
    ("alarm_type", "Impact", "The startled call right after a knock, once the "
                             "meadow has hushed"),
    ("drift_type", "Drift", "Sings between two floors moving against each other"),
    ("ambient_type", "Ambient", "Background bed that never goes dead; 'auto' picks "
                                "the quietest fit from any type"),
)

#: Kept so anything still importing the old flat table keeps working.
_KNOBS = {**_KNOBS_MAIN, **_KNOBS_ADVANCED}


def _style_plot(widget: pg.PlotWidget, xlabel: str = "", ylabel: str = "") -> None:
    widget.setBackground(PANEL)
    for ax in ("left", "bottom"):
        a = widget.getAxis(ax)
        a.setPen(pg.mkPen(EDGE))
        a.setTextPen(pg.mkPen(DIM))
    widget.showGrid(x=False, y=False)
    if xlabel:
        widget.setLabel("bottom", xlabel, color=DIM, size="8pt")
    if ylabel:
        widget.setLabel("left", ylabel, color=DIM, size="8pt")


def _titled(title: str, tag: str, inner: QWidget) -> QWidget:
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(6, 4, 6, 6)
    lay.setSpacing(4)
    lbl = QLabel(f"{tag}  {title}")
    lbl.setStyleSheet(f"color:{FG};font-weight:bold;font-size:11px;")
    lay.addWidget(lbl)
    lay.addWidget(inner, 1)
    box.setStyleSheet(f"background:{PANEL};border-radius:4px;")
    return box


class _SliderRow(QWidget):
    """A labelled float slider that reports changes as (attr, value)."""

    changed = Signal(str, float)

    def __init__(self, attr: str, label: str, lo: float, hi: float,
                 decimals: int, value: float) -> None:
        super().__init__()
        self._attr, self._lo, self._hi = attr, lo, hi
        self._decimals = decimals
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(1)
        head = QHBoxLayout()
        head.setSpacing(4)
        self._name = QLabel(label)
        self._name.setStyleSheet(f"color:{FG};font-size:10px;")
        self._value = QLabel("")
        self._value.setStyleSheet(f"color:{DIM};font-size:10px;")
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        head.addWidget(self._name, 1)
        head.addWidget(self._value)
        lay.addLayout(head)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(self._to_slider(value))
        self._slider.valueChanged.connect(self._on_move)
        lay.addWidget(self._slider)
        self._render(value)

    def _to_slider(self, v: float) -> int:
        frac = (float(v) - self._lo) / max(self._hi - self._lo, 1e-9)
        return int(round(np.clip(frac, 0.0, 1.0) * 1000))

    def _from_slider(self, s: int) -> float:
        return self._lo + (self._hi - self._lo) * (s / 1000.0)

    def _render(self, v: float) -> None:
        self._value.setText(f"{v:.{self._decimals}f}")

    def _on_move(self, s: int) -> None:
        v = self._from_slider(s)
        self._render(v)
        self.changed.emit(self._attr, float(v))

    def set_value(self, v: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self._to_slider(v))
        self._slider.blockSignals(False)
        self._render(float(v))


class _CastPanel(QScrollArea):
    """① The cast — which species the structure chose, and what it is doing."""

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self._inner = QWidget()
        self._lay = QVBoxLayout(self._inner)
        self._lay.setContentsMargins(4, 2, 4, 2)
        self._lay.setSpacing(6)
        self._lay.addStretch(1)
        self.setWidget(self._inner)
        self.setStyleSheet(f"background:{PANEL};border:none;")
        self._cards: dict[int, QLabel] = {}
        self._placeholder = QLabel("waiting for the first identification…")
        self._placeholder.setStyleSheet(f"color:{DIM};font-size:11px;padding:8px;")
        self._lay.insertWidget(0, self._placeholder)

    def update_frame(self, viz: VizFrame) -> None:
        leads = [c for c in viz.cast if c.role == "lead"]
        if not leads:
            return
        self._placeholder.setVisible(False)
        modal = viz.modal
        sync = viz.frame.sync
        for i, entry in enumerate(leads):
            m = entry.mode
            colour = MODE_COLORS[m % len(MODE_COLORS)]
            damp = (modal.damping[m] * 100.0
                    if modal.damping is not None and m < len(modal.damping) else float("nan"))
            s = float(sync[m]) if 0 <= m < sync.size else 0.0
            n_sing = int(viz.singing.get(entry.info.species, 0))
            supporters = [c.info.species for c in viz.cast
                          if c.mode == m and c.role == "chorus"]
            state = ("▲ RESONANCE — resonance layer active, chorus phase-locked"
                     if s > 0.6 else "scattered chorus, individuals free-running")
            state_col = "#ff9f43" if s > 0.6 else DIM
            bar = int(round(s * 22))
            html = (
                f"<div style='color:{colour};font-weight:bold;font-size:10px'>f{m+1}"
                f" &nbsp;·&nbsp; {type_label(entry.info.type or entry.info.group).upper()}"
                f"</div>"
                f"<div style='color:{FG};font-size:13px;font-style:italic;"
                f"font-weight:bold'>{entry.info.species}</div>"
                f"<div style='color:{DIM};font-size:10px'>{entry.info.common or entry.info.group}"
                f" · carrier {entry.info.carrier:,.0f} Hz · {entry.info.license.upper()}</div>"
                f"<div style='color:{FG};font-size:10px'>{entry.mode_freq:.2f} Hz"
                f" &nbsp; ζ={damp:.2f}% &nbsp;→ target {entry.target_carrier:,.0f} Hz</div>"
                f"<div style='color:#7ee787;font-size:10px'>chirps at "
                f"{entry.mode_freq:.2f}/s &nbsp;(1:1, no transposition)</div>"
                f"<div style='color:{FG};font-size:10px'>singing: {n_sing} individuals"
                f" &nbsp;<span style='color:{colour}'>{'█'*bar}</span>"
                f"<span style='color:{EDGE}'>{'█'*(22-bar)}</span>"
                f" <span style='color:{DIM}'>sync {s:.2f}</span></div>"
                f"<div style='color:{state_col};font-size:10px'>{state}</div>"
                f"<div style='color:{DIM};font-size:9px'>with: "
                f"{', '.join(supporters) if supporters else '—'}</div>"
            )
            card = self._cards.get(m)
            if card is None:
                card = QLabel()
                card.setTextFormat(Qt.RichText)
                card.setWordWrap(True)
                # set once: setStyleSheet forces a full style repolish, which
                # is far too expensive to repeat at frame rate
                card.setStyleSheet(
                    f"background:#232a35;border:1px solid {colour};"
                    f"border-radius:5px;padding:6px;")
                self._cards[m] = card
                self._lay.insertWidget(self._lay.count() - 1, card)
            card.setText(html)


class _RadarPanel(pg.PlotWidget):
    """② Resonance radar — the structural spectrum, modes and excitation."""

    def __init__(self) -> None:
        super().__init__()
        _style_plot(self, "structural frequency (Hz)", "PSD")
        self.setLogMode(False, True)
        self._curve = self.plot(pen=pg.mkPen("#5ac8fa", width=2))
        self._mode_lines: list[pg.InfiniteLine] = []
        self._exc = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ff6b6b", width=2))
        self.addItem(self._exc)
        self._region = pg.LinearRegionItem(values=(0, 0), movable=False,
                                           brush=pg.mkBrush(255, 107, 107, 40))
        self._region.setZValue(-10)
        self.addItem(self._region)
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self._pmax = 1e-6

    def update_frame(self, viz: VizFrame) -> None:
        f = viz.frame
        if f.psd is not None and f.psd_freqs is not None and f.psd.size > 4:
            self._curve.setData(f.psd_freqs, np.maximum(f.psd, 1e-12))
            # hold the range against a decaying peak: per-frame autoscaling made
            # the axis leap around and the shape impossible to compare over time
            self._pmax = max(self._pmax * 0.97, float(np.max(f.psd)))
            self.setYRange(np.log10(self._pmax) - 4.0, np.log10(self._pmax) + 0.3)
            self.setXRange(float(f.psd_freqs[0]), float(f.psd_freqs[-1]), padding=0.01)
        freqs = np.asarray(viz.modal.frequencies_hz, dtype=float).ravel()
        while len(self._mode_lines) < freqs.size:
            i = len(self._mode_lines)
            ln = pg.InfiniteLine(angle=90, pen=pg.mkPen(
                MODE_COLORS[i % len(MODE_COLORS)], width=1, style=Qt.DashLine),
                label=f"f{i+1}", labelOpts={"color": MODE_COLORS[i % len(MODE_COLORS)],
                                            "position": 0.92})
            self.addItem(ln)
            self._mode_lines.append(ln)
        for i, ln in enumerate(self._mode_lines):
            if i < freqs.size and np.isfinite(freqs[i]):
                ln.setVisible(True)
                ln.setPos(float(freqs[i]))
                ln.label.setText(f"f{i+1} {freqs[i]:.2f}")
            else:
                ln.setVisible(False)
        if f.exc_freq_hz > 0:
            self._exc.setPos(f.exc_freq_hz)
            if freqs.size:
                near = float(freqs[int(np.argmin(np.abs(freqs - f.exc_freq_hz)))])
                lo, hi = sorted((near, f.exc_freq_hz))
                self._region.setRegion((lo, hi))


class _WaterfallPanel(pg.PlotWidget):
    """③ Frequency interaction — time × frequency × energy as a ridgeline."""

    def __init__(self) -> None:
        super().__init__()
        _style_plot(self, "frequency (Hz)", "time  →  energy")
        self.getAxis("left").setStyle(showValues=False)
        self._slices: list[np.ndarray] = []
        self._curves = []
        for i in range(WATERFALL_SLICES):
            frac = i / max(WATERFALL_SLICES - 1, 1)
            col = QColor("#ff9f43")
            col.setAlphaF(0.16 + 0.84 * (1.0 - frac))
            c = self.plot(pen=pg.mkPen(col, width=1.2))
            c.setZValue(-i)
            self._curves.append(c)
        self._mode_lines: list[pg.InfiniteLine] = []
        self._freqs: np.ndarray | None = None

    def update_frame(self, viz: VizFrame) -> None:
        f = viz.frame
        if f.psd is None or f.psd_freqs is None or f.psd.size < 8:
            return
        self._freqs = f.psd_freqs
        db = 10.0 * np.log10(np.maximum(f.psd, 1e-14))
        # smooth: raw bin-by-bin dB is far too jagged to read as a ridgeline
        k = max(3, (db.size // 96) | 1)
        db = np.convolve(db, np.ones(k) / k, mode="same")
        lo = float(np.percentile(db, 5))
        db = np.maximum(db - lo, 0.0)
        self._slices.insert(0, db)
        del self._slices[WATERFALL_SLICES:]
        span = max(float(np.percentile(np.concatenate(self._slices), 98)), 1e-6)
        # each slice sits a full span above the previous one, so ridges stay legible
        for i, curve in enumerate(self._curves):
            if i < len(self._slices):
                curve.setData(self._freqs,
                              np.minimum(self._slices[i], span * 1.6)
                              + (WATERFALL_SLICES - i) * span * 0.85)
            else:
                curve.setData([], [])
        freqs = np.asarray(viz.modal.frequencies_hz, dtype=float).ravel()
        while len(self._mode_lines) < freqs.size:
            i = len(self._mode_lines)
            ln = pg.InfiniteLine(angle=90, pen=pg.mkPen(
                MODE_COLORS[i % len(MODE_COLORS)], width=1, style=Qt.DotLine))
            self.addItem(ln)
            self._mode_lines.append(ln)
        for i, ln in enumerate(self._mode_lines):
            ok = i < freqs.size and np.isfinite(freqs[i])
            ln.setVisible(bool(ok))
            if ok:
                ln.setPos(float(freqs[i]))


class _ScorePanel(pg.PlotWidget):
    """④ Chorus score — who is singing, and when."""

    def __init__(self) -> None:
        super().__init__()
        _style_plot(self, "time (s)", "")
        self.getAxis("left").setStyle(showValues=False)
        self._lanes: dict[str, dict] = {}
        self._t: list[float] = []

    def _lane(self, name: str, role: str, index: int) -> dict:
        lane = self._lanes.get(name)
        if lane is None:
            colour = QColor(ROLE_COLORS.get(role, "#888888"))
            fill = QColor(colour)
            fill.setAlphaF(0.85)
            curve = self.plot(pen=pg.mkPen(colour, width=1),
                              fillLevel=index, brush=pg.mkBrush(fill))
            short = name if len(name) <= 22 else (
                name.split()[0][:1] + ". " + " ".join(name.split()[1:]))
            label = pg.TextItem(f"{short}", color=DIM, anchor=(0, 0.5))
            label.setFont(QFont("", 7))
            self.addItem(label)
            lane = {"curve": curve, "label": label, "values": [], "index": index}
            self._lanes[name] = lane
        return lane

    def update_frame(self, viz: VizFrame) -> None:
        tracked = [c for c in viz.cast if c.role in ("lead", "resonance", "torsion")]
        if not tracked:
            return
        self._t.append(viz.t)
        keep = [i for i, tt in enumerate(self._t) if tt >= viz.t - SCORE_SECONDS]
        if keep:
            self._t = self._t[keep[0]:]
        n_keep = len(self._t)
        sync = viz.frame.sync
        be = viz.frame.band_energy
        for i, entry in enumerate(tracked):
            idx = len(tracked) - i - 1
            lane = self._lane(entry.info.species, entry.role, idx)
            m = entry.mode
            if entry.role == "lead":
                v = float(be[m]) if 0 <= m < be.size else 0.0
            elif entry.role == "resonance":
                s = float(sync[m]) if 0 <= m < sync.size else 0.0
                v = float(np.clip((s - 0.35) / 0.65, 0.0, 1.0))
            else:
                v = float(viz.frame.torsion)
            lane["values"].append(v)
            lane["values"] = lane["values"][-n_keep:]
            lane["index"] = idx
            vals = np.asarray(lane["values"], dtype=float)
            tt = np.asarray(self._t[-len(vals):], dtype=float)
            lane["curve"].setData(tt, idx + 0.82 * vals)
            lane["curve"].setFillLevel(idx)
            lane["labels_at"] = idx
        # keep a left-hand gutter so the species names never sit inside a fill
        t_end = float(self._t[-1])
        t_start = float(self._t[0])
        gutter = max((t_end - t_start), SCORE_SECONDS * 0.25) * 0.30
        for entry in tracked:
            lane = self._lanes.get(entry.info.species)
            if lane is not None:
                lane["label"].setPos(t_start - gutter * 0.95, lane["index"] + 0.42)
        self.setXRange(t_start - gutter, t_end, padding=0.01)
        self.setYRange(0, max(len(tracked), 1))


class _LegendPanel(QLabel):
    """⑥ What am I hearing?"""

    def __init__(self) -> None:
        super().__init__()
        self.setTextFormat(Qt.RichText)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignTop)
        rows = "".join(
            f"<tr><td style='padding:2px 6px 2px 0'>"
            f"<span style='background:{ROLE_COLORS[k]};color:{ROLE_COLORS[k]}'>"
            f"&nbsp;&nbsp;&nbsp;</span></td>"
            f"<td style='color:{FG};font-size:10px;font-weight:bold'>{k}</td></tr>"
            f"<tr><td></td><td style='color:{DIM};font-size:9px;padding-bottom:5px'>"
            f"{ROLE_MEANING[k]}</td></tr>"
            for k in ("lead", "chorus", "resonance", "torsion", "ambient"))
        self.setText(
            f"<table style='border-collapse:collapse'>{rows}</table>"
            f"<div style='color:#5ac8fa;font-size:10px;font-weight:bold;"
            f"padding-top:6px'>CHANNELS</div>"
            f"<div style='color:{DIM};font-size:9px'>driven axis → the modes and their "
            f"leads · other axis → the supporting chorus swells and widens · az → "
            f"ambient bed density · gx/gy → a rocking floor sings the drift voice · "
            f"gz → torsion. The shaker's own sensor is the excitation reference, "
            f"never a voice.</div>"
            f"<div style='color:#ff9f43;font-size:10px;font-weight:bold;"
            f"padding-top:6px'>STRUCTURAL CHANGE</div>"
            f"<div style='color:{DIM};font-size:9px'>Each mode sings as the animal "
            f"type you chose, and inside that type the mode's frequency picks the "
            f"species. If the structure softens its frequencies drop and the lead "
            f"moves to a lower-pitched species of the same type, so the species "
            f"composition is itself a readout.</div>")
        self.setStyleSheet(f"background:{PANEL};padding:6px;")


class BioacousticChorusTab(QWidget):
    """Live bioacoustic sonification of the structure."""

    MODELS = ("Bioacoustic Chorus",)

    # Requests to the worker. These MUST be signals, not direct calls: the
    # worker lives on another thread, and a direct call would run its body
    # (stopping its QTimer, closing the audio device, writing a WAV) on the
    # GUI thread — a guardrail G1/G4 violation.
    _req_start = Signal(object)
    _req_stop = Signal()
    _req_option = Signal(str, object)
    _req_capture = Signal(bool)
    _req_refit = Signal()

    def __init__(self, recorder_controller=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = recorder_controller
        self._cfg = ChorusConfig()
        self._queue: Queue = Queue(maxsize=VIZ_QUEUE_MAX)
        self._worker = None
        self._thread: QThread | None = None
        self._stopping = False
        self._status_extra: dict = {}
        self._rows: dict[str, _SliderRow] = {}
        self._sensor_mapping: dict | None = None
        self._build_ui()
        self._drain = QTimer(self)
        self._drain.setInterval(DRAIN_MS)
        self._drain.timeout.connect(self._drain_queue)
        self._update_enabled()

    # --------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        self.setStyleSheet(f"background:{BG};color:{FG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # transport
        bar = QFrame()
        bar.setStyleSheet(f"background:{PANEL};border-radius:4px;")
        tl = QHBoxLayout(bar)
        tl.setContentsMargins(10, 6, 10, 6)
        self._btn_start = QPushButton("▶  Start")
        self._btn_start.setStyleSheet("color:#7ee787;font-weight:bold;")
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_capture = QPushButton("●  Capture WAV")
        self._btn_capture.setCheckable(True)
        self._btn_capture.setStyleSheet("color:#ff6b6b;")
        self._btn_capture.toggled.connect(self._on_capture)
        self._btn_snapshot = QPushButton("Save PNG")
        self._btn_snapshot.clicked.connect(self._on_snapshot)
        # SOLO: the fastest way to learn which animal belongs to which mode
        self._solo_buttons = []
        # One button per eigenfrequency.
        for i, label in enumerate(("All",) + tuple(f"f{k + 1}" for k in range(N_FREQS))):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setMaximumWidth(46)
            b.setChecked(i == 0)
            b.clicked.connect(lambda _c=False, k=i - 1: self._on_solo(k))
            self._solo_buttons.append(b)
        for w in (self._btn_start, self._btn_stop, self._btn_capture):
            tl.addWidget(w)
        tl.addWidget(self._btn_snapshot)
        tl.addSpacing(12)
        tl.addWidget(QLabel("Solo:"))
        for b in self._solo_buttons:
            tl.addWidget(b)
        tl.addStretch(1)
        self._status = QLabel("idle")
        self._status.setStyleSheet("color:#ffd93d;font-size:11px;")
        tl.addWidget(self._status)
        root.addWidget(bar)

        # controls + stage
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_controls())
        split.addWidget(self._build_stage())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 1400])
        root.addWidget(split, 1)

    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(400)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        # Where the sensors are — read-only, like every other tab.
        self._map_summary = QLabel("")
        self._map_summary.setWordWrap(True)
        self._map_summary.setStyleSheet(f"color:{DIM};font-size:10px;")
        map_box = QGroupBox("Sensor placement (from Settings)")
        map_box.setStyleSheet(self._group_css())
        mbl = QVBoxLayout(map_box)
        mbl.setContentsMargins(4, 4, 4, 4)
        mbl.addWidget(self._map_summary)
        lay.addWidget(map_box)

        lay.addWidget(self._build_cast_box())

        for group, knobs in _KNOBS_MAIN.items():
            lay.addWidget(self._knob_box(group, knobs))

        # One macro for "how strongly the structure colours the sound", instead
        # of seven separate case voices in the main view.
        exp_box = QGroupBox("Structure expression")
        exp_box.setStyleSheet(self._group_css())
        ebl = QVBoxLayout(exp_box)
        ebl.setContentsMargins(4, 4, 4, 4)
        self._row_expression = _SliderRow(
            "_expression", "Expression", 0.0, 2.0, 2, 1.0)
        self._row_expression.changed.connect(self._on_expression)
        self._row_expression.setToolTip(
            "Scales resonance, approach, beating, torsion, drift, impact and "
            "dropout together. Set each one on its own under Advanced.")
        ebl.addWidget(self._row_expression)
        lay.addWidget(exp_box)

        # --- Advanced: everything else, collapsed ---------------------------
        self._btn_advanced = QPushButton("▸  Advanced")
        self._btn_advanced.setCheckable(True)
        self._btn_advanced.setStyleSheet(
            f"QPushButton{{text-align:left;color:#5ac8fa;background:{PANEL};"
            f"border:1px solid {EDGE};border-radius:4px;padding:5px;font-size:10px}}")
        self._advanced = QWidget()
        adv = QVBoxLayout(self._advanced)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSpacing(8)
        self._advanced.setVisible(False)
        self._btn_advanced.toggled.connect(self._on_advanced_toggled)
        lay.addWidget(self._btn_advanced)
        lay.addWidget(self._advanced)

        for group, knobs in _KNOBS_ADVANCED.items():
            adv.addWidget(self._knob_box(group, knobs))

        ident = self._build_identification_box()
        adv.addWidget(ident)
        lay.addStretch(1)
        scroll.setWidget(inner)
        self._refresh_map_summary()
        return scroll

    @staticmethod
    def _group_css() -> str:
        return (f"QGroupBox{{color:#5ac8fa;font-weight:bold;font-size:10px;"
                f"border:1px solid {EDGE};border-radius:4px;margin-top:7px;"
                f"padding:6px}}"
                f"QGroupBox::title{{subcontrol-origin:margin;left:7px}}")

    def _knob_box(self, group: str, knobs) -> QGroupBox:
        box = QGroupBox(group)
        box.setStyleSheet(self._group_css())
        bl = QVBoxLayout(box)
        bl.setContentsMargins(4, 4, 4, 4)
        bl.setSpacing(2)
        for attr, label, lo, hi, dec in knobs:
            row = _SliderRow(attr, label, lo, hi, dec, getattr(self._cfg, attr))
            row.changed.connect(self._on_knob)
            self._rows[attr] = row
            bl.addWidget(row)
        if True:
            if group == "Frequency map":
                self._chk_autofit = QCheckBox("auto-fit to this structure")
                self._chk_autofit.setChecked(self._cfg.autofit)
                self._chk_autofit.setStyleSheet(f"color:{FG};font-size:10px;")
                self._chk_autofit.setToolTip(
                    "Fit the frequency ends to the structure's own modes, so each "
                    "animal type's whole species palette is used.\n"
                    "The fit is taken once at baseline and then held, so later "
                    "frequency drift still recasts the meadow.")
                self._chk_autofit.toggled.connect(
                    lambda v: self._on_knob("autofit", bool(v)))
                bl.addWidget(self._chk_autofit)
                self._btn_refit = QPushButton("Re-fit now")
                self._btn_refit.clicked.connect(self._on_refit)
                bl.addWidget(self._btn_refit)
        return box

    def _build_cast_box(self) -> QGroupBox:
        """Who sings what: one animal type per mode, one per structural case.

        The choice is made here, before Start, and can be changed live. Inside
        each type the species is still chosen by the structure's frequency, so
        the damage readout survives any combination the user picks.
        """
        box = QGroupBox("Cast — who sings what")
        box.setStyleSheet(self._group_css())
        grid = QGridLayout(box)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        try:
            from ...sonification.chorus.catalog import available_types
            keys = list(available_types()) or list(TYPE_ORDER)
        except Exception:
            keys = list(TYPE_ORDER)
        self._type_keys = keys

        def combo(extra: tuple = ()) -> QComboBox:
            c = QComboBox()
            for k in extra:
                c.addItem(k, k)
            for k in keys:
                c.addItem(TYPES[k][0], k)
                c.setItemData(c.count() - 1, TYPES[k][2], Qt.ToolTipRole)
            c.setStyleSheet(f"font-size:10px;color:{FG};")
            return c

        def select(c: QComboBox, key: str) -> None:
            i = c.findData(key)
            if i < 0 and c.count():
                i = 0
            c.blockSignals(True)
            c.setCurrentIndex(max(i, 0))
            c.blockSignals(False)

        self._mode_type_combos: list[QComboBox] = []
        types = tuple(self._cfg.type_of_mode)
        for m in range(N_FREQS):
            lbl = QLabel(f"f{m + 1}")
            lbl.setToolTip(f"Eigenfrequency {m + 1}: which animal type sings it. The "
                           f"frequency itself picks the species inside the type.")
            lbl.setStyleSheet(f"color:{MODE_COLORS[m]};font-size:10px;font-weight:bold;")
            c = combo()
            select(c, types[min(m, len(types) - 1)] if types else keys[0])
            c.currentIndexChanged.connect(lambda _i, k=m: self._on_mode_type(k))
            grid.addWidget(lbl, m, 0)
            grid.addWidget(c, m, 1)
            self._mode_type_combos.append(c)

        self._case_type_combos: dict[str, QComboBox] = {}
        for r, (attr, label, tip) in enumerate(_CASE_SLOTS, start=N_FREQS):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{ROLE_COLORS.get(attr.split('_')[0], DIM)};"
                              f"font-size:10px;")
            lbl.setToolTip(tip)
            c = combo(extra=("auto",) if attr == "ambient_type" else ())
            c.setToolTip(tip)
            select(c, str(getattr(self._cfg, attr)))
            c.currentIndexChanged.connect(lambda _i, a=attr: self._on_case_type(a))
            grid.addWidget(lbl, r, 0)
            grid.addWidget(c, r, 1)
            self._case_type_combos[attr] = c
        grid.setColumnStretch(1, 1)
        self._refresh_mode_type_rows()
        return box

    def _refresh_mode_type_rows(self) -> None:
        """Only as many mode rows as the identification can return."""
        n = int(self._cfg.n_modes)
        for m, c in enumerate(getattr(self, "_mode_type_combos", [])):
            c.setEnabled(m < n)

    def _on_mode_type(self, _m: int) -> None:
        types = tuple(str(c.currentData()) for c in self._mode_type_combos)
        self._on_knob("type_of_mode", types)

    def _on_case_type(self, attr: str) -> None:
        c = self._case_type_combos.get(attr)
        if c is not None:
            self._on_knob(attr, str(c.currentData()))

    def _build_identification_box(self) -> QGroupBox:
        ident = QGroupBox("Identification")
        ident.setStyleSheet(self._group_css())
        il = QGridLayout(ident)
        il.setContentsMargins(4, 4, 4, 4)
        self._spin_modes = QSpinBox()
        self._spin_modes.setRange(1, N_FREQS)
        self._spin_modes.setValue(min(int(self._cfg.n_modes), N_FREQS))
        self._spin_modes.valueChanged.connect(self._on_n_modes)
        self._spin_reid = QDoubleSpinBox()
        self._spin_reid.setRange(2.0, 60.0)
        self._spin_reid.setValue(self._cfg.reid_interval_s)
        self._spin_reid.valueChanged.connect(
            lambda v: self._on_knob("reid_interval_s", float(v)))
        # The axis combo is gone: the channel follows the excitation axis in the
        # Settings map, which is the direction the rig is actually shaken in.
        # Two places to set one physical fact is how they end up disagreeing.
        for r, (lbl, w) in enumerate((("modes", self._spin_modes),
                                      ("re-ID (s)", self._spin_reid))):
            il.addWidget(QLabel(lbl), r, 0)
            il.addWidget(w, r, 1)
        return ident

    def _build_stage(self) -> QWidget:
        stage = QWidget()
        grid = QGridLayout(stage)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self._cast_panel = _CastPanel()
        self._radar = _RadarPanel()
        # ⑤ is the same live wireframe the Digital Twin and Model Updating use:
        # per-floor pose rebuilt from ax/ay/az/gz, rendered off the GUI thread.
        # It reads the controller itself, so it needs no frames from the worker.
        self._structure = LiveStructureView(interval_ms=250)
        self._structure.set_controller(self._controller)
        self._waterfall = _WaterfallPanel()
        self._score = _ScorePanel()
        self._legend = _LegendPanel()
        grid.addWidget(_titled("THE CAST — which species the structure chose",
                               "①", self._cast_panel), 0, 0)
        grid.addWidget(_titled("RESONANCE RADAR — where the energy is",
                               "②", self._radar), 0, 1)
        grid.addWidget(_titled("STRUCTURE — live wireframe (ax ay az gz)", "⑤",
                               self._structure), 0, 2)
        grid.addWidget(_titled("FREQUENCY INTERACTION — time × frequency × energy",
                               "③", self._waterfall), 1, 0)
        grid.addWidget(_titled("CHORUS SCORE — who is singing, and when",
                               "④", self._score), 1, 1)
        grid.addWidget(_titled("WHAT AM I HEARING?", "⑥", self._legend), 1, 2)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 2)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        self._stage = stage
        return stage

    # ---------------------------------------------------------------- control
    @Slot(str, object)
    @Slot(bool)
    def _on_n_modes(self, v: int) -> None:
        self._on_knob("n_modes", float(v))
        self._refresh_mode_type_rows()

    def _on_advanced_toggled(self, on: bool) -> None:
        self._advanced.setVisible(bool(on))
        self._btn_advanced.setText(("▾  Advanced" if on else "▸  Advanced"))

    @Slot(str, float)
    def _on_expression(self, _attr: str, value) -> None:
        """Scale every case voice together, relative to its default.

        Coalesced to one application per event-loop turn: a slider emits on
        every pixel, and each application pushes seven options to the worker.
        """
        self._pending_expression = float(value)
        if not getattr(self, "_expression_scheduled", False):
            self._expression_scheduled = True
            QTimer.singleShot(0, self._apply_expression)

    def _apply_expression(self) -> None:
        self._expression_scheduled = False
        scale = float(getattr(self, "_pending_expression", 1.0))
        for attr, default in _CASE_VOICES.items():
            self._on_knob(attr, default * scale)
            row = self._rows.get(attr)
            if row is not None:
                row.set_value(default * scale)

    # ---------------------------------------------- placement (from Settings)
    def apply_sensor_map(self, mapping) -> None:
        """Adopt the placement map from Settings. This tab owns no picker.

        The map decides three things here: which sensor is the shaker (excluded
        from identification, because it is the input), which channel to listen on
        (the excitation axis), and where each sensor sits — which is what lets the
        chorus be laid out like the rig instead of in arbitrary stereo positions.
        """
        if hasattr(mapping, "to_mapping"):
            mapping = mapping.to_mapping()
        mapping = dict(mapping) if isinstance(mapping, dict) else None
        self._sensor_mapping = mapping
        self._on_knob("sensor_map", mapping)
        if hasattr(self, "_structure"):
            self._structure.apply_sensor_map(mapping)
        layout = slayout.layout_from_mapping(mapping)
        # Listen on the axis the structure is actually being shaken along.
        self._on_knob("axis", layout.channel)
        if hasattr(self, "_spin_modes"):
            cap = layout.max_modes(int(self._spin_modes.value()))
            if layout.is_valid and cap < int(self._spin_modes.value()):
                self._spin_modes.setValue(cap)
        self._refresh_map_summary()

    def _refresh_map_summary(self) -> None:
        label = getattr(self, "_map_summary", None)
        if label is None:
            return
        layout = slayout.layout_from_mapping(getattr(self, "_sensor_mapping", None))
        if not layout.is_valid:
            label.setText(
                "<span style='color:#ffd93d'>No placement set.</span> The chorus "
                "still sings, but every animal sits centre and the shaker is "
                "treated as a floor. Set it in Settings → Sensor placement map.")
            return
        bits = [layout.describe()]
        if layout.has_base:
            bits.append(f"S{layout.base_sensor_id} is the shaker — excluded from "
                        f"identification, so it cannot be heard as a mode")
        bits.append("plan column → stereo position · floor → distance")
        other = "ay" if layout.channel == "ax" else "ax"
        bits.append(f"{layout.channel} → modes · {other} → cross-axis chorus · "
                    f"az → ambient bed · gx/gy → rocking (drift) · gz → torsion")
        label.setText(" · ".join(bits))

    def _on_knob(self, attr: str, value) -> None:
        if isinstance(value, (tuple, list)):
            setattr(self._cfg, attr, tuple(value))
        else:
            try:
                setattr(self._cfg, attr, type(getattr(self._cfg, attr))(value))
            except Exception:
                setattr(self._cfg, attr, value)
        if self._worker is not None:
            self._req_option.emit(attr, getattr(self._cfg, attr))

    def _update_enabled(self) -> None:
        streaming = False
        if self._controller is not None:
            try:
                streaming = bool(self._controller.is_streaming())
            except Exception:
                streaming = False
        running = self._worker is not None
        self._btn_start.setEnabled(streaming and not running)
        self._btn_stop.setEnabled(running)
        self._btn_capture.setEnabled(running)
        if not streaming and not running:
            self._status.setText("start the live stream first")

    @Slot()
    def on_stream_started(self) -> None:
        self._update_enabled()

    @Slot()
    def on_stream_stopped(self) -> None:
        if self._worker is not None:
            self._on_stop()
        self._update_enabled()

    # -------------------------------------------------------------- lifecycle
    def _on_start(self) -> None:
        if self._worker is not None or self._controller is None:
            return
        from ...sonification.chorus.catalog import catalog_available
        if not catalog_available():
            QMessageBox.warning(self, "Sonification",
                                "The species catalog or grain bank is missing.\n"
                                "Reinstall the package data to enable this model.")
            return
        from ...sonification.chorus.live_worker import ChorusWorker

        self._stopping = False
        thread = QThread(self)
        worker = ChorusWorker(self._controller, self._queue)
        worker.moveToThread(thread)
        self._thread, self._worker = thread, worker

        worker.started_ok.connect(self._on_worker_started)
        worker.stopped.connect(self._on_worker_stopped)
        worker.error.connect(self._on_worker_error)
        worker.status.connect(self._on_status)
        worker.captured.connect(self._on_captured)
        # every request crosses the thread boundary as a queued signal
        self._req_start.connect(worker.start)
        self._req_stop.connect(worker.stop)
        self._req_option.connect(worker.set_option)
        self._req_capture.connect(worker.set_capture)
        self._req_refit.connect(worker.refit_map)
        if hasattr(self._controller, "stream_rate_updated"):
            self._controller.stream_rate_updated.connect(worker.on_stream_rate)
        # teardown: the worker is destroyed with its thread, never leaked.
        # DirectConnection is required — _on_stop blocks the GUI thread in
        # thread.wait(), so a queued quit() would never be delivered and we
        # would deadlock until the timeout. QThread.quit() is thread-safe.
        worker.stopped.connect(thread.quit, Qt.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        # bind the worker locally — self._worker may be cleared before this fires
        cfg = replace(self._cfg)
        thread.started.connect(lambda: self._req_start.emit(cfg))
        thread.start()
        self._drain.start()
        self._status.setText("starting…")
        self._update_enabled()

    def _on_stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        worker, thread = self._worker, self._thread
        self._worker = None
        try:
            if worker is not None:
                if hasattr(self._controller, "stream_rate_updated"):
                    try:
                        self._controller.stream_rate_updated.disconnect(
                            worker.on_stream_rate)
                    except (RuntimeError, TypeError):
                        pass
                # queued: the worker shuts its own timer, device and capture
                # down on its own thread, then emits stopped -> thread.quit
                self._req_stop.emit()
            if thread is not None:
                if not thread.wait(5000):
                    logger.warning("chorus: worker thread did not finish in time")
                    thread.quit()
                    thread.wait(1000)
                thread.deleteLater()
        finally:
            self._thread = None
            self._drain.stop()
            self._structure.stop()
            self._btn_capture.blockSignals(True)
            self._btn_capture.setChecked(False)
            self._btn_capture.blockSignals(False)
            self._status.setText("stopped")
            self._stopping = False
            self._update_enabled()

    def shutdown(self) -> None:
        """Stop the model. Called by MainWindow on application close."""
        self._on_stop()

    @Slot()
    def _on_worker_started(self) -> None:
        self._status.setText("listening — identifying modes…")
        self._structure.start()
        self._update_enabled()

    @Slot()
    def _on_worker_stopped(self) -> None:
        self._update_enabled()

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        logger.error("chorus: %s", message)
        self._status.setText(message)
        QMessageBox.warning(self, "Sonification", message)
        self._on_stop()

    @Slot(dict)
    def _on_status(self, payload: dict) -> None:
        self._status_extra = payload
        # reflect the auto-fitted casting map back into the sliders, otherwise
        # they show defaults while the engine is using fitted values
        mapping = payload.get("map") or {}
        for attr in ("f_lo", "f_hi"):
            if attr not in mapping:
                continue
            value = float(mapping[attr])
            if abs(value - float(getattr(self._cfg, attr))) < 1e-6:
                continue
            setattr(self._cfg, attr, value)
            row = self._rows.get(attr)
            if row is not None:
                row.set_value(value)          # blocks its own signal
        if "autofit" in mapping and hasattr(self, "_chk_autofit"):
            want = bool(mapping["autofit"])
            if want != self._chk_autofit.isChecked():
                self._chk_autofit.blockSignals(True)
                self._chk_autofit.setChecked(want)
                self._chk_autofit.blockSignals(False)
                self._cfg.autofit = want

    @Slot(str)
    def _on_captured(self, path: str) -> None:
        QMessageBox.information(self, "Sonification", f"Capture saved:\n{path}")

    def _on_solo(self, mode: int) -> None:
        for i, b in enumerate(self._solo_buttons):
            b.setChecked(i - 1 == mode)
        self._on_knob("solo_mode", int(mode))

    def _on_refit(self) -> None:
        if self._worker is not None:
            self._req_refit.emit()

    def _on_capture(self, enabled: bool) -> None:
        if self._worker is not None:
            self._req_capture.emit(bool(enabled))

    def _on_snapshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save stage snapshot", "sonification_stage.png", "PNG (*.png)")
        if path:
            self._stage.grab().save(path)

    # ------------------------------------------------------------------ frames
    def _drain_queue(self) -> None:
        latest: VizFrame | None = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except Empty:
                break
        if latest is None:
            return
        try:
            self._cast_panel.update_frame(latest)
            self._radar.update_frame(latest)
            self._waterfall.update_frame(latest)
            self._score.update_frame(latest)
        except Exception:
            logger.debug("chorus: panel update failed", exc_info=True)
        extra = getattr(self, "_status_extra", {})
        bits = [latest.state_text]
        if extra:
            if extra.get("silent"):
                bits.append("SILENT (install sensepi[sonification])")
            bits.append(f"buffer {extra.get('buffer_s', 0):.2f}s")
            if extra.get("rate_hz"):
                bits.append(f"{extra['rate_hz']:.1f} Hz in")
            if extra.get("underruns"):
                bits.append(f"⚠ {extra['underruns']} underruns")
            if extra.get("capturing"):
                bits.append(f"● {extra.get('capture_s', 0):.0f}s")
            ch = extra.get("channels") or ()
            if ch:
                bits.append("+" + " ".join(ch))
        self._status.setText("  ·  ".join(bits))

    def closeEvent(self, event) -> None:      # noqa: N802 (Qt naming)
        self._on_stop()
        super().closeEvent(event)
