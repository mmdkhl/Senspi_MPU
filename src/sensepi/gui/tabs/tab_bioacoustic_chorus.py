"""Bioacoustic Chorus — a live sonification workflow.

The structure's own modal frequencies cast a chorus of real recorded insect and
amphibian voices: mode 1 sings as frogs, mode 2 as crickets, mode 3 as katydids,
each chirping at exactly its mode's natural frequency with no transposition.

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
from ...sonification.chorus.types import (CASE_MEANING, FAMILY_LABEL, ROLE_COLORS,
                                          ROLE_MEANING, ChorusConfig, VizFrame)

logger = logging.getLogger(__name__)

BG = "#12151a"
PANEL = "#1a1f27"
FG = "#e8eaed"
DIM = "#8b93a1"
EDGE = "#2b3440"
MODE_COLORS = ("#5ac8fa", "#7ee787", "#ff9f43", "#c792ea")

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
        ("resonance_voice", "Resonance (cicada)", 0.0, 1.5, 2),
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
    "Casting map": [
        ("f_lo", "f low (Hz)", 0.1, 5.0, 2),
        ("f_hi", "f high (Hz)", 5.0, 40.0, 1),
        ("c_lo", "carrier low (Hz)", 100.0, 2000.0, 0),
        ("c_hi", "carrier high (Hz)", 3000.0, 16000.0, 0),
    ],
}

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
            state = ("▲ RESONANCE — cicada layer active, chorus phase-locked"
                     if s > 0.6 else "scattered chorus, individuals free-running")
            state_col = "#ff9f43" if s > 0.6 else DIM
            bar = int(round(s * 22))
            html = (
                f"<div style='color:{colour};font-weight:bold;font-size:10px'>MODE {m+1}"
                f" &nbsp;·&nbsp; {FAMILY_LABEL.get(entry.info.group, entry.info.group).upper()}"
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


class _StructurePanel(pg.PlotWidget):
    """⑤ Structure — the frame ANIMATED by the measured motion.

    Displacement is rebuilt from modal superposition:

        x(floor, t) = Σ_m  shape[floor, m] · band_energy[m] · sin(2π f_m t)

    so the frame sways at the structure's own frequencies, in its own mode
    shapes, with amplitudes following how excited each mode currently is. The
    phase advances on the GUI timer, which keeps the motion smooth even when
    control frames are dropped.

    Modes above a few Hz would alias at a 25 Hz redraw, so ALL modes are slowed
    by one shared factor. Relative rates therefore stay true — mode 3 still
    visibly moves faster than mode 1 — and the factor is shown on the plot.
    """

    MAX_DISPLAY_HZ = 1.2   # comfortable sway rate on screen

    def __init__(self) -> None:
        super().__init__()
        _style_plot(self)
        self.getAxis("left").setStyle(showValues=False)
        self.getAxis("bottom").setStyle(showValues=False)
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self._left = self.plot(pen=pg.mkPen(DIM, width=3))
        self._right = self.plot(pen=pg.mkPen(DIM, width=3))
        self._floors = [self.plot(pen=pg.mkPen(FG, width=6)) for _ in range(6)]
        self._ghost = [self.plot(pen=pg.mkPen(QColor(90, 200, 250, 60), width=2))
                       for _ in range(6)]
        self._dots = pg.ScatterPlotItem(size=12)
        self.addItem(self._dots)
        self._ground = self.plot(pen=pg.mkPen(DIM, width=4))
        self._ground.setData([-1.25, 1.25], [0, 0])
        self._note = pg.TextItem("", color=DIM, anchor=(0.5, 0))
        self._note.setFont(QFont("", 7))
        self.addItem(self._note)
        self._phase = np.zeros(4)
        self._amp = np.zeros(4)
        self._slow = 1.0
        self._ref = 1e-3
        self._n = 0

    def update_frame(self, viz: VizFrame) -> None:
        freqs = np.asarray(viz.modal.frequencies_hz, dtype=float).ravel()
        be = np.asarray(viz.frame.band_energy, dtype=float).ravel()
        env = np.asarray(viz.frame.env_floor, dtype=float).ravel()
        n_floors = int(env.size) or 3
        shapes = viz.modal.shapes
        if freqs.size == 0:
            return
        k = min(freqs.size, self._phase.size)

        dt = DRAIN_MS / 1000.0
        self._phase[:k] = (self._phase[:k]
                           + 2 * np.pi * freqs[:k] * self._slow * dt) % (2 * np.pi)
        # amplitude follows excitation, smoothed so it breathes rather than jumps
        tgt = np.zeros(4)
        tgt[:k] = be[:k] if be.size >= k else 0.0
        self._amp += 0.25 * (tgt - self._amp)

        # One shared slow-motion factor, keyed to the DOMINANT mode so the main
        # sway is comfortable to watch. Keying it to the fastest mode instead
        # made everything crawl. Relative speeds stay true either way.
        dom = int(np.argmax(self._amp[:k])) if k and self._amp[:k].any() else 0
        f_dom = float(freqs[min(dom, k - 1)]) if k else 1.0
        self._slow = float(np.clip(self.MAX_DISPLAY_HZ / max(f_dom, 1e-6), 0.04, 1.0))

        # modal superposition -> per-floor lateral displacement
        disp = np.zeros(n_floors)
        for m in range(k):
            if shapes is not None and shapes.ndim == 2 and shapes.shape[1] > m \
                    and shapes.shape[0] >= n_floors:
                col = np.asarray(shapes[:n_floors, m], dtype=float)
                denom = max(np.abs(col).max(), 1e-9)
                col = col / denom
            else:                      # no shapes yet: assume a first-mode-like sway
                col = np.linspace(0.35, 1.0, n_floors) ** (m + 1)
            disp += col * self._amp[m] * np.sin(self._phase[m])
        # Normalise against a DECAYING PEAK, never against this frame's own max:
        # per-frame normalisation would rescale the shape to full deflection
        # every frame and cancel the oscillation entirely (it did).
        peak = float(np.abs(disp).max())
        self._ref = max(self._ref * 0.992, peak, 1e-3)
        disp = disp * (0.55 / self._ref)
        if not np.all(np.isfinite(disp)):
            return

        ys = np.arange(n_floors + 1, dtype=float)
        xl = np.concatenate([[-0.8], -0.8 + disp])
        xr = np.concatenate([[0.8], 0.8 + disp])
        self._left.setData(xl, ys)
        self._right.setData(xr, ys)
        spots = []
        for i in range(min(n_floors, len(self._floors))):
            self._floors[i].setData([xl[i + 1], xr[i + 1]], [i + 1, i + 1])
            self._ghost[i].setData([-0.8, 0.8], [i + 1, i + 1])   # rest position
            colour = MODE_COLORS[i % len(MODE_COLORS)]
            mag = float(env[i] / max(env.max(), 1e-12)) if env.size > i else 0.5
            spots.append({"pos": (xr[i + 1], i + 1), "size": 9 + 20 * mag,
                          "brush": pg.mkBrush(colour)})
        for j in range(n_floors, len(self._floors)):
            self._floors[j].setData([], [])
            self._ghost[j].setData([], [])
        self._dots.setData(spots)
        self._note.setText(f"live motion · display-scaled"
                           + (f" · slowed ×{1/self._slow:.0f}" if self._slow < 0.95 else ""))
        self._note.setPos(0.0, -0.32)
        self.setXRange(-1.5, 1.5)
        self.setYRange(-0.45, n_floors + 0.6)


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
            f"<div style='color:#ff9f43;font-size:10px;font-weight:bold;"
            f"padding-top:6px'>STRUCTURAL CHANGE</div>"
            f"<div style='color:{DIM};font-size:9px'>If the structure softens its "
            f"frequencies drop, the target carriers drop, and the cast changes "
            f"species. Roughly 25% stiffness loss turns the katydids into frogs, "
            f"so the species composition is itself a readout.</div>")
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
        # One button per mode the identification can return (the spinbox allows
        # up to 4, and four sensors support four); mode 4 could be heard but
        # never soloed.
        for i, label in enumerate(("All", "1", "2", "3", "4")):
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
            if group == "Casting map":
                self._chk_autofit = QCheckBox("auto-fit to this structure")
                self._chk_autofit.setChecked(self._cfg.autofit)
                self._chk_autofit.setStyleSheet(f"color:{FG};font-size:10px;")
                self._chk_autofit.setToolTip(
                    "Match the two ranges to each other: the frequency ends are "
                    "fitted to the structure's own modes and the carrier ends to "
                    "the catalog, so the modes use the whole species palette.\n"
                    "The fit is taken once at baseline and then held, so later "
                    "frequency drift still recasts the meadow.")
                self._chk_autofit.toggled.connect(
                    lambda v: self._on_knob("autofit", bool(v)))
                bl.addWidget(self._chk_autofit)
                self._btn_refit = QPushButton("Re-fit now")
                self._btn_refit.clicked.connect(self._on_refit)
                bl.addWidget(self._btn_refit)
        return box

    def _build_identification_box(self) -> QGroupBox:
        ident = QGroupBox("Identification")
        ident.setStyleSheet(self._group_css())
        il = QGridLayout(ident)
        il.setContentsMargins(4, 4, 4, 4)
        self._spin_modes = QSpinBox()
        self._spin_modes.setRange(1, 4)
        self._spin_modes.setValue(self._cfg.n_modes)
        self._spin_modes.valueChanged.connect(
            lambda v: self._on_knob("n_modes", float(v)))
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
        self._structure = _StructurePanel()
        self._waterfall = _WaterfallPanel()
        self._score = _ScorePanel()
        self._legend = _LegendPanel()
        grid.addWidget(_titled("THE CAST — which species the structure chose",
                               "①", self._cast_panel), 0, 0)
        grid.addWidget(_titled("RESONANCE RADAR — where the energy is",
                               "②", self._radar), 0, 1)
        grid.addWidget(_titled("STRUCTURE", "⑤", self._structure), 0, 2)
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
        label.setText(" · ".join(bits))

    def _on_knob(self, attr: str, value) -> None:
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
        for attr in ("f_lo", "f_hi", "c_lo", "c_hi"):
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
            self._structure.update_frame(latest)
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
        self._status.setText("  ·  ".join(bits))

    def closeEvent(self, event) -> None:      # noqa: N802 (Qt naming)
        self._on_stop()
        super().closeEvent(event)
