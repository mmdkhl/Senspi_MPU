"""Structure Pulse — the measured building as a graphical score.

One recording is analysed once into a set of **views**; any view can then be
*played*, with a playhead sweeping left to right while the curve's height drives
a synthesiser. The plot on screen and the sound in the room are the same object.

After Iannis Xenakis' UPIC (1977) — x is time, y is frequency — and Curtis
Roads' pulsar synthesis as revived in Marcin Pietruszewski's nuPG, which is why
the default voice is ``pulsar``: a pulsar's repetition rate and its formant are
independent, so a 2 Hz eigenfrequency is heard *at 2 Hz*, as rhythm, instead of
being transposed into something it is not.

Threading: analysis runs on a ``QThread`` worker (G1/G4). Audio is a finished
buffer handed to the device, so playback is a cursor, not a scheduler. No SSH
here (G2) — the tab holds a ``RecorderController`` reference and pulls
thread-safe snapshots. The engine imports no Qt (G7).
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QScrollArea,
                               QInputDialog, QSizePolicy, QSpinBox, QSplitter,
                               QVBoxLayout, QWidget)

from ...analysis import sensor_layout as slayout
from ...sonification.structure_pulse import (BELL_TIMBRES, SCALE_MODES,
                                             VOICE_MODES, BufferPlayer,
                                             PulseConfig, render_view, write_wav)

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
# Mode labels sit on chrome; the traces are drawn on the dark plot.
TRACE_COLORS = theme.MODE_COLORS

PLAYHEAD_MS = 33            # ~30 fps; the playhead only reads a cursor
#: Ticks of a frozen cursor before we conclude the output stream has died and
#: reopen it. ~0.5 s: long enough not to react to a single scheduling hiccup.
STALL_TICKS = 15
DEFAULT_RECORD_S = 60


class _AnalysisWorker(QThread):
    """Recording -> dataset, off the GUI thread.

    Either snapshots ``record_s`` seconds from the live stream, or loads a
    finished session folder from disk.
    """

    done = Signal(object, str)          # dataset | None, message

    def __init__(self, controller, mapping, *, record_s: float = 0.0,
                 session_dir=None, saved_dir=None, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._mapping = mapping
        self._record_s = float(record_s)
        self._session_dir = session_dir
        self._saved_dir = saved_dir
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:                                  # noqa: D102
        from ...sonification.structure_pulse import build_dataset

        try:
            if self._saved_dir is not None:
                from ...sonification.structure_pulse import load_dataset

                ds = load_dataset(self._saved_dir)
                self.done.emit(ds, f"reopened {Path(self._saved_dir).name}")
                return
            if self._session_dir is not None:
                ds = build_dataset(self._session_dir, mapping=self._mapping)
                self.done.emit(ds, f"analysed {Path(self._session_dir).name}")
                return
            ds = self._record_then_analyse()
            if ds is not None:
                self.done.emit(ds, f"recorded {self._record_s:.0f} s")
        except Exception as exc:                            # pragma: no cover
            logger.exception("structure pulse: analysis failed")
            self.done.emit(None, f"analysis failed: {exc}")

    def _record_then_analyse(self):
        """Wait for the live buffer to fill, then analyse what it holds.

        Pull-based: the controller's modal accumulator is already collecting, so
        'recording' here means waiting for ``record_s`` of it and taking a
        thread-safe snapshot. No SSH, no disk, nothing the tab owns.
        """
        ctrl = self._controller
        if ctrl is None:
            self.done.emit(None, "no live stream")
            return None
        want = max(self._record_s, 1.0)
        try:
            if hasattr(ctrl, "require_modal_window_seconds"):
                ctrl.require_modal_window_seconds(max(want * 1.5, 120.0))
        except Exception:
            logger.debug("structure pulse: could not grow the window", exc_info=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < want:
            if self._abort:
                self.done.emit(None, "recording cancelled")
                return None
            self.msleep(100)
        from ...sonification.structure_pulse.analysis import build_dataset_from_capture

        return build_dataset_from_capture(ctrl.snapshot_modal_capture,
                                          seconds=want, mapping=self._mapping)


class StructurePulseTab(QWidget):
    """Record, analyse, choose a view, and play the plot."""

    def __init__(self, recorder_controller=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = recorder_controller
        self._cfg = PulseConfig()
        self._dataset = None
        self._render = None
        self._worker: _AnalysisWorker | None = None
        self._player = BufferPlayer()
        self._audio = None
        self._sensor_mapping: dict | None = None
        self._silent = False
        self._last_pos = -1.0
        self._stall_ticks = 0
        self._last_wrap_pos = 0.0
        self._live_worker = None
        self._live_thread: QThread | None = None
        self._live_next = None          # the render waiting to take over
        self._live_cycle = 0
        #: Plot range held across live cycles, so the view does not jump every
        #: time the structure gets a little louder. Keyed by what is plotted;
        #: changing the view or the channel starts a fresh range.
        self._range_key = None
        self._range_y = None

        self._build_ui()
        # Offer the views immediately. Previously the list was empty until
        # something had been opened or Live had already started, so there was
        # no way to choose Spectrum *before* starting the loop — and once the
        # loop was running the choice could not be made either.
        from ...sonification.structure_pulse.analysis import PREVIEW_VIEWS

        self._populate_views(PREVIEW_VIEWS)
        self._tick = QTimer(self)
        self._tick.setInterval(PLAYHEAD_MS)
        self._tick.timeout.connect(self._on_tick)
        self._refresh_sessions()
        self._update_enabled()

    # --------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        self.setStyleSheet(f"background:{BG};color:{FG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        root.addWidget(self._build_source_bar())

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_controls())
        split.addWidget(self._build_stage())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 1300])
        root.addWidget(split, 1)

    def _build_source_bar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(f"background:{PANEL};border-radius:4px;")
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(10, 5, 10, 5)
        outer.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(row)

        row.addWidget(QLabel("Record"))
        self._spin_record = QSpinBox()
        self._spin_record.setRange(5, 600)
        self._spin_record.setValue(DEFAULT_RECORD_S)
        self._spin_record.setSuffix(" s")
        row.addWidget(self._spin_record)
        self._btn_record = QPushButton("●  Start recording")
        self._btn_record.setStyleSheet(f"color:{theme.semantic('record')};font-weight:bold;")
        self._btn_record.clicked.connect(self._on_record)
        row.addWidget(self._btn_record)

        row.addSpacing(18)
        row.addWidget(QLabel("or open"))
        self._combo_session = QComboBox()
        self._combo_session.setMinimumWidth(140)
        self._combo_session.setSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Preferred)
        self._combo_session.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._combo_session.setToolTip(
            "Sensor recordings are analysed when opened; saved Structure Pulse "
            "sessions reopen instantly with their results intact.")
        row.addWidget(self._combo_session)
        self._btn_open = QPushButton("Open")
        self._btn_open.clicked.connect(self._on_load)
        row.addWidget(self._btn_open)
        self._btn_browse = QPushButton("Browse…")
        self._btn_browse.setToolTip("Open a recording folder from anywhere on disk")
        self._btn_browse.clicked.connect(self._on_browse)
        row.addWidget(self._btn_browse)
        btn_refresh = QPushButton("⟳")
        btn_refresh.setMaximumWidth(32)
        btn_refresh.setToolTip("Rescan output/sensor_recordings")
        btn_refresh.clicked.connect(self._refresh_sessions)
        row.addWidget(btn_refresh)

        row.addSpacing(14)
        self._btn_live = QPushButton("◉  Live")
        self._btn_live.setCheckable(True)
        self._btn_live.setStyleSheet(f"color:{theme.semantic('play')};font-weight:bold;")
        self._btn_live.setToolTip(
            "Record, analyse and play continuously. Each cycle is prepared "
            "while the previous one plays, so the sound keeps running and the "
            "plot follows the structure.")
        self._btn_live.toggled.connect(self._on_live_toggled)
        row.addWidget(self._btn_live)
        self._spin_live = QSpinBox()
        self._spin_live.setRange(3, 120)
        self._spin_live.setValue(10)
        self._spin_live.setSuffix(" s")
        self._spin_live.setToolTip("Length of each live cycle")
        self._spin_live.valueChanged.connect(self._on_live_window)
        row.addWidget(self._spin_live)
        # How many times one cycle is heard before the next replaces it. The
        # capture length is what the structure gives us; this is how fast we
        # read it back. At 1x a 10 s window is swept once over 10 s; at 5x it
        # is swept in 2 s and repeats five times, which turns a slow structural
        # drift into a figure short enough to hear as a shape.
        self._spin_replay = QSpinBox()
        self._spin_replay.setRange(1, 20)
        self._spin_replay.setValue(1)
        self._spin_replay.setPrefix("\u00d7 ")
        self._spin_replay.setToolTip(
            "Replays per cycle.\n\n"
            "1 sweeps the window once, taking as long as the window itself. "
            "Higher compresses the sweep and repeats it until the next cycle "
            "is ready, so a 10 s window at \u00d75 is swept in 2 s, five times.")
        self._spin_replay.valueChanged.connect(self._on_replay_changed)
        row.addWidget(self._spin_replay)

        self._btn_save = QPushButton("Save session")
        self._btn_save.setToolTip(
            "Store the recording AND its results, so it can be replayed later "
            "or re-analysed with different settings.")
        self._btn_save.clicked.connect(self._on_save_session)
        row.addWidget(self._btn_save)

        row.addStretch(1)
        row.addStretch(1)

        # The status gets its OWN line. In the control row it needed a fixed
        # 420 px it could never give back, which pushed the bar's minimum width
        # past the window and pushed the text off the edge. On its own line it
        # can shrink to nothing and still say everything when there is room.
        self._status = QLabel("open a recording, or record from the live stream")
        self._status.setStyleSheet(f"color:{theme.semantic('status')};")
        self._status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._status.setMinimumWidth(0)
        outer.addWidget(self._status)
        return bar

    @staticmethod
    def _group_css() -> str:
        return (f"QGroupBox{{color:{theme.accent()};font-weight:bold;"
                f"border:1px solid {theme.edge()};border-radius:4px;margin-top:7px;"
                f"padding:6px}}"
                f"QGroupBox::title{{subcontrol-origin:margin;left:7px}}")

    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(250)
        scroll.setMaximumWidth(380)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        # --- what is plotted, and therefore what is heard -------------------
        box = QGroupBox("Score")
        box.setStyleSheet(self._group_css())
        g = QGridLayout(box)
        g.setContentsMargins(4, 4, 4, 4)
        g.addWidget(QLabel("View"), 0, 0)
        self._combo_view = QComboBox()
        self._combo_view.currentIndexChanged.connect(self._on_view_changed)
        g.addWidget(self._combo_view, 0, 1)
        g.addWidget(QLabel("Voice"), 1, 0)
        self._combo_voice = QComboBox()
        for v in VOICE_MODES:
            self._combo_voice.addItem(
                {"pulsar": "Pulsar (building's own rate)",
                 "arc": "Arc (UPIC sweep)",
                 "audify": "Audify (speed up)"}.get(v, v), v)
        self._combo_voice.setCurrentIndex(
            max(self._combo_voice.findData(self._cfg.voice), 0))
        self._combo_voice.currentIndexChanged.connect(self._on_dirty)
        g.addWidget(self._combo_voice, 1, 1)
        g.addWidget(QLabel("Sweep"), 2, 0)
        self._spin_dur = QDoubleSpinBox()
        self._spin_dur.setRange(2.0, 120.0)
        self._spin_dur.setValue(self._cfg.duration_s)
        self._spin_dur.setSuffix(" s")
        self._spin_dur.valueChanged.connect(self._on_dirty)
        g.addWidget(self._spin_dur, 2, 1)
        g.addWidget(QLabel("Scale"), 3, 0)
        self._combo_scale = QComboBox()
        self._combo_scale.addItem("Auto (contour for records)", "auto")
        for m in SCALE_MODES:
            self._combo_scale.addItem(
                {"envelope": "Envelope: smooth contour",
                 "magnitude": "Magnitude: ignore sign",
                 "signed": "Signed: the value itself"}.get(m, m), m)
        self._combo_scale.setToolTip(
            "What the pitch follows. A record swings through zero, so the raw "
            "value makes the pitch wobble with the waveform; the envelope "
            "leaves a slow contour instead. Auto picks per view.")
        self._combo_scale.currentIndexChanged.connect(self._on_dirty)
        g.addWidget(self._combo_scale, 3, 1)

        g.addWidget(QLabel("Smoothing"), 4, 0)
        self._spin_smooth = QDoubleSpinBox()
        self._spin_smooth.setRange(0.1, 25.0)
        self._spin_smooth.setSingleStep(0.5)
        self._spin_smooth.setValue(2.0)
        self._spin_smooth.setSuffix(" %")
        self._spin_smooth.setToolTip(
            "How much of the sweep the envelope is averaged over. A record's "
            "contour wanders: a mode shape rises once and sounds like a "
            "gesture. More smoothing turns one into the other.")
        self._spin_smooth.valueChanged.connect(self._on_dirty)
        g.addWidget(self._spin_smooth, 4, 1)

        g.addWidget(QLabel("Channel"), 5, 0)
        self._combo_channel = QComboBox()
        self._combo_channel.setToolTip(
            "Which sensor channel a time-domain view is drawn from. "
            "Rebuilt from the stored recording, so switching costs nothing.")
        self._combo_channel.currentIndexChanged.connect(self._on_view_changed)
        g.addWidget(self._combo_channel, 5, 1)
        g.setColumnStretch(1, 1)
        lay.addWidget(box)

        # --- which traces actually sound -----------------------------------
        # One curve at a time by default. Four sensors playing at once is a
        # chord, not a reading; the point is to hear ONE thing and compare.
        tbox = QGroupBox("Traces")
        tbox.setStyleSheet(self._group_css())
        tl_ = QVBoxLayout(tbox)
        tl_.setContentsMargins(4, 4, 4, 4)
        tl_.setSpacing(2)
        btns = QHBoxLayout()
        for text, fn in (("All", lambda: self._set_all_traces(True)),
                         ("None", lambda: self._set_all_traces(False)),
                         ("First", self._select_first_trace)):
            b = QPushButton(text)
            b.setMaximumWidth(62)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        tl_.addLayout(btns)
        self._trace_box = QWidget()
        self._trace_lay = QVBoxLayout(self._trace_box)
        self._trace_lay.setContentsMargins(0, 2, 0, 0)
        self._trace_lay.setSpacing(1)
        tl_.addWidget(self._trace_box)
        self._trace_checks: list = []
        lay.addWidget(tbox)

        # --- bells ----------------------------------------------------------
        bells = QGroupBox("Bells at the eigenfrequencies")
        bells.setStyleSheet(self._group_css())
        bg = QGridLayout(bells)
        bg.setContentsMargins(4, 4, 4, 4)
        self._chk_bells = QCheckBox("ring on crossing")
        self._chk_bells.setChecked(self._cfg.bells)
        self._chk_bells.toggled.connect(self._on_dirty)
        bg.addWidget(self._chk_bells, 0, 0, 1, 2)
        self._combo_timbre = []
        for i in range(3):
            lbl = QLabel(f"f{i + 1}")
            lbl.setStyleSheet(f"color:{theme.mode_colors()[i]};font-weight:bold;")
            c = QComboBox()
            for t in BELL_TIMBRES:
                c.addItem(t, t)
            c.setCurrentIndex(max(c.findData(
                ("bell", "organ", "triangle")[i]), 0))
            c.currentIndexChanged.connect(self._on_dirty)
            bg.addWidget(lbl, i + 1, 0)
            bg.addWidget(c, i + 1, 1)
            self._combo_timbre.append(c)
        bg.setColumnStretch(1, 1)
        lay.addWidget(bells)

        # --- sound ----------------------------------------------------------
        snd = QGroupBox("Sound")
        snd.setStyleSheet(self._group_css())
        sg_ = QGridLayout(snd)
        sg_.setContentsMargins(4, 4, 4, 4)
        self._spin_flo = QDoubleSpinBox()
        self._spin_flo.setRange(20.0, 2000.0)
        self._spin_flo.setValue(self._cfg.f_lo)
        self._spin_flo.setSuffix(" Hz")
        self._spin_fhi = QDoubleSpinBox()
        self._spin_fhi.setRange(200.0, 16000.0)
        self._spin_fhi.setValue(self._cfg.f_hi)
        self._spin_fhi.setSuffix(" Hz")
        self._spin_master = QDoubleSpinBox()
        self._spin_master.setRange(0.0, 1.0)
        self._spin_master.setSingleStep(0.05)
        self._spin_master.setValue(self._cfg.master)
        for r, (lbl, w, tip) in enumerate((
                ("pitch low", self._spin_flo, "Bottom of the plot maps here"),
                ("pitch high", self._spin_fhi, "Top of the plot maps here"),
                ("master", self._spin_master, ""))):
            w.valueChanged.connect(self._on_dirty)
            if tip:
                w.setToolTip(tip)
            sg_.addWidget(QLabel(lbl), r, 0)
            sg_.addWidget(w, r, 1)
        sg_.setColumnStretch(1, 1)
        lay.addWidget(snd)

        # --- transport ------------------------------------------------------
        tr = QGroupBox("Play")
        tr.setStyleSheet(self._group_css())
        tl = QVBoxLayout(tr)
        tl.setContentsMargins(4, 4, 4, 4)
        self._btn_play = QPushButton("▶  Play")
        self._btn_play.setStyleSheet(f"color:{theme.semantic('play')};font-weight:bold;")
        self._btn_play.clicked.connect(self._on_play)
        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.clicked.connect(self._on_stop)
        self._chk_loop = QCheckBox("loop")
        self._btn_wav = QPushButton("Save WAV…")
        self._btn_wav.clicked.connect(self._on_save_wav)
        for w in (self._btn_play, self._btn_stop, self._chk_loop, self._btn_wav):
            tl.addWidget(w)
        lay.addWidget(tr)

        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._info.setStyleSheet(f"color:{theme.dim()};")
        lay.addWidget(self._info)
        lay.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_stage(self) -> QWidget:
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(theme.PLOT_BG)
        for ax in ("left", "bottom"):
            a = self._plot.getAxis(ax)
            a.setPen(pg.mkPen(theme.PLOT_AXIS))
            a.setTextPen(pg.mkPen(theme.DIM_ON_DARK))
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.addLegend(offset=(-10, 10))
        self._playhead = pg.InfiniteLine(angle=90, movable=False,
                                         pen=pg.mkPen(theme.STATUS_ON_DARK, width=2))
        self._playhead.setVisible(False)
        self._plot.addItem(self._playhead)
        col.addWidget(self._plot, 1)

        self._caption = QLabel("The plot is the score: the playhead sweeps left "
                               "to right and the curve's height becomes sound.")
        self._caption.setWordWrap(True)
        self._caption.setStyleSheet(f"color:{theme.dim()};padding:2px 6px;")
        col.addWidget(self._caption)
        return wrap

    # ------------------------------------------------------------ placement
    def apply_sensor_map(self, mapping) -> None:
        """Adopt the Settings placement map. This tab owns no picker."""
        if hasattr(mapping, "to_mapping"):
            mapping = mapping.to_mapping()
        self._sensor_mapping = dict(mapping) if isinstance(mapping, dict) else None
        # A running loop must hear about it too, or every later cycle would keep
        # using the placement the user has just changed — the shaker row, the
        # channel and the floor labels all come from this map.
        if self._live_worker is not None:
            self._live_worker.configure(mapping=self._sensor_mapping)

    # --------------------------------------------------------------- sources
    def _refresh_sessions(self) -> None:
        """One list of everything openable: saved sessions first, then raw
        recordings. Saved ones carry their results, so they reopen instantly."""
        from ...config.app_config import AppPaths
        from ...dataio import modal_session_loader as msl
        from ...sonification.structure_pulse import describe, list_sessions

        self._combo_session.clear()
        paths = AppPaths()
        try:
            for folder in list_sessions(self._saved_dir_base()):
                self._combo_session.addItem(f"◆ {describe(folder)}",
                                            ("saved", str(folder)))
        except Exception:
            logger.debug("structure pulse: listing saved sessions failed",
                         exc_info=True)
        try:
            for s_ in msl.list_sessions(Path(paths.sensor_recordings))[:40]:
                self._combo_session.addItem(
                    f"{s_.parent.parent.name} · {s_.name}", ("recording", str(s_)))
        except Exception:
            logger.debug("structure pulse: listing recordings failed", exc_info=True)
        if self._combo_session.count() == 0:
            self._combo_session.addItem("nothing to open yet", ("", ""))

    @staticmethod
    def _saved_dir_base() -> Path:
        from ...config.app_config import AppPaths

        return Path(AppPaths().sonification_output) / "structure_pulse"

    def _on_load(self) -> None:
        data = self._combo_session.currentData() or ("", "")
        kind, path = (data if isinstance(data, tuple) else ("", ""))
        if not path:
            self._set_status("nothing selected")
            return
        if kind == "saved":
            self._start_worker(saved_dir=path)
        else:
            self._start_worker(session_dir=path)

    def _on_browse(self) -> None:
        """Open any recording folder, not only the ones under the output tree.

        Accepts either a Smart Recording folder or a saved Structure Pulse
        session, and works out which by looking for the session manifest.
        """
        from ...config.app_config import AppPaths

        start = str(AppPaths().sensor_recordings)
        folder = QFileDialog.getExistingDirectory(
            self, "Open a recording or a saved session", start)
        if not folder:
            return
        path = Path(folder)
        if (path / "pulse.json").is_file():
            self._start_worker(saved_dir=str(path))
            return
        self._start_worker(session_dir=str(path))

    def _on_save_session(self) -> None:
        """Store the recording and its results together."""
        if self._dataset is None:
            self._set_status("nothing to save yet")
            return
        from ...sonification.structure_pulse import save_dataset

        name, ok = QInputDialog.getText(
            self, "Save session", "Name this session (optional):")
        if not ok:
            return
        try:
            folder = save_dataset(self._dataset, self._saved_dir_base(),
                                  name=name.strip())
        except Exception as exc:
            logger.exception("structure pulse: save failed")
            self._set_status(f"save failed: {exc}")
            return
        raw_note = ("with its raw data, can be re-analysed later"
                    if self._dataset.has_raw else "results only")
        self._set_status(f"saved {folder.name} ({raw_note})")
        self._refresh_sessions()
        idx = self._combo_session.findData(("saved", str(folder)))
        if idx >= 0:
            self._combo_session.setCurrentIndex(idx)

    def _on_record(self) -> None:
        if self._worker is not None:
            self._worker.abort()
            return
        streaming = False
        try:
            streaming = bool(self._controller.is_streaming())
        except Exception:
            streaming = False
        if not streaming:
            # Status line, not a dialog: recording is a normal button a user
            # will press before starting the stream, and a modal box on a plain
            # ordering mistake is noise.
            self._set_status(
                "start the live stream first (Live Signals → Start), then record")
            return
        self._start_worker(record_s=float(self._spin_record.value()))

    def _start_worker(self, *, record_s: float = 0.0, session_dir=None,
                      saved_dir=None) -> None:
        if self._worker is not None:
            return
        self._on_stop(end_live=False)
        self._worker = _AnalysisWorker(self._controller, self._sensor_mapping,
                                       record_s=record_s, session_dir=session_dir,
                                       saved_dir=saved_dir, parent=self)
        self._worker.done.connect(self._on_analysis_done)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        if record_s:
            self._btn_record.setText("■  Cancel")
            self._set_status(f"recording {record_s:.0f} s…")
        elif saved_dir is not None:
            self._set_status("reopening…")
        else:
            self._set_status("analysing…")
        self._update_enabled()

    @Slot()
    def _on_worker_finished(self) -> None:
        w, self._worker = self._worker, None
        if w is not None:
            w.deleteLater()
        self._btn_record.setText("●  Start recording")
        self._update_enabled()

    def _describe(self, dataset) -> None:
        """The one-line summary under the controls. Shared by both paths so the
        live loop reports its frequencies exactly like an opened recording."""
        f = np.asarray(dataset.modal.frequencies_hz, dtype=float).ravel()
        z = np.asarray(dataset.modal.damping_ratios, dtype=float).ravel()
        bits = [f"{dataset.duration_s:.0f} s @ {dataset.fs:.0f} Hz",
                f"{len(dataset.sensor_ids)} sensors",
                "ch " + " ".join(dataset.channels)]
        if f.size:
            bits.append("f = " + ", ".join(f"{v:.2f}" for v in f) + " Hz")
        if z.size and np.isfinite(z).any():
            bits.append("ζ = " + ", ".join(
                f"{v * 100:.1f}%" if np.isfinite(v) else "n/a" for v in z))
        if not dataset.modal.ok and dataset.modal.message:
            # Without this the spectrum simply has no f1/f2/f3 and the user is
            # left guessing whether the structure or the software is at fault.
            bits.append(f"no eigenfrequencies: {dataset.modal.message}")
        if getattr(dataset, "saved_path", ""):
            bits.append("saved" + (" · re-analysable" if dataset.has_raw else ""))
        self._info.setText(" · ".join(bits))

    def _populate_views(self, names) -> None:
        """Fill the view list, KEEPING whatever the user had selected.

        Live mode replaces the dataset every cycle; rebuilding the list blindly
        would reset the choice each time and make the tab impossible to steer.
        """
        names = [str(n) for n in names]
        want = str(self._combo_view.currentData() or "")
        have = [self._combo_view.itemData(i)
                for i in range(self._combo_view.count())]
        if have == names:
            return                                  # nothing to do
        self._combo_view.blockSignals(True)
        self._combo_view.clear()
        for name in names:
            self._combo_view.addItem(name, name)
        i = self._combo_view.findData(want)
        self._combo_view.setCurrentIndex(i if i >= 0 else 0)
        self._combo_view.blockSignals(False)

    @Slot(object, str)
    def _on_analysis_done(self, dataset, message: str) -> None:
        if dataset is None:
            self._set_status(message)
            return
        self._dataset = dataset
        self._populate_views(dataset.view_names())
        self._describe(dataset)
        self._set_status(message)
        self._on_view_changed()

    # ----------------------------------------------------------------- render
    def _current_config(self) -> PulseConfig:
        cfg = replace(self._cfg)
        cfg.voice = str(self._combo_voice.currentData() or "pulsar")
        cfg.duration_s = float(self._spin_dur.value())
        # Live: fit `replay` sweeps inside one capture window, so the cycle is
        # heard that many times before the next one is ready.
        if self._live_worker is not None:
            replay = max(1, int(self._spin_replay.value()))
            if replay > 1:
                cfg.duration_s = max(1.0, float(self._spin_live.value()) / replay)
            # Audify is deliberately NOT retimed to the slot. Its speed is
            # what carries the structure into hearing range: the automatic
            # factor puts a 2 Hz mode at ~220 Hz, and forcing a pass to last
            # window/replay instead gives roughly x8, which lands it at 16 Hz
            # where there is nothing to hear. It keeps its own speed and simply
            # repeats until the next cycle arrives.
        cfg.bells = bool(self._chk_bells.isChecked())
        cfg.f_lo = float(self._spin_flo.value())
        cfg.f_hi = float(self._spin_fhi.value())
        cfg.master = float(self._spin_master.value())
        cfg.scale = str(self._combo_scale.currentData() or "auto")
        cfg.envelope_smooth = float(self._spin_smooth.value()) / 100.0
        return cfg.clamped()

    def _apply_timbres(self, view) -> None:
        for mk in getattr(view, "markers", []):
            if 0 <= mk.mode < len(self._combo_timbre):
                mk.timbre = str(self._combo_timbre[mk.mode].currentData())

    def _on_dirty(self, *_a) -> None:
        """A knob moved: re-render so what you hear matches what is set."""
        if self._live_worker is not None:
            # Tell the worker, AND re-render the window we already have. Waiting
            # for the next cycle meant a change did nothing for up to the whole
            # cycle length, which made live mode feel broken.
            self._live_worker.configure(
                view_name=str(self._combo_view.currentData() or ""),
                config=self._current_config())
        if self._dataset is not None:
            self._rerender()

    def _on_view_changed(self, *_a) -> None:
        if self._live_worker is not None:
            # Do this FIRST and unconditionally: before the opening cycle lands
            # there is no dataset yet, and an early return left the loop
            # rendering whatever it started on however the combo was set.
            self._live_worker.configure(
                view_name=str(self._combo_view.currentData() or ""),
                config=self._current_config())
        if self._dataset is None:
            return
        sender = self.sender()
        if sender is not self._combo_channel:
            self._refresh_channels()
        self._refresh_traces()
        self._reset_plot_range()
        self._draw_view()
        if self._live_worker is not None:
            self._live_worker.configure(
                view_name=str(self._combo_view.currentData() or ""),
                config=self._current_config())
        self._rerender()

    def _base_view(self):
        """The view as the analysis built it, before channel or trace choice.

        Returns None when the selection is not in this dataset — a three second
        live window has no mode shapes, say. Silently falling back to another
        view would mean the plot and the label disagreed.
        """
        if self._dataset is None:
            return None
        want = str(self._combo_view.currentData() or "")
        if want and want not in self._dataset.views:
            return None
        return self._dataset.view(want)

    def _current_view(self):
        """What is actually drawn and played: the chosen channel, the chosen
        traces. Everything downstream — plot, render, playhead — uses this."""
        base = self._base_view()
        if base is None:
            return None
        from ...sonification.structure_pulse.analysis import (TIME_KINDS,
                                                              rebuild_time_view)

        view = base
        name = str(self._combo_view.currentData() or "")
        axis = str(self._combo_channel.currentData() or "")
        if name in TIME_KINDS and axis:
            swapped = rebuild_time_view(self._dataset, name, axis,
                                        self._sensor_mapping)
            if swapped is not None:
                view = swapped
        keep = [i for i, chk in enumerate(self._trace_checks) if chk.isChecked()]
        if keep and len(keep) < len(view.curves):
            view = replace(view, curves=[view.curves[i] for i in keep
                                         if i < len(view.curves)])
        return view

    def _refresh_channels(self) -> None:
        """Offer the channels this dataset actually holds, for time views only."""
        from ...sonification.structure_pulse.analysis import TIME_KINDS

        name = str(self._combo_view.currentData() or "")
        is_time = name in TIME_KINDS and bool(getattr(self._dataset, "raw", None))
        self._combo_channel.blockSignals(True)
        self._combo_channel.clear()
        if is_time:
            for ax in (self._dataset.channels or []):
                if name == "Displacement" and str(ax).startswith("g"):
                    continue          # integrating a rotation rate is not a length
                self._combo_channel.addItem(ax, ax)
        self._combo_channel.setEnabled(self._combo_channel.count() > 1)
        self._combo_channel.blockSignals(False)

    def _refresh_traces(self) -> None:
        """One checkbox per curve. Only the first is on: hear one thing."""
        for chk in self._trace_checks:
            chk.setParent(None)
            chk.deleteLater()
        self._trace_checks = []
        base = self._base_view()
        if base is None:
            return
        from ...sonification.structure_pulse.analysis import (TIME_KINDS,
                                                              rebuild_time_view)

        name = str(self._combo_view.currentData() or "")
        axis = str(self._combo_channel.currentData() or "")
        view = base
        if name in TIME_KINDS and axis:
            swapped = rebuild_time_view(self._dataset, name, axis,
                                        self._sensor_mapping)
            if swapped is not None:
                view = swapped
        for i, c in enumerate(view.curves):
            chk = QCheckBox(c.label or f"trace {i + 1}")
            chk.setStyleSheet(
                f"color:{TRACE_COLORS[i % len(TRACE_COLORS)]};")
            chk.setChecked(i == 0)
            chk.toggled.connect(self._on_traces_changed)
            self._trace_lay.addWidget(chk)
            self._trace_checks.append(chk)

    def _set_all_traces(self, on: bool) -> None:
        if not self._trace_checks:
            return
        for i, chk in enumerate(self._trace_checks):
            chk.blockSignals(True)
            # "None" would be silence, so the first always survives
            chk.setChecked(bool(on) or i == 0)
            chk.blockSignals(False)
        self._on_traces_changed()

    def _select_first_trace(self) -> None:
        self._set_all_traces(False)

    def _on_traces_changed(self, *_a) -> None:
        self._reset_plot_range()
        if not any(chk.isChecked() for chk in self._trace_checks):
            if self._trace_checks:              # never leave nothing playing
                self._trace_checks[0].blockSignals(True)
                self._trace_checks[0].setChecked(True)
                self._trace_checks[0].blockSignals(False)
        self._draw_view()
        self._rerender()

    def _rerender(self) -> None:
        view = self._current_view()
        if view is None:
            want = str(self._combo_view.currentData() or "that view")
            self._set_status(
                f"{want} is not available in this window, "
                f"try a longer one" if self._live_worker is not None
                else f"{want} is not in this recording")
            self._player.clear()
            self._render = None
            self._update_enabled()
            return
        self._on_stop(end_live=False)
        self._apply_timbres(view)
        cfg = self._current_config()
        try:
            self._render = render_view(
                view, cfg, modal_frequencies=self._dataset.modal.frequencies_hz,
                data_fs=self._dataset.fs)
        except Exception as exc:
            logger.exception("structure pulse: render failed")
            self._set_status(f"render failed: {exc}")
            return
        if self._render.audio.size == 0:
            self._set_status(self._render.message or "nothing to play")
            self._player.clear()
        elif self._live_worker is not None:
            # live: swap the sound over immediately so plot and audio agree
            self._live_next = None
            self._play_render(self._render)
            return
        else:
            self._player.load(self._render)
            note = self._render.message or ""
            self._set_status(
                f"{view.name} · {cfg.voice} · {self._render.duration_s:.1f} s"
                + (f" · {note}" if note else ""))
        self._update_enabled()

    def _apply_stable_range(self, view) -> None:
        """Keep the vertical scale steady while live.

        Autoranging on every cycle made the plot jump each time the structure
        got louder or quieter — four different scales in four cycles — which is
        unreadable when you are watching it change. The range therefore GROWS to
        fit new data and never shrinks on its own; it is only reset when what is
        plotted actually changes.
        """
        key = (view.name,
               str(self._combo_channel.currentData() or ""),
               tuple(c.label for c in view.curves))
        ys = [c.y[np.isfinite(c.y)] for c in view.curves if c.is_usable]
        ys = [a for a in ys if a.size]
        if not ys:
            return
        lo = float(min(float(a.min()) for a in ys))
        hi = float(max(float(a.max()) for a in ys))
        if hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad

        if key != self._range_key or self._range_y is None:
            self._range_key, self._range_y = key, (lo, hi)     # a new thing
        else:
            prev_lo, prev_hi = self._range_y
            # grow only: a quieter cycle keeps the old scale, so the trace
            # visibly shrinks inside a fixed frame instead of the frame moving
            self._range_y = (min(prev_lo, lo), max(prev_hi, hi))
        vb = self._plot.getViewBox()
        vb.enableAutoRange(axis="y", enable=False)
        vb.setYRange(self._range_y[0], self._range_y[1], padding=0.0)
        vb.enableAutoRange(axis="x", enable=True)

    def _reset_plot_range(self) -> None:
        """Forget the held range — used when the user changes what is shown."""
        self._range_key = None
        self._range_y = None
        vb = self._plot.getViewBox()
        vb.enableAutoRange(axis="y", enable=True)

    def _plot_x(self, data_x: float) -> float:
        """Data x -> the coordinate the PLOT is drawn in.

        pyqtgraph's log mode draws in log10(x), but ``InfiniteLine.setPos``
        takes view coordinates, not data ones. Feeding it the raw period put
        the playhead on a 0.02-2 s axis somewhere past the right-hand edge —
        the line appeared to skip the start, or never showed at all.
        """
        view = self._current_view()
        if view is not None and view.x_log and data_x > 0:
            return float(np.log10(data_x))
        return float(data_x)

    def _draw_view(self) -> None:
        view = self._current_view()
        self._plot.clear()
        self._plot.addItem(self._playhead)
        if view is None:
            self._caption.setText(
                f"{self._combo_view.currentData() or 'This view'} is not in the "
                f"current window.")
            return
        self._plot.setLogMode(x=bool(view.x_log), y=False)
        self._plot.setLabel("bottom", view.x_label, color=theme.DIM_ON_DARK, size="9pt")
        self._plot.setLabel("left", view.y_label, color=theme.DIM_ON_DARK, size="9pt")
        for i, c in enumerate(view.curves):
            if not c.is_usable:
                continue
            pen = pg.mkPen(TRACE_COLORS[i % len(TRACE_COLORS)], width=2)
            self._plot.plot(c.x, c.y, pen=pen, name=c.label or None)
        self._apply_stable_range(view)
        for mk in view.markers:
            colour = (TRACE_COLORS[mk.mode % len(TRACE_COLORS)]
                      if mk.mode >= 0 else theme.DIM_ON_DARK)
            line = pg.InfiniteLine(pos=self._plot_x(float(mk.x)), angle=90,
                                   movable=False,
                                   pen=pg.mkPen(colour, width=1,
                                                style=Qt.DashLine),
                                   label=mk.label,
                                   labelOpts={"color": colour, "position": 0.92,
                                              "movable": False})
            self._plot.addItem(line)
        self._caption.setText(view.note or "")

    # -------------------------------------------------------------- playback
    def _ensure_audio(self) -> bool:
        """Guarantee a LIVE output stream, reopening a dead one.

        The bug this exists to kill: holding a non-None ``AudioOutput`` is not
        the same as holding a working one. A stream can stop under us — a device
        change, a backend abort, a sleep/wake — and the object stays behind
        looking healthy. Playback then went silent for the rest of the session
        and only restarting the app fixed it. So test ``running``, never merely
        ``is not None``.
        """
        from ...sonification.chorus.audio_out import AudioOutput, is_audio_available

        if self._audio is not None and self._audio.running:
            return True
        if self._audio is not None:                  # stale handle: drop it
            try:
                self._audio.stop()
            except Exception:
                logger.debug("structure pulse: closing a dead stream failed",
                             exc_info=True)
            self._audio = None
        if not is_audio_available():
            self._silent = True
            return False
        out = AudioOutput()
        if not out.start(self._player.pull):
            self._silent = True
            return False
        self._audio = out
        self._silent = False
        self._stall_ticks = 0
        return True

    def _on_play(self) -> None:
        if not self._player.has_audio:
            return
        if not self._ensure_audio():
            # Say it in the status line, never in a dialog: this would fire on
            # EVERY press on a machine with no audio device, and a modal dialog
            # in a playback path blocks the thread that is supposed to be
            # animating the playhead.
            self._set_status(
                "no audio device: playhead runs silently "
                "(pip install -r requirements.txt for sound)")
        self._last_pos = -1.0
        self._stall_ticks = 0
        self._player.play(restart=True, loop=bool(self._chk_loop.isChecked()))
        self._playhead.setVisible(True)
        self._tick.start()
        self._update_enabled()

    def _on_stop(self, *, end_live: bool = True) -> None:
        """Stop the sound. In live mode that means stopping the LOOP too.

        Otherwise Stop looks broken: the player halts, then the next cycle
        lands a few seconds later and starts it again.
        """
        if end_live and self._live_worker is not None:
            self._stop_live()
            return
        self._player.stop()
        self._tick.stop()
        self._playhead.setVisible(False)
        self._update_enabled()

    def _on_tick(self) -> None:
        if self._render is None:
            return
        t = self._player.position_s
        if self._silent:                      # no device: advance by wall clock
            t = min(t + PLAYHEAD_MS / 1000.0, self._render.duration_s)
            self._player.seek_seconds(t)
        elif self._player.is_playing:
            # Watchdog. If the player says it is playing but the cursor has not
            # moved, the device is not pulling: the stream died without telling
            # anyone. Reopen it once and carry on from where we stood, rather
            # than leaving the user with a frozen playhead and no sound.
            if abs(t - self._last_pos) < 1e-9:
                self._stall_ticks += 1
                if self._stall_ticks == STALL_TICKS:
                    logger.warning("structure pulse: output stalled, reopening")
                    self._set_status("audio stalled: reopening the device")
                    if self._ensure_audio():
                        self._player.seek_seconds(t)
                        self._player.play(restart=False,
                                          loop=bool(self._chk_loop.isChecked()))
                    else:
                        self._set_status("audio device lost: playing silently")
            else:
                self._stall_ticks = 0
            self._last_pos = t
        self._playhead.setPos(self._plot_x(self._render.x_at(t)))
        # A repeating cycle never stops on its own, so the hand-over happens
        # when the playhead wraps. Swapping mid-sweep would cut the sound in
        # half; at the wrap the next cycle simply starts where this one ended.
        if (self._live_worker is not None and self._live_next is not None
                and self._player.is_playing and t + 0.05 < self._last_wrap_pos):
            nxt, self._live_next = self._live_next, None
            self._last_wrap_pos = 0.0
            self._play_render(nxt)
            return
        self._last_wrap_pos = t
        if not self._player.is_playing:
            if self._live_worker is not None and self._live_next is not None:
                nxt, self._live_next = self._live_next, None
                self._play_render(nxt)    # seamless hand-over to the next cycle
                return
            if self._live_worker is None:
                self._on_stop()
            else:
                self._player.stop()      # wait for the next cycle, stay live

    # ------------------------------------------------------------ live mode
    def _on_live_toggled(self, on: bool) -> None:
        if on:
            self._start_live()
        else:
            self._stop_live()

    def _on_live_window(self, value: int) -> None:
        if self._live_worker is not None:
            self._live_worker.configure(window_s=float(value))
            self._push_live_config()

    def _on_replay_changed(self, _value: int) -> None:
        """Replays per cycle changed: the next cycle is rendered to match.

        The one already playing is left alone. Re-rendering it underneath the
        playhead would cut the sound for a setting whose whole point is how the
        sound is paced.
        """
        if self._live_worker is not None:
            self._push_live_config()

    def _push_live_config(self) -> None:
        """Hand the worker the render settings the next cycle should use."""
        if self._live_worker is None:
            return
        self._live_worker.configure(view_name=self._combo_view.currentText(),
                                    config=self._current_config())

    def _start_live(self) -> None:
        streaming = False
        try:
            streaming = bool(self._controller.is_streaming())
        except Exception:
            streaming = False
        if not streaming:
            self._set_status(
                "live mode needs the stream running (Live Signals → Start)")
            self._btn_live.setChecked(False)
            return
        if self._live_thread is not None:
            return
        from ._pulse_live import LivePulseWorker

        self._on_stop(end_live=False)
        self._live_next = None
        self._live_cycle = 0
        thread = QThread(self)
        worker = LivePulseWorker(self._controller, self._sensor_mapping,
                                 window_s=float(self._spin_live.value()))
        worker.configure(view_name=str(self._combo_view.currentData() or ""),
                         config=self._current_config())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ready.connect(self._on_live_ready)
        worker.progress.connect(self._on_live_progress)
        worker.failed.connect(self._on_live_failed)
        worker.stopped.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._live_worker, self._live_thread = worker, thread
        thread.start()
        self._btn_live.setText("■  Stop live")
        self._set_status("live: filling the first window…")
        self._update_enabled()

    def _stop_live(self, *, wait: bool = False) -> None:
        """End the loop. **Never blocks the GUI thread unless asked to.**

        The old version waited up to four seconds for the worker to finish, on
        the GUI thread — which is itself a freeze, and exactly what the user hit
        when a cycle was mid-analysis: the Stop press appeared to do nothing and
        the window stopped repainting. The worker is told to stop, the tab
        detaches from it immediately, and the thread tidies itself up through
        ``finished``. Only ``shutdown`` waits, where blocking is correct.
        """
        worker, thread = self._live_worker, self._live_thread
        self._live_worker = None
        self._live_thread = None
        self._live_next = None
        if worker is not None:
            worker.stop()
            # Detach now: a cycle already in flight must not land in a tab that
            # has moved on, or it would restart playback after the user stopped.
            for sig in (worker.ready, worker.progress, worker.failed):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
        if thread is not None:
            thread.quit()
            if wait and not thread.wait(5000):
                logger.warning("structure pulse: live thread did not finish")
        self._on_stop(end_live=False)
        self._btn_live.setText("◉  Live")
        if self._btn_live.isChecked():
            self._btn_live.blockSignals(True)
            self._btn_live.setChecked(False)
            self._btn_live.blockSignals(False)
        # Leave the last cycle loaded and playable: stopping live should hand
        # the window back, not throw it away.
        if self._dataset is not None and self._dataset.view_names():
            self._populate_views(self._dataset.view_names())
            self._rerender()
            self._set_status(
                f"live stopped: holding the last {self._dataset.duration_s:.0f} s window")
        self._update_enabled()

    def _set_status(self, text: str) -> None:
        """Status text, elided to whatever width the label currently has."""
        lbl = getattr(self, "_status", None)
        if lbl is None:
            return
        self._status_full = str(text)
        lbl.setToolTip(self._status_full)
        self._elide_status()

    def _elide_status(self) -> None:
        lbl = getattr(self, "_status", None)
        if lbl is None:
            return
        full = getattr(self, "_status_full", "")
        width = max(lbl.width() - 4, 40)
        lbl.setText(lbl.fontMetrics().elidedText(full, Qt.ElideRight, width))

    def resizeEvent(self, event) -> None:          # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._elide_status()

    @Slot(float, int)
    def _on_live_progress(self, frac: float, cycle: int) -> None:
        if self._live_worker is None:
            return
        bits = [f"live · cycle {cycle + 1} capturing {frac * 100:.0f}%"]
        if self._player.is_playing:
            bits.append("playing")
        elif self._render is None:
            # Carry the reason forward. The progress line fires several times a
            # second and would otherwise wipe out the one message explaining
            # why nothing is sounding.
            want = str(self._combo_view.currentData() or "that view")
            bits.append(f"{want} not in this window, try a longer one")
        self._set_status(" · ".join(bits))

    @Slot(object, object, int)
    def _on_live_ready(self, dataset, render, cycle: int) -> None:
        """A fresh cycle arrived. Show it now; play it when the current one ends."""
        if self._live_worker is None:
            return
        first = self._dataset is None or self._combo_view.count() == 0
        self._dataset = dataset
        self._live_cycle = cycle
        # Keep the LIST steady across cycles. A short window may not yield
        # every view — three seconds is not enough to identify mode shapes —
        # and rebuilding from each cycle's contents would make the combo
        # reshuffle under the user's cursor.
        from ...sonification.structure_pulse.analysis import PREVIEW_VIEWS

        names = list(PREVIEW_VIEWS)
        for extra in dataset.view_names():
            if extra not in names:
                names.append(extra)
        self._populate_views(names)
        self._describe(dataset)
        if first:
            self._refresh_channels()
            self._refresh_traces()
        self._draw_view()                 # the plot follows every cycle
        self._update_enabled()
        if render is None or render.audio.size == 0:
            return
        if self._player.is_playing:
            self._live_next = render      # queued; swapped in when this one ends
        else:
            self._play_render(render)

    @Slot(str)
    def _on_live_failed(self, message: str) -> None:
        logger.warning("structure pulse live: %s", message)
        self._set_status(message)

    def _play_render(self, render) -> None:
        """Load a render and start it, reusing the transport path."""
        self._render = render
        self._player.load(render)
        self._ensure_audio()
        self._last_pos = -1.0
        self._stall_ticks = 0
        # Live always repeats. Any render shorter than the capture cycle would
        # otherwise finish early and leave silence until the next one was
        # ready, which is what made audify look broken: its length comes from
        # the data, not from the Sweep box, so one pass can be under a second.
        live_loop = self._live_worker is not None
        self._player.play(restart=True,
                          loop=live_loop or bool(self._chk_loop.isChecked()))
        self._playhead.setVisible(True)
        if not self._tick.isActive():
            self._tick.start()
        self._update_enabled()

    def _on_save_wav(self) -> None:
        if self._render is None or self._render.audio.size == 0:
            return
        from ...config.app_config import AppPaths

        base = Path(AppPaths().sonification_output) / "structure_pulse"
        base.mkdir(parents=True, exist_ok=True)
        view = self._current_view()
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{(view.name if view else 'pulse').lower().replace(' ', '_')}_{stamp}.wav"
        path, _ = QFileDialog.getSaveFileName(self, "Save render",
                                              str(base / name), "WAV (*.wav)")
        if not path:
            return
        try:
            write_wav(path, self._render)
            self._set_status(f"saved {Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self, "Structure Pulse", f"Could not save: {exc}")

    # ---------------------------------------------------------------- state
    def _update_enabled(self) -> None:
        busy = self._worker is not None
        has = self._player.has_audio
        self._btn_play.setEnabled(has and not busy)
        self._btn_stop.setEnabled(self._player.is_playing)
        self._btn_wav.setEnabled(has)
        self._btn_save.setEnabled(self._dataset is not None and not busy)
        live = self._live_worker is not None
        # Always usable: choosing what to sonify is how the user tells the loop
        # what to produce, and that choice has to be possible before any data.
        self._combo_view.setEnabled(not busy)
        # a one-shot open would fight the loop, so park those while live
        self._btn_record.setEnabled(not live and not busy)
        # One source at a time: opening a recording mid-loop would leave two
        # datasets fighting over the same plot.
        self._combo_session.setEnabled(not live and not busy)
        for w in (getattr(self, "_btn_open", None), getattr(self, "_btn_browse", None)):
            if w is not None:
                w.setEnabled(not live and not busy)

    @Slot()
    def on_stream_started(self) -> None:
        self._update_enabled()

    @Slot()
    def on_stream_stopped(self) -> None:
        if self._worker is not None:
            self._worker.abort()
        if self._live_worker is not None:
            self._stop_live()
            self._set_status("stream stopped: live mode ended")

    def shutdown(self) -> None:
        """Stop audio and any worker. Called by MainWindow on close."""
        self._stop_live(wait=True)
        self._tick.stop()
        self._player.stop()
        if self._audio is not None:
            try:
                self._audio.stop()
            except Exception:
                logger.debug("structure pulse: audio stop failed", exc_info=True)
            self._audio = None
        w = self._worker
        if w is not None:
            w.abort()
            w.wait(3000)
            self._worker = None

    def closeEvent(self, event) -> None:          # noqa: N802 (Qt naming)
        self.shutdown()
        super().closeEvent(event)
