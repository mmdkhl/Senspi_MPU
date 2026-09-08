"""The physical structure, live: a wireframe placed by the Settings map.

Shared by the Digital Twin experiment (decision on one calibration) and Model
Updating's Continuous Update output (decision per cycle or rolling average).
One implementation, so both show the same thing.

What is drawn is filtered, doubly-integrated acceleration — **not** measured
displacement — reconstructed off the GUI thread by :class:`WireframeJob` and
handed back by signal. See :mod:`sensepi.digital_twin.motion` for what that
reconstruction is and is not.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ...analysis import sensor_layout as slayout
from ...digital_twin import motion as twin_motion

#: Visual exaggeration of the reconstructed yaw and the hard limit on it.
YAW_DISPLAY_GAIN = 25.0
YAW_DISPLAY_CLIP = 0.35        # rad, ~20 degrees


class WireframeRelay(QObject):
    """Carries a finished wireframe pose from the pool thread to the GUI."""

    ready = Signal(object, float)


class WireframeJob(QRunnable):
    """One reconstruction of the physical structure's pose, off the GUI thread.

    Everything here is thread-safe by construction: ``snapshot_modal_capture``
    copies under the controller's own lock, the layout is a frozen dataclass,
    and nothing touches a widget (G1). The result goes back through a signal.
    """

    def __init__(self, capture, layout, wire_peak: float, relay: WireframeRelay) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._capture = capture
        self._layout = layout
        self._peak = float(wire_peak)
        self._relay = relay

    def run(self) -> None:
        try:
            state, peak = _compute_wireframe_state(self._capture, self._layout, self._peak)
        except Exception:
            state, peak = None, self._peak
        self._relay.ready.emit(state, peak)


def _compute_wireframe_state(capture, layout, wire_peak: float):
    """Pure: capture -> per-floor pose. Returns ``(state | None, new_peak)``."""
    axis = layout.channel
    other = "ay" if axis == "ax" else "ax"
    window = twin_motion.WINDOW_S
    main = capture(axis=axis, last_seconds=window)
    if main is None or getattr(main, "data", None) is None or main.data.size == 0:
        return None, wire_peak
    fs = float(main.fs)
    if not np.isfinite(fs) or fs <= 2.0:
        return None, wire_peak
    ids = list(main.sensor_ids)
    cross = capture(axis=other, last_seconds=window)
    vert = capture(axis="az", last_seconds=window)
    spin = capture(axis="gz", last_seconds=window)

    def row_of(session, sid):
        if session is None or getattr(session, "data", None) is None:
            return None
        try:
            return np.asarray(session.data[list(session.sensor_ids).index(sid)], dtype=float)
        except (ValueError, IndexError):
            return None

    # az is NaN when the channel preset does not stream it (the default
    # AX/AY/GZ preset does not). Say so on the panel instead of drawing a flat
    # vertical without comment.
    az_streamed = False
    per_floor: dict = {}
    for sid in ids:
        floor = layout.story_map.get(int(sid))
        if floor is None:
            floor = 0 if int(sid) == layout.base_sensor_id else None
        if floor is None:
            continue
        main_row = row_of(main, sid)
        if main_row is None:
            continue
        u_main = twin_motion.displacement_at(main_row, fs)
        cross_row = row_of(cross, sid)
        u_cross = twin_motion.displacement_at(cross_row, fs) if cross_row is not None else 0.0
        vert_row = row_of(vert, sid)
        if vert_row is not None and np.isfinite(vert_row).any():
            az_streamed = True
            u_vert = twin_motion.displacement_at(vert_row, fs)
        else:
            u_vert = 0.0
        spin_row = row_of(spin, sid)
        yaw = twin_motion.angle_at(spin_row, fs) if spin_row is not None else 0.0
        ux, uy = ((u_main, u_cross) if axis == "ax" else (u_cross, u_main))
        cell = str(layout.cell_map.get(int(sid), "B2"))
        cx = ("ABC".find(cell[:1].upper()) - 1) * 0.8
        cy = (int(cell[1:2] or 2) - 2) * 0.8
        entry = per_floor.setdefault(int(floor), {"u": [], "yaw": [], "sensors": []})
        entry["u"].append((ux, uy, u_vert))
        entry["yaw"].append(yaw)
        entry["sensors"].append((int(sid), cx, cy))

    if not per_floor:
        return None, wire_peak
    peak = max((abs(v) for e in per_floor.values() for u in e["u"] for v in u), default=0.0)
    new_peak = max(peak, 0.85 * float(wire_peak))
    scale = twin_motion.normalising_scale([new_peak], target=0.8)

    state: dict = {"_az_streamed": az_streamed}
    for floor in range(0, int(layout.n_floors) + 1):
        e = per_floor.get(floor)
        if e is None:
            state[floor] = {"u": (0.0, 0.0, 0.0), "yaw": 0.0, "measured": False, "sensors": ()}
            continue
        us = np.asarray(e["u"], dtype=float)
        # Yaw is a genuine angle (radians) and small — a few milliradians on the
        # rig — so it is exaggerated for visibility and clipped so a noisy gyro
        # can never spin the floor into an unreadable frame.
        yaw = float(np.clip(float(np.mean(e["yaw"])) * YAW_DISPLAY_GAIN,
                            -YAW_DISPLAY_CLIP, YAW_DISPLAY_CLIP))
        state[floor] = {
            "u": tuple(float(v) * scale for v in us.mean(axis=0)),
            "yaw": yaw,
            "measured": True,
            "sensors": tuple(e["sensors"]),
        }
    return state, new_peak


class WireframeCanvas(FigureCanvas):
    """The PHYSICAL structure, live, built from the Settings placement map.

    Every other view in this tab shows the *numerical* model. This one shows the
    real one: nodes placed where the sensors actually are — floor for height,
    plan cell for position — moving with the measured signal.

    What is drawn is filtered, doubly-integrated acceleration, **not** measured
    displacement, and the caption says so. Accelerometers cannot measure
    displacement; integrating twice in open loop drifts without bound. Over a
    short rolling window, band-passed first and detrended between the two
    integrations, it is stable — which is exactly what the couple of seconds of
    accepted latency buys. ``gz`` is an angular *rate*, so it needs one
    integration to give the yaw angle, and that is what makes the floors visibly
    twist rather than only sway.

    Floors with no sensor are drawn dashed and dim. They are never interpolated:
    the same rule the story mapping follows everywhere else in this application.
    """

    _MEASURED = "#123B6D"
    _UNMEASURED = "#B9C2CC"
    _SENSOR = "#E4572E"
    _BACKGROUND = "#F7F9FC"

    def __init__(self, parent: QWidget | None = None) -> None:
        self.fig = Figure(figsize=(5.0, 5.0))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        self.fig.patch.set_facecolor(self._BACKGROUND)
        self._layout = slayout.layout_from_mapping(None)
        self._scale = 1.0
        self._empty()

    def _empty(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor(self._BACKGROUND)
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.draw_idle()

    def set_layout(self, layout) -> None:
        self._layout = layout
        if not layout.is_valid:
            self._empty()

    # -- geometry ---------------------------------------------------------
    def _floor_nodes(self):
        """``{floor: (x, y)}`` for the plan corners, plus the sensor positions."""
        return [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    def update_state(self, state: dict) -> None:
        """Draw one frame.

        ``state`` maps floor -> ``{"u": (ux, uy, uz), "yaw": radians,
        "measured": bool, "sensors": [(sid, cell_x, cell_y)]}``.
        """
        layout = self._layout
        if not layout.is_valid or not state:
            return
        self.ax.clear()
        self.ax.set_facecolor(self._BACKGROUND)
        corners = self._floor_nodes()
        n_floors = max(int(layout.n_floors), 1)

        prev_xy = None
        for floor in range(0, n_floors + 1):
            info = state.get(floor, {})
            ux, uy, uz = info.get("u", (0.0, 0.0, 0.0))
            yaw = float(info.get("yaw", 0.0))
            measured = bool(info.get("measured", False))
            colour = self._MEASURED if measured else self._UNMEASURED
            style = "-" if measured else "--"
            width = 1.6 if measured else 0.9

            c, s = np.cos(yaw), np.sin(yaw)
            xs, ys = [], []
            for (px, py) in corners + [corners[0]]:
                xs.append(px * c - py * s + ux)
                ys.append(px * s + py * c + uy)
            z = float(floor) + uz
            self.ax.plot(xs, ys, [z] * len(xs), style, color=colour, linewidth=width)

            if prev_xy is not None:
                for k in range(len(corners)):
                    self.ax.plot([prev_xy[0][k], xs[k]], [prev_xy[1][k], ys[k]],
                                 [float(floor) - 1 + prev_xy[2], z],
                                 style, color=colour, linewidth=width * 0.8)
            prev_xy = (xs, ys, info.get("u", (0, 0, 0))[2])

            for (sid, cx, cy) in info.get("sensors", ()):
                sx = cx * c - cy * s + ux
                sy = cx * s + cy * c + uy
                self.ax.scatter([sx], [sy], [z], color=self._SENSOR, s=26,
                                depthshade=False)
                self.ax.text(sx, sy, z + 0.12, f"S{sid}", color=self._SENSOR, fontsize=6,
                             ha="center", va="bottom")

        # Nothing but the frame: the caption and title live in Qt labels beside
        # the canvas, where they cannot overlap the drawing.
        self.ax.set_xlim(-2.2, 2.2)
        self.ax.set_ylim(-2.2, 2.2)
        self.ax.set_zlim(0, n_floors + 0.5)
        self.ax.set_box_aspect((1, 1, 1.35))
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_zticks(list(range(0, n_floors + 1)))
        self.ax.tick_params(labelsize=7)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.draw_idle()




class LiveStructureView(QWidget):
    """Title + wireframe + caption, and the timer/job plumbing that feeds it.

    Give it a controller (for ``snapshot_modal_capture``) and a placement map;
    call :meth:`start` / :meth:`stop`. Everything heavy runs in the pool.
    """

    def __init__(self, parent: QWidget | None = None, interval_ms: int = 400) -> None:
        super().__init__(parent)
        self._controller = None
        self._mapping: dict | None = None
        self._peak = 0.0
        self._busy = False
        self._relay = WireframeRelay()
        self._relay.ready.connect(self._on_state)
        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self.refresh)

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        title = QLabel("Physical structure — live")
        title.setStyleSheet("font-weight:600;")
        col.addWidget(title)
        self.canvas = WireframeCanvas(self)
        col.addWidget(self.canvas, stretch=1)
        self.caption = QLabel(
            "Reconstructed from acceleration (not measured displacement) · "
            "dashed floor = no sensor")
        self.caption.setStyleSheet("color:#7A8794;font-size:10px;")
        self.caption.setWordWrap(True)
        col.addWidget(self.caption)

    def set_controller(self, controller) -> None:
        self._controller = controller

    def apply_sensor_map(self, mapping) -> None:
        if hasattr(mapping, "to_mapping"):
            mapping = mapping.to_mapping()
        self._mapping = dict(mapping) if isinstance(mapping, dict) else None
        self.canvas.set_layout(slayout.layout_from_mapping(self._mapping))

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    @Slot()
    def refresh(self) -> None:
        if self._busy:
            return
        layout = slayout.layout_from_mapping(self._mapping)
        capture = getattr(self._controller, "snapshot_modal_capture", None)
        if not layout.is_valid or capture is None:
            return
        self._busy = True
        QThreadPool.globalInstance().start(WireframeJob(capture, layout, self._peak, self._relay))

    @Slot(object, float)
    def _on_state(self, state, peak: float) -> None:
        self._busy = False
        if state:
            self._peak = float(peak)
            self.canvas.update_state(state)
            note = "" if state.get("_az_streamed", True) else " · az not streamed"
            self.caption.setText(
                "Reconstructed from acceleration (not measured displacement) · "
                "dashed floor = no sensor" + note)
