"""Structure Pulse tab: lifecycle, rendering and playhead.

No SSH, no hardware, no audio device. Skips cleanly where Qt cannot create
widgets, the same way the chorus GUI tests do.
"""
from __future__ import annotations

import unittest

import numpy as np

try:
    from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception:                                   # pragma: no cover
    HAVE_QT = False

_NO_WIDGETS = ("a QCoreApplication already exists in this process, so widgets "
               "cannot be created; run this module on its own")


def _ensure_app():
    if not HAVE_QT:
        return None
    app = QApplication.instance()
    if app is None:
        try:
            return QApplication([])
        except Exception:
            return None
    return app if hasattr(app, "topLevelWidgets") else None


MAP = {"n_floors": 3, "axis": "x", "placements": [
    {"sensor_id": 1, "floor": 1, "cell": "B2"},
    {"sensor_id": 2, "floor": 2, "cell": "B2"},
    {"sensor_id": 3, "floor": 3, "cell": "A3"},
    {"sensor_id": 4, "floor": 3, "cell": "C1"}]}


class _Snap:
    def __init__(self, data, fs, ids):
        self.data, self.fs, self.sensor_ids = data, fs, list(ids)


if HAVE_QT:
    class _FakeController(QObject):
        """RecorderController's surface, no SSH and no hardware."""

        stream_started = Signal()
        stream_stopped = Signal()
        stream_rate_updated = Signal(str, float)

        def __init__(self, streaming=True):
            super().__init__()
            self._streaming = streaming
            self.window_s = 120.0
            rng = np.random.default_rng(3)
            t = np.arange(0, 90, 1 / 100.0)
            self._d = {}
            for ax in ("ax", "ay", "az", "gz"):
                rows = []
                for k in range(4):
                    sig = sum(np.sin(2 * np.pi * f * t) * a
                              for f, a in ((2.0, 0.05 * (k + 1)),
                                           (6.0, 0.02), (9.0, 0.01)))
                    rows.append(sig + rng.standard_normal(t.size) * 1e-3)
                self._d[ax] = np.vstack(rows)

        def is_streaming(self):
            return self._streaming

        def require_modal_window_seconds(self, s):
            self.window_s = max(self.window_s, float(s))

        def snapshot_modal_capture(self, *, axis="ax", last_seconds=None,
                                   target_fs=None):
            n = int((last_seconds or 6.0) * 100.0)
            return _Snap(self._d.get(axis, self._d["ax"])[:, -n:], 100.0,
                         [1, 2, 3, 4])


@unittest.skipUnless(HAVE_QT, "PySide6 not importable")
class TestStructurePulseTab(unittest.TestCase):
    def setUp(self):
        self.app = _ensure_app()
        if self.app is None:
            self.skipTest(_NO_WIDGETS)
        from sensepi.gui.tabs.tab_structure_pulse import StructurePulseTab

        self.ctrl = _FakeController()
        self.tab = StructurePulseTab(recorder_controller=self.ctrl)
        self.tab.apply_sensor_map(MAP)

    def tearDown(self):
        try:
            self.tab.shutdown()
        finally:
            self._pump(60)

    def _pump(self, ms):
        loop = QEventLoop()
        QTimer.singleShot(int(ms), loop.quit)
        loop.exec()

    def _dataset(self, seconds=20.0):
        """Build a dataset straight from the fake controller's capture."""
        from sensepi.sonification.structure_pulse import build_dataset_from_capture

        return build_dataset_from_capture(self.ctrl.snapshot_modal_capture,
                                          seconds=seconds, mapping=MAP)

    # ------------------------------------------------------------------ build
    def test_builds_with_pulsar_as_the_default_voice(self):
        self.assertEqual(self.tab._combo_voice.currentData(), "pulsar")
        self.assertEqual(self.tab._cfg.voice, "pulsar")

    def test_nothing_is_playable_before_a_recording_is_analysed(self):
        self.assertFalse(self.tab._btn_play.isEnabled())
        self.assertFalse(self.tab._btn_wav.isEnabled())
        # The view list IS offered up front — it is how the user tells a live
        # loop what to produce — but nothing is playable until data arrives.
        self.assertGreater(self.tab._combo_view.count(), 0)
        self.assertIsNone(self.tab._render)

    def test_play_before_analysis_is_a_no_op(self):
        self.tab._on_play()                 # must not raise
        self.assertFalse(self.tab._player.is_playing)

    # ---------------------------------------------------------------- dataset
    def test_a_live_capture_yields_every_view(self):
        ds = self._dataset()
        self.assertTrue(ds.modal.ok, ds.modal.message)
        names = ds.view_names()
        for expect in ("Time history", "Spectrum", "Response spectrum"):
            self.assertIn(expect, names)
        f = np.asarray(ds.modal.frequencies_hz).ravel()
        self.assertGreaterEqual(f.size, 1)
        self.assertAlmostEqual(float(f[0]), 2.0, delta=0.4)

    def test_analysis_populates_the_tab_and_renders(self):
        self.tab._on_analysis_done(self._dataset(), "test")
        self._pump(120)
        self.assertGreater(self.tab._combo_view.count(), 0)
        self.assertIsNotNone(self.tab._render)
        self.assertGreater(self.tab._render.audio.size, 0)
        self.assertTrue(self.tab._btn_play.isEnabled())
        self.assertIn("Hz", self.tab._info.text())

    def test_every_view_renders_without_raising(self):
        self.tab._on_analysis_done(self._dataset(), "test")
        self._pump(80)
        for i in range(self.tab._combo_view.count()):
            self.tab._combo_view.setCurrentIndex(i)
            self._pump(60)
            name = self.tab._combo_view.itemText(i)
            r = self.tab._render
            self.assertIsNotNone(r, name)
            self.assertTrue(np.isfinite(r.audio).all(), name)

    # ------------------------------------------------------------ the mapping
    def test_switching_voice_rerenders(self):
        self.tab._on_analysis_done(self._dataset(), "test")
        self._pump(80)
        first = self.tab._render.audio.copy()
        self.tab._combo_voice.setCurrentIndex(
            self.tab._combo_voice.findData("arc"))
        self._pump(120)
        self.assertEqual(self.tab._current_config().voice, "arc")
        self.assertFalse(np.array_equal(first, self.tab._render.audio))

    def test_sweep_duration_is_honoured(self):
        self.tab._on_analysis_done(self._dataset(), "test")
        self._pump(80)
        self.tab._spin_dur.setValue(5.0)
        self._pump(150)
        self.assertAlmostEqual(self.tab._render.duration_s, 5.0, places=5)

    def test_timbre_choice_reaches_the_markers(self):
        self.tab._on_analysis_done(self._dataset(), "test")
        self._pump(80)
        idx = self.tab._combo_view.findText("Spectrum")
        if idx < 0:
            self.skipTest("no spectrum view")
        self.tab._combo_view.setCurrentIndex(idx)
        self._pump(80)
        combo = self.tab._combo_timbre[0]
        combo.setCurrentIndex(combo.findData("woodblock"))
        self._pump(150)
        view = self.tab._current_view()
        f1 = [m for m in view.markers if m.mode == 0]
        self.assertTrue(f1)
        self.assertEqual(f1[0].timbre, "woodblock")

    def test_bells_can_be_turned_off(self):
        self.tab._on_analysis_done(self._dataset(), "test")
        self._pump(80)
        idx = self.tab._combo_view.findText("Spectrum")
        if idx < 0:
            self.skipTest("no spectrum view")
        self.tab._combo_view.setCurrentIndex(idx)
        self._pump(80)
        self.assertTrue(self.tab._render.strikes)
        self.tab._chk_bells.setChecked(False)
        self._pump(150)
        self.assertEqual(self.tab._render.strikes, [])

    # -------------------------------------------------------------- playhead
    def test_playhead_follows_the_plot_from_left_to_right(self):
        self.tab._on_analysis_done(self._dataset(), "test")
        self._pump(80)
        self.tab._spin_dur.setValue(4.0)
        self._pump(150)
        r = self.tab._render
        view = self.tab._current_view()
        x0, x1 = view.x_range()
        self.assertAlmostEqual(r.x_at(0.0), x0, places=4)
        self.assertAlmostEqual(r.x_at(r.duration_s), x1, places=3)
        # strictly advancing
        xs = [r.x_at(t) for t in np.linspace(0, r.duration_s, 12)]
        self.assertTrue(all(b >= a for a, b in zip(xs[:-1], xs[1:])))

    def test_play_runs_and_stop_is_idempotent(self):
        self.tab._on_analysis_done(self._dataset(), "test")
        self._pump(80)
        self.tab._spin_dur.setValue(3.0)
        self._pump(120)
        self.tab._on_play()
        self._pump(200)
        self.assertTrue(self.tab._playhead.isVisible() or self.tab._silent)
        self.tab._on_stop()
        self.tab._on_stop()                     # twice must be safe
        self.assertFalse(self.tab._player.is_playing)

    # -------------------------------------------------------------- lifecycle
    def test_shutdown_is_safe_twice_and_stops_everything(self):
        self.tab._on_analysis_done(self._dataset(), "test")
        self._pump(80)
        self.tab._on_play()
        self._pump(120)
        self.tab.shutdown()
        self.tab.shutdown()
        self.assertFalse(self.tab._player.is_playing)
        self.assertIsNone(self.tab._worker)

    def test_recording_without_a_stream_does_not_start_a_worker(self):
        self.ctrl._streaming = False
        try:
            self.tab._on_record()
        except Exception:
            pass                                 # a dialog may be suppressed
        self.assertIsNone(self.tab._worker)


@unittest.skipUnless(HAVE_QT, "PySide6 not importable")
class TestContainerWiring(unittest.TestCase):
    """The Sonification tab must treat both models alike."""

    def setUp(self):
        self.app = _ensure_app()
        if self.app is None:
            self.skipTest(_NO_WIDGETS)

    def test_both_models_are_present_and_uniform(self):
        from sensepi.gui.tabs.tab_sonification import SonificationTab

        son = SonificationTab(recorder_controller=None)
        try:
            names = [son._tabs.tabText(i) for i in range(son._tabs.count())]
            self.assertEqual(names, ["Bioacoustic Chorus", "Structure Pulse"])
            self.assertEqual(len(son.models), 2)
            for model in son.models:
                for hook in ("on_stream_started", "on_stream_stopped", "shutdown"):
                    self.assertTrue(callable(getattr(model, hook, None)),
                                    f"{type(model).__name__}.{hook}")
            son.apply_sensor_map(MAP)            # must reach both, no raise
            son.on_stream_started()
            son.on_stream_stopped()
        finally:
            son.shutdown()


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_QT, "PySide6 not importable")
class TestTraceAndChannelSelection(unittest.TestCase):
    """One curve at a time, on a channel the user picks."""

    def setUp(self):
        self.app = _ensure_app()
        if self.app is None:
            self.skipTest(_NO_WIDGETS)
        from sensepi.gui.tabs.tab_structure_pulse import StructurePulseTab
        from sensepi.sonification.structure_pulse import build_dataset_from_arrays

        self.ctrl = _FakeController()
        self.tab = StructurePulseTab(recorder_controller=self.ctrl)
        self.tab.apply_sensor_map(MAP)
        chans = {ax: self.ctrl._d[ax][:, -2000:] for ax in ("ax", "ay", "az", "gz")}
        ds = build_dataset_from_arrays(chans, fs=100.0, sensor_ids=[1, 2, 3, 4],
                                       mapping=MAP, source="unit")
        self.tab._on_analysis_done(ds, "test")
        self._pump(120)

    def tearDown(self):
        try:
            self.tab.shutdown()
        finally:
            self._pump(50)

    def _pump(self, ms):
        loop = QEventLoop()
        QTimer.singleShot(int(ms), loop.quit)
        loop.exec()

    def _select(self, name):
        i = self.tab._combo_view.findText(name)
        if i < 0:
            self.skipTest(f"no {name} view")
        self.tab._combo_view.setCurrentIndex(i)
        self._pump(120)

    def test_only_one_trace_sounds_by_default(self):
        self._select("Time history")
        self.assertGreater(len(self.tab._trace_checks), 1)
        on = [c for c in self.tab._trace_checks if c.isChecked()]
        self.assertEqual(len(on), 1)
        self.assertEqual(len(self.tab._current_view().curves), 1)

    def test_turning_traces_on_adds_them_to_the_sound(self):
        self._select("Time history")
        self.tab._set_all_traces(True)
        self._pump(150)
        n = len(self.tab._trace_checks)
        self.assertEqual(len(self.tab._current_view().curves), n)

    def test_never_leaves_nothing_selected(self):
        self._select("Time history")
        self.tab._set_all_traces(False)
        self._pump(120)
        self.assertGreaterEqual(
            len([c for c in self.tab._trace_checks if c.isChecked()]), 1)
        self.assertGreaterEqual(len(self.tab._current_view().curves), 1)
        # unticking the last one by hand must also be refused
        for c in self.tab._trace_checks:
            c.setChecked(False)
        self._pump(150)
        self.assertGreaterEqual(len(self.tab._current_view().curves), 1)

    def test_mode_shapes_can_be_soloed_one_at_a_time(self):
        self._select("Mode shapes")
        checks = self.tab._trace_checks
        self.assertGreaterEqual(len(checks), 2)
        for i in range(len(checks)):
            for j, c in enumerate(checks):
                c.blockSignals(True)
                c.setChecked(i == j)
                c.blockSignals(False)
            self.tab._on_traces_changed()
            self._pump(100)
            view = self.tab._current_view()
            self.assertEqual(len(view.curves), 1)
            self.assertEqual(view.curves[0].label, checks[i].text())

    def test_time_history_can_be_shown_on_another_channel(self):
        self._select("Time history")
        combo = self.tab._combo_channel
        self.assertGreater(combo.count(), 1)
        seen = {}
        for k in range(combo.count()):
            combo.setCurrentIndex(k)
            self._pump(120)
            view = self.tab._current_view()
            seen[str(combo.itemData(k))] = view.curves[0].y.copy()
            self.assertIn(str(combo.itemData(k)), view.name)
        axes = list(seen)
        self.assertFalse(np.array_equal(seen[axes[0]], seen[axes[1]]),
                         "swapping the channel must change the data")

    def test_a_gyro_channel_is_labelled_as_a_rotation_rate(self):
        self._select("Time history")
        combo = self.tab._combo_channel
        k = combo.findData("gz")
        if k < 0:
            self.skipTest("no gz in this dataset")
        combo.setCurrentIndex(k)
        self._pump(120)
        self.assertIn("deg/s", self.tab._current_view().y_label)

    def test_displacement_never_offers_a_gyro_channel(self):
        """Integrating a rotation rate twice is not a displacement."""
        self._select("Displacement")
        offered = [str(self.tab._combo_channel.itemData(k))
                   for k in range(self.tab._combo_channel.count())]
        self.assertTrue(offered)
        self.assertFalse([a for a in offered if a.startswith("g")], offered)

    def test_spectrum_and_response_offer_no_channel_choice(self):
        for name in ("Spectrum", "Response spectrum"):
            self._select(name)
            self.assertFalse(self.tab._combo_channel.isEnabled(), name)

    def test_response_spectrum_has_no_eigenfrequency_bells(self):
        """A response curve IS the sound; ringing f1/f2/f3 over it was wrong."""
        self._select("Response spectrum")
        self.assertEqual(self.tab._current_view().markers, [])
        self.assertEqual(self.tab._render.strikes, [])

    def test_spectrum_keeps_its_bells(self):
        self._select("Spectrum")
        self.assertTrue(self.tab._current_view().markers)
        self.assertTrue(self.tab._render.strikes)

    def test_changing_traces_rerenders(self):
        self._select("Time history")
        first = self.tab._render.audio.copy()
        self.tab._set_all_traces(True)
        self._pump(200)
        self.assertFalse(np.array_equal(first, self.tab._render.audio))


@unittest.skipUnless(HAVE_QT, "PySide6 not importable")
class TestAudioRecovery(unittest.TestCase):
    """A dead output stream must be reopened, not silently tolerated.

    The reported symptom: play once, change the data, play again — silence for
    the rest of the session until the app was restarted.
    """

    def setUp(self):
        self.app = _ensure_app()
        if self.app is None:
            self.skipTest(_NO_WIDGETS)
        from sensepi.gui.tabs.tab_structure_pulse import StructurePulseTab
        from sensepi.sonification.structure_pulse import build_dataset_from_arrays

        self.ctrl = _FakeController()
        self.tab = StructurePulseTab(recorder_controller=self.ctrl)
        self.tab.apply_sensor_map(MAP)
        chans = {ax: self.ctrl._d[ax][:, -1500:] for ax in ("ax", "gz")}
        self.tab._on_analysis_done(
            build_dataset_from_arrays(chans, fs=100.0, sensor_ids=[1, 2, 3, 4],
                                      mapping=MAP, source="unit"), "t")
        self._pump(120)

    def tearDown(self):
        try:
            self.tab.shutdown()
        finally:
            self._pump(50)

    def _pump(self, ms):
        loop = QEventLoop()
        QTimer.singleShot(int(ms), loop.quit)
        loop.exec()

    def test_ensure_audio_reopens_a_dead_stream(self):
        self.tab._spin_dur.setValue(3.0)
        self._pump(150)
        if not self.tab._ensure_audio():
            self.skipTest("no audio device in this environment")
        self.assertTrue(self.tab._audio.running)
        self.tab._audio.stop()                      # the stream dies
        self.assertFalse(self.tab._audio.running)
        self.assertTrue(self.tab._ensure_audio(), "must reopen, not give up")
        self.assertTrue(self.tab._audio.running)

    def test_play_after_a_dead_stream_still_makes_sound(self):
        self.tab._spin_dur.setValue(3.0)
        self._pump(150)
        if not self.tab._ensure_audio():
            self.skipTest("no audio device in this environment")
        self.tab._on_play()
        self._pump(400)
        self.assertGreater(self.tab._player.position_s, 0.0)
        self.tab._on_stop()
        self.tab._audio.stop()                      # device goes away
        # change the data, exactly as the user did
        i = self.tab._combo_view.findText("Spectrum")
        if i >= 0:
            self.tab._combo_view.setCurrentIndex(i)
            self._pump(150)
        self.tab._on_play()
        self._pump(500)
        self.assertGreater(self.tab._player.position_s, 0.0,
                           "silent after a dead stream — the reported bug")


@unittest.skipUnless(HAVE_QT, "PySide6 not importable")
class TestPlayheadOnLogAxis(unittest.TestCase):
    """The reported skip: on a period axis the line never showed."""

    def setUp(self):
        self.app = _ensure_app()
        if self.app is None:
            self.skipTest(_NO_WIDGETS)
        from sensepi.gui.tabs.tab_structure_pulse import StructurePulseTab
        from sensepi.sonification.structure_pulse import build_dataset_from_arrays

        self.ctrl = _FakeController()
        self.tab = StructurePulseTab(recorder_controller=self.ctrl)
        self.tab.apply_sensor_map(MAP)
        chans = {ax: self.ctrl._d[ax][:, -2500:] for ax in ("ax", "gz")}
        self.tab._on_analysis_done(
            build_dataset_from_arrays(chans, fs=100.0, sensor_ids=[1, 2, 3, 4],
                                      mapping=MAP, source="unit"), "t")
        self._pump(150)

    def tearDown(self):
        try:
            self.tab.shutdown()
        finally:
            self._pump(50)

    def _pump(self, ms):
        loop = QEventLoop()
        QTimer.singleShot(int(ms), loop.quit)
        loop.exec()

    def test_playhead_is_placed_in_plot_coordinates_on_a_log_axis(self):
        i = self.tab._combo_view.findText("Response spectrum")
        if i < 0:
            self.skipTest("no response spectrum")
        self.tab._combo_view.setCurrentIndex(i)
        self._pump(200)
        view = self.tab._current_view()
        self.assertTrue(view.x_log)
        x0, x1 = view.x_range()
        lo, hi = np.log10(x0), np.log10(x1)
        r = self.tab._render
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            pos = self.tab._plot_x(r.x_at(frac * r.duration_s))
            self.assertGreaterEqual(pos, lo - 1e-6)
            self.assertLessEqual(pos, hi + 1e-6)
            self.assertAlmostEqual((pos - lo) / (hi - lo), frac, places=3)

    def test_a_linear_view_is_unaffected(self):
        i = self.tab._combo_view.findText("Time history")
        if i < 0:
            self.skipTest("no time history")
        self.tab._combo_view.setCurrentIndex(i)
        self._pump(150)
        self.assertFalse(self.tab._current_view().x_log)
        self.assertAlmostEqual(self.tab._plot_x(12.5), 12.5)

    def test_smoothing_control_calms_a_wandering_contour(self):
        from sensepi.sonification.structure_pulse.render import _norm01, _shape

        i = self.tab._combo_view.findText("Time history")
        if i < 0:
            self.skipTest("no time history")
        self.tab._combo_view.setCurrentIndex(i)
        self._pump(150)
        c = self.tab._current_view().curves[0]
        y = np.interp(np.linspace(0, 1, 4000), np.linspace(0, 1, c.x.size), c.y)
        turns = {}
        for pct in (1.0, 15.0):
            n = _norm01(_shape(y, "auto", 4000, 20.0, pct / 100.0, "time"))
            d = np.diff(np.sign(np.diff(n[::20])))
            turns[pct] = int(np.count_nonzero(d))
        self.assertLess(turns[15.0], turns[1.0],
                        f"more smoothing must mean fewer reversals: {turns}")


@unittest.skipUnless(HAVE_QT, "PySide6 not importable")
class TestLiveMode(unittest.TestCase):
    """Record, analyse and play on a rolling loop."""

    def setUp(self):
        self.app = _ensure_app()
        if self.app is None:
            self.skipTest(_NO_WIDGETS)
        from sensepi.gui.tabs.tab_structure_pulse import StructurePulseTab

        self.ctrl = _FakeController()
        self.tab = StructurePulseTab(recorder_controller=self.ctrl)
        self.tab.apply_sensor_map(MAP)

    def tearDown(self):
        try:
            self.tab.shutdown()
        finally:
            self._pump(80)

    def _pump(self, ms):
        loop = QEventLoop()
        QTimer.singleShot(int(ms), loop.quit)
        loop.exec()

    def test_live_refuses_to_start_without_a_stream(self):
        self.ctrl._streaming = False
        self.tab._btn_live.setChecked(True)
        self._pump(120)
        self.assertIsNone(self.tab._live_thread)
        self.assertFalse(self.tab._btn_live.isChecked())

    def test_live_cycles_deliver_data_and_play(self):
        self.tab._spin_live.setValue(3)
        self.tab._spin_dur.setValue(3.0)
        self.tab._btn_live.setChecked(True)
        for _ in range(20):
            self._pump(400)
            if self.tab._live_cycle >= 1:
                break
        self.assertGreaterEqual(self.tab._live_cycle, 1, "no live cycle arrived")
        self.assertIsNotNone(self.tab._dataset)
        self.assertGreater(self.tab._combo_view.count(), 0)
        self.assertIsNotNone(self.tab._render)
        self.assertGreater(self.tab._render.audio.size, 0)
        self.tab._btn_live.setChecked(False)
        self._pump(400)
        self.assertIsNone(self.tab._live_thread)

    def test_stopping_the_stream_ends_live_mode(self):
        self.tab._spin_live.setValue(3)
        self.tab._btn_live.setChecked(True)
        self._pump(300)
        self.assertIsNotNone(self.tab._live_thread)
        self.tab.on_stream_stopped()
        self._pump(300)
        self.assertIsNone(self.tab._live_thread)
        self.assertFalse(self.tab._btn_live.isChecked())

    def test_shutdown_stops_a_running_live_loop(self):
        self.tab._spin_live.setValue(3)
        self.tab._btn_live.setChecked(True)
        self._pump(300)
        self.tab.shutdown()
        self.assertIsNone(self.tab._live_thread)
        self.assertFalse(self.tab._player.is_playing)

    def test_changing_the_view_retargets_the_next_cycle(self):
        self.tab._spin_live.setValue(3)
        self.tab._btn_live.setChecked(True)
        for _ in range(20):
            self._pump(400)
            if self.tab._live_cycle >= 1:
                break
        if self.tab._live_cycle < 1:
            self.skipTest("no cycle arrived in time")
        i = self.tab._combo_view.findText("Spectrum")
        if i >= 0:
            self.tab._combo_view.setCurrentIndex(i)
            self._pump(150)
            self.assertEqual(self.tab._live_worker._view_name, "Spectrum")
        self.tab._btn_live.setChecked(False)
        self._pump(300)

    def test_view_list_is_offered_before_the_first_cycle(self):
        """Waiting a whole window for an empty combo made live unusable."""
        self.tab._spin_live.setValue(3)
        self.tab._btn_live.setChecked(True)
        self._pump(150)
        self.assertGreater(self.tab._combo_view.count(), 0)
        self.assertTrue(self.tab._combo_view.isEnabled())
        self.tab._btn_live.setChecked(False)
        self._pump(300)

    def test_selecting_a_view_before_any_data_retargets_the_worker(self):
        self.tab._spin_live.setValue(3)
        self.tab._btn_live.setChecked(True)
        self._pump(150)
        i = self.tab._combo_view.findText("Spectrum")
        self.assertGreaterEqual(i, 0)
        self.tab._combo_view.setCurrentIndex(i)
        self._pump(150)
        self.assertEqual(self.tab._live_worker._view_name, "Spectrum")
        self.tab._btn_live.setChecked(False)
        self._pump(300)

    def test_the_view_list_does_not_reshuffle_between_cycles(self):
        self.tab._spin_live.setValue(3)
        self.tab._spin_dur.setValue(3.0)
        self.tab._btn_live.setChecked(True)
        self._pump(150)
        before = [self.tab._combo_view.itemData(i)
                  for i in range(self.tab._combo_view.count())]
        i = self.tab._combo_view.findText("Spectrum")
        self.tab._combo_view.setCurrentIndex(i)
        for _ in range(16):
            self._pump(400)
            if self.tab._live_cycle >= 1:
                break
        if self.tab._live_cycle < 1:
            self.tab._btn_live.setChecked(False)
            self._pump(300)
            self.skipTest("no cycle arrived in time")
        after = [self.tab._combo_view.itemData(i)
                 for i in range(self.tab._combo_view.count())]
        self.assertEqual(before, after, "the combo must not reshuffle")
        self.assertEqual(self.tab._combo_view.currentData(), "Spectrum",
                         "the user's choice must survive a cycle")
        self.tab._btn_live.setChecked(False)
        self._pump(300)

    def test_changing_the_view_while_live_changes_the_sound_at_once(self):
        self.tab._spin_live.setValue(3)
        self.tab._spin_dur.setValue(3.0)
        self.tab._btn_live.setChecked(True)
        for _ in range(16):
            self._pump(400)
            if self.tab._live_cycle >= 1:
                break
        if self.tab._live_cycle < 1:
            self.tab._btn_live.setChecked(False)
            self._pump(300)
            self.skipTest("no cycle arrived in time")
        got = {}
        for name in ("Spectrum", "Time history"):
            i = self.tab._combo_view.findText(name)
            if i < 0:
                continue
            self.tab._combo_view.setCurrentIndex(i)
            self._pump(300)
            self.assertEqual(self.tab._live_worker._view_name, name)
            self.assertIsNotNone(self.tab._render, name)
            got[name] = self.tab._render.audio.copy()
        if len(got) == 2:
            a, b = got.values()
            n = min(len(a), len(b))
            self.assertFalse(np.array_equal(a[:n], b[:n]),
                             "the sound must follow the view immediately")
        self.tab._btn_live.setChecked(False)
        self._pump(300)

    def test_a_view_this_window_cannot_make_is_reported_not_substituted(self):
        """A 3 s window has no mode shapes; silently playing something else
        would make the plot and the label disagree."""
        self.tab._spin_live.setValue(3)
        self.tab._spin_dur.setValue(3.0)
        self.tab._btn_live.setChecked(True)
        for _ in range(16):
            self._pump(400)
            if self.tab._live_cycle >= 1:
                break
        if self.tab._live_cycle < 1:
            self.tab._btn_live.setChecked(False)
            self._pump(300)
            self.skipTest("no cycle arrived in time")
        if "Mode shapes" in (self.tab._dataset.views or {}):
            self.tab._btn_live.setChecked(False)
            self._pump(300)
            self.skipTest("this window did yield mode shapes")
        i = self.tab._combo_view.findText("Mode shapes")
        self.assertGreaterEqual(i, 0, "the list should still offer it")
        self.tab._combo_view.setCurrentIndex(i)
        self._pump(300)
        self.assertIsNone(self.tab._base_view())
        self.assertIsNone(self.tab._render)
        self.assertFalse(self.tab._btn_play.isEnabled())
        self._pump(700)
        # the label elides to a fixed width; the full text is the tooltip
        self.assertIn("not in this window", self.tab._status.toolTip())
        self.tab._btn_live.setChecked(False)
        self._pump(300)

    def test_stopping_live_hands_the_last_window_back(self):
        self.tab._spin_live.setValue(3)
        self.tab._spin_dur.setValue(3.0)
        self.tab._btn_live.setChecked(True)
        for _ in range(16):
            self._pump(400)
            if self.tab._live_cycle >= 1:
                break
        if self.tab._live_cycle < 1:
            self.tab._btn_live.setChecked(False)
            self._pump(300)
            self.skipTest("no cycle arrived in time")
        self.tab._btn_live.setChecked(False)
        self._pump(500)
        self.assertIsNone(self.tab._live_thread)
        self.assertIsNotNone(self.tab._dataset, "the window must survive")
        self.assertGreater(self.tab._combo_view.count(), 0)
        self.assertTrue(self.tab._btn_play.isEnabled(),
                        "it should still be playable after stopping")

    def _to_cycle(self, n=1, tries=18):
        for _ in range(tries):
            self._pump(400)
            if self.tab._live_cycle >= n:
                return True
        return False

    def test_play_during_live_does_not_orphan_the_worker_thread(self):
        """Regression: a stray state reset in the play handler set
        ``_live_thread`` to None while the thread kept running, so it was never
        joined and the process aborted on exit with
        'QThread: Destroyed while thread is still running'."""
        self.tab._spin_live.setValue(3)
        self.tab._spin_dur.setValue(3.0)
        self.tab._btn_live.setChecked(True)
        if not self._to_cycle():
            self.tab._btn_live.setChecked(False)
            self._pump(300)
            self.skipTest("no cycle arrived in time")
        thread = self.tab._live_thread
        self.assertIsNotNone(thread)
        self.tab._on_play()
        self._pump(200)
        self.assertIs(self.tab._live_thread, thread,
                      "pressing Play must not drop the live thread")
        self.assertIsNotNone(self.tab._live_worker)
        self.tab._btn_live.setChecked(False)
        self._pump(400)
        self.assertIsNone(self.tab._live_thread)

    def test_stop_during_live_ends_the_loop(self):
        """Otherwise Stop looks broken: the next cycle restarts the sound."""
        self.tab._spin_live.setValue(3)
        self.tab._spin_dur.setValue(3.0)
        self.tab._btn_live.setChecked(True)
        if not self._to_cycle():
            self.tab._btn_live.setChecked(False)
            self._pump(300)
            self.skipTest("no cycle arrived in time")
        self.tab._on_stop()
        self._pump(300)
        self.assertIsNone(self.tab._live_thread)
        self.assertFalse(self.tab._btn_live.isChecked())
        self._pump(1500)                       # a cycle would have landed by now
        self.assertFalse(self.tab._player.is_playing,
                         "Stop must not be undone by the next cycle")

    def test_the_two_sources_cannot_run_at_once(self):
        self.tab._spin_live.setValue(3)
        self.tab._btn_live.setChecked(True)
        self._pump(300)
        for name in ("_btn_open", "_btn_browse", "_btn_record", "_combo_session"):
            w = getattr(self.tab, name)
            self.assertFalse(w.isEnabled(), f"{name} must be parked during live")
        self.tab._btn_live.setChecked(False)
        self._pump(400)
        for name in ("_btn_open", "_btn_browse", "_btn_record", "_combo_session"):
            self.assertTrue(getattr(self.tab, name).isEnabled(),
                            f"{name} must come back after live")

    def test_shutdown_is_clean_at_any_moment_after_starting(self):
        for delay in (0, 60, 250):
            tab = type(self.tab)(recorder_controller=_FakeController())
            tab.apply_sensor_map(MAP)
            tab._spin_live.setValue(3)
            tab._btn_live.setChecked(True)
            self._pump(delay)
            tab.shutdown()
            self.assertIsNone(tab._live_thread, f"delay {delay}")
            self.assertIsNone(tab._live_worker, f"delay {delay}")

    def test_hammering_the_controls_during_live_never_raises(self):
        self.tab._spin_live.setValue(3)
        self.tab._spin_dur.setValue(3.0)
        self.tab._btn_live.setChecked(True)
        if not self._to_cycle():
            self.tab._btn_live.setChecked(False)
            self._pump(300)
            self.skipTest("no cycle arrived in time")
        for _ in range(3):
            for name in ("Spectrum", "Time history", "Response spectrum"):
                i = self.tab._combo_view.findText(name)
                if i >= 0:
                    self.tab._combo_view.setCurrentIndex(i)
                    self._pump(60)
            self.tab._set_all_traces(True)
            self._pump(60)
            self.tab._set_all_traces(False)
            self._pump(60)
            self.tab._spin_smooth.setValue(8.0)
            self._pump(60)
        self.assertIsNotNone(self.tab._live_thread, "the loop must survive")
        self.tab._btn_live.setChecked(False)
        self._pump(400)

    def test_live_then_recording_playback_still_works(self):
        """Live is additive: opening a recording afterwards must behave."""
        from sensepi.sonification.structure_pulse import build_dataset_from_arrays

        self.tab._spin_live.setValue(3)
        self.tab._btn_live.setChecked(True)
        self._pump(400)
        self.tab._btn_live.setChecked(False)
        self._pump(400)
        self.assertIsNone(self.tab._live_thread)
        chans = {ax: self.ctrl._d[ax][:, -2500:] for ax in ("ax", "gz")}
        self.tab._on_analysis_done(
            build_dataset_from_arrays(chans, fs=100.0, sensor_ids=[1, 2, 3, 4],
                                      mapping=MAP, source="unit"), "opened")
        self._pump(200)
        self.assertIsNotNone(self.tab._render)
        self.assertGreater(self.tab._render.audio.size, 0)
        self.assertTrue(self.tab._btn_play.isEnabled())

    def test_the_plot_scale_does_not_jump_between_cycles(self):
        """Autoranging every cycle made the view leap each time the structure
        got louder — four different scales in four cycles, unreadable."""
        self.tab.resize(1200, 700)
        self.tab.show()
        self._pump(120)
        self.tab._spin_live.setValue(3)
        self.tab._spin_dur.setValue(3.0)
        self.tab._btn_live.setChecked(True)
        seen = []
        for n in (1, 2, 3):
            if not self._to_cycle(n):
                break
            self._pump(120)
            seen.append(tuple(round(v, 6)
                              for v in self.tab._plot.getViewBox().viewRange()[1]))
        self.tab._btn_live.setChecked(False)
        self._pump(300)
        if len(seen) < 2:
            self.skipTest("not enough cycles arrived")
        for a, b in zip(seen[:-1], seen[1:]):
            self.assertLessEqual(b[0], a[0] + 1e-9, "the range must never shrink")
            self.assertGreaterEqual(b[1], a[1] - 1e-9, "the range must never shrink")

    def test_changing_the_view_starts_a_fresh_scale(self):
        from sensepi.sonification.structure_pulse import build_dataset_from_arrays

        chans = {ax: self.ctrl._d[ax][:, -2500:] for ax in ("ax", "gz")}
        self.tab._on_analysis_done(
            build_dataset_from_arrays(chans, fs=100.0, sensor_ids=[1, 2, 3, 4],
                                      mapping=MAP, source="unit"), "t")
        self._pump(150)
        self.assertIsNotNone(self.tab._range_key)
        first = self.tab._range_key
        i = self.tab._combo_view.findText("Spectrum")
        if i < 0:
            self.skipTest("no spectrum view")
        self.tab._combo_view.setCurrentIndex(i)
        self._pump(200)
        self.assertNotEqual(self.tab._range_key, first,
                            "a different view must get its own scale")

    def test_the_status_line_cannot_resize_the_layout(self):
        self.tab.resize(1200, 700)
        self.tab.show()
        self._pump(120)
        width = self.tab._status.width()
        self.tab._set_status("x" * 400)
        self._pump(60)
        self.assertEqual(self.tab._status.width(), width,
                         "a long status must not widen the bar")
        self.assertLessEqual(len(self.tab._status.text()), 400)
        self.assertIn("xxx", self.tab._status.toolTip())

    def test_a_sensor_map_change_reaches_a_running_loop(self):
        self.tab.apply_sensor_map(MAP)
        self.tab._spin_live.setValue(3)
        self.tab._btn_live.setChecked(True)
        self._pump(300)
        self.assertIsNotNone(self.tab._live_worker)
        other = {"n_floors": 5, "axis": "y",
                 "placements": [{"sensor_id": 1, "floor": 0, "cell": "B2"},
                                {"sensor_id": 2, "floor": 3, "cell": "B2"}]}
        self.tab.apply_sensor_map(other)
        self._pump(100)
        self.assertEqual(self.tab._live_worker._mapping["n_floors"], 5)
        self.tab.apply_sensor_map(None)
        self._pump(100)
        self.assertIsNone(self.tab._live_worker._mapping,
                          "clearing the map must reach the loop too")
        self.tab._btn_live.setChecked(False)
        self._pump(300)

    def test_live_survives_degenerate_windows(self):
        """Zeros, NaN and a constant must not take the loop down."""
        from PySide6.QtCore import QObject, Signal

        class _Bad(QObject):
            stream_started = Signal()
            stream_stopped = Signal()
            stream_rate_updated = Signal(str, float)

            class _B:
                _AXIS_COLUMNS = {"ax": 1, "gz": 6}

            def __init__(self, mode):
                super().__init__()
                self._modal_buffer = self._B()
                self.mode = mode

            def is_streaming(self):
                return True

            def require_modal_window_seconds(self, s):
                pass

            def snapshot_modal_capture(self, *, axis="ax", last_seconds=None,
                                       target_fs=None):
                n = int((last_seconds or 6) * 100)

                class S:
                    def __init__(s2, d, fs, ids):
                        s2.data, s2.fs, s2.sensor_ids = d, fs, ids

                if self.mode == "zeros":
                    return S(np.zeros((4, n)), 100.0, [1, 2, 3, 4])
                if self.mode == "nan":
                    return S(np.full((4, n), np.nan), 100.0, [1, 2, 3, 4])
                return S(np.ones((4, n)) * 3.3, 100.0, [1, 2, 3, 4])

        for mode in ("zeros", "nan", "const"):
            tab = type(self.tab)(recorder_controller=_Bad(mode))
            tab.apply_sensor_map(MAP)
            tab._spin_live.setValue(3)
            tab._btn_live.setChecked(True)
            for _ in range(6):
                self._pump(400)
            tab.shutdown()
            self.assertIsNone(tab._live_thread, mode)

    def test_live_identifies_eigenfrequencies_with_the_default_window(self):
        """Reported: the Spectrum tab showed frequencies, Structure Pulse did
        not. Identification refuses <= 10 s, and a '10 s' request yields 9.99 s
        of samples, so the default live window failed by a hair and the
        spectrum came back with no f1/f2/f3 at all."""
        self.tab._spin_live.setValue(10)          # the default
        self.tab._spin_dur.setValue(5.0)
        self.tab._btn_live.setChecked(True)
        got = False
        for _ in range(40):
            self._pump(400)
            if self.tab._live_cycle >= 1:
                got = True
                break
        if not got:
            self.tab._btn_live.setChecked(False)
            self._pump(300)
            self.skipTest("no cycle arrived in time")
        self.assertTrue(self.tab._dataset.modal.ok,
                        self.tab._dataset.modal.message)
        self.assertGreaterEqual(
            np.asarray(self.tab._dataset.modal.frequencies_hz).size, 1)
        i = self.tab._combo_view.findText("Spectrum")
        if i >= 0:
            self.tab._combo_view.setCurrentIndex(i)
            self._pump(300)
            self.assertTrue(self.tab._current_view().markers,
                            "the spectrum must carry f1/f2/f3")
            self.assertTrue(self.tab._render.strikes, "and ring them")
        self.assertIn("f =", self.tab._info.text())
        self.tab._btn_live.setChecked(False)
        self._pump(400)

    def test_the_cycle_rate_is_independent_of_the_analysed_window(self):
        from sensepi.gui.tabs._pulse_live import MIN_ANALYSIS_S

        self.assertGreater(MIN_ANALYSIS_S, 10.0,
                           "identification refuses 10 s and under")

    def test_the_top_bar_fits_in_a_small_window(self):
        """The status line used to demand a fixed 420 px it could never give
        back, pushing the bar past the window and the text off the edge."""
        self.tab.show()
        self._pump(120)
        bar = self.tab._status.parent()
        self.assertLess(bar.minimumSizeHint().width(), 900,
                        "the bar must not demand more than a modest window")
        self.tab._set_status("live · cycle 9 capturing 71% · playing · " + "x" * 160)
        for width in (1400, 1000, 760):
            self.tab.resize(width, 700)
            self._pump(80)
            self.assertLessEqual(self.tab._status.width(), self.tab.width())
        self.assertIn("xxx", self.tab._status.toolTip())

    def test_views_are_selectable_before_anything_is_loaded(self):
        """The combo used to be empty until a recording was opened or a live
        cycle had landed, so Spectrum could not be chosen before starting."""
        fresh = type(self.tab)(recorder_controller=_FakeController())
        try:
            self.assertGreater(fresh._combo_view.count(), 0)
            self.assertTrue(fresh._combo_view.isEnabled())
            i = fresh._combo_view.findText("Spectrum")
            self.assertGreaterEqual(i, 0)
            fresh._combo_view.setCurrentIndex(i)
            self._pump(80)
            self.assertEqual(fresh._combo_view.currentData(), "Spectrum")
        finally:
            fresh.shutdown()

    def test_a_view_chosen_before_live_is_what_the_loop_renders(self):
        i = self.tab._combo_view.findText("Spectrum")
        self.assertGreaterEqual(i, 0)
        self.tab._combo_view.setCurrentIndex(i)
        self._pump(80)
        self.tab._spin_live.setValue(10)
        self.tab._btn_live.setChecked(True)
        self._pump(200)
        self.assertEqual(self.tab._live_worker._view_name, "Spectrum")
        self.tab._btn_live.setChecked(False)
        self._pump(300)

    def test_stopping_live_never_blocks_the_gui_thread(self):
        """Regression: Stop waited up to four seconds for the worker on the GUI
        thread. That IS a freeze — the press looked dead and the window stopped
        repainting, so the only way out was quitting the app."""
        import time

        self.tab._spin_live.setValue(10)
        worst = 0.0
        for _ in range(3):
            self.tab._btn_live.setChecked(True)
            self._pump(250)                    # stop mid-capture, the bad case
            self.assertIsNotNone(self.tab._live_thread)
            t0 = time.monotonic()
            self.tab._btn_live.setChecked(False)
            worst = max(worst, time.monotonic() - t0)
            self._pump(200)
            self.assertIsNone(self.tab._live_thread)
        self.assertLess(worst, 1.0,
                        f"stopping blocked the GUI for {worst:.2f}s")

    def test_stopping_detaches_so_a_late_cycle_cannot_restart_playback(self):
        self.tab._spin_live.setValue(10)
        self.tab._spin_dur.setValue(4.0)
        self.tab._btn_live.setChecked(True)
        self._pump(300)
        worker = self.tab._live_worker
        self.tab._btn_live.setChecked(False)
        self._pump(200)
        self.assertIsNone(self.tab._live_worker)
        # a cycle still in flight must land nowhere
        try:
            worker.ready.emit(None, None, 99)
        except RuntimeError:
            pass                                # already torn down: also fine
        self._pump(200)
        self.assertFalse(self.tab._player.is_playing)
