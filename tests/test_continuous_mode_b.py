"""§9 headless Mode-B smoke: drive the redesigned two-stage continuous loop with a
synthetic noisy capture (a cluster + an injected outlier + a persistent step) and
assert the acceptance behaviours:

  * the loop NEVER hard-stops on noisy data (no ``error`` signal; every cycle runs);
  * sigma shrinks while readings cluster and widens on an outlier;
  * a lone outlier barely moves the consolidated estimate;
  * a sustained step migrates the estimate within a few cycles.

The worker's ``run()`` is called synchronously (no QThread) with a scripted
``_build_sensor_exp_dict`` and a trivial patched forward model, so the test is
deterministic and fast. It still exercises the REAL Stage-1 tracker + Stage-2
``run_bayesian_calibration`` wiring. Needs openseespy (the worker imports the
calibrator); skipped elsewhere.
"""

import os
import types
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import openseespy.opensees  # noqa: F401
    from PySide6.QtCore import QCoreApplication
    from sensepi.gui.tabs import tab_model_updating as mu
    _HAS = True
except Exception:  # pragma: no cover - env-dependent
    _HAS = False


_F0 = np.array([2.0, 6.0, 8.3])   # seeded structural modes


def _fake_session():
    # The worker only checks .success and .duration_s before _build_sensor_exp_dict
    # (which we patch), so a light namespace is enough.
    return types.SimpleNamespace(success=True, duration_s=30.0, message="ok")


def _scripted_frequencies():
    """Per-cycle raw peaks: 8 clustered, 1 outlier on mode 3, then a step on mode 1."""
    rng = np.random.default_rng(1)
    seq = []
    for _ in range(8):                       # cycles 1-8: tight cluster
        seq.append(list(_F0 + rng.normal(0, 0.01, size=3)))
    # cycle 9: mode-3 in-gate outlier (8.9, down-weighted -> widens sigma, barely
    # moves) PLUS a far 15 Hz spurious peak (unassociated -> logged, not folded).
    seq.append([2.0, 6.0, 8.9, 15.0])
    for _ in range(20):                      # cycles 10-29: mode 1 steps 2.0 -> 2.6
        seq.append(list(np.array([2.6, 6.0, 8.3]) + rng.normal(0, 0.01, size=3)))
    return seq


@unittest.skipUnless(_HAS, "needs openseespy + PySide6 (canonical env: senspi_mpu)")
class TestContinuousModeBSmoke(unittest.TestCase):
    def setUp(self):
        self.app = QCoreApplication.instance() or QCoreApplication([])
        from opensees_model_updating.calibration import calibrator
        self.calibrator = calibrator
        self._orig_apply = calibrator.apply_calibration_vector
        self._orig_extract = calibrator.extract_modal_results
        # Trivial forward model: frequencies scale with E only (g(theta)).
        calibrator.apply_calibration_vector = lambda bp, x: {
            "E": float(bp["E"]) * float(x[0]),
            "floor_masses": [float(bp["floor_masses"][i]) * float(x[i + 1])
                             for i in range(int(bp["nStory"]))],
            "_x": np.asarray(x, dtype=float),
        }
        calibrator.extract_modal_results = lambda p, **k: {
            "freqs": list(_F0 * float(p["_x"][0])),
            "mode_shapes_ux_master": [],
        }
        # Scripted identification: feed one cycle's peaks per call.
        self._seq = _scripted_frequencies()
        self._idx = 0
        self._orig_build = mu._build_sensor_exp_dict
        mu._build_sensor_exp_dict = self._fake_build

    def tearDown(self):
        self.calibrator.apply_calibration_vector = self._orig_apply
        self.calibrator.extract_modal_results = self._orig_extract
        mu._build_sensor_exp_dict = self._orig_build

    def _fake_build(self, session, params):
        freqs = self._seq[min(self._idx, len(self._seq) - 1)]
        self._idx += 1
        # fdd_freqs/fdd_spectrum/method back the per-iteration FDD-spectrum figure.
        fdd = types.SimpleNamespace(
            success=True, message="ok", frequencies_hz=[float(f) for f in freqs],
            method="fdd", fdd_freqs=np.linspace(0.5, 12.0, 64),
            fdd_spectrum=np.ones(64))
        story = types.SimpleNamespace(
            coverage_stories=[], mode_shapes_available=False, mode_shapes_ux={})
        return {}, fdd, story

    def _params(self):
        n = 3
        return {
            "nStory": n, "nCalibModes": 3, "E": 1.0,
            "floor_masses": [1.0] * n,
            "E_scale_lb": 0.5, "E_scale_ub": 1.5,
            "m_scale_lb": 0.5, "m_scale_ub": 1.5,
            "w_freq": 1.0, "w_mode": 0.35, "max_nfev": 60,
            "sensor_axis": "ax", "sensor_n_modes": 3, "use_mode_shapes": False,
            "sensor_f_min": 0.5, "sensor_f_max": 12.0, "sensor_target_fs": None,
        }

    def _run_worker(self, n_cycles):
        settings = {"duration_s": 10.0, "interval_s": 10.0, "max_cycles": n_cycles,
                    "calibration_method": "bayesian"}
        worker = mu._ContinuousUpdateWorker(self._params(), settings,
                                            lambda **kw: _fake_session())
        errors, logs = [], []
        worker.error.connect(errors.append)
        worker.log.connect(logs.append)
        worker.result.connect(lambda payload: None)
        worker.run()   # synchronous (no QThread); interval==duration -> no real sleep
        return worker, errors, "".join(logs)

    def test_loop_runs_and_tracks(self):
        n = 29
        worker, errors, log = self._run_worker(n)

        # 1) The loop NEVER hard-stopped on noisy data.
        self.assertEqual(errors, [], f"loop hard-stopped: {errors}")
        self.assertEqual(len(worker._history), n, "every cycle should complete a fit")

        hist = worker._history
        f3 = [h["f_hat"][2] for h in hist]      # mode-3 consolidated frequency
        s3 = [h["f_sigma"][2] for h in hist]    # mode-3 sigma
        f1 = [h["f_hat"][0] for h in hist]      # mode-1 consolidated frequency

        # 2) sigma widened at the outlier cycle (index 8) and shrank again afterwards.
        self.assertGreater(s3[8], s3[7], "outlier should widen sigma")
        self.assertLess(s3[-1], s3[8], "clustering after the outlier should shrink sigma")

        # 3) The 8.9 outlier was down-weighted -> mode-3 barely moved, stays near 8.3.
        self.assertAlmostEqual(f3[8], f3[7], delta=0.1)
        self.assertLess(abs(f3[8] - 8.3), 0.2)  # still locked near the real 8.3 Hz mode

        # 4) The sustained step on mode 1 (2.0 -> 2.6) migrated within a few cycles.
        self.assertAlmostEqual(f1[-1], 2.6, delta=0.1)
        self.assertGreater(f1[-1], f1[8] + 0.3)

        # 5) Transparent logging surfaced both the down-weighted outlier (folded, not
        #    dropped) and the far spurious peak (unassociated) — nothing silently lost.
        self.assertIn("OUTLIER", log)
        self.assertIn("unassociated", log)
        self.assertIn("consolidated", log)

    def test_least_squares_method_also_runs(self):
        # The deterministic Stage-2 path stays selectable and must not hard-stop.
        settings = {"duration_s": 10.0, "interval_s": 10.0, "max_cycles": 5,
                    "calibration_method": "least_squares"}
        worker = mu._ContinuousUpdateWorker(self._params(), settings,
                                            lambda **kw: _fake_session())
        errors = []
        worker.error.connect(errors.append)
        worker.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(worker._history), 5)


if __name__ == "__main__":
    unittest.main()
