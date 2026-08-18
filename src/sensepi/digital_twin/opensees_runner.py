"""Dedicated OpenSees runner for the Digital Twin Experiment tab.

This module is intentionally isolated from the existing Model Updating transient
analysis.  It runs the calibrated model in real wall-clock time and streams only
small display chunks to the GUI so the 3D animation and time-history plots do
not slow down the numerical analysis.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def load_ground_motion(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load the prescribed shaker/OpenSees acceleration in analysis units."""
    gm_file = Path(str(params["gmFile"]))
    if not gm_file.exists():
        raise FileNotFoundError(f"Ground-motion file not found: {gm_file}")

    dt = float(params["dtGM"])
    factor = float(params["gmFactor"])
    values = np.loadtxt(gm_file, dtype=float, ndmin=1).reshape(-1)
    if values.size < 2:
        raise ValueError("Ground motion file must contain at least 2 acceleration points.")
    times = np.arange(values.size, dtype=float) * dt
    return times, values * factor


def _ground_accel_at_time(
    ground_t: np.ndarray,
    ground_a: np.ndarray,
    t: float,
) -> float:
    return float(
        np.interp(
            float(t),
            ground_t,
            ground_a,
            left=float(ground_a[0]),
            right=float(ground_a[-1]),
        )
    )


def run_digital_twin_stream(
    params: dict[str, Any],
    modal_data: dict[str, Any],
    *,
    frame_callback: Callable[[dict[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    target_fps: float = 15.0,
    realtime: bool = True,
    realtime_offset_s: float = 0.0,
    sfac_anim: float = 20.0,
    show_info: bool = False,
) -> dict[str, Any]:
    """Run the calibrated model in real time for the Digital Twin tab only.

    Numerical integration still uses every ``dtGM`` step.  The GUI is updated
    at approximately ``target_fps`` and receives only the new samples since the
    previous update.  This separation is important: plotting/3D rendering must
    never dictate the OpenSees integration step.

    OpenSees ``UniformExcitation`` returns relative nodal response.  The runner
    therefore stores both relative acceleration and an absolute floor
    acceleration formed as ``a_abs = a_rel + a_ground`` for comparison with the
    floor-mounted MPU accelerometers.
    """
    import openseespy.opensees as ops

    from opensees_model_updating.analysis.transient import (
        set_rayleigh_damping_from_modal,
        setup_dynamic_excitation,
    )

    ctx = modal_data["ctx"]
    master_nodes = list(ctx["master_nodes"])
    if not master_nodes:
        raise ValueError("The OpenSees model has no story master nodes.")

    dt = float(params["dtGM"])
    if dt <= 0.0:
        raise ValueError("dtGM must be greater than zero.")

    ground_t, ground_a = load_ground_motion(params)
    n_steps = int(ground_a.size - 1)
    tmax = n_steps * dt

    # Decouple display refresh from numerical integration.  Examples:
    # dt=0.01 s, 15 fps -> one GUI frame about every 7 OpenSees steps.
    # dt=0.001 s, 15 fps -> one GUI frame about every 67 OpenSees steps.
    target_fps = max(2.0, min(float(target_fps), 30.0))
    emit_every = max(1, int(round((1.0 / target_fps) / dt)))

    alpha_m, beta_k_init = set_rayleigh_damping_from_modal(
        modal_data,
        float(params["zeta"]),
        int(params["numModes"]),
    )
    if show_info:
        print("Digital Twin Rayleigh damping:")
        print("alphaM    =", alpha_m)
        print("betaKinit =", beta_k_init)
        print(f"Digital Twin GUI target = {target_fps:.1f} fps; emit every {emit_every} integration step(s)")

    setup_dynamic_excitation(params)

    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-10, 100, 0)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    vis_elems = list(ctx.get("vis_elems", []))
    vis_nodes = list(ctx.get("vis_nodes", []))
    node_xyz = dict(ctx.get("node_xyz", {}))

    t_hist: list[float] = []
    floor_u_hist: list[list[float]] = []
    floor_a_rel_hist: list[list[float]] = []
    floor_a_abs_hist: list[list[float]] = []
    ground_accel_hist: list[float] = []
    ok = 0
    stopped = False
    last_emitted_index = 0

    def deformed_segments() -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        # Compute each node displacement once per display frame rather than once
        # for every element endpoint.  This noticeably reduces OpenSees/Python
        # calls for the 3D animation.
        deformed: dict[int, tuple[float, float, float]] = {}
        for node in vis_nodes:
            xyz = node_xyz.get(node)
            if xyz is None:
                xyz = ops.nodeCoord(node)
            x, y, z = xyz
            deformed[node] = (
                float(x + sfac_anim * ops.nodeDisp(node, 1)),
                float(y + sfac_anim * ops.nodeDisp(node, 2)),
                float(z + sfac_anim * ops.nodeDisp(node, 3)),
            )
        return [(deformed[n1], deformed[n2]) for n1, n2 in vis_elems if n1 in deformed and n2 in deformed]

    def emit(kind: str) -> None:
        nonlocal last_emitted_index
        if frame_callback is None:
            return

        end = len(t_hist)
        start = last_emitted_index
        if end > start:
            t_chunk = np.asarray(t_hist[start:end], dtype=float)
            a_chunk = np.asarray(floor_a_abs_hist[start:end], dtype=float)
            g_chunk = np.asarray(ground_accel_hist[start:end], dtype=float)
        else:
            t_chunk = np.array([], dtype=float)
            a_chunk = np.empty((0, len(master_nodes)), dtype=float)
            g_chunk = np.array([], dtype=float)
        last_emitted_index = end

        wall_lag = 0.0
        if t_hist:
            wall_lag = max(0.0, time.perf_counter() - (wall_start + t_hist[-1]))

        frame_callback(
            {
                "kind": kind,
                "time": float(ops.getTime()),
                "tmax": float(tmax),
                "t_chunk": t_chunk,
                "floor_a_abs_chunk": a_chunk,
                "ground_accel_chunk": g_chunk,
                "deformed_segments": deformed_segments(),
                "realtime_lag_s": float(wall_lag),
                "ok": int(ok),
                "stopped": bool(stopped),
            }
        )

    # When the GUI starts from a detected physical shaker onset, detection itself
    # takes a short sustained-motion window.  Shifting the real-time origin back
    # by that measured delay lets OpenSees catch up immediately, then continue on
    # the same wall clock as the physical experiment.
    realtime_offset_s = max(0.0, float(realtime_offset_s))
    wall_start = time.perf_counter() - realtime_offset_s
    emit("frame")

    for i in range(n_steps):
        if stop_requested is not None and bool(stop_requested()):
            stopped = True
            break

        ok = ops.analyze(1, dt)
        if ok != 0:
            print(f"Digital Twin transient analysis failed at step {i + 1}, time = {ops.getTime()}")
            break

        t = float(ops.getTime())
        ag = _ground_accel_at_time(ground_t, ground_a, t)
        floor_u = [float(ops.nodeDisp(node, 1)) for node in master_nodes]
        floor_a_rel = [float(ops.nodeAccel(node, 1)) for node in master_nodes]
        floor_a_abs = [value + ag for value in floor_a_rel]

        t_hist.append(t)
        floor_u_hist.append(floor_u)
        floor_a_rel_hist.append(floor_a_rel)
        floor_a_abs_hist.append(floor_a_abs)
        ground_accel_hist.append(ag)

        if realtime:
            target = wall_start + t
            sleep_s = target - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        # Emit only after the corresponding wall-clock instant has been reached.
        # This prevents the displayed numerical response from appearing one step
        # ahead of the physical experiment.
        if (i + 1) % emit_every == 0:
            emit("frame")

    # Send any samples accumulated since the previous display frame.
    emit("final")

    return {
        "t_hist": np.asarray(t_hist, dtype=float),
        "floor_u_hist": np.asarray(floor_u_hist, dtype=float),
        "floor_a_rel_hist": np.asarray(floor_a_rel_hist, dtype=float),
        "floor_a_abs_hist": np.asarray(floor_a_abs_hist, dtype=float),
        "ground_accel_hist": np.asarray(ground_accel_hist, dtype=float),
        "stopped": bool(stopped),
        "ok": int(ok),
    }
