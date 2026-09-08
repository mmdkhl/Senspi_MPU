# -*- coding: utf-8 -*-
"""Transient (time-history) analysis for the OpenSees frame model."""

from ..paths import output_str as _out
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops

from ..utils.formatters import r3


def set_rayleigh_damping_from_modal(modal_data, zeta, numModes):
    """
    Compute and set Rayleigh damping coefficients from modal data.

    Parameters
    ----------
    modal_data : dict
        Output from extract_modal_results().
    zeta : float
        Target damping ratio.
    numModes : int
        Number of modes (must be >= 2).

    Returns
    -------
    tuple : (alphaM, betaKinit)
    """
    if numModes < 2:
        raise ValueError("At least 2 modes are required to compute Rayleigh damping.")

    lam = modal_data["lam"]
    w1 = math.sqrt(lam[0])
    w2 = math.sqrt(lam[1])

    alphaM = 2.0 * zeta * w1 * w2 / (w1 + w2)
    betaKinit = 2.0 * zeta / (w1 + w2)

    ops.rayleigh(alphaM, 0.0, betaKinit, 0.0)
    return alphaM, betaKinit


def setup_dynamic_excitation(params):
    """
    Set up ground motion time series and uniform excitation pattern.

    Parameters
    ----------
    params : dict
        Must contain gmFile, dtGM, gmFactor.
    """
    gmFile = params["gmFile"]
    dtGM = params["dtGM"]
    gmFactor = params["gmFactor"]

    if not os.path.exists(gmFile):
        raise FileNotFoundError(f"Ground-motion file not found: {gmFile}")

    ops.timeSeries('Path', 20, '-dt', dtGM, '-filePath', gmFile, '-factor', gmFactor)
    ops.pattern('UniformExcitation', 20, 1, '-accel', 20)


def setup_recorders(ctx, prefix="", output_base=None):
    """
    Set up OpenSees node recorders for floor displacements and accelerations.

    Parameters
    ----------
    ctx : dict
        Model context returned by build_model().
    prefix : str
        Filename prefix for recorder output files.

    Returns
    -------
    int : roof master node tag
    """
    master_nodes = ctx["master_nodes"]
    roof_master = master_nodes[-1]

    # These filenames cross into the OpenSees C++ layer, where a relative path
    # is resolved against the process working directory and a wrong one is
    # silent — the file is simply written somewhere else. An absolute path
    # removes the possibility.
    ops.recorder('Node', '-file', _out(f'{prefix}time_floor_disp_X.out', output_base),
                 '-time', '-node', *master_nodes, '-dof', 1, 'disp')

    ops.recorder('Node', '-file', _out(f'{prefix}time_floor_accel_X.out', output_base),
                 '-time', '-node', *master_nodes, '-dof', 1, 'accel')

    ops.recorder('Node', '-file', _out(f'{prefix}time_roof_disp_X.out', output_base),
                 '-time', '-node', roof_master, '-dof', 1, 'disp')

    ops.recorder('Node', '-file', _out(f'{prefix}time_roof_accel_X.out', output_base),
                 '-time', '-node', roof_master, '-dof', 1, 'accel')

    return roof_master


def get_deformed_xyz(node_tag, sfac=1.0):
    """Get the deformed coordinates of a node."""
    x, y, z = ops.nodeCoord(node_tag)
    ux = ops.nodeDisp(node_tag, 1)
    uy = ops.nodeDisp(node_tag, 2)
    uz = ops.nodeDisp(node_tag, 3)
    return x + sfac * ux, y + sfac * uy, z + sfac * uz


def render_transient_to_png(transient_data, modal_data=None, overlay_response=None, sfac_anim=20):
    """
    Render Roof Displacement and Roof Acceleration time-history plots to PNG bytes.

    Parameters
    ----------
    transient_data : dict
        Output from run_transient_analysis_collect_data() with keys t_hist, u_hist, a_hist.
    modal_data : dict or None
        Unused — kept for API compatibility.
    overlay_response : dict or None
        Uncalibrated response to overlay (keys: t_hist, u_hist, a_hist).
    sfac_anim : float
        Unused — kept for API compatibility.

    Returns
    -------
    bytes : PNG image data
    """
    import io
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    t_hist = transient_data["t_hist"]
    u_hist = transient_data["u_hist"]
    a_hist = transient_data["a_hist"]

    fs = 15  # base font size — stays readable after ~50 % downscale

    fig = Figure(figsize=(7, 5.5))
    FigureCanvasAgg(fig)

    # --- Roof displacement (top) ---
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t_hist, u_hist, color="red", linewidth=1.8, label="Calibrated")
    if overlay_response is not None:
        ax1.plot(
            overlay_response["t_hist"], overlay_response["u_hist"],
            color="black", linestyle="--", linewidth=1.0, label="Uncalibrated",
        )
        ax1.legend(fontsize=fs - 2, loc="upper right")
    ax1.set_title("Roof Displacement", fontsize=fs + 1)
    ax1.set_xlabel("Time (s)", fontsize=fs)
    ax1.set_ylabel("Displacement (m)", fontsize=fs)
    if len(t_hist) > 0:
        ax1.set_xlim(0.0, float(t_hist[-1]))
    ax1.tick_params(labelsize=fs - 1)
    ax1.grid(True, alpha=0.25)

    # --- Roof acceleration (bottom) ---
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t_hist, a_hist, color="red", linewidth=1.8, label="Calibrated")
    if overlay_response is not None:
        ax2.plot(
            overlay_response["t_hist"], overlay_response["a_hist"],
            color="black", linestyle="--", linewidth=1.0, label="Uncalibrated",
        )
        ax2.legend(fontsize=fs - 2, loc="upper right")
    ax2.set_title("Roof Acceleration", fontsize=fs + 1)
    ax2.set_xlabel("Time (s)", fontsize=fs)
    ax2.set_ylabel("Acceleration (m/s²)", fontsize=fs)
    if len(t_hist) > 0:
        ax2.set_xlim(0.0, float(t_hist[-1]))
    ax2.tick_params(labelsize=fs - 1)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout(pad=1.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def run_transient_analysis_collect_data(params, modal_data, show_info=False,
                                        recorder_prefix="", output_base=None):
    """
    Run transient analysis and collect roof response data (no visualization).

    Parameters
    ----------
    params : dict
        All model/analysis parameters.
    modal_data : dict
        Output from extract_modal_results().
    show_info : bool
        Print progress to console.
    recorder_prefix : str
        Prefix for recorder output files.

    Returns
    -------
    dict with keys: t_hist, u_hist, a_hist
    """
    ctx = modal_data["ctx"]
    zeta = params["zeta"]
    gmFile = params["gmFile"]
    dtGM = params["dtGM"]

    alphaM, betaKinit = set_rayleigh_damping_from_modal(modal_data, zeta, params["numModes"])

    if show_info:
        print("Rayleigh damping:")
        print("alphaM    =", alphaM)
        print("betaKinit =", betaKinit)

    setup_dynamic_excitation(params)
    roof_master = setup_recorders(ctx, prefix=recorder_prefix, output_base=output_base)

    ops.wipeAnalysis()
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-10, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    with open(gmFile, 'r', encoding="utf-8") as f:
        npts = sum(1 for line in f if line.strip())

    if npts < 2:
        raise ValueError("Ground motion file must contain at least 2 acceleration points.")

    nSteps = npts - 1
    t_hist = []
    u_hist = []
    a_hist = []
    ok = 0

    for i in range(nSteps):
        ok = ops.analyze(1, dtGM)

        if ok != 0:
            print(f"Transient analysis failed at step {i+1}, time = {ops.getTime()}")
            break

        t_hist.append(ops.getTime())
        u_hist.append(ops.nodeDisp(roof_master, 1))
        a_hist.append(ops.nodeAccel(roof_master, 1))

    if ok == 0:
        print("Transient analysis completed successfully.")
        print("Final roof displacement X =", r3(ops.nodeDisp(roof_master, 1)), "m")
    else:
        print("Transient analysis failed before completion.")

    return {
        "t_hist": np.array(t_hist, dtype=float),
        "u_hist": np.array(u_hist, dtype=float),
        "a_hist": np.array(a_hist, dtype=float),
    }


def run_transient_analysis_stream(
    params,
    modal_data,
    show_info=False,
    recorder_prefix="",
    output_base=None,
    frame_callback=None,
    plot_every=10,
    anim_every=10,
    realtime=True,
    sfac_anim=20.0,
):
    """
    Run transient analysis one OpenSees step at a time and stream frames to a GUI.

    This is the PySide6-friendly version of the older live visualization
    workflow from DigitalTwin_V8.py. It does not create its own Matplotlib
    window. Instead, every few time steps it calls ``frame_callback(frame)``
    with roof response arrays and deformed 3D line segments.

    If ``realtime`` is True, the loop is deliberately slowed so that a
    20-second ground-motion file takes about 20 seconds to play in the GUI.

    Parameters
    ----------
    params : dict
        Model and analysis parameters. Must contain gmFile, dtGM, zeta,
        numModes, etc.
    modal_data : dict
        Output from extract_modal_results(). The OpenSees model corresponding
        to this modal_data must still be in memory.
    show_info : bool
        Print progress to console.
    recorder_prefix : str
        Prefix for recorder output files.
    frame_callback : callable or None
        Function receiving dictionaries with keys ``kind``, ``time``,
        ``t_hist``, ``u_hist``, ``a_hist`` and ``deformed_segments``.
    plot_every : int
        Emit roof response updates every this many OpenSees steps.
    anim_every : int
        Update 3D deformed geometry every this many OpenSees steps.
    realtime : bool
        If True, sleep between steps so wall-clock time follows analysis time.
    sfac_anim : float
        Deformation scale factor for the 3D animation.

    Returns
    -------
    dict
        Same structure as run_transient_analysis_collect_data():
        ``t_hist``, ``u_hist`` and ``a_hist`` as NumPy arrays.
    """
    ctx = modal_data["ctx"]
    zeta = params["zeta"]
    gmFile = params["gmFile"]
    dtGM = params["dtGM"]

    plot_every = max(1, int(plot_every))
    anim_every = max(1, int(anim_every))
    emit_every = max(1, min(plot_every, anim_every))

    alphaM, betaKinit = set_rayleigh_damping_from_modal(modal_data, zeta, params["numModes"])

    if show_info:
        print("Rayleigh damping:")
        print("alphaM    =", alphaM)
        print("betaKinit =", betaKinit)

    setup_dynamic_excitation(params)
    roof_master = setup_recorders(ctx, prefix=recorder_prefix, output_base=output_base)

    ops.wipeAnalysis()
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-10, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    with open(gmFile, 'r', encoding="utf-8") as f:
        npts = sum(1 for line in f if line.strip())

    if npts < 2:
        raise ValueError("Ground motion file must contain at least 2 acceleration points.")

    nSteps = npts - 1
    tmax = nSteps * dtGM
    vis_elems = list(ctx.get("vis_elems", []))

    t_hist = []
    u_hist = []
    a_hist = []
    ok = 0

    def _deformed_segments():
        segments = []
        for n1, n2 in vis_elems:
            x1, y1, z1 = get_deformed_xyz(n1, sfac_anim)
            x2, y2, z2 = get_deformed_xyz(n2, sfac_anim)
            segments.append(((float(x1), float(y1), float(z1)), (float(x2), float(y2), float(z2))))
        return segments

    def _emit_frame(kind="frame"):
        if frame_callback is None:
            return
        frame_callback({
            "kind": kind,
            "time": float(ops.getTime()),
            "tmax": float(tmax),
            "t_hist": np.asarray(t_hist, dtype=float),
            "u_hist": np.asarray(u_hist, dtype=float),
            "a_hist": np.asarray(a_hist, dtype=float),
            "deformed_segments": _deformed_segments(),
            "ok": int(ok),
        })

    # Send one initial, undeformed frame so the GUI can draw the model before
    # the first dynamic step is completed.
    _emit_frame(kind="frame")

    t0_wall = time.perf_counter()

    for i in range(nSteps):
        ok = ops.analyze(1, dtGM)

        if ok != 0:
            print(f"Transient analysis failed at step {i+1}, time = {ops.getTime()}")
            break

        t = ops.getTime()
        u = ops.nodeDisp(roof_master, 1)
        a = ops.nodeAccel(roof_master, 1)

        t_hist.append(float(t))
        u_hist.append(float(u))
        a_hist.append(float(a))

        if i % emit_every == 0:
            _emit_frame(kind="frame")

        if realtime:
            target_wall_time = t0_wall + float(t)
            sleep_time = target_wall_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Always emit one final frame, even if the final step number was not on the
    # regular plotting interval.
    _emit_frame(kind="final")

    if ok == 0:
        print("Transient analysis completed successfully.")
        print("Final roof displacement X =", r3(ops.nodeDisp(roof_master, 1)), "m")
    else:
        print("Transient analysis failed before completion.")

    return {
        "t_hist": np.array(t_hist, dtype=float),
        "u_hist": np.array(u_hist, dtype=float),
        "a_hist": np.array(a_hist, dtype=float),
    }


def run_transient_analysis_with_visualization(params, modal_data, show_info=False, overlay_response=None):
    """
    Run transient analysis with real-time matplotlib visualization.

    Parameters
    ----------
    params : dict
        All model/analysis parameters.
    modal_data : dict
        Output from extract_modal_results().
    show_info : bool
        Print progress to console.
    overlay_response : dict or None
        Uncalibrated response to overlay (keys: t_hist, u_hist, a_hist).
    """
    ctx = modal_data["ctx"]
    zeta = params["zeta"]
    gmFile = params["gmFile"]
    dtGM = params["dtGM"]

    alphaM, betaKinit = set_rayleigh_damping_from_modal(modal_data, zeta, params["numModes"])

    if show_info:
        print("Rayleigh damping:")
        print("alphaM    =", alphaM)
        print("betaKinit =", betaKinit)

    setup_dynamic_excitation(params)
    roof_master = setup_recorders(ctx)

    plt.ion()

    sfac_anim = 20
    plot_every = 10
    anim_every = 10

    vis_elems = ctx["vis_elems"]
    vis_nodes = ctx["vis_nodes"]
    node_xyz = ctx["node_xyz"]

    with open(gmFile, 'r', encoding="utf-8") as f:
        npts = sum(1 for line in f if line.strip())

    if npts < 2:
        raise ValueError("Ground motion file must contain at least 2 acceleration points.")

    Tmax = (npts - 1) * dtGM
    nSteps = npts - 1

    fig = plt.figure(figsize=(16, 7))
    try:
        fig.canvas.manager.window.move(100, 10)
    except Exception:
        pass

    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4])

    ax1 = fig.add_subplot(gs[0, 0])
    line1, = ax1.plot([], [], color='red', linewidth=1.7, label='Calibrated')
    ax1.set_title("Roof Displacement", fontsize=12)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Displacement (m)", fontsize=12)
    ax1.set_xlim(0.0, Tmax)
    ax1.grid(False)
    if overlay_response is not None:
        ax1.plot(
            overlay_response["t_hist"], overlay_response["u_hist"],
            color='black', linestyle='--', linewidth=0.7,
            label='Uncalibrated'
        )
        ax1.legend(fontsize=9, loc="upper right")

    ax2 = fig.add_subplot(gs[1, 0])
    line2, = ax2.plot([], [], color='red', linewidth=1.7, label='Calibrated')
    ax2.set_title("Roof Acceleration", fontsize=12)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Acceleration (m/s²)", fontsize=12)
    ax2.set_xlim(0.0, Tmax)
    ax2.grid(False)
    if overlay_response is not None:
        ax2.plot(
            overlay_response["t_hist"], overlay_response["a_hist"],
            color='black', linestyle='--', linewidth=0.7,
            label='Uncalibrated'
        )
        ax2.legend(fontsize=9, loc="upper right")

    ax_anim = fig.add_subplot(gs[:, 1], projection='3d')
    ax_anim.set_title("Calibrated Numerical Model 3D Response (OpenSees)")
    ax_anim.set_xlabel("X")
    ax_anim.set_ylabel("Y")
    ax_anim.set_zlabel("Z")

    ax_anim.grid(False)
    ax_anim.xaxis.pane.fill = False
    ax_anim.yaxis.pane.fill = False
    ax_anim.zaxis.pane.fill = False
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])
    ax_anim.set_zticks([])
    ax_anim.view_init(elev=25, azim=-70)

    all_x = [node_xyz[n][0] for n in vis_nodes]
    all_y = [node_xyz[n][1] for n in vis_nodes]
    all_z = [node_xyz[n][2] for n in vis_nodes]

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    zmin, zmax = min(all_z), max(all_z)

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    zmid = 0.5 * (zmin + zmax)

    half = 0.55 * max(xmax - xmin, ymax - ymin, zmax - zmin)

    ax_anim.set_xlim(xmid - half, xmid + half)
    ax_anim.set_ylim(ymid - half, ymid + half)
    ax_anim.set_zlim(zmid - half, zmid + half)
    ax_anim.set_box_aspect((1, 1, 1))

    for n1, n2 in vis_elems:
        x1, y1, z1 = node_xyz[n1]
        x2, y2, z2 = node_xyz[n2]
        ax_anim.plot([x1, x2], [y1, y2], [z1, z2],
                     linestyle='--', linewidth=1.0, color='0.6')

    defo_lines = []
    for _ in vis_elems:
        ln, = ax_anim.plot([], [], [], color='navy', linewidth=3.0)
        defo_lines.append(ln)

    def update_live_model():
        for k, (n1, n2) in enumerate(vis_elems):
            x1, y1, z1 = get_deformed_xyz(n1, sfac_anim)
            x2, y2, z2 = get_deformed_xyz(n2, sfac_anim)
            defo_lines[k].set_data([x1, x2], [y1, y2])
            defo_lines[k].set_3d_properties([z1, z2])

    fig.tight_layout()

    ops.wipeAnalysis()
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-10, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    t_hist = []
    u_hist = []
    a_hist = []

    ok = 0

    update_live_model()
    fig.canvas.draw()
    fig.canvas.flush_events()

    t0_wall = time.perf_counter()

    for i in range(nSteps):
        ok = ops.analyze(1, dtGM)

        if ok != 0:
            print(f"Transient analysis failed at step {i+1}, time = {ops.getTime()}")
            break

        t = ops.getTime()
        u = ops.nodeDisp(roof_master, 1)
        a = ops.nodeAccel(roof_master, 1)

        t_hist.append(t)
        u_hist.append(u)
        a_hist.append(a)

        if i % plot_every == 0:
            line1.set_data(t_hist, u_hist)
            ax1.relim()
            ax1.autoscale_view(scalex=False, scaley=True)

            line2.set_data(t_hist, a_hist)
            ax2.relim()
            ax2.autoscale_view(scalex=False, scaley=True)

        if i % anim_every == 0:
            update_live_model()

        if i % min(plot_every, anim_every) == 0:
            fig.canvas.draw()
            fig.canvas.flush_events()

        target_wall_time = t0_wall + t
        sleep_time = target_wall_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    if ok == 0:
        print("Transient analysis completed successfully.")
        print("Final roof displacement X =", r3(ops.nodeDisp(roof_master, 1)), "m")
    else:
        print("Transient analysis failed before completion.")

    plt.ioff()
    plt.show()
