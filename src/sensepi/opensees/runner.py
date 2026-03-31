"""OpenSees execution backend for the Digital Twin tab."""

from __future__ import annotations

import math
from pathlib import Path

from .models import (
    DigitalTwinAnalysisParams,
    DigitalTwinAnalysisResult,
    DigitalTwinModalResult,
    DigitalTwinTransientResult,
)


class DigitalTwinDependencyError(RuntimeError):
    """Raised when optional OpenSees dependencies are not available."""


def _load_ground_motion(path: Path) -> list[float]:
    values: list[float] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            text = raw.strip()
            if not text:
                continue
            values.append(float(text.replace(",", ".")))
    if not values:
        raise ValueError(f"Ground-motion file is empty: {path}")
    return values


def _write_modal_outputs(result: DigitalTwinModalResult, output_dir: Path) -> None:
    periods_path = output_dir / "periods.out"
    with periods_path.open("w", encoding="utf-8") as fh:
        fh.write("Mode  Lambda(rad^2/s^2)  Frequency(Hz)  Period(s)\n")
        for index, (lam, freq, period) in enumerate(
            zip(result.lambdas, result.frequencies_hz, result.periods_s),
            start=1,
        ):
            fh.write(f"{index}  {lam}  {freq}  {period}\n")

    mode_shapes_path = output_dir / "mode_shapes.out"
    with mode_shapes_path.open("w", encoding="utf-8") as fh:
        fh.write("Mode  Node  UX  UY  UZ  RX  RY  RZ\n")
        for mode, per_node in result.mode_shapes.items():
            for node, values in per_node.items():
                ux, uy, uz, rx, ry, rz = values
                fh.write(f"{mode}  {node}  {ux}  {uy}  {uz}  {rx}  {ry}  {rz}\n")


def _write_transient_outputs(result: DigitalTwinTransientResult, output_dir: Path) -> None:
    roof_disp_path = output_dir / "time_roof_disp_X.out"
    with roof_disp_path.open("w", encoding="utf-8") as fh:
        for time_s, value in zip(result.time_s, result.roof_disp_x_m):
            fh.write(f"{time_s} {value}\n")

    roof_accel_path = output_dir / "time_roof_accel_X.out"
    with roof_accel_path.open("w", encoding="utf-8") as fh:
        for time_s, value in zip(result.time_s, result.roof_accel_x_ms2):
            fh.write(f"{time_s} {value}\n")

    floor_disp_path = output_dir / "time_floor_disp_X.out"
    with floor_disp_path.open("w", encoding="utf-8") as fh:
        for row_index, time_s in enumerate(result.time_s):
            row = [f"{time_s}"]
            for node in sorted(result.floor_disp_x_m):
                row.append(str(result.floor_disp_x_m[node][row_index]))
            fh.write(" ".join(row) + "\n")

    floor_accel_path = output_dir / "time_floor_accel_X.out"
    with floor_accel_path.open("w", encoding="utf-8") as fh:
        for row_index, time_s in enumerate(result.time_s):
            row = [f"{time_s}"]
            for node in sorted(result.floor_accel_x_ms2):
                row.append(str(result.floor_accel_x_ms2[node][row_index]))
            fh.write(" ".join(row) + "\n")


def run_digital_twin_analysis(
    params: DigitalTwinAnalysisParams,
) -> DigitalTwinAnalysisResult:
    """Run the generalized OpenSees digital twin analysis."""

    params.validate()
    params.output_dir.mkdir(parents=True, exist_ok=True)

    gm_values = _load_ground_motion(params.gm_file)

    try:
        import openseespy.opensees as ops
    except Exception as exc:  # pragma: no cover - dependency guard
        raise DigitalTwinDependencyError(
            "OpenSeesPy is not installed. Install `openseespy` to run the Digital Twin analysis."
        ) from exc

    ops.wipe()
    ops.model("Basic", "-ndm", 3, "-ndf", 6)

    n_story = len(params.story_heights)
    z_levels = [0.0]
    for height in params.story_heights:
        z_levels.append(z_levels[-1] + float(height))

    e = float(params.youngs_modulus_pa)
    nu = float(params.poissons_ratio)
    g = e / (2.0 * (1.0 + nu))

    t_column = float(params.column_thickness_m)
    b_column = float(params.column_width_m)
    a_column = t_column * b_column
    iy_column = b_column * t_column**3 / 12.0
    iz_column = t_column * b_column**3 / 12.0
    j_column = (1.0 / 3.0) * b_column * t_column**3

    b_beam = float(params.beam_width_m)
    h_beam = float(params.beam_height_m)
    a_beam = b_beam * h_beam
    iy_beam = h_beam * b_beam**3 / 12.0
    iz_beam = b_beam * h_beam**3 / 12.0
    j_beam = (1.0 / 3.0) * h_beam * b_beam**3

    lx = float(params.lx_m)
    ly = float(params.ly_m)
    base_nodes = [1, 2, 3, 4]
    base_coords = [
        (0.0, 0.0, 0.0),
        (lx, 0.0, 0.0),
        (lx, ly, 0.0),
        (0.0, ly, 0.0),
    ]
    for node, (x, y, z) in zip(base_nodes, base_coords):
        ops.node(node, x, y, z)

    story_node_tags: dict[int, list[int]] = {}
    master_nodes: list[int] = []
    for story in range(1, n_story + 1):
        z = z_levels[story]
        nds = [story * 10 + 1, story * 10 + 2, story * 10 + 3, story * 10 + 4]
        coords = [
            (0.0, 0.0, z),
            (lx, 0.0, z),
            (lx, ly, z),
            (0.0, ly, z),
        ]
        for node, (x, y, zc) in zip(nds, coords):
            ops.node(node, x, y, zc)
        story_node_tags[story] = nds
        master = 1000 + story
        ops.node(master, lx / 2.0, ly / 2.0, z)
        master_nodes.append(master)

    for node in base_nodes:
        ops.fix(node, 1, 1, 1, 1, 1, 1)
    for master in master_nodes:
        ops.fix(master, 0, 0, 1, 1, 1, 0)
    for story in range(1, n_story + 1):
        ops.rigidDiaphragm(3, 1000 + story, *story_node_tags[story])

    floor_jm = [
        mass * (lx**2 + ly**2) / 12.0 for mass in params.floor_masses
    ]
    for story in range(1, n_story + 1):
        mass = float(params.floor_masses[story - 1])
        ops.mass(1000 + story, mass, mass, 0.0, 0.0, 0.0, floor_jm[story - 1])

    ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
    ops.geomTransf("Linear", 2, 0.0, 1.0, 0.0)
    ops.geomTransf("Linear", 3, 1.0, 0.0, 0.0)

    col_tag = 1
    for index in range(4):
        ops.element(
            "elasticBeamColumn",
            col_tag,
            base_nodes[index],
            story_node_tags[1][index],
            a_column,
            e,
            g,
            j_column,
            iy_column,
            iz_column,
            1,
        )
        col_tag += 1
    for story in range(1, n_story):
        lower = story_node_tags[story]
        upper = story_node_tags[story + 1]
        for index in range(4):
            ops.element(
                "elasticBeamColumn",
                col_tag,
                lower[index],
                upper[index],
                a_column,
                e,
                g,
                j_column,
                iy_column,
                iz_column,
                1,
            )
            col_tag += 1

    for story in range(1, n_story + 1):
        n1, n2, n3, n4 = story_node_tags[story]
        base_tag = story * 100
        beam_rows = [
            (base_tag + 1, n1, n2, 2),
            (base_tag + 2, n4, n3, 2),
            (base_tag + 3, n1, n4, 3),
            (base_tag + 4, n2, n3, 3),
        ]
        for tag, i_node, j_node, transf in beam_rows:
            ops.element(
                "elasticBeamColumn",
                tag,
                i_node,
                j_node,
                a_beam,
                e,
                g,
                j_beam,
                iy_beam,
                iz_beam,
                transf,
            )

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for story in range(1, n_story + 1):
        pcol = float(params.floor_masses[story - 1]) * 9.81 / 4.0
        for node in story_node_tags[story]:
            ops.load(node, 0.0, 0.0, -pcol, 0.0, 0.0, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-12, 50, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    gravity_ok = ops.analyze(1)
    if gravity_ok != 0:
        ops.wipe()
        raise RuntimeError("Gravity analysis failed.")
    ops.loadConst("-time", 0.0)

    lam = list(ops.eigen("-fullGenLapack", int(params.num_modes)))
    periods: list[float] = []
    freqs: list[float] = []
    mode_shapes: dict[int, dict[int, tuple[float, float, float, float, float, float]]] = {}
    for mode_index, eigenvalue in enumerate(lam, start=1):
        omega = math.sqrt(float(eigenvalue))
        periods.append((2.0 * math.pi / omega) if omega > 0.0 else 0.0)
        freqs.append(omega / (2.0 * math.pi))
        per_node: dict[int, tuple[float, float, float, float, float, float]] = {}
        for node in master_nodes:
            per_node[node] = tuple(
                float(ops.nodeEigenvector(node, mode_index, dof))
                for dof in range(1, 7)
            )
        mode_shapes[mode_index] = per_node

    modal_result = DigitalTwinModalResult(
        periods_s=periods,
        frequencies_hz=freqs,
        lambdas=[float(value) for value in lam],
        master_nodes=master_nodes,
        mode_shapes=mode_shapes,
    )
    _write_modal_outputs(modal_result, params.output_dir)

    if len(lam) >= 2:
        w1 = math.sqrt(float(lam[0]))
        w2 = math.sqrt(float(lam[1]))
        alpha_m = 2.0 * params.damping_ratio * w1 * w2 / (w1 + w2)
        beta_kinit = 2.0 * params.damping_ratio / (w1 + w2)
        ops.rayleigh(alpha_m, 0.0, beta_kinit, 0.0)

    dt = float(params.dt_seconds)
    path_tag = 20
    pattern_tag = 20
    ops.timeSeries(
        "Path",
        path_tag,
        "-dt",
        dt,
        "-values",
        *[float(value) * float(params.gm_factor) for value in gm_values],
    )
    ops.pattern("UniformExcitation", pattern_tag, 1, "-accel", path_tag)

    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-10, 100, 0)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    roof_master = master_nodes[-1]
    time_hist: list[float] = []
    roof_disp_hist: list[float] = []
    roof_accel_hist: list[float] = []
    floor_disp_hist: dict[int, list[float]] = {node: [] for node in master_nodes}
    floor_accel_hist: dict[int, list[float]] = {node: [] for node in master_nodes}

    for step in range(len(gm_values) - 1):
        ok = ops.analyze(1, dt)
        if ok != 0:
            ops.wipe()
            raise RuntimeError(
                f"Transient analysis failed at step {step + 1}, time={ops.getTime():.4f}s."
            )

        current_time = float(ops.getTime())
        time_hist.append(current_time)
        roof_disp_hist.append(float(ops.nodeDisp(roof_master, 1)))
        roof_accel_hist.append(float(ops.nodeAccel(roof_master, 1)))
        for node in master_nodes:
            floor_disp_hist[node].append(float(ops.nodeDisp(node, 1)))
            floor_accel_hist[node].append(float(ops.nodeAccel(node, 1)))

    transient_result = DigitalTwinTransientResult(
        time_s=time_hist,
        roof_disp_x_m=roof_disp_hist,
        roof_accel_x_ms2=roof_accel_hist,
        floor_disp_x_m=floor_disp_hist,
        floor_accel_x_ms2=floor_accel_hist,
    )
    _write_transient_outputs(transient_result, params.output_dir)
    ops.wipe()

    return DigitalTwinAnalysisResult(
        params=params,
        modal=modal_result,
        transient=transient_result,
        output_dir=params.output_dir,
    )
