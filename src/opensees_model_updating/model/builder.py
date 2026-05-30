# -*- coding: utf-8 -*-
"""
Frame model builder - wraps build_model() from DigitalTwin_V8.py.

Creates a 3D 1-bay N-story aluminum frame in OpenSeesPy.
Units: N, m, sec
"""

import openseespy.opensees as ops

from ..utils.formatters import (
    deep_round,
    story_column_layout_to_text,
    column_orientation_layout_to_text,
    additional_masses_to_text,
)


def build_model(params, show_info=False):
    """
    Build the 3D frame OpenSees model.

    Parameters
    ----------
    params : dict
        All model parameters. Required keys:
        Lx, Ly, nStory, story_heights, story_column_layout,
        column_orientation_layout, additional_masses,
        E, nu, t_column, b_column, b_beam, h_beam, floor_masses
    show_info : bool
        Print model summary to console.

    Returns
    -------
    dict : model context (ctx) with node/element data
    """
    ops.wipe()
    ops.model('Basic', '-ndm', 3, '-ndf', 6)

    Lx = params["Lx"]
    Ly = params["Ly"]
    story_heights = params["story_heights"]
    nStory = params["nStory"]
    story_column_layout = params["story_column_layout"]
    column_orientation_layout = params.get(
        "column_orientation_layout",
        {story: {c: "weak" for c in (1, 2, 3, 4)} for story in range(1, nStory + 1)}
    )
    additional_masses = params.get(
        "additional_masses",
        {story: [0.0, 0.0, 0.0, 0.0, 0.0] for story in range(1, nStory + 1)}
    )

    E = params["E"]
    nu = params["nu"]
    G = E / (2.0 * (1.0 + nu))

    t_column = params["t_column"]
    b_column = params["b_column"]
    b_beam = params["b_beam"]
    h_beam = params["h_beam"]

    A = t_column * b_column
    Iy = b_column * t_column**3 / 12.0
    Iz = t_column * b_column**3 / 12.0
    J = (1.0 / 3.0) * b_column * t_column**3

    Ab = b_beam * h_beam
    Iyb = h_beam * b_beam**3 / 12.0
    Izb = b_beam * h_beam**3 / 12.0
    Jb = (1.0 / 3.0) * h_beam * b_beam**3

    floor_masses = params["floor_masses"]
    total_dynamic_masses = [
        floor_masses[story - 1] + sum(additional_masses.get(story, [0.0, 0.0, 0.0, 0.0, 0.0]))
        for story in range(1, nStory + 1)
    ]
    floor_weights = [m * 9.81 for m in total_dynamic_masses]
    floor_Jm = [floor_masses[i] * (Lx**2 + Ly**2) / 12.0 for i in range(nStory)]

    z_levels = [0.0]
    for h in story_heights:
        z_levels.append(z_levels[-1] + h)

    base_nodes = [1, 2, 3, 4]
    base_coords = [
        (0.0, 0.0, 0.0),
        (Lx,  0.0, 0.0),
        (Lx,  Ly,  0.0),
        (0.0, Ly,  0.0),
    ]

    for nd, (x, y, z) in zip(base_nodes, base_coords):
        ops.node(nd, x, y, z)

    story_node_tags = {}
    master_nodes = []
    mass_assignment = {}
    gravity_corner_masses_by_story = {}

    for story in range(1, nStory + 1):
        z = z_levels[story]
        nds = [story * 10 + 1, story * 10 + 2, story * 10 + 3, story * 10 + 4]
        crds = [
            (0.0, 0.0, z),
            (Lx,  0.0, z),
            (Lx,  Ly,  z),
            (0.0, Ly,  z),
        ]

        for nd, (x, y, zc) in zip(nds, crds):
            ops.node(nd, x, y, zc)

        story_node_tags[story] = nds

        master = 1000 + story
        ops.node(master, Lx / 2.0, Ly / 2.0, z)
        master_nodes.append(master)

    for nd in base_nodes:
        ops.fix(nd, 1, 1, 1, 1, 1, 1)

    for master in master_nodes:
        ops.fix(master, 0, 0, 1, 1, 1, 0)

    for story in range(1, nStory + 1):
        ops.rigidDiaphragm(3, 1000 + story, *story_node_tags[story])

    # -------------------------------------------------------------------------
    # Mass placement
    # -------------------------------------------------------------------------
    # The floor mass entered in the Basic Model tab is treated as the
    # self-weight/structural mass of each floor. It is always assigned at the
    # diaphragm master node at the center of the slab and is never redistributed
    # to the corners.
    #
    # The table in the Mass + Analysis tab is now for additional masses in kg.
    # These additional masses can be placed at the center or at C1-C4. They are
    # included in modal/transient analysis, but they are not changed by the
    # calibration optimizer.
    for story in range(1, nStory + 1):
        self_mass = floor_masses[story - 1]
        extra = additional_masses.get(story, [0.0, 0.0, 0.0, 0.0, 0.0])
        extra_center_mass = extra[0]
        extra_corner_masses = extra[1:]

        master = 1000 + story

        center_translational_mass = self_mass + extra_center_mass
        center_Jm = self_mass * (Lx**2 + Ly**2) / 12.0

        ops.mass(
            master,
            center_translational_mass, center_translational_mass, 0.0,
            0.0, 0.0, center_Jm
        )

        for nd, m_corner in zip(story_node_tags[story], extra_corner_masses):
            ops.mass(
                nd,
                m_corner, m_corner, 0.0,
                0.0, 0.0, 0.0
            )

        gravity_corner_masses = [
            (self_mass + extra_center_mass) / 4.0 + m_extra
            for m_extra in extra_corner_masses
        ]
        gravity_corner_masses_by_story[story] = gravity_corner_masses

        mass_assignment[story] = {
            "self_weight_mass_center_kg": self_mass,
            "additional_masses_center_c1_c2_c3_c4_kg": extra[:],
            "total_dynamic_mass_kg": self_mass + sum(extra),
            "center_node": master,
            "center_translational_mass_kg": center_translational_mass,
            "corner_nodes": story_node_tags[story][:],
            "additional_corner_masses_kg": extra_corner_masses,
            "gravity_corner_masses_kg": gravity_corner_masses,
            "center_rotational_mass_Jz": center_Jm,
        }

    # Geometric transformations
    # Tag 1: original vertical-column orientation. For a vertical column,
    # vecxz=(1,0,0) makes global X response use column Iy (weak-axis response).
    ops.geomTransf('Linear', 1, 1.0, 0.0, 0.0)

    # Beam transformations
    ops.geomTransf('Linear', 2, 0.0, 1.0, 0.0)
    ops.geomTransf('Linear', 3, 1.0, 0.0, 0.0)

    # Tag 4: strong-axis stiffness in X.
    # For a vertical element, vecxz=(0,1,0) makes global X response use column Iz.
    ops.geomTransf('Linear', 4, 0.0, 1.0, 0.0)

    col_data = []
    col_tag = 1

    for story in range(1, nStory + 1):
        if story == 1:
            lower_nodes = base_nodes
            upper_nodes = story_node_tags[1]
        else:
            lower_nodes = story_node_tags[story - 1]
            upper_nodes = story_node_tags[story]

        existing_cols = story_column_layout.get(story, [1, 2, 3, 4])

        for col_id in existing_cols:
            idx = col_id - 1
            iNode = lower_nodes[idx]
            jNode = upper_nodes[idx]
            orientation = column_orientation_layout.get(story, {}).get(col_id, "weak")
            col_data.append((col_tag, iNode, jNode, story, col_id, orientation))
            col_tag += 1

    for tag, iNode, jNode, _, _, orientation in col_data:
        transfTag = 1 if orientation == "weak" else 4
        ops.element('elasticBeamColumn', tag, iNode, jNode, A, E, G, J, Iy, Iz, transfTag)

    beam_data = []
    for story in range(1, nStory + 1):
        n1, n2, n3, n4 = story_node_tags[story]
        base_tag = story * 100
        beam_data.extend([
            (base_tag + 1, n1, n2, 2, story, "X-bottom"),
            (base_tag + 2, n4, n3, 2, story, "X-top"),
            (base_tag + 3, n1, n4, 3, story, "Y-left"),
            (base_tag + 4, n2, n3, 3, story, "Y-right"),
        ])

    for tag, iNode, jNode, transfTag, _, _ in beam_data:
        ops.element('elasticBeamColumn', tag, iNode, jNode, Ab, E, G, Jb, Iyb, Izb, transfTag)

    vis_elems = [(i, j) for _, i, j, _, _, _ in col_data] + [(i, j) for _, i, j, _, _, _ in beam_data]
    vis_nodes = base_nodes[:]
    for story in range(1, nStory + 1):
        vis_nodes.extend(story_node_tags[story])

    node_xyz = {nd: ops.nodeCoord(nd) for nd in vis_nodes}

    if show_info:
        print("\n=== MODEL SUMMARY ===")
        print("Stories:", nStory)
        print("Story heights:", story_heights)
        print("Self-weight floor masses:", floor_masses)
        print("Story column layout:", story_column_layout_to_text(story_column_layout, nStory))
        print("Column orientation layout:", column_orientation_layout_to_text(column_orientation_layout, nStory))
        print("Additional masses:", additional_masses_to_text(additional_masses, nStory))
        print("Mass assignment:", deep_round(mass_assignment))

    return {
        "z_levels": z_levels,
        "floor_weights": floor_weights,
        "floor_Jm": floor_Jm,
        "gravity_corner_masses_by_story": gravity_corner_masses_by_story,
        "base_nodes": base_nodes,
        "story_node_tags": story_node_tags,
        "master_nodes": master_nodes,
        "col_data": col_data,
        "beam_data": beam_data,
        "vis_elems": vis_elems,
        "vis_nodes": vis_nodes,
        "node_xyz": node_xyz,
        "mass_assignment": mass_assignment,
    }
